from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    Reasoning,
    TableReason,
    ColumnReason,
)
from app.services.neo4j_service import Neo4jService
from app.services.prompt_builder import PromptBuilder
from app.services.llm_service import LLMService
from app.services.postgres_service import PostgresService
from app.services.table_index_service import TableIndexService
from app.services.graph_rag_service import GraphRAGService
from app.services.tot_service import ToTService
from app.services.embedding_service import EmbeddingService
from app.utils.sql_validator import validate_sql, validate_columns_in_sql
from app.utils.logger import log_step

router = APIRouter()

neo4j_service = Neo4jService()
prompt_builder = PromptBuilder()
llm_service = LLMService()
postgres_service = PostgresService()
table_index_service = TableIndexService(neo4j_service)
graph_rag_service = GraphRAGService(neo4j_service)
tot_service = ToTService(llm_service)
embedding_service = EmbeddingService()

MAX_RETRIES = 5
USE_TOT = False


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse, responses={
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
})
async def query(request: QueryRequest) -> QueryResponse:
    log_step("START", f"Received question", question=request.question, top_k=request.top_k)

    try:
        schema_context = await neo4j_service.get_schema_context(request.question)
    except Exception as exc:
        log_step("ERROR", "Neo4j schema fetch failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="Schema service unavailable",
                detail=str(exc),
            ).model_dump(),
        )

    table_names = [f"{t.schema_name}.{t.table_name}" for t in schema_context.tables]
    log_step("NEO4J", f"Found {len(schema_context.tables)} tables", tables=table_names)

    # --- Build reasoning ---
    reasoning_tables: list[TableReason] = []
    reasoning_columns: dict[str, list[ColumnReason]] = {}
    retry_log: list[str] = []

    q_vec = None
    if embedding_service.is_ready:
        try:
            q_vec = await embedding_service.embed_question(request.question)
        except Exception:
            pass

    for table in schema_context.tables:
        tier_label = "REPORT (t3_)" if table.table_name.startswith("t3_") else \
                     "MID (t2_)" if table.table_name.startswith("t2_") else "RAW (t1_)"
        description = table.description or ""
        top_cols: list[str] = []
        col_scores: list[ColumnReason] = []
        try:
            cols_with_scores = await table_index_service.get_relevant_columns_with_scores(
                request.question, table.table_name, top_k=8, q_vec=q_vec
            )
            top_cols = [c for c, _ in cols_with_scores]
            col_scores = [
                ColumnReason(
                    column=c,
                    score=round(s, 4),
                    reason=f"Semantic match score {round(s, 4)} with question — column name/description aligns with '{request.question}'"
                ) for c, s in cols_with_scores
            ]
        except Exception:
            pass

        match_words = []
        for c in top_cols:
            for w in request.question.lower().split():
                if w in c.lower() or c.lower() in w:
                    match_words.append(f"'{c}' contains keyword '{w}'")
        match_str = "; ".join(match_words[:3]) if match_words else f"top semantic match for '{request.question}'"

        reasoning_tables.append(TableReason(
            table=f"silver_layer.{table.table_name}",
            tier=tier_label,
            description=description or "No description",
            top_columns=top_cols,
            reason=f"{tier_label} — {description[:80] if description else 'no description'}. "
                   f"Relevant columns: {', '.join(top_cols[:3]) or '(none)'}. "
                   f"Basis: {match_str}"
        ))
        if col_scores:
            reasoning_columns[table.table_name] = col_scores

    relevant_columns: dict[str, list[str]] = {}
    for table in schema_context.tables:
        try:
            cols = await table_index_service.get_relevant_columns(
                request.question, table.table_name, top_k=8, q_vec=q_vec
            )
            if cols:
                relevant_columns[table.table_name] = cols
        except Exception:
            pass

    if relevant_columns:
        filtered_context = neo4j_service.filter_columns_by_relevance(
            schema_context, relevant_columns
        )
        log_step("COLUMNS", f"Filtered to relevant columns per table",
                 tables_with_filters=list(relevant_columns.keys()))
    else:
        filtered_context = schema_context

    enrichment = None
    try:
        enrichment = await graph_rag_service.enrich(
            [t.table_name for t in schema_context.tables]
        )
    except Exception as exc:
        log_step("GRAPH_RAG", "Enrichment failed, continuing without it", error=str(exc))

    system_prompt = prompt_builder.build_system_prompt()
    user_prompt = prompt_builder.build_user_prompt(
        request.question, filtered_context, enrichment
    )
    log_step(
        "PROMPT",
        f"Built prompts (system={len(system_prompt)} chars, "
        f"user={len(user_prompt)} chars)",
    )

    known_columns = _build_known_columns(schema_context)
    known_tables = _build_known_tables(schema_context)
    table_columns = _build_table_columns(schema_context)

    generated_sql: str | None = None

    if USE_TOT:
        log_step("TOT", "Using Tree of Thoughts SQL generation")
        try:
            generated_sql = await tot_service.generate_best(
                system_prompt,
                user_prompt,
                known_columns,
                known_tables,
                table_columns,
                schema_context,
                num_candidates=3,
            )
        except Exception as exc:
            log_step("TOT", "ToT generation failed", error=str(exc))

    if not generated_sql:
        last_error: str | None = None
        last_auto_fixed: str | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                retry_log.append(f"Retry {attempt - 1}: {last_error or 'unknown error'}")
            correction_hint = ""
            if last_error:
                correction_hint = (
                    f"\n\nCORRECTION (attempt {attempt - 1} errors):\n"
                    f"{last_error}\n\n"
                    f"IMPORTANT: Fix ALL column names to match the schema. "
                    f"Only use columns listed under the table you query. "
                    f"Return ONLY corrected SQL."
                )
            current_prompt = user_prompt + correction_hint

            try:
                generated_sql = await llm_service.generate_sql(
                    system_prompt, current_prompt
                )
            except ValueError as exc:
                log_step(
                    "RETRY",
                    f"LLM unable to generate SQL on attempt {attempt}",
                    error=str(exc),
                )
                last_error = str(exc)
                continue
            except Exception as exc:
                log_step("ERROR", "LLM service error", error=str(exc))
                raise HTTPException(
                    status_code=500,
                    detail=ErrorResponse(
                        error="LLM service error", detail=str(exc)
                    ).model_dump(),
                )

            log_step(
                "LLM",
                f"SQL generated (attempt {attempt})",
                sql=generated_sql.replace("\n", " "),
            )

            generated_sql = _auto_qualify_tables(generated_sql)

            try:
                validate_sql(generated_sql)
            except ValueError as exc:
                log_step("RETRY", f"SQL validation failed", error=str(exc))
                last_error = str(exc)
                continue

            try:
                generated_sql = validate_columns_in_sql(
                    generated_sql,
                    known_columns,
                    known_tables,
                    table_columns,
                )
            except ValueError as exc:
                log_step("RETRY", f"Column validation failed", error=str(exc))
                last_error = str(exc)
                last_auto_fixed = _try_auto_fix_sql(
                    generated_sql, known_columns, known_tables, table_columns
                )
                continue

            break
        else:
            if last_auto_fixed:
                log_step("RETRY", "All LLM retries failed, trying auto-fixed SQL as fallback", sql=last_auto_fixed.replace("\n", " "))
                generated_sql = last_auto_fixed
            elif last_error and ("UNABLE_TO_GENERATE" in (last_error or "") or "could not generate" in (last_error or "")):
                log_step("RETRY", "t3_-only schema failed, fetching ALL t3_ tables from Neo4j")
                retry_log.append("Fallback: t3_-only schema insufficient, fetched all available t3_ tables")
                try:
                    broad_context = await neo4j_service.get_all_t3_tables()
                    broad_filtered = broad_context
                    broad_enrich = await graph_rag_service.enrich(
                        [t.table_name for t in broad_context.tables]
                    ) if enrichment is None else enrichment
                    broad_user = prompt_builder.build_user_prompt(
                        request.question, broad_filtered, broad_enrich
                    )
                    broad_known_cols = _build_known_columns(broad_context)
                    broad_known_tbls = _build_known_tables(broad_context)
                    broad_tbl_cols = _build_table_columns(broad_context)
                    log_step("RETRY", f"Retrying with {len(broad_context.tables)} tables (was {len(schema_context.tables)})")
                    for attempt2 in range(1, 4):
                        hint = f"\n\nYou previously responded with UNABLE_TO_GENERATE. The schema now includes upstream tables. Generate the SQL.\n"
                        sql2 = await llm_service.generate_sql(system_prompt, broad_user + hint)
                        if sql2:
                            sql2 = _auto_qualify_tables(sql2)
                            try:
                                validate_sql(sql2)
                                sql2 = validate_columns_in_sql(sql2, broad_known_cols, broad_known_tbls, broad_tbl_cols)
                                generated_sql = sql2
                                log_step("RETRY", f"Broad schema fallback succeeded on attempt {attempt2}", sql=sql2.replace("\n", " "))
                                break
                            except ValueError:
                                continue
                except Exception as broad_exc:
                    log_step("RETRY", "Broad schema fallback failed", error=str(broad_exc))
                if not generated_sql:
                    raise HTTPException(
                        status_code=422,
                        detail=ErrorResponse(
                            error=f"Failed to generate valid SQL after {MAX_RETRIES} attempts (including broad schema fallback)",
                            detail=last_error,
                        ).model_dump(),
                    )
            else:
                raise HTTPException(
                    status_code=422,
                    detail=ErrorResponse(
                        error=f"Failed to generate valid SQL after {MAX_RETRIES} attempts",
                        detail=last_error,
                    ).model_dump(),
                )

    results: list[dict] = []
    for pg_attempt in range(1, 3):
        try:
            results = await postgres_service.execute_query(
                generated_sql, request.top_k
            )
            break
        except Exception as exc:
            if pg_attempt == 1:
                retry_log.append(f"PG error ({pg_attempt}): {str(exc)[:100]}")
            log_step(
                "ERROR",
                f"PostgreSQL execution failed (attempt {pg_attempt})",
                sql=generated_sql.replace("\n", " "),
                error=str(exc),
            )
            if pg_attempt == 1:
                fixed = _fix_table_aliases(generated_sql)
                if fixed != generated_sql:
                    log_step("RETRY", "Attempting alias fix", fixed_sql=fixed.replace("\n", " "))
                    generated_sql = fixed
                    continue
                alt_fixed = _try_auto_fix_sql(
                    generated_sql, known_columns, known_tables, table_columns
                )
                if alt_fixed is not None and alt_fixed != generated_sql:
                    log_step("RETRY", "Attempting column auto-fix", fixed_sql=alt_fixed.replace("\n", " "))
                    generated_sql = alt_fixed
                    continue
                pg_fixed = _fix_pg_errors(generated_sql, str(exc), table_columns)
                if pg_fixed != generated_sql:
                    log_step("RETRY", "Attempting PG-level fix", fixed_sql=pg_fixed.replace("\n", " "))
                    generated_sql = pg_fixed
                    continue
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error="Query execution failed",
                    detail=f"SQL: {generated_sql}\nError: {exc}",
                ).model_dump(),
            )

    log_step("POSTGRES", f"Query returned {len(results)} rows")

    final_explanation: str | None = None
    try:
        used_tables_list = []
        import re as _re
        for m in _re.finditer(r'(?:FROM|JOIN)\s+\w+\.(\w+)', generated_sql, _re.IGNORECASE):
            used_tables_list.append(m.group(1))
        used_cols_list = []
        for tbl_context in schema_context.tables:
            for c in tbl_context.columns:
                if c.name.lower() in generated_sql.lower():
                    used_cols_list.append(f"{tbl_context.table_name}.{c.name}")
        used_cols_list = list(set(used_cols_list))

        ctx_summary = "\n".join(
            f"{t.table_name} ({t.description or 'no desc'}): "
            + ", ".join(c.name for c in t.columns[:8])
            for t in schema_context.tables
        )

        explain_prompt = (
            f"Question: {request.question}\n\n"
            f"SQL generated:\n{generated_sql}\n\n"
            f"Table(s) used: {', '.join(used_tables_list)}\n"
            f"Column(s) used: {', '.join(used_cols_list)}\n\n"
            f"All tables available in schema:\n{ctx_summary}\n\n"
            "Explain in 2-3 sentences: Which table(s) did you choose and why? "
            "Which columns did you use and why? Keep it concise for a business user."
        )
        explanation = await llm_service.generate_sql(
            "You are a helpful assistant that explains SQL generation choices clearly.",
            explain_prompt,
        )
        explanation = explanation.replace("UNABLE_TO_GENERATE", "").strip()
        if explanation:
            final_explanation = explanation
            log_step("EXPLAIN", f"Generated explanation", text=explanation[:100])
    except Exception as exc:
        log_step("EXPLAIN", "Failed to generate explanation", error=str(exc))

    schema_tables_used = sorted(
        {f"{t.schema_name}.{t.table_name}" for t in schema_context.tables}
    )

    log_step("DONE", "Request completed", row_count=len(results))

    final_attempt = attempt if 'attempt' in dir() and isinstance(attempt, int) else 1
    sql_gen_desc = f"Generated on attempt {final_attempt}/{MAX_RETRIES}"
    if retry_log:
        sql_gen_desc += f" with {len(retry_log)} retries: {'; '.join(retry_log[:3])}"

    reasoning = Reasoning(
        table_selection=reasoning_tables,
        column_selection=reasoning_columns,
        final_explanation=final_explanation,
        sql_generation=sql_gen_desc,
        retries=retry_log,
    )
    return QueryResponse(
        question=request.question,
        generated_sql=generated_sql,
        results=results,
        row_count=len(results),
        schema_tables_used=schema_tables_used,
        reasoning=reasoning,
    )


def _try_auto_fix_sql(
    sql: str,
    known_columns: set[str],
    known_tables: set[str],
    table_columns: dict[str, set[str]],
) -> str | None:
    try:
        return validate_columns_in_sql(sql, known_columns, known_tables, table_columns)
    except Exception:
        return None


def _quote_if_mixed(name: str) -> str:
    return f'"{name}"' if any(c.isupper() for c in name) else name

def _auto_qualify_tables(sql: str) -> str:
    import re
    sql_raw = str(sql)
    def _qualify_from(m: re.Match) -> str:
        preceding = sql_raw[max(0, m.start() - 25):m.start()]
        if re.search(r'EXTRACT\s*\(', preceding, re.IGNORECASE):
            return m.group(0)
        raw = m.group(1)
        alias = m.group(2) or ""
        if "." in raw:
            schema, tbl = raw.split(".", 1)
            return f" FROM {schema}.{_quote_if_mixed(tbl)}{' ' + alias if alias else ''} "
        return f" FROM silver_layer.{_quote_if_mixed(raw)}{' ' + alias if alias else ''} "
    def _qualify_join(m: re.Match) -> str:
        raw = m.group(1)
        alias = m.group(2) or ""
        if "." in raw:
            schema, tbl = raw.split(".", 1)
            return f" JOIN {schema}.{_quote_if_mixed(tbl)}{' ' + alias if alias else ''} "
        return f" JOIN silver_layer.{_quote_if_mixed(raw)}{' ' + alias if alias else ''} "
    result = re.sub(
        r'\bFROM\s+(\w+(?:\.\w+)?)(\s+(?:AS\s+)?\w+)?',
        _qualify_from,
        f" {sql_raw}",
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r'\bJOIN\s+(\w+(?:\.\w+)?)(\s+(?:AS\s+)?\w+)?',
        _qualify_join,
        result,
        flags=re.IGNORECASE,
    )
    return result.strip()


def _fix_table_aliases(sql: str) -> str:
    import re
    aliases: dict[str, str] = {}
    for m in re.finditer(
        r"""(?:^|\s)(?:FROM|JOIN)\s+(?:\w+\.)?"?(\w+)"?(?:\s+(?:AS\s+)?(\w+))?""",
        sql, re.IGNORECASE | re.MULTILINE,
    ):
        preceding = sql[max(0, m.start() - 20):m.start()]
        if re.search(r'EXTRACT\s*\(', preceding, re.IGNORECASE):
            continue
        tbl = m.group(1)
        alias = m.group(2) or tbl
        aliases[tbl] = alias

    if not aliases:
        return sql

    result = sql
    for tbl, alias in aliases.items():
        if tbl != alias:
            result = re.sub(
                r"(?<![.\w])" + re.escape(tbl) + r"\s*\.",
                alias + ".",
                result,
            )
    return result


def _fix_pg_errors(sql: str, error: str, table_columns: dict[str, set[str]]) -> str:
    import re
    pg_alias = {
        "current_status": "status",
        "current_premise_name": "premise_name",
        "origin_cp_id": "premise_id",
        "delivery_datetime": "drs_created_date",
        "delivery_date": "drs_created_date",
        "trip_end_hub_inscan_at": "booking_datetime",
    }
    result = sql
    for match in re.finditer(r'column\s+"(\w+)"\s+does not exist', error, re.IGNORECASE):
        bad_col = match.group(1)
        if bad_col in pg_alias:
            good_col = pg_alias[bad_col]
            result = re.sub(r'\b' + re.escape(bad_col) + r'\b', good_col, result)
    if re.search(r"operator does not exist.*text.*timestamp", error, re.IGNORECASE):
        result = re.sub(r'\bdelivery_date\b', 'delivery_date::timestamp', result)
    if re.search(r"operator does not exist.*character varying.*timestamp", error, re.IGNORECASE):
        def _replace_varchar_cmp(m: re.Match) -> str:
            col = m.group(1)
            return (
                f"NULLIF(trim({col}), '') ~ '^[1-9]' "
                f"AND NULLIF(trim({col}), '')::timestamp "
                f"= {m.group(2)}"
            )
        result = re.sub(
            r"(\w+)\s*=\s*(CURRENT_DATE\s*-\s*INTERVAL\s+'[^']+'(?:\s*::\s*\w+)?)",
            _replace_varchar_cmp,
            result,
            count=1,
        )
    if re.search(r"date/time field value out of range", error, re.IGNORECASE):
        for m in re.finditer(r"(\w+)\s*::\s*(?:DATE|TIMESTAMP)(?:\s|$)", result, re.IGNORECASE):
            col = m.group(1)
            result = re.sub(
                r'WHERE\s+',
                "WHERE NULLIF(trim(" + col + "), '') ~ '^[1-9]' AND ",
                result,
                count=1,
            )
            break
    return result


def _build_known_columns(schema_context: object) -> set[str]:
    from app.models.schemas import SchemaContext
    ctx: SchemaContext = schema_context
    cols: set[str] = set()
    for table in ctx.tables:
        for col in table.columns:
            cols.add(col.name)
    return cols


def _build_known_tables(schema_context: object) -> set[str]:
    from app.models.schemas import SchemaContext
    ctx: SchemaContext = schema_context
    return {t.table_name.lower() for t in ctx.tables}


def _build_table_columns(schema_context: object) -> dict[str, set[str]]:
    from app.models.schemas import SchemaContext
    ctx: SchemaContext = schema_context
    result: dict[str, set[str]] = {}
    for table in ctx.tables:
        result[table.table_name.lower()] = {col.name for col in table.columns}
    return result
