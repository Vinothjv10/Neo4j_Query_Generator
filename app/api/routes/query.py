"""
Main query pipeline orchestrating:
  1. GNN-boosted table retrieval  (always active when gnn_schema_service is ready)
  2. DAIL-SQL few-shot examples   (always active when examples exist)
  3. ReAct agent SQL generation   (active when USE_REACT_AGENT=True)
  4. Standard LLM retry pipeline  (fallback / default)
"""

from __future__ import annotations

import os

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
from app.services.react_agent_service import ReactAgentService
from app.services.dail_sql_service import DailSQLService
from app.services.gnn_schema_service import GNNSchemaService
from app.utils.sql_validator import validate_sql, validate_columns_in_sql
from app.utils.logger import log_step
from app.config import settings

router = APIRouter()

neo4j_service = Neo4jService()
prompt_builder = PromptBuilder()
llm_service = LLMService()
postgres_service = PostgresService()
table_index_service = TableIndexService(neo4j_service)
graph_rag_service = GraphRAGService(neo4j_service)
tot_service = ToTService(llm_service)
embedding_service = EmbeddingService()
react_agent_service = ReactAgentService(
    llm_service, table_index_service, graph_rag_service, neo4j_service
)
dail_sql_service = DailSQLService(embedding_service)
gnn_schema_service = GNNSchemaService(neo4j_service)

MAX_RETRIES = 5
USE_TOT = False
# Set USE_REACT_AGENT=true in environment to enable the ReAct agent pipeline
USE_REACT_AGENT = settings.use_react_agent


@router.get("/health")
async def health() -> dict:
    example_count = await dail_sql_service.example_count()
    return {
        "status": "ok",
        "gnn_ready": gnn_schema_service.is_ready,
        "dail_sql_examples": example_count,
        "react_agent_enabled": USE_REACT_AGENT,
    }


@router.post("/query", response_model=QueryResponse, responses={
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
})
async def query(request: QueryRequest) -> QueryResponse:
    log_step("START", "Received question",
             question=request.question,
             top_k=request.top_k,
             react_agent=USE_REACT_AGENT)

    # ── Step 1: Schema context retrieval (with GNN boost) ────────────────────
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

    # ── Step 2: GNN structural re-ranking ────────────────────────────────────
    gnn_boost_applied = False
    if gnn_schema_service.is_ready:
        seed_names = [t.table_name for t in schema_context.tables]
        prop_scores = gnn_schema_service.propagate_from_seeds(seed_names)
        if prop_scores:
            # Sort schema_context.tables by GNN combined score (structural + propagation)
            def _gnn_score(tbl_name: str) -> float:
                struct = gnn_schema_service.get_structural_score(tbl_name)
                prop = prop_scores.get(tbl_name, 0.0)
                return 0.6 * struct + 0.4 * prop

            sorted_tables = sorted(
                schema_context.tables,
                key=lambda t: _gnn_score(t.table_name),
                reverse=True,
            )
            schema_context.tables[:] = sorted_tables
            gnn_boost_applied = True
            log_step("GNN", "Applied structural re-ranking",
                     top3=[t.table_name for t in schema_context.tables[:3]])

    # ── Step 3: Embed question (reuse across column ranking + DAIL-SQL) ──────
    q_vec = None
    if embedding_service.is_ready:
        try:
            q_vec = await embedding_service.embed_question(request.question)
        except Exception:
            pass

    # ── Step 4: DAIL-SQL — retrieve few-shot examples ────────────────────────
    few_shot_examples: list[dict] = []
    try:
        few_shot_examples = await dail_sql_service.retrieve_examples(
            request.question, q_embed=q_vec, top_k=3
        )
        if few_shot_examples:
            log_step("DAIL_SQL", f"Injecting {len(few_shot_examples)} few-shot examples",
                     scores=[e["score"] for e in few_shot_examples])
    except Exception as exc:
        log_step("DAIL_SQL", "Example retrieval failed, continuing", error=str(exc))

    # ── Step 5: Column ranking + reasoning metadata ───────────────────────────
    reasoning_tables: list[TableReason] = []
    reasoning_columns: dict[str, list[ColumnReason]] = {}
    retry_log: list[str] = []

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
                    reason=(
                        f"Semantic match score {round(s, 4)} — "
                        f"column name/description aligns with '{request.question}'"
                    ),
                ) for c, s in cols_with_scores
            ]
        except Exception:
            pass

        match_words = []
        for c in top_cols:
            for w in request.question.lower().split():
                if w in c.lower() or c.lower() in w:
                    match_words.append(f"'{c}' contains keyword '{w}'")
        match_str = (
            "; ".join(match_words[:3]) if match_words
            else f"top semantic match for '{request.question}'"
        )

        gnn_score_str = ""
        if gnn_boost_applied:
            gs = gnn_schema_service.get_structural_score(table.table_name)
            gnn_score_str = f" GNN structural score={gs:.3f}."

        reasoning_tables.append(TableReason(
            table=f"silver_layer.{table.table_name}",
            tier=tier_label,
            description=description or "No description",
            top_columns=top_cols,
            reason=(
                f"{tier_label} — {description[:80] if description else 'no description'}."
                f"{gnn_score_str} "
                f"Relevant columns: {', '.join(top_cols[:3]) or '(none)'}. "
                f"Basis: {match_str}"
            ),
        ))
        if col_scores:
            reasoning_columns[table.table_name] = col_scores

    # ── Step 6: Column filtering for prompt ───────────────────────────────────
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
        log_step("COLUMNS", "Filtered to relevant columns per table",
                 tables_with_filters=list(relevant_columns.keys()))
    else:
        filtered_context = schema_context

    # ── Step 7: Graph RAG enrichment ─────────────────────────────────────────
    enrichment = None
    try:
        enrichment = await graph_rag_service.enrich(
            [t.table_name for t in schema_context.tables]
        )
    except Exception as exc:
        log_step("GRAPH_RAG", "Enrichment failed, continuing without it", error=str(exc))

    # ── Step 8: SQL Generation ────────────────────────────────────────────────
    generated_sql: str | None = None
    agent_trace: list[str] = []
    agent_mode_used = False

    known_columns = _build_known_columns(schema_context)
    known_tables = _build_known_tables(schema_context)
    table_columns = _build_table_columns(schema_context)

    # ── 8a: ReAct Agent path ──────────────────────────────────────────────────
    if USE_REACT_AGENT:
        log_step("AGENT", "Using ReAct agent for SQL generation")
        try:
            generated_sql, agent_trace = await react_agent_service.generate_sql(
                request.question
            )
            agent_mode_used = True
            log_step("AGENT", "Agent produced SQL", sql=generated_sql.replace("\n", " "))
            generated_sql = _auto_qualify_tables(generated_sql)
            # Validate agent output — fallback to standard if it fails
            try:
                validate_sql(generated_sql)
                generated_sql = validate_columns_in_sql(
                    generated_sql, known_columns, known_tables, table_columns
                )
            except ValueError as val_exc:
                log_step("AGENT", "Agent SQL failed validation, falling back",
                         error=str(val_exc))
                retry_log.append(f"Agent SQL validation failed: {str(val_exc)[:120]}")
                agent_mode_used = False
                generated_sql = None
        except Exception as agent_exc:
            log_step("AGENT", "ReAct agent failed, falling back to standard pipeline",
                     error=str(agent_exc))
            retry_log.append(f"ReAct agent failed: {str(agent_exc)[:120]}")

    # ── 8b: Standard retry pipeline (ToT or LLM loop) ────────────────────────
    if not generated_sql:
        system_prompt = prompt_builder.build_system_prompt()
        user_prompt = prompt_builder.build_user_prompt(
            request.question,
            filtered_context,
            enrichment,
            few_shot_examples=few_shot_examples or None,
        )
        log_step(
            "PROMPT",
            f"Built prompts (system={len(system_prompt)} chars, "
            f"user={len(user_prompt)} chars, "
            f"few_shot={len(few_shot_examples)})",
        )

        if USE_TOT:
            log_step("TOT", "Using Tree of Thoughts SQL generation")
            try:
                generated_sql = await tot_service.generate_best(
                    system_prompt, user_prompt,
                    known_columns, known_tables, table_columns,
                    schema_context, num_candidates=3,
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
                    log_step("RETRY", f"LLM unable to generate SQL on attempt {attempt}",
                             error=str(exc))
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

                log_step("LLM", f"SQL generated (attempt {attempt})",
                         sql=generated_sql.replace("\n", " "))

                generated_sql = _auto_qualify_tables(generated_sql)

                try:
                    validate_sql(generated_sql)
                except ValueError as exc:
                    log_step("RETRY", "SQL validation failed", error=str(exc))
                    last_error = str(exc)
                    continue

                try:
                    generated_sql = validate_columns_in_sql(
                        generated_sql, known_columns, known_tables, table_columns,
                    )
                except ValueError as exc:
                    log_step("RETRY", "Column validation failed", error=str(exc))
                    last_error = str(exc)
                    last_auto_fixed = _try_auto_fix_sql(
                        generated_sql, known_columns, known_tables, table_columns
                    )
                    continue

                break
            else:
                if last_auto_fixed:
                    log_step("RETRY", "All LLM retries failed, using auto-fixed SQL",
                             sql=last_auto_fixed.replace("\n", " "))
                    generated_sql = last_auto_fixed
                elif last_error and (
                    "UNABLE_TO_GENERATE" in (last_error or "")
                    or "could not generate" in (last_error or "")
                ):
                    log_step("RETRY",
                             "t3_-only schema failed, fetching ALL t3_ tables from Neo4j")
                    retry_log.append(
                        "Fallback: t3_-only schema insufficient, fetched all available t3_ tables"
                    )
                    try:
                        broad_context = await neo4j_service.get_all_t3_tables()
                        broad_enrich = (
                            await graph_rag_service.enrich(
                                [t.table_name for t in broad_context.tables]
                            ) if enrichment is None else enrichment
                        )
                        broad_user = prompt_builder.build_user_prompt(
                            request.question, broad_context, broad_enrich,
                            few_shot_examples=few_shot_examples or None,
                        )
                        broad_known_cols = _build_known_columns(broad_context)
                        broad_known_tbls = _build_known_tables(broad_context)
                        broad_tbl_cols = _build_table_columns(broad_context)
                        log_step("RETRY",
                                 f"Retrying with {len(broad_context.tables)} tables "
                                 f"(was {len(schema_context.tables)})")
                        for attempt2 in range(1, 4):
                            hint = (
                                "\n\nYou previously responded with UNABLE_TO_GENERATE. "
                                "The schema now includes upstream tables. Generate the SQL.\n"
                            )
                            sql2 = await llm_service.generate_sql(
                                system_prompt, broad_user + hint
                            )
                            if sql2:
                                sql2 = _auto_qualify_tables(sql2)
                                try:
                                    validate_sql(sql2)
                                    sql2 = validate_columns_in_sql(
                                        sql2, broad_known_cols, broad_known_tbls, broad_tbl_cols
                                    )
                                    generated_sql = sql2
                                    log_step("RETRY",
                                             f"Broad schema fallback succeeded on attempt {attempt2}",
                                             sql=sql2.replace("\n", " "))
                                    break
                                except ValueError:
                                    continue
                    except Exception as broad_exc:
                        log_step("RETRY", "Broad schema fallback failed", error=str(broad_exc))
                    if not generated_sql:
                        raise HTTPException(
                            status_code=422,
                            detail=ErrorResponse(
                                error=(
                                    f"Failed to generate valid SQL after {MAX_RETRIES} attempts "
                                    f"(including broad schema fallback)"
                                ),
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

    # ── Step 9: PostgreSQL execution with auto-fix retries ────────────────────
    results: list[dict] = []
    for pg_attempt in range(1, 3):
        try:
            results = await postgres_service.execute_query(generated_sql, request.top_k)
            break
        except Exception as exc:
            if pg_attempt == 1:
                retry_log.append(f"PG error ({pg_attempt}): {str(exc)[:100]}")
            log_step("ERROR", f"PostgreSQL execution failed (attempt {pg_attempt})",
                     sql=generated_sql.replace("\n", " "), error=str(exc))
            if pg_attempt == 1:
                fixed = _fix_table_aliases(generated_sql)
                if fixed != generated_sql:
                    generated_sql = fixed
                    continue
                alt_fixed = _try_auto_fix_sql(
                    generated_sql, known_columns, known_tables, table_columns
                )
                if alt_fixed is not None and alt_fixed != generated_sql:
                    generated_sql = alt_fixed
                    continue
                pg_fixed = _fix_pg_errors(generated_sql, str(exc), table_columns)
                if pg_fixed != generated_sql:
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

    # ── Step 10: DAIL-SQL — store successful example ──────────────────────────
    try:
        used_tables_list = []
        import re as _re_store
        for m in _re_store.finditer(
            r'(?:FROM|JOIN)\s+\w+\.(\w+)', generated_sql, _re_store.IGNORECASE
        ):
            used_tables_list.append(m.group(1))
        await dail_sql_service.store_example(
            question=request.question,
            sql=generated_sql,
            tables_used=used_tables_list,
            q_embed=q_vec,
        )
    except Exception as exc:
        log_step("DAIL_SQL", "Failed to store example", error=str(exc))

    # ── Step 11: LLM explanation ──────────────────────────────────────────────
    final_explanation: str | None = None
    try:
        import re as _re_explain
        used_tables_list2 = []
        for m in _re_explain.finditer(
            r'(?:FROM|JOIN)\s+\w+\.(\w+)', generated_sql, _re_explain.IGNORECASE
        ):
            used_tables_list2.append(m.group(1))
        used_cols_list = list({
            f"{tbl.table_name}.{c.name}"
            for tbl in schema_context.tables
            for c in tbl.columns
            if c.name.lower() in generated_sql.lower()
        })
        ctx_summary = "\n".join(
            f"{t.table_name} ({t.description or 'no desc'}): "
            + ", ".join(c.name for c in t.columns[:8])
            for t in schema_context.tables
        )
        explain_prompt = (
            f"Question: {request.question}\n\n"
            f"SQL generated:\n{generated_sql}\n\n"
            f"Table(s) used: {', '.join(used_tables_list2)}\n"
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
            log_step("EXPLAIN", "Generated explanation", text=explanation[:100])
    except Exception as exc:
        log_step("EXPLAIN", "Failed to generate explanation", error=str(exc))

    # ── Step 12: Assemble response ────────────────────────────────────────────
    schema_tables_used = sorted(
        {f"{t.schema_name}.{t.table_name}" for t in schema_context.tables}
    )
    log_step("DONE", "Request completed", row_count=len(results))

    # Compute attempt count for sql_generation description
    final_attempt = locals().get("attempt", 1)
    if not isinstance(final_attempt, int):
        final_attempt = 1
    sql_gen_desc = (
        f"Agent mode (ReAct)" if agent_mode_used
        else f"Generated on attempt {final_attempt}/{MAX_RETRIES}"
    )
    if retry_log:
        sql_gen_desc += f" | {len(retry_log)} issue(s): {'; '.join(retry_log[:3])}"

    reasoning = Reasoning(
        table_selection=reasoning_tables,
        column_selection=reasoning_columns,
        final_explanation=final_explanation,
        sql_generation=sql_gen_desc,
        retries=retry_log,
        agent_mode=agent_mode_used,
        agent_trace=agent_trace,
        few_shot_count=len(few_shot_examples),
        gnn_boost_applied=gnn_boost_applied,
    )
    return QueryResponse(
        question=request.question,
        generated_sql=generated_sql,
        results=results,
        row_count=len(results),
        schema_tables_used=schema_tables_used,
        reasoning=reasoning,
    )


# ── SQL helpers ───────────────────────────────────────────────────────────────

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
        _qualify_from, f" {sql_raw}", flags=re.IGNORECASE,
    )
    result = re.sub(
        r'\bJOIN\s+(\w+(?:\.\w+)?)(\s+(?:AS\s+)?\w+)?',
        _qualify_join, result, flags=re.IGNORECASE,
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
            _replace_varchar_cmp, result, count=1,
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
    if re.search(r"function sum\(text\)", error, re.IGNORECASE):
        for m in re.finditer(r"SUM\s*\(\s*(\w+)\)", result, re.IGNORECASE):
            col = m.group(1)
            result = re.sub(
                r'\bSUM\s*\(\s*' + re.escape(col) + r'\s*\)',
                f"SUM(CAST(REGEXP_REPLACE(NULLIF(trim({col}), ''), '[^0-9.]', '', 'g') AS NUMERIC))",
                result,
                count=1,
            )
            break
    return result


def _build_known_columns(schema_context: object) -> set[str]:
    from app.models.schemas import SchemaContext
    ctx: SchemaContext = schema_context  # type: ignore[assignment]
    return {col.name for table in ctx.tables for col in table.columns}


def _build_known_tables(schema_context: object) -> set[str]:
    from app.models.schemas import SchemaContext
    ctx: SchemaContext = schema_context  # type: ignore[assignment]
    return {t.table_name.lower() for t in ctx.tables}


def _build_table_columns(schema_context: object) -> dict[str, set[str]]:
    from app.models.schemas import SchemaContext
    ctx: SchemaContext = schema_context  # type: ignore[assignment]
    return {table.table_name.lower(): {col.name for col in table.columns}
            for table in ctx.tables}
