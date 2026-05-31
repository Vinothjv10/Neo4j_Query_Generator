from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ErrorResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.neo4j_service import Neo4jService
from app.services.prompt_builder import PromptBuilder
from app.services.llm_service import LLMService
from app.services.postgres_service import PostgresService
from app.services.table_index_service import TableIndexService
from app.services.graph_rag_service import GraphRAGService
from app.services.tot_service import ToTService
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

    enrichment = None
    try:
        enrichment = await graph_rag_service.enrich(
            [t.table_name for t in schema_context.tables]
        )
    except Exception as exc:
        log_step("GRAPH_RAG", "Enrichment failed, continuing without it", error=str(exc))

    system_prompt = prompt_builder.build_system_prompt()
    user_prompt = prompt_builder.build_user_prompt(
        request.question, schema_context, enrichment
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
                if alt_fixed != generated_sql:
                    log_step("RETRY", "Attempting column auto-fix", fixed_sql=alt_fixed.replace("\n", " "))
                    generated_sql = alt_fixed
                    continue
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error="Query execution failed",
                    detail=f"SQL: {generated_sql}\nError: {exc}",
                ).model_dump(),
            )

    log_step("POSTGRES", f"Query returned {len(results)} rows")

    schema_tables_used = sorted(
        {f"{t.schema_name}.{t.table_name}" for t in schema_context.tables}
    )

    log_step("DONE", "Request completed", row_count=len(results))
    return QueryResponse(
        question=request.question,
        generated_sql=generated_sql,
        results=results,
        row_count=len(results),
        schema_tables_used=schema_tables_used,
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


def _auto_qualify_tables(sql: str) -> str:
    import re
    sql_raw = str(sql)
    def _qualify_from(m: re.Match) -> str:
        preceding = sql_raw[max(0, m.start() - 25):m.start()]
        if re.search(r'EXTRACT\s*\(', preceding, re.IGNORECASE):
            return m.group(0)
        full_tbl = m.group(1)
        alias = m.group(2) or ""
        if "." in full_tbl:
            return m.group(0)
        return f" FROM silver_layer.{full_tbl}{' ' + alias if alias else ''} "
    def _qualify_join(m: re.Match) -> str:
        full_tbl = m.group(1)
        alias = m.group(2) or ""
        if "." in full_tbl:
            return m.group(0)
        return f" JOIN silver_layer.{full_tbl}{' ' + alias if alias else ''} "
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
        r"(?:^|\s)(?:FROM|JOIN)\s+(?:\w+\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?",
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
    return {t.table_name for t in ctx.tables}


def _build_table_columns(schema_context: object) -> dict[str, set[str]]:
    from app.models.schemas import SchemaContext
    ctx: SchemaContext = schema_context
    result: dict[str, set[str]] = {}
    for table in ctx.tables:
        result[table.table_name] = {col.name for col in table.columns}
    return result
