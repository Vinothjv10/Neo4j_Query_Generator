import logging

from neo4j import AsyncGraphDatabase

from app.config import settings
from app.models.schemas import ColumnInfo, SchemaContext, TableInfo
from app.utils.logger import log_step

logger = logging.getLogger(__name__)

STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "what", "how", "many", "much",
    "show", "me", "give", "list", "find", "get", "of", "in", "on",
    "by", "for", "with", "and", "or", "to", "from",
}


class Neo4jService:
    def __init__(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._table_index = None
        self._graph_rag = None

    def set_services(self, table_index, graph_rag) -> None:
        self._table_index = table_index
        self._graph_rag = graph_rag

    async def close(self) -> None:
        await self._driver.close()

    async def get_schema_context(self, question: str) -> SchemaContext:
        matched_tables: list[TableInfo] = []

        if self._table_index:
            log_step("NEO4J", "Using semantic TF-IDF search for table retrieval")
            try:
                entries = await self._table_index.search(question, top_k=5)
                if entries:
                    table_names = [e.table_name for e in entries]
                    log_step(
                        "NEO4J",
                        f"Semantic search matched tables",
                        tables=table_names,
                    )
                    matched_tables = await self._fetch_tables_by_names(table_names)
            except Exception as exc:
                log_step("NEO4J", "Semantic search failed, falling back", error=str(exc))

        if not matched_tables:
            log_step("NEO4J", "Falling back to keyword-based table retrieval")
            matched_tables = await self._keyword_search(question)

        if not matched_tables:
            log_step("NEO4J", "No tables matched, fetching default t3_ tables")
            ctx = await self._fetch_primary_tables()
            matched_tables = ctx.tables

        t3_tables = [t for t in matched_tables if t.table_name.startswith("t3_")]
        max_t3 = 5
        if len(t3_tables) > max_t3:
            kept = [t.table_name for t in t3_tables[:max_t3]]
            log_step("NEO4J", f"Capped t3_ tables from {len(t3_tables)} to {max_t3}", kept=kept)
            matched_tables = t3_tables[:max_t3]

        log_step("NEO4J", f"Returning {len(matched_tables)} tables")
        return SchemaContext(tables=matched_tables)

    async def _keyword_search(self, question: str) -> list[TableInfo]:
        keywords = self._extract_keywords(question)
        log_step("NEO4J", f"Keyword search with", keywords=keywords)
        if not keywords:
            return []

        all_tables: list[TableInfo] = []
        seen: set[str] = set()

        for keyword in keywords:
            tables, _ = await self._fetch_t3_by_keyword(keyword)
            for table in tables:
                key = f"{table.schema_name}.{table.table_name}"
                if key not in seen:
                    seen.add(key)
                    all_tables.append(table)
        return all_tables

    def _extract_keywords(self, question: str) -> list[str]:
        tokens = question.lower().split()
        return [t for t in tokens if t not in STOPWORDS and t.isalpha()]

    async def _fetch_t3_by_keyword(
        self, keyword: str
    ) -> tuple[list[TableInfo], set[str]]:
        query = """
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE t.name STARTS WITH "t3_"
          AND (toLower(c.description) CONTAINS toLower($keyword)
               OR toLower(t.name) CONTAINS toLower($keyword))
        WITH DISTINCT t
        MATCH (t)-[:HAS_COLUMN]->(all_c:Column)
        OPTIONAL MATCH (all_c)-[:MAPS_TO]->(mapped:Column)
        OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Table)
        WITH t, all_c, mapped, dep
        RETURN t.name AS table_name,
               COALESCE(t.schema, "silver_layer") AS schema_name,
               t.description AS table_desc,
               collect(DISTINCT {
                 name: all_c.name,
                 type: all_c.data_type,
                 description: all_c.description,
                 maps_to: mapped.name
               }) AS columns,
               collect(DISTINCT dep.name) AS related_tables
        """
        async with self._driver.session() as session:
            result = await session.run(query, {"keyword": keyword})
            records = await result.data()

        deps: set[str] = set()
        tables: list[TableInfo] = []
        for record in records:
            columns = [
                ColumnInfo(
                    name=col["name"],
                    data_type=col.get("type") or "unknown",
                    description=self._build_col_desc(
                        col.get("description"), col.get("maps_to")
                    ),
                )
                for col in record.get("columns", [])
                if col.get("name")
            ]
            related = [r for r in record.get("related_tables", []) if r]
            deps.update(related)
            tables.append(
                TableInfo(
                    table_name=record["table_name"],
                    schema_name=record.get("schema_name", "silver_layer"),
                    description=record.get("table_desc"),
                    columns=columns,
                    related_tables=related,
                )
            )
        return tables, deps

    async def _fetch_tables_by_names(
        self, table_names: list[str]
    ) -> list[TableInfo]:
        query = """
        MATCH (t:Table)
        WHERE t.name IN $table_names
        MATCH (t)-[:HAS_COLUMN]->(c:Column)
        OPTIONAL MATCH (c)-[:MAPS_TO]->(mapped:Column)
        OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Table)
        WITH t, c, mapped, dep
        RETURN t.name AS table_name,
               COALESCE(t.schema, "silver_layer") AS schema_name,
               t.description AS table_desc,
               collect(DISTINCT {
                 name: c.name,
                 type: c.data_type,
                 description: c.description,
                 maps_to: mapped.name
               }) AS columns,
               collect(DISTINCT dep.name) AS related_tables
        """
        async with self._driver.session() as session:
            result = await session.run(query, {"table_names": table_names})
            records = await result.data()

        tables: list[TableInfo] = []
        for record in records:
            columns = [
                ColumnInfo(
                    name=col["name"],
                    data_type=col.get("type") or "unknown",
                    description=self._build_col_desc(
                        col.get("description"), col.get("maps_to")
                    ),
                )
                for col in record.get("columns", [])
                if col.get("name")
            ]
            tables.append(
                TableInfo(
                    table_name=record["table_name"],
                    schema_name=record.get("schema_name", "silver_layer"),
                    description=record.get("table_desc"),
                    columns=columns,
                    related_tables=[
                        r for r in record.get("related_tables", []) if r
                    ],
                )
            )
        return tables

    async def _fetch_primary_tables(self) -> SchemaContext:
        query = """
        MATCH (t:Table)
        WHERE t.name STARTS WITH "t3_"
        OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
        OPTIONAL MATCH (c)-[:MAPS_TO]->(mapped:Column)
        OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Table)
        WITH t, c, mapped, dep
        RETURN t.name AS table_name,
               COALESCE(t.schema, "silver_layer") AS schema_name,
               t.description AS table_desc,
               collect(DISTINCT {
                 name: c.name,
                 type: c.data_type,
                 description: c.description,
                 maps_to: mapped.name
               }) AS columns,
               collect(DISTINCT dep.name) AS related_tables
        """
        async with self._driver.session() as session:
            result = await session.run(query)
            records = await result.data()

        tables: list[TableInfo] = []
        for record in records:
            columns = [
                ColumnInfo(
                    name=col["name"],
                    data_type=col.get("type") or "unknown",
                    description=self._build_col_desc(
                        col.get("description"), col.get("maps_to")
                    ),
                )
                for col in record.get("columns", [])
                if col.get("name")
            ]
            tables.append(
                TableInfo(
                    table_name=record["table_name"],
                    schema_name=record.get("schema_name", "silver_layer"),
                    description=record.get("table_desc"),
                    columns=columns,
                    related_tables=[
                        r for r in record.get("related_tables", []) if r
                    ],
                )
            )

        if len(tables) > 5:
            tables = tables[:5]

        return SchemaContext(tables=tables)

    def filter_columns_by_relevance(
        self,
        ctx: SchemaContext,
        relevant_columns: dict[str, list[str]],
    ) -> SchemaContext:
        filtered_tables: list[TableInfo] = []
        for table in ctx.tables:
            relevant = relevant_columns.get(table.table_name)
            if relevant:
                col_set = set(relevant)
                filtered_cols = [c for c in table.columns if c.name in col_set]
                if not filtered_cols:
                    filtered_cols = table.columns[:5]
                filtered_tables.append(
                    TableInfo(
                        table_name=table.table_name,
                        schema_name=table.schema_name,
                        description=table.description,
                        columns=filtered_cols,
                        related_tables=table.related_tables,
                    )
                )
            else:
                filtered_tables.append(table)
        return SchemaContext(tables=filtered_tables)

    def _build_col_desc(
        self, description: str | None, maps_to: str | None
    ) -> str | None:
        if not description and not maps_to:
            return None
        parts: list[str] = []
        if description:
            parts.append(description)
        if maps_to:
            parts.append(f"[maps to: {maps_to}]")
        return " ".join(parts)
