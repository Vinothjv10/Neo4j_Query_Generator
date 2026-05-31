from app.services.neo4j_service import Neo4jService
from app.utils.logger import log_step


class GraphRAGService:
    def __init__(self, neo4j_service: Neo4jService) -> None:
        self._neo4j = neo4j_service

    @staticmethod
    def _keep_tier(name: str) -> bool:
        return name.startswith("t3_") or name.startswith("t2_")

    async def enrich(
        self, table_names: list[str]
    ) -> dict:
        result: dict = {
            "join_hints": [],
            "column_mappings": [],
            "dependency_chains": [],
            "sibling_tables": [],
        }

        for tbl in table_names:
            deps = await self._fetch_dependencies(tbl)
            if deps:
                result["dependency_chains"].append(
                    {"table": tbl, "depends_on": deps}
                )

            maps = await self._fetch_column_mappings(tbl)
            if maps:
                result["column_mappings"].extend(maps)

            siblings = await self._fetch_siblings(tbl)
            if siblings:
                result["sibling_tables"].extend(
                    {"table": tbl, "joinable_with": siblings}
                )

            hints = await self._fetch_join_hints(tbl)
            if hints:
                result["join_hints"].extend(hints)

        result["join_hints"] = [
            h for h in result["join_hints"]
            if self._keep_tier(h["source_table"]) and self._keep_tier(h["target_table"])
        ]
        result["column_mappings"] = [
            m for m in result["column_mappings"]
            if self._keep_tier(m.get("source_table", "")) and self._keep_tier(m.get("target_table", ""))
        ]
        result["sibling_tables"] = [
            s for s in result["sibling_tables"]
            if isinstance(s, dict) and self._keep_tier(s.get("table", ""))
        ]

        log_step(
            "GRAPH_RAG",
            f"Enriched context: {len(result['dependency_chains'])} dep chains, "
            f"{len(result['column_mappings'])} col mappings, "
            f"{len(result['sibling_tables'])} sibling groups",
        )
        return result

    async def _fetch_dependencies(self, table_name: str) -> list[str]:
        async with self._neo4j._driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Table {name: $name})-[:DEPENDS_ON]->(dep:Table)
                RETURN dep.name AS dep_name, dep.description AS dep_desc
                """,
                name=table_name,
            )
            records = await result.data()
            return [r["dep_name"] for r in records if r.get("dep_name")]

    async def _fetch_column_mappings(self, table_name: str) -> list[dict]:
        async with self._neo4j._driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Table {name: $name})-[:HAS_COLUMN]->(c:Column)-[:MAPS_TO]->(other:Column)
                OPTIONAL MATCH (other)<-[:HAS_COLUMN]-(other_table:Table)
                RETURN c.name AS source_col,
                       other.name AS target_col,
                       other_table.name AS target_table
                """,
                name=table_name,
            )
            records = await result.data()
            return [
                {
                    "source_table": table_name,
                    "source_col": r["source_col"],
                    "target_table": r.get("target_table") or "?",
                    "target_col": r["target_col"],
                }
                for r in records
                if r.get("source_col") and r.get("target_col")
            ]

    async def _fetch_siblings(self, table_name: str) -> list[dict]:
        async with self._neo4j._driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Table {name: $name})-[:HAS_COLUMN]->(c:Column)
                MATCH (c)-[:MAPS_TO]->(other:Column)<-[:HAS_COLUMN]-(other_table:Table)
                WHERE other_table.name <> $name
                RETURN other_table.name AS sibling,
                       c.name AS via_column,
                       other.name AS mapped_column
                """,
                name=table_name,
            )
            records = await result.data()
            seen = set()
            unique = []
            for r in records:
                key = r.get("sibling", "")
                if key and key not in seen:
                    seen.add(key)
                    unique.append(
                        {
                            "table": r["sibling"],
                            "via_column": r.get("via_column"),
                            "mapped_to": r.get("mapped_column"),
                        }
                    )
            return unique

    async def _fetch_join_hints(self, table_name: str) -> list[dict]:
        async with self._neo4j._driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Table {name: $name})-[:HAS_COLUMN]->(c:Column)-[:MAPS_TO]->(other:Column)
                OPTIONAL MATCH (other)<-[:HAS_COLUMN]-(other_table:Table)
                WHERE other_table.name IS NOT NULL AND other_table.name <> $name
                RETURN c.name AS source_col,
                       other.name AS target_col,
                       other_table.name AS target_table
                """,
                name=table_name,
            )
            records = await result.data()
            return [
                {
                    "source_table": table_name,
                    "source_col": r["source_col"],
                    "target_table": r["target_table"],
                    "target_col": r["target_col"],
                    "join_condition": (
                        f"{table_name}.{r['source_col']} = "
                        f"{r['target_table']}.{r['target_col']}"
                    ),
                }
                for r in records
                if r.get("source_col") and r.get("target_table") and r.get("target_col")
            ]
