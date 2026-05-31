import logging
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.neo4j_service import Neo4jService
from app.utils.logger import log_step

logger = logging.getLogger(__name__)


class TableIndexEntry:
    def __init__(
        self,
        table_name: str,
        schema_name: str,
        description: str | None,
        document: str,
    ) -> None:
        self.table_name = table_name
        self.schema_name = schema_name
        self.description = description or ""
        self.document = document


class TableIndexService:
    def __init__(self, neo4j_service: Neo4jService) -> None:
        self._neo4j = neo4j_service
        self._entries: list[TableIndexEntry] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: Any = None
        self._built = False

    async def build_index(self) -> None:
        log_step("INDEX", "Building TF-IDF table index from Neo4j")
        raw_tables = await self._fetch_all_tables_raw()

        documents: list[str] = []
        self._entries = []

        for t in raw_tables:
            doc_parts: list[str] = [
                t["name"],
                t.get("description") or "",
            ]
            for c in t.get("columns", []):
                doc_parts.append(c["name"])
                desc = c.get("description") or ""
                if desc:
                    doc_parts.append(desc)

            doc = " ".join(doc_parts)
            documents.append(doc)
            self._entries.append(
                TableIndexEntry(
                    table_name=t["name"],
                    schema_name=t.get("schema", "silver_layer"),
                    description=t.get("description"),
                    document=doc,
                )
            )

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(documents)
        self._built = True
        log_step(
            "INDEX",
            f"Index built: {len(self._entries)} tables, "
            f"{self._tfidf_matrix.shape[1]} features",
        )

    async def search(
        self, question: str, top_k: int = 5, min_score: float = 0.02
    ) -> list[TableIndexEntry]:
        if not self._built:
            await self.build_index()

        assert self._vectorizer is not None
        assert self._tfidf_matrix is not None

        query_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        top_indices = scores.argsort()[::-1]
        results: list[TableIndexEntry] = []
        for idx in top_indices:
            if scores[idx] >= min_score:
                results.append(self._entries[idx])
            if len(results) >= top_k:
                break

        log_step(
            "INDEX",
            f"Semantic search returned {len(results)} tables "
            f"(top score={float(scores[top_indices[0]]):.4f})",
            question=question,
        )
        return results

    async def get_enriched_document(self, table_name: str) -> str | None:
        for entry in self._entries:
            if entry.table_name == table_name:
                return entry.document
        return None

    async def _fetch_all_tables_raw(self) -> list[dict]:
        async with self._neo4j._driver.session() as session:
            result = await session.run("""
                MATCH (t:Table)
                OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                WITH t, c
                ORDER BY c.name
                RETURN t.name AS name,
                       t.schema AS schema,
                       t.description AS description,
                       collect(DISTINCT {
                           name: c.name,
                           data_type: c.data_type,
                           description: c.description
                       }) AS columns
            """)
            return await result.data()
