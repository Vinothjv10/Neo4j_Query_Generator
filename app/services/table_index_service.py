import logging
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

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
        self._embedding_service = None

    def set_embedding_service(self, svc: Any) -> None:
        self._embedding_service = svc

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

        if self._embedding_service and self._embedding_service.is_ready:
            log_step("INDEX", "Embedding index already available, skipping rebuild")
        elif self._embedding_service:
            try:
                await self._embedding_service.build_index(raw_tables)
            except Exception as exc:
                log_step("INDEX", "Embedding index build failed, TF-IDF only",
                         error=str(exc))

    async def search(
        self, question: str, top_k: int = 5, min_score: float = 0.02
    ) -> list[TableIndexEntry]:
        if not self._built:
            await self.build_index()

        entries = await self._tfidf_search(question, top_k * 2, min_score)

        if self._embedding_service and self._embedding_service.is_ready:
            entries = await self._hybrid_search(question, entries, top_k)

        entries = await self._prioritize_t3(entries, top_k)
        return entries[:top_k]

    async def _prioritize_t3(
        self, entries: list[TableIndexEntry], top_k: int
    ) -> list[TableIndexEntry]:
        t3 = [e for e in entries if e.table_name.startswith("t3_")]
        t2 = [e for e in entries if e.table_name.startswith("t2_")]
        t1 = [e for e in entries if e.table_name.startswith("t1_")]
        others = [e for e in entries if not e.table_name[0:2].startswith("t")]

        if t3:
            log_step("INDEX", f"T3-prioritized: {len(t3)}/{len(entries)} tables kept")
            return (t3 + others)[:top_k]
        if t2:
            log_step("INDEX", f"No t3_ found, falling back to {len(t2)} t2_ tables")
            return (t2 + others)[:top_k]
        if t1:
            log_step("INDEX", f"No t3_/t2_ found, falling back to {len(t1)} t1_ tables")
            return (t1 + others)[:top_k]
        return entries

    async def _tfidf_search(
        self, question: str, top_k: int, min_score: float
    ) -> list[TableIndexEntry]:
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
            f"TF-IDF search returned {len(results)} tables "
            f"(top score={float(scores[top_indices[0]]):.4f})" if len(scores) > 0 else "no scores",
            question=question,
        )
        return results

    async def _hybrid_search(
        self,
        question: str,
        tfidf_entries: list[TableIndexEntry],
        top_k: int,
    ) -> list[TableIndexEntry]:
        embed_results = await self._embedding_service.search_tables(
            question, top_k=top_k * 2, min_score=0.0
        )
        if not embed_results:
            return tfidf_entries

        embed_map: dict[str, float] = dict(embed_results)

        seen: set[str] = set()
        for entry in tfidf_entries:
            seen.add(entry.table_name)

        for name, score in embed_results:
            if name not in seen:
                seen.add(name)
                fake = TableIndexEntry(name, "silver_layer", None, "")
                tfidf_entries.append(fake)

        combined: list[tuple[TableIndexEntry, float]] = []
        for entry in tfidf_entries:
            embed_score = embed_map.get(entry.table_name, 0.0)
            combined.append((entry, embed_score))

        embed_only_scores = np.array(
            [s for _, s in combined], dtype=np.float32
        ).reshape(-1, 1)
        if embed_only_scores.max() > embed_only_scores.min():
            scaler = MinMaxScaler()
            embed_only_scores = scaler.fit_transform(embed_only_scores).flatten()
        else:
            embed_only_scores = embed_only_scores.flatten()

        final: list[tuple[TableIndexEntry, float]] = []
        for entry, _ in combined:
            score = embed_map.get(entry.table_name, 0.0)
            final.append((entry, score))

        final.sort(key=lambda x: x[1], reverse=True)

        log_step("INDEX", f"Hybrid search returned {len(final[:top_k])} tables",
                 method="embedding_primary")
        return [e for e, _ in final[:top_k]]

    async def get_relevant_columns(
        self, question: str, table_name: str, top_k: int = 5,
        q_vec: np.ndarray | None = None
    ) -> list[str]:
        if not self._embedding_service or not self._embedding_service.is_ready:
            return []
        results = await self._embedding_service.rank_columns(
            question, table_name, top_k, q_vec=q_vec
        )
        return [c for c, _ in results]

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
