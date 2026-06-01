"""
DAIL-SQL: Few-shot example retrieval for SQL generation.

Inspired by DAIL-SQL (Alibaba DAMO, VLDB 2024).

Stores successful (question, SQL) pairs in an async SQLite database.
On each new query, retrieves the top-K most similar past examples using
a hybrid score:

    similarity = ALPHA * cosine_semantic(q_embed, example_embed)
                + (1 - ALPHA) * skeleton_jaccard(q_skeleton, example_skeleton)

Retrieved examples are injected as few-shot context into the LLM prompt,
dramatically improving SQL accuracy by showing the model correct patterns
from your own schema.

Database location: configurable via DAIL_SQL_DB env var, defaults to
/tmp/dail_sql_store.db
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.logger import log_step
from app.utils.sql_skeleton import extract_skeleton, skeleton_similarity

try:
    import aiosqlite
    _AIOSQLITE_AVAILABLE = True
except ImportError:
    _AIOSQLITE_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "aiosqlite not installed — DAIL-SQL example store disabled. "
        "Run: pip install aiosqlite"
    )

if TYPE_CHECKING:
    from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DAIL_SQL_DB", "/tmp/dail_sql_store.db")
ALPHA = 0.6          # weight: semantic similarity vs skeleton similarity
TOP_K_EXAMPLES = 3   # number of few-shot examples to inject into prompt
MIN_SIMILARITY = 0.3 # minimum combined score to include an example


def _vec_to_bytes(vec: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, vec)
    return buf.getvalue()


def _bytes_to_vec(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b))


class DailSQLService:
    def __init__(self, embedding_service: "EmbeddingService | None" = None) -> None:
        self._embed_svc = embedding_service
        self._db_path = DB_PATH

    def set_embedding_service(self, svc: "EmbeddingService") -> None:
        self._embed_svc = svc

    @property
    def is_available(self) -> bool:
        return _AIOSQLITE_AVAILABLE

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create the SQLite table if it does not already exist."""
        if not _AIOSQLITE_AVAILABLE:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sql_examples (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    question  TEXT    NOT NULL,
                    sql       TEXT    NOT NULL,
                    skeleton  TEXT    NOT NULL,
                    q_embed   BLOB,
                    tables    TEXT,
                    created_at REAL   NOT NULL
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_skeleton ON sql_examples(skeleton)"
            )
            await db.commit()
        log_step("DAIL_SQL", f"Example store initialised", path=self._db_path)

    # ── Write path ───────────────────────────────────────────────────────────

    async def store_example(
        self,
        question: str,
        sql: str,
        tables_used: list[str] | None = None,
        q_embed: np.ndarray | None = None,
    ) -> None:
        """Persist a successful (question, SQL) pair for future few-shot use."""
        if not _AIOSQLITE_AVAILABLE:
            return
        skeleton = extract_skeleton(sql)
        embed_bytes = _vec_to_bytes(q_embed) if q_embed is not None else None
        tables_json = json.dumps(tables_used or [])

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO sql_examples
                   (question, sql, skeleton, q_embed, tables, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (question, sql, skeleton, embed_bytes, tables_json, time.time()),
            )
            await db.commit()

        log_step("DAIL_SQL", "Stored example",
                 question=question[:60], skeleton=skeleton[:80])

    # ── Read path ────────────────────────────────────────────────────────────

    async def retrieve_examples(
        self,
        question: str,
        q_embed: np.ndarray | None = None,
        top_k: int = TOP_K_EXAMPLES,
    ) -> list[dict]:
        """
        Return the top-k most similar stored examples.

        Each returned dict has keys: question, sql, skeleton, score, tables.
        """
        if not _AIOSQLITE_AVAILABLE:
            return []
        # Fetch all stored examples (up to 500 rows — small enough for in-process ranking)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT question, sql, skeleton, q_embed, tables FROM sql_examples ORDER BY created_at DESC LIMIT 500"
            ) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return []

        q_skel = extract_skeleton(question)

        scored: list[tuple[float, dict]] = []
        for row in rows:
            ex_question, ex_sql, ex_skeleton, ex_embed_bytes, ex_tables_json = row

            # Skeleton Jaccard similarity
            skel_sim = skeleton_similarity(q_skel, ex_skeleton)

            # Semantic similarity (if embeddings available)
            sem_sim = 0.0
            if q_embed is not None and ex_embed_bytes:
                try:
                    ex_vec = _bytes_to_vec(ex_embed_bytes).reshape(1, -1)
                    q_vec = q_embed.reshape(1, -1)
                    sem_sim = float(cosine_similarity(q_vec, ex_vec)[0][0])
                except Exception:
                    sem_sim = 0.0

            combined = ALPHA * sem_sim + (1.0 - ALPHA) * skel_sim

            if combined >= MIN_SIMILARITY:
                try:
                    tables = json.loads(ex_tables_json) if ex_tables_json else []
                except Exception:
                    tables = []
                scored.append((combined, {
                    "question": ex_question,
                    "sql": ex_sql,
                    "skeleton": ex_skeleton,
                    "score": round(combined, 4),
                    "tables": tables,
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored[:top_k]]

        log_step("DAIL_SQL", f"Retrieved {len(results)} few-shot examples",
                 scores=[r["score"] for r in results])
        return results

    # ── Count ────────────────────────────────────────────────────────────────

    async def example_count(self) -> int:
        if not _AIOSQLITE_AVAILABLE:
            return -1
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute("SELECT COUNT(*) FROM sql_examples") as cur:
                    row = await cur.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
