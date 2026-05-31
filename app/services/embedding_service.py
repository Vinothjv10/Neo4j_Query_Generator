import json
import os

import numpy as np
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
from sklearn.metrics.pairwise import cosine_similarity
import httpx

from app.config import settings
from app.utils.logger import log_step

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
EMBED_MODEL = "text-embedding-005"
EMBED_DIM = 768
BATCH_SIZE = 200


class EmbeddingService:
    def __init__(self) -> None:
        self._token: str | None = None
        self._table_names: list[str] = []
        self._table_docs: list[str] = []
        self._table_embeddings: np.ndarray | None = None
        self._column_embeddings: dict[str, dict[str, np.ndarray]] = {}
        self._ready = False

    async def _ensure_token(self) -> str:
        if self._token:
            return self._token
        creds_path = settings.google_application_credentials
        if not creds_path or not os.path.exists(creds_path):
            raise FileNotFoundError(
                f"Service account file not found: {creds_path}"
            )
        abs_path = os.path.abspath(creds_path)
        credentials = service_account.Credentials.from_service_account_file(
            abs_path, scopes=SCOPES,
        )
        auth_req = GoogleAuthRequest()
        credentials.refresh(auth_req)
        self._token = credentials.token
        return self._token

    def _embed_url(self) -> str:
        loc = settings.vertex_ai_location or "us-east5"
        proj = settings.vertex_ai_project or "saturam"
        return (
            f"https://{loc}-aiplatform.googleapis.com/v1/"
            f"projects/{proj}/locations/{loc}/"
            f"publishers/google/models/{EMBED_MODEL}:predict"
        )

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        token = await self._ensure_token()
        url = self._embed_url()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        results: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i:i + BATCH_SIZE]
            payload = {
                "instances": [{"content": t} for t in chunk],
                "parameters": {"outputDimensionality": EMBED_DIM},
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                log_step("EMBED", f"Embedding API error (HTTP {resp.status_code})",
                         error=resp.text[:200])
                results.extend([] for _ in chunk)
                continue
            data = resp.json()
            predictions = data.get("predictions", [])
            for pred in predictions:
                emb = pred.get("embeddings") or pred
                vals = emb.get("values") if isinstance(emb, dict) else emb
                results.append(list(vals) if vals else [])
        return results

    async def build_index(self, tables_raw: list[dict]) -> None:
        log_step("EMBED", "Building embedding index for tables and columns",
                 count=len(tables_raw))

        self._table_names = []
        self._table_docs = []
        self._column_embeddings = {}

        all_table_texts: list[str] = []
        table_index_map: list[int] = []

        for t in tables_raw:
            name = t["name"]
            desc = t.get("description") or ""
            col_names = [c["name"] for c in t.get("columns", []) if c.get("name")]
            col_descs = []
            for c in t.get("columns", []):
                if c.get("name"):
                    cdesc = c.get("description") or ""
                    col_descs.append(f"{c['name']}: {cdesc}" if cdesc else c['name'])

            table_doc = f"{name} {' '.join(col_names)} {desc}"
            self._table_names.append(name)
            self._table_docs.append(table_doc)
            all_table_texts.append(table_doc)
            table_index_map.append(len(all_table_texts) - 1)

            col_doc_map: dict[str, np.ndarray] = {}
            for c in t.get("columns", []):
                if c.get("name"):
                    cdesc = c.get("description") or ""
                    cdoc = f"{c['name']}: {cdesc}" if cdesc else c["name"]
                    col_doc_map[c["name"]] = np.zeros(EMBED_DIM, dtype=np.float32)
                    all_table_texts.append(cdoc)
            self._column_embeddings[name] = col_doc_map

        all_vectors = await self._embed_batch(all_table_texts)
        if not all_vectors:
            log_step("EMBED", "Embedding failed, index not built")
            return

        table_count = len(self._table_names)
        table_vecs = np.array(all_vectors[:table_count], dtype=np.float32)

        col_idx = table_count
        for name in self._table_names:
            col_map = self._column_embeddings[name]
            for col_name in col_map:
                if col_idx < len(all_vectors):
                    col_map[col_name] = np.array(
                        all_vectors[col_idx], dtype=np.float32
                    )
                    col_idx += 1

        self._table_embeddings = table_vecs
        self._ready = True
        log_step("EMBED", f"Index built: {len(self._table_names)} tables, "
                 f"{col_idx - table_count} columns")

    async def embed_question(self, question: str) -> np.ndarray | None:
        vecs = await self._embed_batch([question])
        if not vecs:
            return None
        return np.array(vecs[0], dtype=np.float32)

    async def search_tables(
        self, question: str, top_k: int = 3, min_score: float = 0.15,
        q_vec: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        if not self._ready:
            return []

        if q_vec is None:
            q_vec = await self.embed_question(question)
        if q_vec is None:
            return []

        q_vec_2d = q_vec.reshape(1, -1)
        scores = cosine_similarity(q_vec_2d, self._table_embeddings).flatten()

        top_indices = scores.argsort()[::-1]
        results: list[tuple[str, float]] = []
        for idx in top_indices:
            if scores[idx] >= min_score:
                results.append((self._table_names[idx], float(scores[idx])))
            if len(results) >= top_k:
                break

        log_step("EMBED", f"Table search returned {len(results)} tables",
                 top_score=float(scores[top_indices[0]]) if len(scores) > 0 else 0)
        return results

    async def rank_columns(
        self, question: str, table_name: str, top_k: int = 5,
        q_vec: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        if not self._ready:
            return []

        col_map = self._column_embeddings.get(table_name)
        if not col_map or not col_map:
            return []

        col_names = list(col_map.keys())
        col_vecs = np.array([col_map[c] for c in col_names], dtype=np.float32)

        if np.all(col_vecs == 0):
            return [(c, 0.0) for c in col_names[:top_k]]

        if q_vec is None:
            q_vec = await self.embed_question(question)
        if q_vec is None:
            return [(c, 0.0) for c in col_names[:top_k]]

        q_vec_2d = q_vec.reshape(1, -1)
        scores = cosine_similarity(q_vec_2d, col_vecs).flatten()

        top_indices = scores.argsort()[::-1]
        results: list[tuple[str, float]] = []
        for idx in top_indices:
            results.append((col_names[idx], float(scores[idx])))
            if len(results) >= top_k:
                break

        return results

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def table_names(self) -> list[str]:
        return self._table_names
