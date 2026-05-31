from decimal import Decimal
from datetime import date, datetime
import asyncpg

from app.config import settings
from app.utils.logger import log_step


class PostgresService:
    async def execute_query(self, sql: str, top_k: int = 100) -> list[dict]:
        final_sql = self._ensure_limit(sql, top_k)
        log_step("POSTGRES", f"Executing query", top_k=top_k)

        conn = await asyncpg.connect(settings.postgres_dsn)
        try:
            records = await conn.fetch(final_sql)
            log_step("POSTGRES", f"Query returned {len(records)} rows")
        finally:
            await conn.close()

        return [self._row_to_dict(row) for row in records]

    def _ensure_limit(self, sql: str, top_k: int) -> str:
        if "limit" not in sql.lower():
            return f"{sql.rstrip(';')} LIMIT {top_k}"
        return sql

    def _row_to_dict(self, row: asyncpg.Record) -> dict:
        result: dict = {}
        for key, value in dict(row).items():
            result[key] = self._serialize(value)
        return result

    def _serialize(self, value: object) -> object:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return value
