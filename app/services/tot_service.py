import logging
from dataclasses import dataclass, field

from app.models.schemas import SchemaContext
from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder
from app.utils.logger import log_step
from app.utils.sql_validator import (
    validate_sql,
    validate_columns_in_sql,
)

logger = logging.getLogger(__name__)


@dataclass
class CandidateSQL:
    sql: str
    score: float = 0.0
    validation_errors: list[str] = field(default_factory=list)
    passed_select_check: bool = False
    passed_column_check: bool = False


class ToTService:
    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    async def generate_best(
        self,
        system_prompt: str,
        user_prompt: str,
        known_columns: set[str],
        known_tables: set[str],
        table_columns: dict[str, set[str]],
        schema_context: SchemaContext,
        num_candidates: int = 3,
    ) -> str:
        log_step(
            "TOT",
            f"Generating {num_candidates} SQL candidates via Tree of Thoughts",
        )

        candidates: list[CandidateSQL] = []

        strategies = [
            ("direct", user_prompt),
            (
                "aggregate_focus",
                user_prompt
                + "\n\nStrategy: Use aggregate functions (COUNT, SUM) "
                "with GROUP BY on the relevant dimension column.",
            ),
            (
                "join_context",
                user_prompt
                + "\n\nStrategy: If the question mentions multiple concepts, "
                "join relevant tables using MAPS_TO column relationships.",
            ),
        ]

        for i in range(min(num_candidates, len(strategies))):
            name, prompt = strategies[i]
            sql = await self._generate_candidate(name, system_prompt, prompt)
            if not sql:
                continue

            candidate = CandidateSQL(sql=sql)
            candidate = self._validate_candidate(candidate, known_columns, known_tables, table_columns)
            candidate.score = self._score_candidate(candidate, name, schema_context)
            candidates.append(candidate)

            log_step(
                "TOT",
                f"Candidate '{name}' score={candidate.score:.2f} "
                f"valid={candidate.passed_column_check}",
                sql=sql.replace("\n", " "),
            )

        if not candidates:
            log_step("TOT", "No candidates generated, falling back to single generation")
            return ""

        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]
        log_step(
            "TOT",
            f"Best candidate: score={best.score:.2f}, "
            f"valid={best.passed_column_check}",
            sql=best.sql.replace("\n", " "),
        )
        return best.sql

    async def _generate_candidate(
        self, name: str, system_prompt: str, user_prompt: str
    ) -> str | None:
        try:
            sql = await self._llm.generate_sql(system_prompt, user_prompt)
            return sql
        except Exception as exc:
            log_step("TOT", f"Candidate '{name}' generation failed", error=str(exc))
            return None

    def _validate_candidate(
        self,
        candidate: CandidateSQL,
        known_columns: set[str],
        known_tables: set[str],
        table_columns: dict[str, set[str]],
    ) -> CandidateSQL:
        try:
            validate_sql(candidate.sql)
            candidate.passed_select_check = True
        except ValueError as exc:
            candidate.validation_errors.append(str(exc))
            return candidate

        try:
            candidate.sql = validate_columns_in_sql(
                candidate.sql, known_columns, known_tables, table_columns
            )
            candidate.passed_column_check = True
        except ValueError as exc:
            candidate.validation_errors.append(str(exc))

        return candidate

    def _score_candidate(
        self,
        candidate: CandidateSQL,
        strategy_name: str,
        schema_context: SchemaContext,
    ) -> float:
        score = 0.0

        if candidate.passed_select_check:
            score += 2.0
        if candidate.passed_column_check:
            score += 3.0

        strategy_bonus = {
            "direct": 1.0,
            "aggregate_focus": 0.5,
            "join_context": 0.5,
        }
        score += strategy_bonus.get(strategy_name, 0.0)

        penalty = len(candidate.validation_errors) * 1.0
        score -= penalty

        sql_upper = candidate.sql.upper()
        if "COUNT" in sql_upper or "SUM" in sql_upper or "AVG" in sql_upper:
            score += 0.5

        tables_used = [
            t.table_name
            for t in schema_context.tables
            if t.table_name in candidate.sql
        ]
        if tables_used:
            score += min(len(tables_used) * 0.3, 1.0)

        return max(score, 0.0)
