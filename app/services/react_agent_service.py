"""
ReAct Agentic SQL Agent.

Uses Claude's tool_use capability to interactively explore the Neo4j schema
graph before committing to SQL generation.  The agent can call:

  search_tables(query)           → top matching tables with tier + description
  get_columns(table_name)        → full column list with types + descriptions
  check_join_path(t1, t2)        → MAPS_TO join conditions between two tables

The agent runs up to MAX_AGENT_STEPS tool-call rounds.  When Claude stops
calling tools (stop_reason == "end_turn") the final text block is returned
as the SQL.  Falls back gracefully to the standard pipeline on failure.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from app.utils.logger import log_step

if TYPE_CHECKING:
    from app.services.llm_service import LLMService
    from app.services.table_index_service import TableIndexService
    from app.services.graph_rag_service import GraphRAGService
    from app.services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 10

# ── Tool definitions sent to Claude ─────────────────────────────────────────

AGENT_TOOLS: list[dict] = [
    {
        "name": "search_tables",
        "description": (
            "Search for database tables relevant to a concept, keyword, or domain term. "
            "Returns table names, tiers (t3_=report/preferred, t2_=enriched, t1_=raw), "
            "and descriptions. Call this first to discover which tables contain the data you need. "
            "You can call it multiple times with different queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query describing the data you need, e.g. "
                        "'NDR reason wrong address', 'hub inscan volume', "
                        "'delivery status RTO', 'eway bill generation'"
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_columns",
        "description": (
            "Get all columns for a specific table including column names, data types, "
            "and descriptions. Always call this before writing SQL to verify that the "
            "columns you plan to use actually exist in the table."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": (
                        "Exact table name as returned by search_tables, "
                        "e.g. 't3_delivery_mis_report', 't3_booking_vs_delivery_report'"
                    ),
                }
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "check_join_path",
        "description": (
            "Find the join condition between two tables using Neo4j MAPS_TO relationships. "
            "Returns the column(s) to use in the JOIN ON clause. "
            "Call this when you need to join two tables together."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table1": {
                    "type": "string",
                    "description": "First table name",
                },
                "table2": {
                    "type": "string",
                    "description": "Second table name",
                },
            },
            "required": ["table1", "table2"],
        },
    },
]

# ── System prompt for the agent ──────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """\
You are an expert PostgreSQL analyst for Innofulfill, a logistics platform.
Your task: generate a precise SELECT query for the given business question.

Use the provided tools to EXPLORE THE SCHEMA before writing SQL:
1. Call search_tables to find relevant tables for the question.
2. Call get_columns on each promising table to see exact column names and types.
3. If joining tables, call check_join_path to find the correct join condition.
4. Once you have verified all tables and columns exist, write the final SQL.

DOMAIN GLOSSARY:
  NDR = non-delivery report  |  RTO = return to origin  |  AWB = tracking number
  CP = channel partner       |  DRS = delivery run sheet |  OFD = out for delivery
  Inscan = receipt at hub    |  Outscan = dispatch from hub
  t3_ = report/aggregate (ALWAYS PREFER as FROM target)
  t2_ = enriched/joined      |  t1_ = raw source data

SQL RULES:
  • Fully qualify every table: silver_layer.table_name
  • Mixed-case table names must be double-quoted: silver_layer."t3_Eway_Report"
  • SELECT only — no INSERT, UPDATE, DELETE, DROP, ALTER, CREATE
  • LIMIT 100 unless it's a COUNT/aggregation query
  • PostgreSQL syntax: CURRENT_DATE, INTERVAL, date_trunc(), EXTRACT()
  • GROUP BY any non-aggregated column in SELECT
  • Use meaningful aliases: COUNT(*) AS total_shipments

OUTPUT FORMAT:
  When you have explored enough to write the SQL, respond with ONLY the raw SQL.
  No markdown fences. No explanation. No commentary.
  If you truly cannot generate SQL, respond with exactly: UNABLE_TO_GENERATE
"""


# ── Service class ────────────────────────────────────────────────────────────

class ReactAgentService:
    def __init__(
        self,
        llm_service: "LLMService",
        table_index_service: "TableIndexService",
        graph_rag_service: "GraphRAGService",
        neo4j_service: "Neo4jService",
    ) -> None:
        self._llm = llm_service
        self._tis = table_index_service
        self._grag = graph_rag_service
        self._neo4j = neo4j_service

    async def generate_sql(
        self, question: str
    ) -> tuple[str, list[str]]:
        """
        Run the ReAct agent loop.

        Returns
        -------
        (sql, agent_trace)
            sql         – Raw SQL string produced by the agent.
            agent_trace – Human-readable list of tool calls and results.

        Raises
        ------
        ValueError
            If the agent exceeds MAX_AGENT_STEPS or returns UNABLE_TO_GENERATE.
        """
        messages: list[dict] = [{"role": "user", "content": question}]
        trace: list[str] = []

        for step in range(MAX_AGENT_STEPS):
            response = await self._llm.generate_with_tools(
                AGENT_SYSTEM_PROMPT, messages, AGENT_TOOLS,
                max_tokens=4096, temperature=0.0,
            )

            stop_reason: str = response.get("stop_reason", "end_turn")
            content: list[dict] = response.get("content", [])

            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(block)

            # Append the assistant's full turn (including tool_use blocks)
            messages.append({"role": "assistant", "content": content})

            if stop_reason == "end_turn" or not tool_calls:
                sql_text = " ".join(text_parts).strip()
                log_step("AGENT", f"Agent finished in {step + 1} step(s)", chars=len(sql_text))
                if not sql_text:
                    raise ValueError("ReAct agent produced no SQL text")
                if "UNABLE_TO_GENERATE" in sql_text:
                    raise ValueError(
                        "ReAct agent could not find a suitable table/column combination"
                    )
                return self._clean_sql(sql_text), trace

            # Execute every tool call and collect results
            tool_results: list[dict] = []
            for tc in tool_calls:
                name = tc.get("name", "")
                inputs = tc.get("input", {})
                tool_id = tc.get("id", "")

                trace.append(f"[step {step + 1}] {name}({json.dumps(inputs, ensure_ascii=False)})")
                log_step("AGENT", f"Tool: {name}", inputs=inputs)

                result_str = await self._dispatch_tool(name, inputs)
                short = result_str[:300] + ("…" if len(result_str) > 300 else "")
                trace.append(f"  → {short}")
                log_step("AGENT", f"Tool result: {name}", result=short)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_str,
                })

            # Feed tool results back as a user turn
            messages.append({"role": "user", "content": tool_results})

        raise ValueError(
            f"ReAct agent exceeded {MAX_AGENT_STEPS} steps without producing SQL"
        )

    # ── Tool dispatcher ──────────────────────────────────────────────────────

    async def _dispatch_tool(self, name: str, inputs: dict) -> str:
        try:
            if name == "search_tables":
                return await self._tool_search_tables(inputs.get("query", ""))
            if name == "get_columns":
                return await self._tool_get_columns(inputs.get("table_name", ""))
            if name == "check_join_path":
                return await self._tool_check_join_path(
                    inputs.get("table1", ""), inputs.get("table2", "")
                )
            return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as exc:
            logger.exception("Tool %s failed: %s", name, exc)
            return json.dumps({"error": str(exc)})

    # ── Tool implementations ─────────────────────────────────────────────────

    async def _tool_search_tables(self, query: str) -> str:
        if not query.strip():
            return json.dumps({"error": "query must not be empty"})
        entries = await self._tis.search(query, top_k=8, allow_t1=False)
        results = []
        for e in entries:
            if e.table_name.startswith("t3_"):
                tier = "t3_REPORT (preferred)"
            elif e.table_name.startswith("t2_"):
                tier = "t2_ENRICHED"
            else:
                tier = "t1_RAW"
            results.append({
                "table": e.table_name,
                "tier": tier,
                "description": (e.description or "No description")[:120],
            })
        return json.dumps({"tables": results, "count": len(results)}, ensure_ascii=False)

    async def _tool_get_columns(self, table_name: str) -> str:
        if not table_name.strip():
            return json.dumps({"error": "table_name must not be empty"})
        tables = await self._neo4j._fetch_tables_by_names([table_name])
        if not tables:
            return json.dumps({
                "error": f"Table '{table_name}' not found in schema",
                "hint": "Use search_tables to discover the correct table name",
            })
        tbl = tables[0]
        cols = [
            {
                "name": c.name,
                "type": c.data_type,
                "description": (c.description or "")[:80],
            }
            for c in tbl.columns
        ]
        return json.dumps({
            "table": table_name,
            "schema": tbl.schema_name,
            "description": tbl.description or "",
            "column_count": len(cols),
            "columns": cols,
        }, ensure_ascii=False)

    async def _tool_check_join_path(self, table1: str, table2: str) -> str:
        if not table1 or not table2:
            return json.dumps({"error": "Both table1 and table2 are required"})
        enrichment = await self._grag.enrich([table1, table2])
        join_hints = enrichment.get("join_hints", [])
        relevant = [
            h for h in join_hints
            if h.get("source_table") in {table1, table2}
            and h.get("target_table") in {table1, table2}
        ]
        if not relevant:
            # Try reverse direction
            all_hints = join_hints
            relevant = all_hints[:4]  # show any available hints as fallback
        return json.dumps({
            "table1": table1,
            "table2": table2,
            "join_hints": relevant,
            "note": "Use these column pairs in your JOIN ON clause",
        }, ensure_ascii=False)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_sql(raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = [ln for ln in raw.splitlines() if not ln.startswith("```")]
            raw = "\n".join(lines).strip()
        import re as _re
        # Strip any preamble before the first SELECT
        select_idx = _re.search(r'\bSELECT\b', raw, _re.IGNORECASE)
        if select_idx:
            before = raw[:select_idx.start()].strip()
            if before:
                raw = raw[select_idx.start():]
        return raw
