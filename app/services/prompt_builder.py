from app.models.schemas import SchemaContext


class PromptBuilder:
    @staticmethod
    def build_system_prompt() -> str:
        return (
            "You are an expert SQL analyst for a logistics company using "
            "the Innofulfill platform. Generate precise, read-only PostgreSQL "
            "SELECT queries from business questions and schema context.\n\n"

            "--- INNOFULFILL DOMAIN ---\n"
            "Booking (First Mile) — shipment creation, AWB, pickup\n"
            "HubOps (Middle Mile) — sorting, inscan, outscan, bagging. "
            "Statuses: RETURN_INITIATED, RTO_DELIVERED (UPPER_SNAKE_CASE)\n"
            "Dispatch (Last Mile) — delivery runs, NDR, RTO. "
            "Statuses: rto, rto_delivered (lowercase)\n"
            "Rathsetu (TMS) — vehicle trips, route planning\n\n"

            "--- RTO LIFECYCLE ---\n"
            "Initiated → Out for Delivery → Delivery Failed → Delivered\n"
            "Dispatch statuses: rto → rto_out_for_delivery → rto_undelivered → rto_delivered\n"
            "Hubops statuses: RETURN_INITIATED → RTO_OUT_FOR_DELIVERY → "
            "RTO_UNDELIVERED → RTO_DELIVERED\n"
            "RTO_DELIVERED means shipment reached sender's CP.\n\n"

            "--- TABLE TIERS ---\n"
            "t1_ = raw data, t2_ = joined/harmonized, "
            "t3_ = report/aggregate — PREFER t3_ as FROM target\n\n"

            "--- DOMAIN GLOSSARY ---\n"
            "AWB = tracking number, DRS = delivery run sheet, POD = proof of delivery\n"
            "CP = Channel Partner (franchise/agent), NDR = non-delivery report\n"
            "Inscan = receipt at hub, Outscan = dispatch from hub, OFD = out for delivery\n"
            "NDD = next delivery date, TAT = turnaround time, EDD = estimated delivery date\n"
            "Terminal statuses: DELIVERED, RTO_DELIVERED\n\n"

            "--- RULES ---\n"
            "1. Return ONLY raw SQL. No markdown, no code fences, no explanation.\n"
            "2. Fully qualify names: schema_name.table_name\n"
            "3. ONLY use columns listed below. Never invent columns.\n"
            "4. SELECT only. No INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE.\n"
            "5. Prefer t3_ tables as FROM target.\n"
            "6. Add meaningful aliases for aggregations (e.g. COUNT(*) AS total).\n"
            "7. LIMIT 100 unless COUNT query.\n"
            "8. PostgreSQL syntax: CURRENT_DATE, INTERVAL, ::DATE, date_trunc, EXTRACT.\n"
            "9. GROUP BY column if question says 'by X' or 'on X level'.\n"
            "10. For RTO: WHERE last_terminal_status = 'RTO_DELIVERED' or "
            "status = 'rto_delivered'.\n"
            "11. If cannot answer, return exactly: UNABLE_TO_GENERATE"
        )

    @staticmethod
    def build_user_prompt(
        question: str,
        schema_context: SchemaContext,
        enrichment: dict | None = None,
    ) -> str:
        lines: list[str] = []

        has_t3 = any(t.table_name.startswith("t3_") for t in schema_context.tables)
        if has_t3:
            lines.append(
                "NOTE: t3_ tables are the primary reporting tables. "
                "Prefer them as your FROM target.\n"
            )

        for table in schema_context.tables:
            desc = table.description or "No description"
            tier = "REPORT" if table.table_name.startswith("t3_") else \
                   "MID" if table.table_name.startswith("t2_") else "RAW"
            lines.append(
                f"[{tier}] {table.schema_name}.{table.table_name} — {desc}"
            )
            col_parts = []
            for col in table.columns:
                col_desc = (col.description or "")[:60]
                if col_desc:
                    col_parts.append(f"{col.name} ({col.data_type}): {col_desc}")
                else:
                    col_parts.append(f"{col.name} ({col.data_type})")
            if col_parts:
                lines.append("  Columns: " + ", ".join(col_parts))
            else:
                lines.append("  Columns: (none)")
            relevant_deps = [
                t for t in table.related_tables
                if t.startswith("t3_") or t.startswith("t2_")
            ]
            if relevant_deps:
                joined = ", ".join(
                    f"{t} (via DEPENDS_ON)" for t in relevant_deps
                )
                lines.append(f"  Joins with: {joined}")
            lines.append("")

        if enrichment:
            join_hints = enrichment.get("join_hints", [])
            if join_hints:
                lines.append("--- JOIN HINTS (for combining tables) ---")
                for h in join_hints[:4]:
                    lines.append(
                        f"  {h['source_table']}.{h['source_col']} = "
                        f"{h['target_table']}.{h['target_col']}"
                    )
                lines.append("")

        lines.append(f"Business Question: {question}")
        return "\n".join(lines)
