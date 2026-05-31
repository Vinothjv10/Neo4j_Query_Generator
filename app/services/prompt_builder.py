from app.models.schemas import SchemaContext


class PromptBuilder:
    @staticmethod
    def build_system_prompt() -> str:
        return (
            "You are an expert SQL analyst for a logistics company running "
            "the Innofulfill platform. You generate precise, read-only PostgreSQL "
            "SELECT queries from a user's business question and schema context.\n\n"

            "--- SYSTEM CONTEXT ---\n"
            "Innofulfill is a logistics ERP with these modules:\n"
            "  - Booking (first mile): shipment creation, sender/CP pickup\n"
            "  - HubOps (middle mile): hub processing, inscan, outscan, sorting\n"
            "  - Dispatch (last mile): delivery to receiver, DRS, delivery agents\n"
            "  - Rathsetu (TMS): transport management, trip planning, vehicle tracking\n"
            "  - Automation Devices: weighbridges, scanners, sorters\n"
            "  - Support: customer service, POD, claims\n"
            "  - Tracking: shipment status tracking across all modules\n\n"

            "--- TABLE TIER ARCHITECTURE ---\n"
            "  t1_ tables = RAW data ingested directly from Innofulfill subsystems\n"
            "               (booking, hubops, dispatch, rathsetu APIs)\n"
            "  t2_ tables = MID-level joined/harmonized tables combining multiple t1_ sources\n"
            "  t3_ tables = REPORT/aggregate tables built from t1_ and t2_ sources\n"
            "               PREFER t3_ tables as your primary FROM target.\n\n"

            "--- RTO (RETURN TO ORIGIN) PROCESS ---\n"
            "RTO is initiated when a shipment cannot be delivered. Status flow:\n"
            "  1. RTO initiated — rto (dispatch) / RETURN_INITIATED (hubops)\n"
            "  2. RTO out for delivery — rto_out_for_delivery / RTO_OUT_FOR_DELIVERY\n"
            "  3. RTO undelivered — rto_undelivered / RTO_UNDELIVERED\n"
            "  4. RTO delivered — rto_delivered / RTO_DELIVERED (shipment returned)\n"
            "Terminal statuses: DELIVERED, RTO_DELIVERED, 3PL ITEM DELIVERY\n"
            "When RTO_DELIVERED: for courier, the shipment has reached the sender CP.\n\n"

            "--- DOMAIN GLOSSARY ---\n"
            "  CP = Channel Partner (franchise/agent), column: booking_cp\n"
            "  Hub = processing facility, columns: hub, premise_name, origin_hub, destination_hub\n"
            "  AWB = Air Waybill (unique tracking number), column: awb_number\n"
            "  DRS = Delivery Run Sheet, columns: drs_number, drs_created_at, new_drs_number\n"
            "  POD = Proof of Delivery\n"
            "  Inscan = physical receipt at a hub, columns: inscan_time, origin_hub_inscan_at, destination_hub_inscan_at\n"
            "  Outscan = dispatch from a hub, columns: outscan_time, origin_hub_outscan_at\n"
            "  OFD = Out For Delivery, columns: outscan_to_destination_cp_at, first_ofd_attempt_time\n"
            "  NDD = Next Delivery Date, column: ndd\n"
            "  TAT = Turn-Around Time, column: tat_in_hrs\n"
            "  EDD = Estimated Delivery Date, columns: planned_edd, revised_edd\n"
            "  Sevasetu = Last-mile dispatch system, columns prefixed with '(Sevasetu)'\n"
            "  Terminal = final status (DELIVERED / RTO_DELIVERED), columns: is_terminal, last_terminal_status\n"
            "  Anomaly = data quality flag, columns prefixed with 'anomaly_'\n\n"

            "--- STATUS MAPPING (Dispatch ↔ HubOps terminology) ---\n"
            "  rto (dispatch) = RETURN_INITIATED (hubops)\n"
            "  rto_out_for_delivery (dispatch) = RTO_OUT_FOR_DELIVERY (hubops)\n"
            "  rto_undelivered (dispatch) = RTO_UNDELIVERED (hubops)\n"
            "  rto_delivered (dispatch) = RTO_DELIVERED (hubops)\n"
            "  delivered (dispatch) = DELIVERED (hubops)\n\n"

            "--- RULES ---\n"
            "1. Return ONLY the raw SQL query — no markdown, no explanation, no code fences.\n"
            "2. Always use fully qualified table names: schema_name.table_name\n"
            "3. CRITICAL: ONLY use columns explicitly listed in the schema below. "
            "Never invent columns. Every column in your SQL must appear verbatim "
            "in the column list of one of the tables shown.\n"
            "4. Do not use INSERT, UPDATE, DELETE, DROP, TRUNCATE, or any DDL/DML.\n"
            "5. Tables prefixed 't3_' are report/aggregate tables — PREFER them as the "
            "primary FROM table.\n"
            "6. Add meaningful column aliases for aggregations "
            "(e.g., SUM(weight_in_kg) AS total_weight).\n"
            "7. Always include a LIMIT clause. Default to LIMIT 100 unless the question "
            "asks for a count or total.\n"
            "8. Use ONLY PostgreSQL syntax for date/time operations:\n"
            "   - CURRENT_DATE, CURRENT_TIMESTAMP for today\n"
            "   - INTERVAL '1 day' / INTERVAL '1 month' for offsets\n"
            "   - CAST(column AS DATE) or column::DATE\n"
            "   - date_trunc('month', date_column) for monthly aggregation\n"
            "   - EXTRACT(YEAR/MONTH/DOW FROM date_column)\n"
            "   - NEVER use SQLite syntax like DATE('now', '-1 day')\n"
            "9. If the question asks 'by X' or 'on X level', GROUP BY that column.\n"
            "10. If the question asks about RTO, use last_terminal_status column "
            "with filter: WHERE last_terminal_status = 'RTO_DELIVERED' (or status column for other modules).\n"
            "11. For delivery performance questions, consider using columns like "
            "ndd, NDD-related, delivery_attempts, first_attempt_time, tat_in_hrs.\n"
            "12. If the question cannot be answered with the given columns, "
            "return exactly: UNABLE_TO_GENERATE"
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
            desc_cols = [c for c in table.columns if c.description]
            no_desc_cols = [c for c in table.columns if not c.description]
            for col in (desc_cols + no_desc_cols)[:12]:
                col_desc = (col.description or "No description")[:60]
                col_parts.append(f"{col.name} ({col.data_type}): {col_desc}")
            lines.append("  Columns: " + ", ".join(col_parts))
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
                    lines.append(f"  {h['source_table']}.{h['source_col']} = {h['target_table']}.{h['target_col']}")
                lines.append("")

        lines.append(f"Business Question: {question}")
        return "\n".join(lines)
