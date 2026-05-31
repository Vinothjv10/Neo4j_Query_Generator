import re

from app.utils.logger import log_step

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
    "ALTER", "CREATE", "EXEC", "EXECUTE",
]


def validate_sql(sql: str) -> None:
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        log_step("VALIDATOR", "REJECTED - not a SELECT query", sql=sql.replace("\n", " "))
        raise ValueError("Only SELECT queries are allowed.")

    upper = stripped.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if kw in upper:
            log_step("VALIDATOR", f"REJECTED - forbidden keyword {kw}", sql=sql.replace("\n", " "))
            raise ValueError(f"Forbidden SQL keyword detected: {kw}")

    log_step("VALIDATOR", "PASSED - SELECT-only check")


def validate_columns_in_sql(
    sql: str,
    known_columns: set[str],
    known_tables: set[str] | None = None,
    table_columns: dict[str, set[str]] | None = None,
) -> str:
    if table_columns:
        return _validate_columns_per_table(sql, table_columns)
    refs = _extract_column_refs(sql)
    refs = [c for c in refs if known_tables is None or c not in known_tables]
    unknown = [c for c in refs if c not in known_columns]
    if unknown:
        msg = "Columns not found in schema: " + ", ".join(suggestions)
        log_step("VALIDATOR", f"REJECTED - unknown columns", unknown=suggestions)
        raise ValueError(msg)
    log_step("VALIDATOR", f"PASSED - column validation ({len(refs)} refs checked)")
    return sql


def _validate_columns_per_table(sql: str, table_columns: dict[str, set[str]]) -> str:
    import re as _re

    tables_in_sql: set[str] = set()
    alias_to_table: dict[str, str] = {}
    for m in _re.finditer(
        r"(?:^|\s)(?:FROM|JOIN)\s+(?:(?:silver_layer|public|gold_layer|bronze_layer)\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?",
        sql, _re.IGNORECASE | _re.MULTILINE,
    ):
        preceding = sql[max(0, m.start() - 20):m.start()]
        if _re.search(r'EXTRACT\s*\(', preceding, _re.IGNORECASE):
            continue
        tbl = m.group(1)
        alias = (m.group(2) or tbl).upper()
        tables_in_sql.add(tbl)
        alias_to_table[alias] = tbl
        alias_to_table[tbl.upper()] = tbl

    qualified_refs: list[tuple[str, str]] = []
    for m in _re.finditer(r"([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)", sql):
        q = m.group(1)
        if q.lower() in {"silver_layer", "public", "gold_layer", "bronze_layer"}:
            continue
        qualified_refs.append((q, m.group(2)))

    all_refs = set(_extract_column_refs(sql))
    qualified_cols = {c for _, c in qualified_refs}
    bare_refs = all_refs - qualified_cols
    bare_refs = {
        c for c in bare_refs
        if c.upper() not in {"CASE", "WHEN", "THEN", "ELSE", "END", "NULL", "1", "0"}
        and c not in tables_in_sql
    }

    unknown: list[str] = []

    for qualifier, col in qualified_refs:
        table = alias_to_table.get(qualifier.upper())
        if table and table in table_columns:
            if col not in table_columns[table]:
                close = _did_you_mean(col, table_columns[table])
                hint = f" Did you mean '{close}'?" if close else ""
                unknown.append(f"{qualifier}.{col}{hint}")
        elif table:
            unknown.append(f"{qualifier}.{col} (table '{table}' not in context)")

    if tables_in_sql:
        tbls_list = sorted(tables_in_sql)
        all_tbl_cols: set[str] = set()
        for tbl in tables_in_sql:
            all_tbl_cols.update(table_columns.get(tbl, set()))
        for col in bare_refs:
            if col not in all_tbl_cols:
                close = _did_you_mean(col, all_tbl_cols)
                hint = f" Did you mean '{close}'?" if close else ""
                unknown.append(f"{col} (not in tables {tbls_list}){hint}")
    else:
        all_known: set[str] = set()
        for cols in table_columns.values():
            all_known.update(cols)
        for col in bare_refs:
            if col not in all_known:
                unknown.append(col)

    if unknown:
        fixed_sql = _auto_fix_columns(sql, unknown, tables_in_sql, table_columns, alias_to_table)
        if fixed_sql != sql:
            log_step("VALIDATOR", "AUTO-FIX applied columns", fixed_sql=fixed_sql.replace("\n", " "))
            return fixed_sql
        detail = _build_rich_error(sql, unknown, tables_in_sql, table_columns)
        log_step("VALIDATOR", "REJECTED - column not in target table", unknown=unknown)
        raise ValueError(detail)
    log_step("VALIDATOR", f"PASSED - per-table column validation ({len(all_refs)} refs checked)")
    return sql


def _build_rich_error(
    sql: str,
    unknown: list[str],
    tables_in_sql: set[str],
    table_columns: dict[str, set[str]],
) -> str:
    lines: list[str] = []
    lines.append("Your SQL uses columns that do not exist on the queried tables.")
    lines.append("")

    parsed_unknown: list[dict] = []
    for entry in unknown:
        parts = entry.split(" (")
        col = parts[0].strip()
        msg = parts[1].rstrip(")") if len(parts) > 1 else ""
        parsed_unknown.append({"column": col, "message": msg})

    for u in parsed_unknown:
        col = u["column"]
        tbl_name = None
        if "." in col:
            qual, raw_col = col.split(".", 1)
            for tbl in tables_in_sql:
                if qual.lower() == tbl.lower():
                    tbl_name = tbl
                    break
            if not tbl_name:
                for alias_tbl in tables_in_sql:
                    if qual.lower() in alias_tbl.lower() or alias_tbl.lower() in qual.lower():
                        tbl_name = alias_tbl
                        break
        else:
            raw_col = col

        if tbl_name:
            lines.append(f"  ERROR: Column '{raw_col}' does not exist on table '{tbl_name}'")
            avail = sorted(table_columns.get(tbl_name, set()))
            close = _did_you_mean(raw_col, set(avail))
            lines.append(f"  Available columns on '{tbl_name}': {', '.join(avail[:15])}")
            if len(avail) > 15:
                lines.append(f"    ... and {len(avail) - 15} more columns")
            if close:
                lines.append(f"  SUGGESTION: Use '{close}' instead of '{raw_col}'")
            lines.append("")
        else:
            lines.append(f"  ERROR: Unknown column '{col}'")
            all_avail: set[str] = set()
            for tbl in tables_in_sql:
                all_avail.update(table_columns.get(tbl, set()))
            close = _did_you_mean(col, all_avail)
            if close:
                lines.append(f"  SUGGESTION: Use '{close}' instead of '{col}'")
            lines.append("")

    lines.append("Fix the SQL and return ONLY corrected SQL using columns from the schema above.")

    if tables_in_sql:
        lines.append("")
        lines.append("SCHEMA REMINDER for the table you are querying:")
        for tbl in tables_in_sql:
            if tbl in table_columns:
                avail = sorted(table_columns[tbl])
                lines.append(f"  '{tbl}' columns: {', '.join(avail[:20])}")
                if len(avail) > 20:
                    lines.append(f"    ... and {len(avail) - 20} more")

    return "\n".join(lines)


def _auto_fix_columns(
    sql: str,
    unknown: list[str],
    tables_in_sql: set[str],
    table_columns: dict[str, set[str]],
    alias_to_table: dict[str, str],
) -> str:
    import re as _re

    _ALIAS_MAP: dict[str, str] = {
        "awb_number": "documentno",
        "tracking_id": "documentno",
    }

    replacements: dict[str, str] = {}

    for entry in unknown:
        col = entry.split(" (")[0].split(".")[-1].strip()
        if col in _ALIAS_MAP:
            mapped = _ALIAS_MAP[col]
            for tbl in tables_in_sql:
                if mapped in table_columns.get(tbl, set()):
                    replacements[col] = mapped
                    break
            if col in replacements:
                continue
        best_candidate = None
        for tbl in tables_in_sql:
            tbl_cols = table_columns.get(tbl, set())
            if col not in tbl_cols:
                candidates = _find_column_fuzzy(col, tbl_cols)
                if candidates:
                    best_candidate = candidates[0]
                    break
        if best_candidate:
            replacements[col] = best_candidate
        else:
            all_cols: set[str] = set()
            for cols in table_columns.values():
                all_cols.update(cols)
            candidates = _find_column_fuzzy(col, all_cols)
            if candidates:
                replacements[col] = candidates[0]

    result = sql
    for old_col, new_col in replacements.items():
        result = _re.sub(
            r"\b" + _re.escape(old_col) + r"\b",
            new_col,
            result,
        )
    return result


def _find_column_fuzzy(col: str, candidates: set[str]) -> list[str]:
    col_lower = col.lower().replace("_", "")
    scored: list[tuple[int, str]] = []
    for c in candidates:
        c_lower = c.lower().replace("_", "")
        if col_lower == c_lower:
            return [c]
        if col_lower in c_lower or c_lower in col_lower:
            scored.append((abs(len(col_lower) - len(c_lower)), c))
    scored.sort()
    return [c for _, c in scored[:1]]


def _extract_column_refs(sql: str) -> list[str]:
    stripped = re.sub(r"--.*", "", sql)
    stripped = re.sub(r"'.*?'", "", stripped)
    stripped = re.sub(r"\".*?\"", "", stripped)
    aliases = _collect_table_aliases(stripped)
    col_refs: set[str] = set()

    schemas = {"silver_layer", "public", "gold_layer", "bronze_layer"}
    for m in re.finditer(
        r"([a-zA-Z_]\w*)\s*\.\s*([a-zA-Z_]\w*)",
        stripped,
    ):
        if m.group(1).lower() not in schemas:
            col_refs.add(m.group(2))

    for m in re.finditer(
        r"([a-zA-Z_]\w+)\s*::",
        stripped,
    ):
        candidate = m.group(1).upper()
        if candidate not in {"TIMESTAMP", "DATE", "TEXT", "INTEGER", "BIGINT",
                              "FLOAT", "DOUBLE", "NUMERIC", "BOOLEAN", "VARCHAR",
                              "INTERVAL"}:
            col_refs.add(m.group(1))

    for m in re.finditer(r"\b(?:COUNT|SUM|AVG|MIN|MAX|COALESCE)\s*\(\s*(?:DISTINCT\s+)?(?:(?:\w+\.)?(\w+))", stripped, re.IGNORECASE):
        col_refs.add(m.group(1))

    select_section = _get_select_section(stripped)
    if select_section:
        select_aliases: set[str] = set()
        for m in re.finditer(r"\bAS\s+([a-zA-Z_]\w+)", select_section, re.IGNORECASE):
            select_aliases.add(m.group(1).upper())
        for m in re.finditer(
            r"(?:AS\s+)?([a-zA-Z_]\w+)\s*(?:,|FROM|FROM\s+\w+|$)",
            select_section,
        ):
            candidate = m.group(1).upper()
            if (
                candidate
                not in {"SELECT", "AS", "FROM", "DISTINCT", "COUNT", "SUM",
                         "AVG", "MIN", "MAX", "COALESCE", "CASE", "WHEN",
                         "THEN", "ELSE", "END", "NULL"}
                and candidate not in aliases
                and candidate not in select_aliases
            ):
                col_refs.add(m.group(1))

    where_section = _get_where_section(stripped)
    if where_section:
        for m in re.finditer(r"([a-zA-Z_]\w+)", where_section):
            candidate = m.group(1).upper()
            if candidate not in {
                "AND", "OR", "IN", "NOT", "NULL", "IS", "BETWEEN", "LIKE",
                "TRUE", "FALSE", "CURRENT_DATE", "CURRENT_TIMESTAMP", "DATE",
                "INTERVAL", "DAY", "MONTH", "YEAR", "EXTRACT", "FROM",
                "TO_CHAR", "DATE_TRUNC", "CAST", "AS",
                "CASE", "WHEN", "THEN", "ELSE", "END",
            } and candidate not in aliases:
                col_refs.add(m.group(1))

    group_cols = _get_groupby_section(stripped)
    if group_cols:
        for m in re.finditer(r"([a-zA-Z_]\w+)", group_cols):
            candidate = m.group(1).upper()
            if candidate not in aliases | {"CASE", "WHEN", "THEN", "ELSE", "END", "NULL"}:
                col_refs.add(m.group(1))

    order_cols = _get_orderby_section(stripped)
    if order_cols:
        for m in re.finditer(r"([a-zA-Z_]\w+)", order_cols):
            candidate = m.group(1).upper()
            if candidate not in {"ASC", "DESC", "CASE", "WHEN", "THEN", "ELSE", "END", "NULL"} | aliases:
                col_refs.add(m.group(1))

    return list(col_refs)


def _collect_table_aliases(sql: str) -> set[str]:
    aliases: set[str] = set()
    for m in re.finditer(
        r"(?:^|\s)(?:FROM|JOIN)\s+(?:\w+\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?",
        sql,
        re.IGNORECASE | re.MULTILINE,
    ):
        preceding = sql[max(0, m.start() - 20):m.start()]
        if re.search(r'EXTRACT\s*\(', preceding, re.IGNORECASE):
            continue
        if m.group(2):
            aliases.add(m.group(2).upper())
        aliases.add(m.group(1).upper())
    return aliases


def _get_select_section(sql: str) -> str:
    m = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else ""


def _get_where_section(sql: str) -> str:
    m = re.search(r"WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else ""


def _get_groupby_section(sql: str) -> str:
    m = re.search(r"GROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|\s+HAVING|\s+LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else ""


def _get_orderby_section(sql: str) -> str:
    m = re.search(r"ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else ""


def _did_you_mean(col: str, known: set[str]) -> str | None:
    col_lower = col.lower()
    candidates = [k for k in known if col_lower in k.lower() or k.lower() in col_lower]
    if candidates:
        return min(candidates, key=lambda c: abs(len(c) - len(col)))
    return None
