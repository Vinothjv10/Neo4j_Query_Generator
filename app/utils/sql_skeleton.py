"""
SQL Skeleton Extraction for DAIL-SQL few-shot retrieval.

A skeleton reduces a SQL query to its structural shape by replacing:
  - concrete column names  → _
  - string/numeric values  → ?
  - table names            → TABLE
  - aggregation args       → *

This enables Jaccard similarity between skeletons of different queries
that have the same logical structure (e.g. both do GROUP BY + COUNT).

Example
-------
Input:
    SELECT hub, COUNT(*) AS total FROM silver_layer.t3_hub_report
    WHERE inscan_date >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY hub ORDER BY total DESC LIMIT 10

Skeleton:
    SELECT _ , AGG(*) FROM TABLE WHERE _ >= DATE_EXPR GROUP BY _ ORDER BY _ LIMIT ?
"""

from __future__ import annotations

import re


# Tokens that represent structural SQL keywords (kept verbatim in skeleton)
_CLAUSE_KEYWORDS = frozenset({
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL",
    "ON", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "UNION", "EXCEPT",
    "INTERSECT", "WITH", "AS", "AND", "OR", "NOT", "IN", "IS", "NULL", "BETWEEN",
    "CASE", "WHEN", "THEN", "ELSE", "END", "DISTINCT", "TOP", "ASC", "DESC",
    "EXTRACT", "INTERVAL", "CURRENT_DATE", "CURRENT_TIMESTAMP",
    "DATE_TRUNC", "COALESCE", "NULLIF", "CAST", "TRIM",
})

# Aggregation function names
_AGG_FUNCTIONS = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE"})

# Date/time expression markers
_DATE_EXPR_RE = re.compile(
    r"CURRENT_DATE|CURRENT_TIMESTAMP|NOW\(\)|DATE_TRUNC\s*\([^)]*\)|"
    r"INTERVAL\s+'[^']+'",
    re.IGNORECASE,
)

# String literals
_STRING_RE = re.compile(r"'[^']*'")

# Numeric literals (standalone integers/floats not part of identifiers)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# Schema-qualified table name: schema.table or just t3_xxx
_QUALIFIED_TABLE_RE = re.compile(
    r'\b(?:\w+\.)?"?(?:t[123]_\w+)"?\b', re.IGNORECASE
)

# Column alias definition: AS alias_name — we want to drop the alias name
_ALIAS_RE = re.compile(r'\bAS\s+\w+\b', re.IGNORECASE)


def extract_skeleton(sql: str) -> str:
    """
    Reduce *sql* to a structural skeleton string.

    Returns a normalised uppercase skeleton suitable for Jaccard comparison.
    """
    s = sql.strip()

    # 1. Replace date expressions before any other substitution
    s = _DATE_EXPR_RE.sub("DATE_EXPR", s)

    # 2. Replace string literals
    s = _STRING_RE.sub("?", s)

    # 3. Replace schema-qualified table references → TABLE
    s = _QUALIFIED_TABLE_RE.sub("TABLE", s)

    # 4. Replace aggregation calls → AGG(*)
    for fn in _AGG_FUNCTIONS:
        s = re.sub(
            rf'\b{fn}\s*\([^)]*\)', "AGG(*)", s, flags=re.IGNORECASE
        )

    # 5. Drop AS aliases
    s = _ALIAS_RE.sub("", s)

    # 6. Replace remaining identifiers that look like column names
    #    (lowercase words / underscore words not in SQL keyword set)
    tokens = re.split(r"(\s+|[,()=<>!]+|\bWHERE\b|\bON\b)", s, flags=re.IGNORECASE)
    result_tokens: list[str] = []
    for tok in tokens:
        upper = tok.strip().upper()
        if not upper or upper in _CLAUSE_KEYWORDS or upper in {"TABLE", "AGG(*)", "?", "DATE_EXPR"}:
            result_tokens.append(tok)
        elif re.match(r'^[A-Z_][A-Z0-9_]*$', upper) and upper not in _CLAUSE_KEYWORDS:
            # Likely a column name or identifier — replace with _
            result_tokens.append("_")
        else:
            result_tokens.append(tok)

    s = "".join(result_tokens)

    # 7. Replace remaining numeric literals
    s = _NUMBER_RE.sub("?", s)

    # 8. Collapse whitespace and upper-case
    s = re.sub(r'\s+', ' ', s).strip().upper()

    return s


def skeleton_similarity(s1: str, s2: str) -> float:
    """
    Jaccard similarity between the token sets of two skeletons.
    Returns a float in [0, 1].
    """
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 and not t2:
        return 1.0
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    return intersection / union if union > 0 else 0.0
