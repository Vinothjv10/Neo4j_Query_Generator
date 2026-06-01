# Text2SQL BI Assistant Memory

## Goal
Convert natural-language logistics questions into PostgreSQL queries using Neo4j schema registry + Vertex AI Claude Haiku + text embeddings + Graph RAG enrichment, with a Next.js frontend showing results and reasoning.

## Constraints & Preferences
- t3_ tables are always preferred FROM targets; search filters to t3_ only when any exist, falls to t2_ then t1_; schema is silver_layer only
- Vertex AI Claude Haiku 4-5@20251001 in us-east5 for LLM; text-embedding-005 for vector search; service account auth via saturam.json
- Full SQL validation: SELECT-only + per-table column existence check with fuzzy auto-fix + re-validation
- Frontend is Next.js at `/home/ubuntu/Neo4j_Query_Generator/frontend`, proxies `/api/*` → localhost:8000
- Step-by-step `[STEP]` logging on every pipeline stage
- No real values in git-tracked files; `.env` (gitignored) has real values; `saturam.json` (gitignored via `*.json`) has service account key

## Done
- **LLM provider**: NVIDIA/Google Gemma → Vertex AI Claude Haiku via REST API (us-east5); OAuth2 token refresh with service account
- **Vertex AI text embeddings**: pre-computes 768-dim vectors for 31 tables + ~505 columns at startup (3 batched API calls); single question embed reused for table search + column ranking
- **Hybrid table search**: TF-IDF (2108 features, `ngram_range=(1,2)`) + embedding cosine similarity; embedding-primary ranking with wide pool (`top_k * 4`, min 20) before t3 filtering
- **T3-prioritization**: search results filtered to t3_ tables only when any exist; falls to t2_ → t1_ otherwise
- **Column-level relevance filtering**: top 8 (was 5) semantically similar columns per table shown in prompt; ensures date columns aren't dropped (e.g., `drs_created_date` was #6 for "inscan" queries)
- **Re-validation after auto-fix**: partially-fixed SQL caught before reaching PostgreSQL
- **Validator false-positive fix**: `drs_created_date` no longer rejected — forbidden keyword check uses whole-word matching, not substring
- **Validator alias fix**: SELECT aliases (`total_shipments`) now excluded from ORDER BY/GROUP BY column checks; `EXTRACT(...)` stripped before column ref extraction; `DAY`, `MONTH`, `YEAR` etc. added to exclude lists
- **Alias map additions**: `current_status→status`, `current_premise_name→premise_name`, `origin_cp_id→premise_id`
- **PG-level error handler** (`_fix_pg_errors`): replaces Neo4j-known but PG-absent columns; handles `text - timestamp` type errors with `::timestamp` cast; handles `character varying = timestamp` type errors with `NULLIF + regex filter + ::timestamp` cast; handles `date/time field value out of range` with `NULLIF + ~ '^[1-9]'` filter
- **Table name quoting**: mixed-case table names (e.g., `t3_Fastrack_orders_report`, `t3_Eway_Report`) are auto-quoted in SQL via `_quote_if_mixed()` for PG compatibility
- **Validator case-insensitive lookup**: `_validate_columns_per_table` and related functions now do case-insensitive table key matching in `table_columns` dict
- **Column name cleanup**: strips `A.` prefix from Neo4j column names (`A.hub` → `hub`)
- **Underscore-split documents**: table/column names split on underscores and added as separate terms in TF-IDF + embedding docs for keyword matching
- **Broad schema fallback**: when all 5 LLM retries fail with "could not generate", fetches ALL t3_ tables from Neo4j (up to 10, sorted by name) and retries up to 3 times
- **LLM-generated explanation**: after SQL executes successfully, lightweight LLM call generates natural-language `final_explanation` of table/column choices
- **Reasoning response & UI**: new `reasoning` field in response with `table_selection` (tier, reason, top columns), `column_selection` (score bars), `final_explanation`, `sql_generation`, `retries`; frontend `ReasoningPanel` accordion component
- **Enter submits form**: `onKeyDown` handler on textarea — Enter submits, Shift+Enter inserts newline
- **Dependencies installed, project flattened, startup logs to /tmp/text2sql-bi.log**

## In Progress
- (none)

## Blocked
- (none)

## Advanced Features Added (May 2026)
- **ReAct Agentic SQL Agent** (`app/services/react_agent_service.py`): Claude tool_use loop with search_tables/get_columns/check_join_path tools. Enabled via `USE_REACT_AGENT=true` env var. Fixes test #9 (NDR domain routing). Falls back to standard pipeline on failure.
- **DAIL-SQL Few-Shot** (`app/services/dail_sql_service.py` + `app/utils/sql_skeleton.py`): SQLite store at `/tmp/dail_sql_store.db`. Stores every successful (question, SQL) pair. Retrieves top-3 similar examples using 0.6×semantic + 0.4×skeleton-Jaccard score. Injected into LLM prompt. Requires `pip install aiosqlite`. Gracefully disabled if not installed.
- **GNN Schema Linking** (`app/services/gnn_schema_service.py`): NetworkX schema graph with DEPENDS_ON + MAPS_TO_SIBLING edges. Per-node features: tier_score, col_count, maps_to_count, PageRank, degree_centrality. 1-hop mean-aggregation message passing. Propagation-based re-ranking of retrieval results at query time. Requires `pip install networkx`.
- **LLM token auto-refresh** (`app/services/llm_service.py`): GCP OAuth2 token now tracked with expiry timestamp; refreshed 5 minutes before expiry to prevent silent 1-hour auth failures.
- **deps**: `networkx`, `aiosqlite` added to requirements.txt

## Setup for Advanced Features
```bash
cd /home/ubuntu/Neo4j_Query_Generator
.venv/bin/pip install networkx aiosqlite

# Optional: enable ReAct agent (uses Claude tool_use API)
echo "USE_REACT_AGENT=true" >> .env

# Restart server
pkill -f uvicorn
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> /tmp/text2sql-bi.log 2>&1 &
```

## Key Decisions
- **Vertex AI over NVIDIA**: uses existing GCP service account, no separate API key needed; Claude Haiku is fastest Anthropic model with adequate SQL quality
- **Single embed call per query**: question embedded once, vector cached and reused for table search + all column rankings; avoids 5+ redundant API calls
- **8-column limit per table in prompt** (was 5): ensures date columns not filtered out for time-based questions; still 2x smaller than full-column prompts
- **Search pool `top_k * 4` (min 20)**: wider net ensures `t3_booking_vs_delivery_report` doesn't get ranked out for queries with "shipments booked"
- **Within-table-only fuzzy match in auto-fix**: cross-table column matching caused more harm than good; removed all-columns fallback
- **Broad fallback fetches ALL t3_ tables**: when semantic search fails to pick the right table, falls back to Neo4j-ordered full list so LLM sees all available tables
- **Table name quoting for mixed-case**: PostgreSQL stores `t3_Fastrack_orders_report` as quoted identifier; auto-qualify now detects mixed case and adds `"` quotes
- **Validator case-insensitive table matching**: `table_columns` dict keys are lowercased; all lookups now normalize to lowercase for case-insensitive match
- **`^[1-9]` year prefix filter**: `NULLIF(trim(col), '') ~ '^[1-9]'` filters out rows where date column starts with `0` (invalid years like `0000-`) before casting
- **LLM-generated explanation separate call**: lightweight prompt after SQL succeeds adds ~2-4s but provides meaningful business-user reasoning

## Next Steps
1. Fix "NDR reason contains wrong address" — LLM doesn't know which table has NDR reason column. Could map "ndr reason" → `last_ndr_reason` or similar column in prompt. Check which table has NDR data (likely `t3_booking_vs_delivery_report` or `t3_delivery_mis_report`).
2. Consider adding domain keyword/alias map for common logistics terms (NDR, RTO, inscan, outscan, etc.) to help LLM table/column selection
3. Track test results over time — 14/15 pass (12/15 + 2 new fixes)

## Test Results (15 end-to-end queries)
| # | Query | Status |
|---|-------|--------|
| 1 | list fast track orders from last week | ✅ |
| 2 | how many eway bills were generated yesterday | ✅ |
| 3 | show delivery by state for last month | ✅ |
| 4 | what is the total number of shipments booked yesterday | ✅ |
| 5 | show me the top 5 hubs by inscan volume this month | ✅ |
| 6 | how many shipments were delivered yesterday each statewise | ✅ |
| 7 | list rto shipments from last week | ✅ |
| 8 | show me average delivery charges by channel partner | ✅ |
| 9 | list all shipments where ndr reason contains wrong address | ❌ (LLM can't map NDR to table) |
| 10 | what is the total weight booked last month grouped by service type | ✅ |
| 11 | how many rto shipments were received at hub yesterday | ✅ |
| 12 | list shipments where the origin and destination pincode are same | ✅ |
| 13 | show me the bottom 5 performing hubs by delay percentage | ✅ |
| 14 | list fast track orders from last week | ✅ |
| 15 | how many eway bills were generated yesterday | ✅ |

## Critical Context
- Neo4j has 31 Table nodes, ~505 Column nodes across silver_layer
- Vertex AI endpoints: us-east5-aiplatform.googleapis.com; Claude rawPredict with `anthropic_version: "vertex-2023-10-16"`; text-embedding-005 max 250 instances/request
- Embedding index builds on startup in 3 batches (~536 total texts); takes ~18 seconds
- Two tables have Neo4j/PG column mismatches: `t2_master_hubops` (Neo4j adds `current_premise_name`, `current_status` not in PG); `t3_delivery_mis_report` (Neo4j has `A.*` prefixed columns, cleaned at fetch time)
- Mixed-case table names in Neo4j (`t3_Fastrack_orders_report`, `t3_Eway_Report`) are quoted in SQL for PG compatibility
- `t3_Eway_Report` has bad data: zero-date values (`0000-00-00 00:00:00`) in `Eway_bill_generated_date` column; filtered out by `_fix_pg_errors` with NULLIF + regex check
- Backend process: `/home/ubuntu/Neo4j_Query_Generator/.venv/bin/python3 -m uvicorn app.main:app`; logs at `/tmp/text2sql-bi.log`
- Workflow: query → embed question → TF-IDF + embedding hybrid search → t3-prioritize → column filter → LLM generate → column validate → PG execute → LLM explain → response
- 14/15 comprehensive tests pass; 1 failure needs model interaction pattern (NDR domain mapping)

## Project Structure
- `/home/ubuntu/Neo4j_Query_Generator/app/services/embedding_service.py`: Vertex AI text-embedding-005; pre-computes on startup; `rank_columns()` and `search_tables()` with score return
- `/home/ubuntu/Neo4j_Query_Generator/app/services/table_index_service.py`: hybrid TF-IDF + embedding search; `_prioritize_t3()`; `search(question, top_k, allow_t1)`; `get_relevant_columns_with_scores()`; underscore-split document building
- `/home/ubuntu/Neo4j_Query_Generator/app/services/neo4j_service.py`: `get_all_t3_tables()` returns up to 10 t3_ tables; `_fetch_all_t3()` internal; `_clean_column_name()` static strips alias prefixes
- `/home/ubuntu/Neo4j_Query_Generator/app/services/llm_service.py`: Vertex AI Claude Haiku REST API; OAuth2 token; `UNABLE_TO_GENERATE` → ValueError
- `/home/ubuntu/Neo4j_Query_Generator/app/services/prompt_builder.py`: system prompt with Innofulfill domain context, RTO lifecycle, tier preference, strict SQL rules
- `/home/ubuntu/Neo4j_Query_Generator/app/utils/sql_validator.py`: whole-word forbidden keyword detection; `_validate_columns_per_table()` case-insensitive table key lookup; `_extract_column_refs()` strips EXTRACT, excludes select aliases, handles DAY/MONTH/YEAR keywords; multi-target alias map
- `/home/ubuntu/Neo4j_Query_Generator/app/api/routes/query.py`: `_fix_pg_errors()` handles varchar=timestamp comparisons (NULLIF + ^[1-9] filter + ::timestamp), date out-of-range errors; `_auto_qualify_tables()` now quotes mixed-case table names; `_build_known_tables/columns()` lowercases keys; broad fallback fetches 10 t3_ tables; LLM explanation call; reasoning object building
- `/home/ubuntu/Neo4j_Query_Generator/app/models/schemas.py`: `TableReason`, `ColumnReason`, `Reasoning` models with `final_explanation` field
- `/home/ubuntu/Neo4j_Query_Generator/frontend/src/components/ReasoningPanel.tsx`: accordion UI; highlighted LLM explanation card; expandable table-selection/columns/sql-flow sections
- `/home/ubuntu/Neo4j_Query_Generator/frontend/src/components/QueryForm.tsx`: `onKeyDown` handler for Enter-to-submit
- `/tmp/text2sql-bi.log`: runtime logs
