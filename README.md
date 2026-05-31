# Text2SQL BI — Natural Language to SQL Query Generator

Turn plain English questions like *"how many shipments yesterday"* into PostgreSQL
queries automatically, using Neo4j schema registry + Vertex AI LLM + vector
embeddings.

---

## Architecture Overview

```
User Question
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  1. TABLE RETRIEVAL  (Neo4j + TF-IDF + Vector Embeddings)   │
│     Finds the right database tables for the question        │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  2. COLUMN FILTERING  (Vector Embeddings cosine similarity)  │
│     Picks only the most relevant columns per table           │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  3. GRAPH RAG ENRICHMENT  (Neo4j relationships)              │
│     Finds join paths between tables (MAPS_TO, DEPENDS_ON)   │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  4. PROMPT BUILDING  (Template assembly)                     │
│     Builds a compact prompt with schema + joins              │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  5. SQL GENERATION  (Vertex AI Claude Haiku)                 │
│     LLM reads prompt and generates PostgreSQL query          │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  6. VALIDATION  (SELECT-only + column existence check)       │
│     Rejects invalid SQL, auto-fixes column names             │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  7. EXECUTION  (PostgreSQL via asyncpg)                      │
│     Runs the query and returns results to the user           │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
   JSON Response
```

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.12** | Backend language |
| **FastAPI** | Web framework (REST API) |
| **Neo4j** | Schema registry — stores table/column metadata and relationships |
| **PostgreSQL** | Target database — runs the generated SQL queries |
| **Vertex AI (GCP)** | LLM inference (Claude Haiku) + Text Embeddings (text-embedding-005) |
| **scikit-learn** | TF-IDF vectorizer trained on logistics table/column descriptions for domain-specific table search |
| **NumPy** | Vector math for cosine similarity |
| **asyncpg** | Async PostgreSQL driver |
| **Pydantic** | Settings management + request/response models |
| **Google Auth** | Service account authentication for Vertex AI |

---

## File Structure

```
/home/ubuntu/Neo4j_Query_Generator/
├── .env                        # Real credentials (gitignored)
├── .env.example                # Variable names only (git-safe)
├── .gitignore
├── saturam.json                # GCP service account key (gitignored)
├── requirements.txt
├── README.md
│
├── app/
│   ├── main.py                 # FastAPI app, startup, lifespan
│   ├── config.py               # Settings from .env file
│   │
│   ├── api/
│   │   └── routes/
│   │       └── query.py        # The main /api/v1/query endpoint
│   │
│   ├── models/
│   │   └── schemas.py          # Request/Response Pydantic models
│   │
│   ├── services/
│   │   ├── neo4j_service.py    # Fetches schema from Neo4j
│   │   ├── table_index_service.py  # TF-IDF + hybrid search
│   │   ├── embedding_service.py    # Vertex AI text embeddings
│   │   ├── graph_rag_service.py    # Join hints from Neo4j
│   │   ├── prompt_builder.py       # Builds LLM prompts
│   │   ├── llm_service.py          # Calls Claude via Vertex AI
│   │   ├── postgres_service.py     # Executes SQL on PostgreSQL
│   │   └── tot_service.py          # Tree of Thoughts (disabled)
│   │
│   └── utils/
│       ├── logger.py           # [STEP] logging utility
│       └── sql_validator.py    # Validates generated SQL
│
└── frontend/                   # Next.js UI (optional)
```

---

## Detailed Step-by-Step Query Processing

### Step 0: Startup (When Server Starts)

When `uvicorn app.main:app` runs, the **lifespan** function executes:

1. **Connect to Neo4j** — opens a persistent connection to the graph database
2. **Fetch all 31 tables** — queries Neo4j for every `Table` node + their `Column` nodes
3. **Build TF-IDF index** — creates a searchable index using `scikit-learn`'s `TfidfVectorizer`. Each table becomes a "document" made of: `table_name + description + all_column_names + column_descriptions`
4. **Build Embedding index** — sends all table + column documents to **Vertex AI text-embedding-005** in batches of 200. Each document becomes a 768-dimensional vector. Stored in memory as NumPy arrays.
5. **Ready** — server accepts requests at `http://localhost:8000`

Log output:
```
[STEP] INDEX | Building TF-IDF table index from Neo4j
[STEP] INDEX | Index built: 31 tables, 2108 features
[STEP] EMBED | Building embedding index for tables and columns | count=31
[STEP] EMBED | Index built: 31 tables, 505 columns
```

### Step 1: Table Retrieval (`neo4j_service.py` + `table_index_service.py`)

When a user sends `{"question": "how many shipments yesterday"}`:

1. **TF-IDF Search** — the question is vectorized using the same TF-IDF vectorizer built at startup. Cosine similarity finds the closest-matching table documents. Returns up to 9 candidate tables.

2. **Embedding Search** — the question is sent to Vertex AI `text-embedding-005` (1 API call). The returned 768-dim vector is compared against all 31 pre-computed table vectors using cosine similarity. Tables with score ≥ 0.15 are kept.

3. **Hybrid Merge** — TF-IDF candidates + embedding candidates are merged. The final list is sorted by embedding similarity score. Top 5 tables are selected.

4. **Fetch Full Schema** — for each selected table name, Neo4j is queried for all columns + data types + descriptions + MAPS_TO relationships.

5. **Result** — a `SchemaContext` object with 5 `TableInfo` objects, each containing ALL their columns.

Log output:
```
[STEP] NEO4J | Using semantic TF-IDF search for table retrieval
[STEP] INDEX | TF-IDF search returned 9 tables (top score=0.1670)
[STEP] INDEX | Hybrid search returned 5 tables | method=embedding_primary
[STEP] NEO4J | Returning 5 tables
```

### Step 2: Column Filtering (`embedding_service.py`)

Before building the prompt, the system filters columns to keep only the most relevant ones.

1. The **same question vector** from Step 1 is reused (no extra API call)
2. For each of the 5 tables, the query vector is compared against that table's pre-computed column vectors using cosine similarity
3. Top 5 most semantically similar columns are kept per table
4. A **filtered copy** of `SchemaContext` is created with only those columns

This is why the prompt is small — instead of showing 12 arbitrary columns per table (which might include irrelevant ones), it shows only the 5 columns that best match the question's intent.

Log output:
```
[STEP] COLUMNS | Filtered to relevant columns per table | tables_with_filters=['t3_booking_vs_delivery_report', ...]
```

### Step 3: Graph RAG Enrichment (`graph_rag_service.py`)

In parallel, the system queries Neo4j for join information:

1. **DEPENDS_ON** — finds which tables the selected tables depend on
2. **MAPS_TO** — finds column-level mappings (e.g., `t3_booking.awb_number` maps to `t2_hubops.documentno`)
3. **Join Hints** — generates `table1.column = table2.column` pairs for all t3_/t2_ table combinations

Log output:
```
[STEP] GRAPH_RAG | Enriched context: 4 dep chains, 43 col mappings
```

### Step 4: Prompt Building (`prompt_builder.py`)

Two prompts are built:

**System Prompt** (2,474 chars) — the "personality" and rules:
- "You are an expert SQL analyst for a logistics company"
- Domain glossary (CP, AWB, DRS, POD, etc.)
- RTO process description
- Rules: only SELECT, use PostgreSQL syntax, prefer t3_ tables, return only raw SQL

**User Prompt** (~2,800 chars) — the actual context + question:
- Table list with only filtered columns (5 per table)
- Column descriptions and data types
- Join hints from Graph RAG (max 4)
- The business question

The prompt is intentionally small (~5.2k total vs ~9.4k before) so the LLM processes
it faster.

Log output:
```
[STEP] PROMPT | Built prompts (system=2474 chars, user=2768 chars)
```

### Step 5: SQL Generation (`llm_service.py`)

1. **Authenticate** — loads the GCP service account key from `saturam.json`, gets an OAuth2 access token
2. **Call API** — sends a POST request to:
   ```
   POST https://us-east5-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict
   ```
3. **Request body** — Anthropic Messages API format:
   ```json
   {
     "anthropic_version": "vertex-2023-10-16",
     "messages": [{"role": "user", "content": "<user_prompt>"}],
     "system": "<system_prompt>",
     "max_tokens": 2048,
     "temperature": 0.10
   }
   ```
4. **Parse response** — extracts the `content[0].text` field from the Anthropic response
5. **Clean** — removes markdown code fences if present

Log output:
```
[STEP] LLM | Calling Claude on Vertex AI | model=claude-haiku-4-5@20251001
[STEP] LLM | SQL generated (attempt 1) | sql=SELECT COUNT(*) ...
```

### Step 6: Validation (`sql_validator.py`)

Three checks run on the generated SQL:

**a) SELECT-only check** — rejects if the SQL contains INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE, EXEC

**b) Column existence check** — for each table referenced in FROM/JOIN:
- Parses all qualified (`table.col`) and bare (`col`) column references
- Checks every column exists in the schema for its table
- If unknown columns found, builds a **rich error message** with:
  - Which column is wrong on which table
  - Available columns on that table (first 20)
  - A fuzzy-match suggestion
  - Full SCHEMA REMINDER section

**c) Auto-fix fallback** — if column check fails:
- Hardcoded aliases: `awb_number`→`documentno`, `tracking_id`→`documentno`
- Fuzzy matching: strips underscores, lowercases, checks substring containment
- If auto-fix succeeds, the fixed SQL is used

If validation fails, the error is sent back to the LLM as a "CORRECTION" hint for the
next retry attempt (up to 5 retries allowed).

Log output:
```
[STEP] VALIDATOR | PASSED - SELECT-only check
[STEP] VALIDATOR | PASSED - per-table column validation
```

### Step 7: PostgreSQL Execution (`postgres_service.py`)

1. **Add LIMIT** — appends `LIMIT {top_k}` if not already present
2. **Connect** — opens async connection to PostgreSQL using the DSN from `.env`
3. **Execute** — runs the SQL query via `asyncpg.fetch()`
4. **Serialize** — converts `Decimal`→`float`, `date/datetime`→ISO string
5. **Return** — list of dictionaries (one per row)

If execution fails, two fallback attempts try:
- Fixing table aliases (if table is aliased but bare name used in WHERE)
- Auto-fixing column names

Log output:
```
[STEP] POSTGRES | Executing query | top_k=5
[STEP] POSTGRES | Query returned 1 rows
```

### Step 8: Response

The API returns a JSON response:

```json
{
  "question": "how many shipments yesterday",
  "generated_sql": "SELECT COUNT(*) ...",
  "results": [{"shipment_count": 10825}],
  "row_count": 1,
  "schema_tables_used": ["silver_layer.t3_booking_vs_delivery_report", ...]
}
```

---

## Configuration

All configuration lives in `.env` (gitignored):

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

POSTGRES_DSN=postgresql://user:password@localhost:5432/dbname

GOOGLE_APPLICATION_CREDENTIALS=./saturam.json
VERTEX_AI_PROJECT=saturam
VERTEX_AI_LOCATION=us-east5
VERTEX_AI_MODEL=claude-haiku-4-5@20251001
```

These are read by `config.py` via `pydantic-settings`, which automatically maps
`VERTEX_AI_PROJECT` ↔ `vertex_ai_project` (snake_case ↔ UPPER_CASE).

---

## Key Design Decisions

### Why Hybrid Search (TF-IDF + Embeddings)?
- **TF-IDF** is instant (no network calls), good for keyword matching
- **Embeddings** understand semantics (e.g., "shipment" matches "booking" if they're semantically related)
- Together: TF-IDF catches exact keyword matches, embeddings catch semantic intent

### Why Pre-compute Embeddings?
- All 31 tables + 505 columns are embedded once at startup (3 API calls × 200 items)
- At query time, only the **question** needs embedding (1 API call)
- Column similarity is computed locally via NumPy (microseconds)
- Result: fast queries (~10s total, mostly LLM inference time)

### Why Only 5 Columns Per Table in Prompt?
- Shows only the most relevant columns → LLM doesn't get confused by irrelevant columns
- Smaller prompt → faster LLM inference → lower latency
- Validation still checks ALL columns → no regression risk

### Why Claude Haiku on Vertex AI?
- Fast inference (Haiku is Anthropic's fastest model)
- Accessed through GCP Vertex AI REST API (no separate API key needed)
- Service account authentication (secure, no hardcoded keys)
- Available in `us-east5` region for this project

---

## API Endpoints

### `GET /api/v1/health`
Returns `{"status": "ok"}` if the server is running.

### `POST /api/v1/query`
Request:
```json
{
  "question": "how many shipments yesterday",
  "top_k": 100
}
```

Response:
```json
{
  "question": "how many shipments yesterday",
  "generated_sql": "SELECT COUNT(*) ...",
  "results": [{"shipment_count": 10825}],
  "row_count": 1,
  "schema_tables_used": ["silver_layer.t3_booking_vs_delivery_report", ...]
}
```

Error response (422):
```json
{
  "detail": {
    "error": "Failed to generate valid SQL after 5 attempts",
    "detail": "Column 'trip_end_hub_id' does not exist..."
  }
}
```

---

## Running the Application

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies (one-time)
pip install -r requirements.txt

# 3. Set up .env with your credentials
cp .env.example .env
# Edit .env with real values

# 4. Place GCP service account key
mv your-key.json saturam.json

# 5. Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 6. Test it
curl http://localhost:8000/api/v1/health
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "how many shipments yesterday"}'
```

---

## Neo4j Schema (Graph Database)

The Neo4j database acts as a **schema registry** with this structure:

```
(:Table {name, schema, description})
    │
    ├── [:HAS_COLUMN] → (:Column {name, data_type, description})
    │       │
    │       └── [:MAPS_TO] → (:Column)  ← (column-level mapping between tables)
    │
    └── [:DEPENDS_ON] → (:Table)        ← (table-level dependency)
```

There are currently **31 Table** nodes, **~500 Column** nodes, and **126 MAPS_TO**
relationships across the graph.

---

## Logging

Every step logs with a `[STEP]` tag for easy grep filtering:

```bash
# Watch live logs
tail -f /tmp/text2sql-bi.log

# Filter specific steps
grep "LLM\|POSTGRES\|VALIDATOR" /tmp/text2sql-bi.log

# Check for errors
grep "ERROR\|RETRY" /tmp/text2sql-bi.log
```

---

## Troubleshooting

**Q: Server won't start**
- Check `.env` file exists and has all required variables
- Check `saturam.json` has valid GCP service account credentials
- Verify Neo4j is running: `curl http://localhost:7474`

**Q: "Service account file not found"**
- Check `GOOGLE_APPLICATION_CREDENTIALS` path in `.env`
- Verify the file exists and is readable

**Q: "Vertex AI API error"**
- Check the model is available in your region
- Verify the service account has `aiplatform.user` role
- Check GCP project has Vertex AI API enabled

**Q: "Neo4j schema fetch failed"**
- Verify Neo4j credentials in `.env`
- Check Neo4j is running and accessible

**Q: "Query execution failed"**
- Verify PostgreSQL credentials in `.env`
- Check the SQL syntax manually in a psql client
- Look for hallucinated column names in the generated SQL
