# How This Application Works — End to End

> **Who is this for?** Anyone who wants to understand what happens when you type a question and get a table of data back. No coding knowledge needed.

---

## The Big Picture in One Line

> You type a **plain English question** → the app figures out which database table to look in → writes a **SQL query** automatically → runs it → shows you the **results**.

---

## The Cast of Characters

Before we walk through the flow, meet the main pieces:

| Piece | What it is | Think of it as... |
|---|---|---|
| **Browser / Frontend** | The webpage you use | A chat window |
| **FastAPI Backend** | The Python server | The brain |
| **Neo4j** | A graph database holding schema info | A map of all tables and columns |
| **PostgreSQL** | The actual data warehouse | The filing cabinet with real data |
| **Vertex AI (Claude)** | Google's AI hosting (runs Claude Haiku) | A very smart translator |
| **Embeddings** | Numbers that represent meaning | A way to measure "how similar two texts are" |

---

## Step-by-Step Walk-Through

### You Type a Question

```
┌─────────────────────────────────┐
│   Browser (Next.js frontend)    │
│                                 │
│  "show top 5 hubs by inscan     │
│   volume this month"            │
│                                 │
│         [Submit]                │
└──────────────┬──────────────────┘
               │  POST /api/v1/query
               ▼
```

You type your question in the text box and press **Enter** or click **Submit**.
The frontend sends it to the backend server over the internet (HTTP request).

---

### Step 1 — Find the Right Tables

**Problem:** There are 31 tables in the database. Which one has "inscan" data?

The backend uses **two search methods at the same time** and combines them:

```
Your question
     │
     ├──► TF-IDF Search (keyword matching)
     │         Breaks question into words, finds tables
     │         whose names/descriptions contain those words
     │
     └──► Embedding Search (meaning matching)
               Converts your question into 768 numbers
               Compares those numbers to 768 numbers stored
               for each table → finds the most "similar" tables

Both results are merged → Top candidates picked
```

**Then it filters:** Tables starting with `t3_` are always preferred.
- `t3_` = Report tables (pre-aggregated, fast) ← **always try these first**
- `t2_` = Enriched tables (joined data)
- `t1_` = Raw source tables (last resort)

**GNN Boost (new):** After the search, a graph analysis runs. Tables that are more "central" in the schema (many connections) get a small score boost — like Google PageRank but for database tables.

**Result:** 3-5 table names are selected, e.g., `t3_hub_report`.

---

### Step 2 — Understand the Table Structure

The backend asks **Neo4j** (the schema map): *"What columns does this table have?"*

```
Backend ──► Neo4j ──► Returns:
                        Table: t3_hub_report
                        Columns:
                          - hub (VARCHAR)           ← hub name
                          - inscan_count (INTEGER)  ← what we need
                          - inscan_date (DATE)      ← for filtering
                          - outscan_count (INTEGER)
                          - ... (20 more columns)
```

**Column Filtering:** Not all 20+ columns are sent to the AI. Only the **top 8 most relevant** columns for your question are selected (using the same "meaning matching" / embedding technique). This keeps the AI focused and prevents confusion.

---

### Step 3 — Check for Similar Past Questions (DAIL-SQL) Domain-Adaptive Instruction Learning for SQL or Data-Aware Instruction Learning for SQL

**New feature:** Before calling the AI, the system checks if anyone has asked a similar question before.

```
Your question: "show top 5 hubs by inscan volume this month"
                          │
                          ▼
               SQLite example database
               (stores successful past queries)
                          │
          ┌───────────────┴───────────────────┐
          │ Similar past question found?       │
          │                                   │
          YES                                 NO
          │                                   │
          ▼                                   ▼
  Attach it as an example          Skip, go straight to AI
  for the AI to learn from
```

If a similar example is found, it's injected at the top of the AI's instructions like this:
> *"Here's how a similar question was answered before: [question + SQL]. Use this as a style guide."*

This dramatically improves accuracy because the AI sees real working examples from your own database.

---

### Step 4 — Build the AI Prompt

The backend assembles a complete instruction package for the AI:

```
┌─────────────────────────────────────────────────┐
│  SYSTEM INSTRUCTIONS (always the same)          │
│  - You are a logistics SQL expert               │
│  - Prefer t3_ tables                            │
│  - RTO means return-to-origin                   │
│  - NDR means non-delivery report                │
│  - etc.                                         │
├─────────────────────────────────────────────────┤
│  SCHEMA (changes per question)                  │
│  - Table: t3_hub_report                         │
│    Columns: hub, inscan_count, inscan_date...   │
│  - Join hints from Neo4j                        │
├─────────────────────────────────────────────────┤
│  EXAMPLES (if similar past queries exist)       │
│  - Example 1: "top hubs last week"              │
│    SQL: SELECT hub, COUNT(*)...                 │
├─────────────────────────────────────────────────┤
│  YOUR QUESTION                                  │
│  "show top 5 hubs by inscan volume this month"  │
└─────────────────────────────────────────────────┘
```

---

### Step 5 — AI Writes the SQL

#### Option A: Standard Mode (default)

```
Backend ──► Claude Haiku (on Google Vertex AI) ──► Returns SQL:

SELECT hub, SUM(inscan_count) AS total_inscan
FROM silver_layer.t3_hub_report
WHERE inscan_date >= date_trunc('month', CURRENT_DATE)
GROUP BY hub
ORDER BY total_inscan DESC
LIMIT 5
```

If the AI makes a mistake, the backend tries up to **5 times**, each time telling the AI exactly what went wrong.

#### Option B: ReAct Agent Mode (advanced, opt-in)

When enabled (`USE_REACT_AGENT=true`), the AI gets **tools** it can call, like a human exploring the database:

```
AI THINKING PROCESS:

Step 1: "I need inscan data. Let me search."
        → calls search_tables("hub inscan volume")
        → gets back: [t3_hub_report, t3_hub_ops_summary, ...]

Step 2: "Let me check what columns t3_hub_report has."
        → calls get_columns("t3_hub_report")
        → gets back: hub, inscan_count, inscan_date, ...

Step 3: "I have everything I need. Writing SQL now."
        → produces final SELECT query
```

This is like the difference between:
- **Standard mode**: Telling someone directions from a map (may have errors)
- **Agent mode**: Letting them explore the city themselves before giving directions (more accurate)

**The agent mode specifically fixes the "NDR wrong address" query (test #9)** because the AI can search for "NDR" directly and discover `t3_delivery_mis_report` has the `last_ndr_reason` column.

---

### Step 6 — Validate the SQL

Before running anything, the backend checks the AI's SQL:

```
Generated SQL
     │
     ├──► Safety check: Is it SELECT only?
     │         (Reject any INSERT, UPDATE, DELETE, DROP, etc.)
     │
     ├──► Column check: Do these columns actually exist?
     │         (Compare against the real column list from Neo4j)
     │
     ├──► Auto-fix: Can we fix small mistakes?
     │         (Replace column aliases, fix known typos)
     │
     └──► Qualify tables: Add "silver_layer." prefix if missing
               (PostgreSQL needs the full schema.table name)

If valid ──► Send to PostgreSQL
If invalid ──► Go back to Step 5 (retry with error message)
```

---

### Step 7 — Run the Query in PostgreSQL

```
Validated SQL
     │
     ▼
PostgreSQL (the real data warehouse)
     │
     ├──► Executes the SELECT query
     │
     ├──► If PostgreSQL gives an error:
     │         Auto-fix runs (e.g., fix date type issues, column aliases)
     │         Retry once
     │
     └──► Returns rows of data

Example result:
  hub          | total_inscan
  -------------|-------------
  Mumbai Hub   | 4,821
  Delhi Hub    | 3,905
  Pune Hub     | 2,344
  Bangalore Hub| 2,109
  Chennai Hub  | 1,876
```

---

### Step 8 — Store This as a Learning Example (DAIL-SQL)

After a successful query, the system saves it for future use:

```
Question + SQL + embedding ──► SQLite database
                                (grows over time)

Next time someone asks: "top hubs by scan count this week"
→ System finds this example and shows it to the AI
→ AI writes better SQL faster, with fewer retries
```

---

### Step 9 — Generate a Human-Readable Explanation

A second quick AI call generates a plain-English explanation of what the query did:

```
"I used the t3_hub_report table because it contains pre-aggregated
hub-level inscan data. I filtered for the current month using
date_trunc('month', CURRENT_DATE) and ranked hubs by total
inscan volume using SUM and ORDER BY DESC."
```

---

### Step 10 — Send Results Back to You

```
Backend packages everything:
  ✓ The SQL that was generated
  ✓ The table rows (results)
  ✓ Row count
  ✓ Which tables were used
  ✓ Reasoning panel data (why these tables/columns)
  ✓ Human explanation
  ✓ Agent trace (if agent mode was used)

──► Frontend receives it
──► Displays data table
──► Shows SQL with syntax highlighting
──► Shows "Reasoning" accordion (expandable)
──► You see your answer!
```

---

## The Complete Flow as One Diagram

```
YOU TYPE A QUESTION
         │
         ▼
   [Browser/Frontend]
         │ HTTP POST /api/v1/query
         ▼
   [FastAPI Backend]
         │
         ├──[Step 1] Search Neo4j schema map
         │           TF-IDF (Term Frequency – Inverse Document Frequency) + Embedding + GNN boost(Graph Neural Network)
         │           → Pick best 3-5 tables
         │
         ├──[Step 2] Fetch table columns from Neo4j
         │           → Filter to top 8 relevant columns
         │
         ├──[Step 3] Check DAIL-SQL example store
         │           → Attach similar past queries if found
         │
         ├──[Step 4] Build AI prompt
         │           (schema + examples + your question)
         │
         ├──[Step 5] Send to Claude AI on Vertex AI
         │    │
         │    ├── Standard: AI writes SQL directly
         │    └── Agent: AI uses tools to explore, then writes SQL
         │
         ├──[Step 6] Validate SQL (safety + column check)
         │           Retry up to 5 times if errors
         │
         ├──[Step 7] Run SQL in PostgreSQL
         │           Auto-fix common errors, retry once
         │
         ├──[Step 8] Save to DAIL-SQL example store
         │           (for future similar questions)
         │
         ├──[Step 9] AI generates plain-English explanation
         │
         └──[Step 10] Send results to browser
                      │
                      ▼
              YOU SEE THE DATA TABLE
```

---

## What Happens When Things Go Wrong

| Problem | What the system does |
|---|---|
| AI writes wrong column name | Auto-fix tries to correct it; retries up to 5 times |
| AI uses wrong table | Broad fallback: fetch ALL t3_ tables and try again |
| PostgreSQL type error | `_fix_pg_errors()` patches the SQL and retries once |
| Neo4j is down | Returns HTTP 503 "Schema service unavailable" |
| AI completely confused | Returns HTTP 422 "Failed to generate valid SQL" |
| Agent mode fails | Automatically falls back to standard pipeline |

---

## The Three New Advanced Features

### 1. ReAct Agent (opt-in)
**What:** The AI has tools. It explores the schema like a human would.
**When to use:** Complex questions, domain-specific terms (NDR, RTO inscan)
**Enable it:** Add `USE_REACT_AGENT=true` to your `.env` file

### 2. DAIL-SQL Few-Shot Learning
**What:** Every successful query is saved. Similar past queries are shown to the AI.
**When it helps:** After ~10 queries, the AI starts seeing relevant examples
**Requires:** `pip install aiosqlite` (then restart server)

### 3. GNN Schema Graph
**What:** Builds a mini "importance map" of your schema using graph math (PageRank). Tables with more connections to other tables get a small ranking boost.
**When it helps:** For tables that semantically don't match the question well but are structurally important
**Requires:** `pip install networkx` (then restart server)

---

## Install & Start Commands

```bash
# In your WSL terminal (ubuntu@Vinoth:~/Neo4j_Query_Generator)

# 1. Install new dependencies (correct command for this setup)
.venv/bin/python3 -m pip install networkx aiosqlite

# 2. Restart the server
pkill -f uvicorn
.venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
  >> /tmp/text2sql-bi.log 2>&1 &

# 3. Watch the logs
tail -f /tmp/text2sql-bi.log

# 4. Test a query
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "show top 5 hubs by inscan volume this month"}' \
  | python3 -m json.tool

# 5. To enable the ReAct Agent
echo "USE_REACT_AGENT=true" >> .env
# then restart the server again
```

---

## Glossary of Terms

| Term | What it means |
|---|---|
| **SQL** | The language databases understand — like English but for data |
| **Schema** | The structure/blueprint of a database (table names, column names) |
| **Embedding** | Turning a word/sentence into a list of numbers so a computer can measure similarity |
| **TF-IDF** | A keyword search technique — counts important words |
| **Cosine Similarity** | A way to measure "how similar" two embedding vectors are (0 = nothing in common, 1 = identical) |
| **Neo4j** | A graph database — stores relationships between things (Table has Column, Table depends on Table) |
| **PostgreSQL** | A relational database — stores rows and columns of real data |
| **Claude Haiku** | Anthropic's fastest AI model, hosted by Google on Vertex AI |
| **Vertex AI** | Google Cloud's AI hosting platform |
| **t3_ tables** | Report-level tables — pre-joined, pre-aggregated, preferred for queries |
| **GNN** | Graph Neural Network — uses a graph's structure to compute importance scores |
| **DAIL-SQL** | A technique from research: use past successful queries as examples for the AI |
| **ReAct** | Reason + Act — an AI technique where the model uses tools in a loop |
| **NDR** | Non-Delivery Report — when a shipment fails to deliver |
| **RTO** | Return to Origin — shipment sent back to the sender |
| **AWB** | Airway Bill — the shipment tracking number |
| **CP** | Channel Partner — a franchise/agent location |
| **Inscan** | Shipment scanned in at a hub |
| **Outscan** | Shipment scanned out from a hub |
