# Text2SQL BI

A FastAPI application that converts natural language business questions into SQL queries using Neo4j as a schema registry, NVIDIA LLM for text-to-SQL generation, and PostgreSQL as the execution engine.

## Project Structure

```
text2sql-bi/          ← Backend (FastAPI + Python)
├── app/
│   ├── main.py               # FastAPI entry point
│   ├── config.py             # Pydantic settings from .env
│   ├── api/routes/query.py   # POST /query and GET /health
│   ├── services/
│   │   ├── neo4j_service.py      # Schema fetcher from Neo4j
│   │   ├── prompt_builder.py     # LLM prompt builder
│   │   ├── llm_service.py        # NVIDIA API caller
│   │   └── postgres_service.py   # SQL executor on PostgreSQL
│   ├── models/schemas.py     # Pydantic models
│   └── utils/sql_validator.py # SELECT-only validator
├── .env.example
├── requirements.txt
└── README.md

frontend/             ← Frontend (Next.js + Tailwind)
├── src/
│   ├── app/
│   │   ├── page.tsx         # Main page with query form + results
│   │   ├── layout.tsx       # Root layout
│   │   ├── api.ts           # Backend API client
│   │   └── globals.css      # Tailwind imports
│   └── components/
│       ├── QueryForm.tsx    # Question input + submit
│       ├── SqlResult.tsx    # Generated SQL display
│       ├── DataTable.tsx    # Results table
│       ├── ErrorAlert.tsx   # Error display
│       └── LoadingSpinner.tsx
├── package.json
├── tsconfig.json
├── next.config.js           # Proxy /api/* → localhost:8000
├── tailwind.config.ts
└── postcss.config.js
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- Neo4j (running instance with schema graph populated)
- PostgreSQL (running instance with business data)
- NVIDIA API key

## Setup

### 1. Clone and navigate

```bash
cd text2sql-bi
```

### 2. Backend setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Create .env from template
cp .env.example .env
# Edit .env with your credentials

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API docs will be available at http://localhost:8000/docs

### 3. Frontend setup

Open a **second terminal**:

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The frontend will be available at http://localhost:3000

> The Next.js dev server proxies `/api/*` requests to `http://localhost:8000/api/*` (configured in `next.config.js`), so the frontend and backend can run on different ports without CORS issues.

## Neo4j Graph Model

### Node Labels

- **Table** — properties: `name` (str), `schema` (str), `description` (str)
- **Column** — properties: `name` (str), `data_type` (str), `description` (str)

### Relationships

- `(Table)-[:HAS_COLUMN]->(Column)` — links a table to its columns.
- `(Table)-[:JOINS_WITH]->(Table)` — indicates a foreign-key or logical join relationship.

### Example Cypher Seed Data

```cypher
CREATE (t:Table {name: "orders", schema: "public", description: "Customer orders"})
CREATE (c1:Column {name: "id", data_type: "integer", description: "Primary key"})
CREATE (c2:Column {name: "customer_id", data_type: "integer", description: "FK to customers"})
CREATE (c3:Column {name: "total", data_type: "decimal", description: "Order total amount"})
CREATE (c4:Column {name: "created_at", data_type: "timestamp", description: "Order creation date"})
CREATE (t)-[:HAS_COLUMN]->(c1), (t)-[:HAS_COLUMN]->(c2),
       (t)-[:HAS_COLUMN]->(c3), (t)-[:HAS_COLUMN]->(c4)
```

## API Usage

### POST /api/v1/query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me total revenue by month in 2024", "top_k": 10}'
```

### GET /api/v1/health

```bash
curl http://localhost:8000/api/v1/health
```
