export interface TableReason {
  table: string;
  tier: string;
  description: string;
  top_columns: string[];
  reason: string;
}

export interface ColumnReason {
  column: string;
  score: number;
  reason: string;
}

export interface Reasoning {
  table_selection: TableReason[];
  column_selection: Record<string, ColumnReason[]>;
  final_explanation: string | null;
  sql_generation: string;
  retries: string[];
}

export interface QueryResponse {
  question: string;
  generated_sql: string;
  results: Record<string, unknown>[];
  row_count: number;
  schema_tables_used: string[];
  reasoning: Reasoning | null;
}

export async function submitQuery(
  question: string,
  top_k: number
): Promise<QueryResponse> {
  const res = await fetch("/api/v1/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k }),
  });

  if (!res.ok) {
    const body = await res.json();
    const detail =
      body?.detail?.error || body?.detail || body?.error || "Request failed";
    throw new Error(detail);
  }

  return res.json();
}
