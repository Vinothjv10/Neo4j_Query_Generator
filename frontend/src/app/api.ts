export interface QueryResponse {
  question: string;
  generated_sql: string;
  results: Record<string, unknown>[];
  row_count: number;
  schema_tables_used: string[];
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
