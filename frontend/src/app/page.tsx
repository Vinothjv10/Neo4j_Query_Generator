"use client";

import { useState } from "react";
import QueryForm from "@/components/QueryForm";
import SqlResult from "@/components/SqlResult";
import DataTable from "@/components/DataTable";
import ErrorAlert from "@/components/ErrorAlert";
import ReasoningPanel from "@/components/ReasoningPanel";
import { submitQuery, QueryResponse } from "./api";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (question: string, topK: number) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await submitQuery(question, topK);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-10">
      <header className="mb-10 text-center">
        <div className="inline-flex items-center justify-center gap-2 rounded-full border border-gray-800 bg-gray-900/60 px-3 py-1 text-xs text-gray-500 mb-4">
          <span className="h-2 w-2 rounded-full bg-green-500" />
          Neo4j &middot; NVIDIA &middot; PostgreSQL
        </div>
        <h1 className="text-4xl font-bold tracking-tight text-white">
          Text2SQL <span className="text-blue-400">BI</span>
        </h1>
        <p className="mt-2 text-sm text-gray-500 max-w-lg mx-auto">
          Ask business questions in plain English and get SQL query results
          instantly.
        </p>
      </header>

      <section className="rounded-xl border border-gray-800 bg-gray-950 p-6 shadow-sm">
        <QueryForm onSubmit={handleSubmit} loading={loading} />
      </section>

      {loading && !result && (
        <div className="mt-8 space-y-4 animate-pulse">
          <div className="h-6 w-48 rounded bg-gray-800" />
          <div className="h-24 rounded-lg bg-gray-800/60" />
          <div className="h-6 w-32 rounded bg-gray-800" />
          <div className="h-48 rounded-lg bg-gray-800/60" />
        </div>
      )}

      {error && (
        <div className="mt-6 animate-fade-in">
          <ErrorAlert message={error} />
        </div>
      )}

      {result && (
        <div className="mt-8 space-y-6 animate-fade-in">
          <div className="flex items-center gap-2 text-sm text-gray-500 border-b border-gray-800 pb-3">
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
            </svg>
            <span className="font-medium text-gray-400">Question:</span>
            <span className="text-gray-300">{result.question}</span>
          </div>

          <div className="rounded-xl border border-gray-800 bg-gray-950 p-6 shadow-sm">
            <SqlResult
              sql={result.generated_sql}
              tablesUsed={result.schema_tables_used}
            />
          </div>

          {result.reasoning && (
            <div className="rounded-xl border border-gray-800 bg-gray-950 p-6 shadow-sm">
              <ReasoningPanel reasoning={result.reasoning} />
            </div>
          )}

          <div className="rounded-xl border border-gray-800 bg-gray-950 p-6 shadow-sm">
            <h3 className="mb-3 text-sm font-semibold text-gray-300">
              Results
            </h3>
            <DataTable
              results={result.results}
              rowCount={result.row_count}
            />
          </div>
        </div>
      )}
    </main>
  );
}
