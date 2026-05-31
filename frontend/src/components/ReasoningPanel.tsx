"use client";

import { useState } from "react";
import { Reasoning } from "@/app/api";

interface ReasoningPanelProps {
  reasoning: Reasoning;
}

export default function ReasoningPanel({ reasoning }: ReasoningPanelProps) {
  const [openSection, setOpenSection] = useState<string | null>(null);

  const toggle = (id: string) => setOpenSection(openSection === id ? null : id);

  return (
    <div className="space-y-3 text-sm">
      <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
        <svg className="h-4 w-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
        </svg>
        Why this answer?
      </h3>

      {/* LLM Explanation */}
      {reasoning.final_explanation && (
        <div className="rounded-lg border border-blue-500/30 bg-blue-500/5 px-4 py-3">
          <p className="text-sm leading-relaxed text-gray-200 whitespace-pre-wrap">
            {reasoning.final_explanation}
          </p>
        </div>
      )}

      {/* Table Selection */}
      <div className="rounded-lg border border-gray-700 overflow-hidden">
        <button
          onClick={() => toggle("tables")}
          className="flex w-full items-center justify-between px-3 py-2.5 bg-gray-800/60 hover:bg-gray-800 transition-colors text-left"
        >
          <span className="font-medium text-gray-300">Tables selected ({reasoning.table_selection.length})</span>
          <svg className={`h-4 w-4 text-gray-500 transition-transform ${openSection === "tables" ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {openSection === "tables" && (
          <div className="divide-y divide-gray-800">
            {reasoning.table_selection.map((t, i) => (
              <div key={i} className="px-3 py-2.5 space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-green-400 bg-green-400/10 px-1.5 py-0.5 rounded">{t.tier}</span>
                  <span className="font-mono text-gray-200 text-xs">{t.table}</span>
                </div>
                <p className="text-xs text-gray-500 line-clamp-2">{t.reason}</p>
                {t.top_columns.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {t.top_columns.map((c) => (
                      <span key={c} className="text-xs font-mono text-blue-400 bg-blue-400/10 px-1.5 py-0.5 rounded">{c}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Column Selection */}
      {Object.keys(reasoning.column_selection).length > 0 && (
        <div className="rounded-lg border border-gray-700 overflow-hidden">
          <button
            onClick={() => toggle("columns")}
            className="flex w-full items-center justify-between px-3 py-2.5 bg-gray-800/60 hover:bg-gray-800 transition-colors text-left"
          >
            <span className="font-medium text-gray-300">Columns ranked by relevance</span>
            <svg className={`h-4 w-4 text-gray-500 transition-transform ${openSection === "columns" ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {openSection === "columns" && (
            <div className="divide-y divide-gray-800">
              {Object.entries(reasoning.column_selection).map(([table, cols]) => (
                <div key={table} className="px-3 py-2 space-y-1">
                  <p className="text-xs font-mono text-gray-400">{table}</p>
                  {cols.map((c, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <div className="h-1.5 w-16 rounded-full bg-gray-700 overflow-hidden flex-shrink-0">
                        <div
                          className="h-full rounded-full bg-blue-500"
                          style={{ width: `${Math.min(c.score * 100, 100)}%` }}
                        />
                      </div>
                      <span className="font-mono text-gray-200 w-32 truncate">{c.column}</span>
                      <span className="text-gray-500">({(c.score * 100).toFixed(0)}%)</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* SQL Generation */}
      <div className="rounded-lg border border-gray-700 overflow-hidden">
        <button
          onClick={() => toggle("sql")}
          className="flex w-full items-center justify-between px-3 py-2.5 bg-gray-800/60 hover:bg-gray-800 transition-colors text-left"
        >
          <span className="font-medium text-gray-300">SQL generation flow</span>
          <svg className={`h-4 w-4 text-gray-500 transition-transform ${openSection === "sql" ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {openSection === "sql" && (
          <div className="px-3 py-2.5 space-y-1.5">
            <p className="text-xs text-gray-400">{reasoning.sql_generation}</p>
            {reasoning.retries.length > 0 && (
              <div className="space-y-1">
                <p className="text-xs font-medium text-yellow-400">Retries / fixes applied:</p>
                {reasoning.retries.map((r, i) => (
                  <p key={i} className="text-xs text-gray-500 font-mono">• {r}</p>
                ))}
              </div>
            )}
            {reasoning.retries.length === 0 && (
              <p className="text-xs text-green-400/70">No retries needed — SQL validated on first attempt.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
