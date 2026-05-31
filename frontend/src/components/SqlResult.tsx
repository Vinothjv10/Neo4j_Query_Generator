"use client";

import { useState } from "react";

interface SqlResultProps {
  sql: string;
  tablesUsed: string[];
}

export default function SqlResult({ sql, tablesUsed }: SqlResultProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">Generated SQL</h3>
        <button
          onClick={handleCopy}
          className="inline-flex items-center gap-1.5 rounded-md border border-gray-700 bg-gray-800 px-2.5 py-1 text-xs text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-colors"
        >
          {copied ? (
            <>
              <svg className="h-3.5 w-3.5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Copied
            </>
          ) : (
            <>
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>

      <pre className="overflow-x-auto rounded-lg border border-gray-700 bg-gray-900 p-4 text-sm leading-relaxed font-mono">
        <code>
          {sql.split(/\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|CROSS|ON|AND|OR|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|AS|IN|NOT|NULL|IS|BETWEEN|LIKE|COUNT|SUM|AVG|MIN|MAX|COALESCE|CASE|WHEN|THEN|ELSE|END|DISTINCT|EXTRACT|DATE_TRUNC|TO_CHAR|WITH|UNION|ALL)\b/gi).map((part, i) => {
            const upper = part.toUpperCase();
            const kw = [
              "SELECT","FROM","WHERE","JOIN","LEFT","RIGHT","INNER","OUTER",
              "CROSS","ON","AND","OR","GROUP BY","ORDER BY","HAVING","LIMIT",
              "AS","IN","NOT","NULL","IS","BETWEEN","LIKE","COUNT","SUM","AVG",
              "MIN","MAX","COALESCE","CASE","WHEN","THEN","ELSE","END","DISTINCT",
              "EXTRACT","DATE_TRUNC","TO_CHAR","WITH","UNION","ALL",
            ];
            if (kw.includes(upper.replace(/\s+/g, " ").trim())) {
              return <span key={i} className="text-purple-400">{part}</span>;
            }
            return part;
          })}
        </code>
      </pre>

      {tablesUsed.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-gray-500">Tables:</span>
          {tablesUsed.map((t) => (
            <span
              key={t}
              className="inline-flex items-center rounded-md border border-gray-700 bg-gray-800/60 px-2 py-0.5 text-xs font-mono text-gray-300"
            >
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
