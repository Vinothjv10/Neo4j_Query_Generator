"use client";

import { FormEvent, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";

interface QueryFormProps {
  onSubmit: (question: string, topK: number) => Promise<void>;
  loading: boolean;
}

const EXAMPLES = [
  "Show me total revenue by month in 2024",
  "List top 10 customers by total orders",
  "What was the average order value last quarter?",
];

export default function QueryForm({ onSubmit, loading }: QueryFormProps) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(100);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    onSubmit(question.trim(), topK);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !loading) {
      e.preventDefault();
      if (question.trim()) onSubmit(question.trim(), topK);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div>
        <label
          htmlFor="question"
          className="block text-sm font-medium text-gray-300 mb-1.5"
        >
          Business Question
        </label>
        <textarea
          id="question"
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder='e.g. "Show me total revenue by month in 2024"'
          className="block w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-sm text-gray-100 placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none transition-colors"
          disabled={loading}
        />
      </div>

      <div className="flex flex-wrap gap-1.5">
        <span className="self-center mr-1 text-xs text-gray-500">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setQuestion(ex)}
            disabled={loading}
            className="rounded-md border border-gray-700/60 bg-gray-800/50 px-2.5 py-1 text-xs text-gray-400 hover:border-gray-600 hover:text-gray-200 disabled:opacity-40 transition-colors"
          >
            {ex}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-end gap-4">
        <div className="w-32">
          <label
            htmlFor="top_k"
            className="block text-xs font-medium text-gray-400 mb-1"
          >
            Max Rows
          </label>
          <input
            id="top_k"
            type="number"
            min={1}
            max={5000}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="block w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2.5 text-sm text-gray-100 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 transition-colors"
            disabled={loading}
          />
        </div>

        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-blue-500 active:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50 transition-all"
        >
          {loading ? (
            <LoadingSpinner />
          ) : (
            <>
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
              Generate SQL
            </>
          )}
        </button>
      </div>
    </form>
  );
}
