interface DataTableProps {
  results: Record<string, unknown>[];
  rowCount: number;
}

export default function DataTable({ results, rowCount }: DataTableProps) {
  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-8 text-gray-500">
        <svg className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5m6 4.125l2.25 2.25m0 0l2.25-2.25M12 11.625V16.5M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z" />
        </svg>
        <p className="text-sm italic">No results returned.</p>
      </div>
    );
  }

  const columns = Object.keys(results[0]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-500">
          {rowCount} row{rowCount !== 1 ? "s" : ""} returned
        </p>
      </div>
      <div className="overflow-hidden rounded-lg border border-gray-700">
        <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 bg-gray-900">
                {columns.map((col) => (
                  <th
                    key={col}
                    className="sticky top-0 bg-gray-900 whitespace-nowrap px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-gray-400"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map((row, i) => (
                <tr
                  key={i}
                  className={
                    i % 2 === 0
                      ? "border-b border-gray-800"
                      : "border-b border-gray-800 bg-gray-900/40"
                  }
                >
                  {columns.map((col) => (
                    <td
                      key={col}
                      className="whitespace-nowrap px-4 py-2 text-sm text-gray-100"
                    >
                      {String(row[col] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
