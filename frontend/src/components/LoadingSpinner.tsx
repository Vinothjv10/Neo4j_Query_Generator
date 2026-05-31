export default function LoadingSpinner() {
  return (
    <div className="flex items-center gap-3">
      <div className="relative h-5 w-5">
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-blue-400 animate-spin" />
        <div className="absolute inset-1 rounded-full border-2 border-transparent border-t-blue-600 animate-spin [animation-duration:0.6s]" />
      </div>
      <span className="text-sm text-gray-400 animate-pulse">
        Generating query...
      </span>
    </div>
  );
}
