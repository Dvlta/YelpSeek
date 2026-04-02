import { ResultCard } from "./ResultCard";
import type { SearchResponse } from "../types";

interface Props {
  data: SearchResponse | null;
  loading: boolean;
  error: string | null;
}

function SkeletonCard() {
  return (
    <div className="bg-white border border-gray-100 rounded-2xl p-5 animate-pulse">
      <div className="h-4 bg-gray-200 rounded w-1/2 mb-3" />
      <div className="h-3 bg-gray-100 rounded w-1/4 mb-4" />
      <div className="h-3 bg-gray-100 rounded w-full mb-2" />
      <div className="h-3 bg-gray-100 rounded w-5/6" />
    </div>
  );
}

export function ResultList({ data, loading, error }: Props) {
  if (loading) {
    return (
      <div className="w-full max-w-2xl space-y-3">
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    );
  }

  if (error) {
    return (
      <p className="text-sm text-red-500">{error}</p>
    );
  }

  if (!data) return null;

  if (data.results.length === 0) {
    return <p className="text-sm text-gray-400">No results found. Try a different search.</p>;
  }

  return (
    <div className="w-full max-w-2xl">
      <p className="text-xs text-gray-400 mb-3">
        {data.total_results} results · {data.latency_ms}ms
      </p>
      <div className="space-y-3">
        {data.results.map((r) => (
          <ResultCard key={r.business_id} result={r} />
        ))}
      </div>
    </div>
  );
}
