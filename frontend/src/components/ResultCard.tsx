import { Star, MapPin } from "lucide-react";
import type { RestaurantResult } from "../types";

interface Props {
  result: RestaurantResult;
}

function Stars({ rating }: { rating: number }) {
  return (
    <div className="flex items-center gap-1">
      <Star size={14} className="text-yellow-400 fill-yellow-400" />
      <span className="text-sm font-medium text-gray-700">{rating.toFixed(1)}</span>
    </div>
  );
}

export function ResultCard({ result }: Props) {
  const categories = result.categories.split(",").map((c) => c.trim()).slice(0, 3);
  const snippet = result.snippet.length > 180
    ? result.snippet.slice(0, 180) + "..."
    : result.snippet;
  const matchPct = Math.round(result.similarity_score * 100);

  return (
    <div className="bg-white border border-gray-100 rounded-2xl p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h3 className="text-base font-semibold text-gray-900 truncate">{result.name}</h3>
          <div className="flex items-center gap-3 mt-1">
            <Stars rating={result.stars} />
            <span className="text-xs text-gray-400">{result.review_count.toLocaleString()} reviews</span>
          </div>
        </div>
        <span className="shrink-0 text-xs font-medium text-gray-500 bg-gray-100 px-2 py-1 rounded-lg">
          {matchPct}% match
        </span>
      </div>

      <div className="flex flex-wrap gap-1.5 mt-3">
        {categories.map((cat) => (
          <span key={cat} className="text-xs text-gray-500 bg-gray-50 border border-gray-100 px-2 py-0.5 rounded-full">
            {cat}
          </span>
        ))}
      </div>

      {snippet && (
        <p className="mt-3 text-sm text-gray-500 leading-relaxed">"{snippet}"</p>
      )}

      <div className="flex items-center gap-1 mt-3 text-xs text-gray-400">
        <MapPin size={12} />
        <span>{result.address}</span>
      </div>
    </div>
  );
}
