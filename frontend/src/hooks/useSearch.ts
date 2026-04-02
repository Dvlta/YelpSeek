import { useState } from "react";
import { search } from "../lib/api";
import type { SearchResponse } from "../types";

export function useSearch() {
  const [data, setData] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(query: string) {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await search(query);
      setData(result);
    } catch {
      setError("Search failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  return { data, loading, error, run };
}
