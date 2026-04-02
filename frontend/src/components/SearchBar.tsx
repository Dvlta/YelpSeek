import { useEffect, useRef, useState } from "react";
import { Search, X, Loader2 } from "lucide-react";

interface Props {
  onSearch: (query: string) => void;
  loading: boolean;
  initialQuery?: string;
}

export function SearchBar({ onSearch, loading, initialQuery = "" }: Props) {
  const [value, setValue] = useState(initialQuery);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    setValue(initialQuery);
  }, [initialQuery]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onSearch(value.trim());
  }

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl">
      <div className="flex items-center gap-2 px-4 py-3 bg-white border border-gray-200 rounded-2xl shadow-sm hover:shadow-md transition-shadow">
        <Search className="text-gray-400 shrink-0" size={20} />
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="cozy birthday dinner, not too loud..."
          className="flex-1 text-gray-800 placeholder-gray-400 outline-none text-base bg-transparent"
        />
        {value && !loading && (
          <button
            type="button"
            onClick={() => setValue("")}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X size={18} />
          </button>
        )}
        {loading ? (
          <Loader2 className="text-gray-400 animate-spin shrink-0" size={20} />
        ) : (
          <button
            type="submit"
            disabled={!value.trim()}
            className="px-4 py-1.5 bg-gray-900 text-white text-sm rounded-xl disabled:opacity-40 hover:bg-gray-700 transition-colors"
          >
            Search
          </button>
        )}
      </div>
    </form>
  );
}
