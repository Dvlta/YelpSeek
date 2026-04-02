import { useState } from "react";
import { SearchBar } from "./components/SearchBar";
import { ExampleQueries } from "./components/ExampleQueries";
import { ResultList } from "./components/ResultList";
import { useSearch } from "./hooks/useSearch";

export default function App() {
  const { data, loading, error, run } = useSearch();
  const [activeQuery, setActiveQuery] = useState("");

  function handleSearch(query: string) {
    setActiveQuery(query);
    run(query);
  }

  const hasResults = data !== null || loading || error !== null;

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Navbar */}
      <nav className="bg-gray-900 px-6 py-4 flex items-center justify-between">
        <span className="text-white font-semibold text-lg tracking-tight">YelpSeek</span>
        <a
          href="https://github.com/dvlta/YelpSeek"
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-400 hover:text-white text-sm transition-colors"
        >
          GitHub
        </a>
      </nav>

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center px-4 py-12 gap-8">
        {/* Hero */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">Find restaurants by experience</h1>
          <p className="text-gray-500 mt-2 text-base">
            Semantic search across 5,000+ Philadelphia restaurants
          </p>
        </div>

        {/* Search */}
        <SearchBar onSearch={handleSearch} loading={loading} initialQuery={activeQuery} />

        {/* Example queries — hide once results are showing */}
        {!hasResults && <ExampleQueries onSelect={handleSearch} />}

        {/* Results */}
        <ResultList data={data} loading={loading} error={error} />
      </main>
    </div>
  );
}
