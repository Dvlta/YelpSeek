const EXAMPLES = [
  "cozy place for a birthday dinner, not too loud",
  "late night ramen when everything else is closed",
  "outdoor brunch spot that's dog friendly",
  "business lunch downtown, fast service",
  "best happy hour with creative cocktails",
  "hidden gem locals actually go to",
  "romantic date night, impressive but not stuffy",
  "vegan options that don't taste like vegan food",
];

interface Props {
  onSelect: (query: string) => void;
}

export function ExampleQueries({ onSelect }: Props) {
  return (
    <div className="w-full max-w-2xl">
      <p className="text-xs text-gray-400 mb-2 uppercase tracking-wide">Try</p>
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            onClick={() => onSelect(q)}
            className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-full transition-colors"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
