"""
Stage 2: Generate Training Pairs
Produces (query, chunk_id) pairs from review chunks for dual-encoder training.

Strategy A (pseudo): Extract first sentence of each review chunk as a pseudo-query,
                     paired with that specific chunk.
Strategy B (synthetic): Use GPT to generate 5 queries per restaurant,
                        paired with all chunks for that restaurant.
"""

import argparse
import json
import os
import random
import re

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training pairs from restaurant documents.")
    parser.add_argument("--input", type=str, default="data/processed/restaurant_docs.parquet",
                        help="Path to restaurant_docs.parquet (default: data/processed/restaurant_docs.parquet)")
    parser.add_argument("--output", type=str, default="data/processed/training_pairs.jsonl",
                        help="Output JSONL path (default: data/processed/training_pairs.jsonl)")
    parser.add_argument("--strategy", type=str, default="pseudo", choices=["pseudo", "synthetic", "both"],
                        help="Query generation strategy (default: pseudo)")
    parser.add_argument("--openai-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of restaurants to process (useful for testing)")
    parser.add_argument("--min-pairs", type=int, default=3,
                        help="Minimum pairs per restaurant to include (default: 3)")
    parser.add_argument("--max-chunks-per-query", type=int, default=5,
                        help="Max chunks to pair with each synthetic query (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Strategy A: Pseudo-queries from review first sentences
# ---------------------------------------------------------------------------

def extract_pseudo_query(review_text: str) -> str | None:
    """Return the first sentence of a review if it's a reasonable query length."""
    sentences = re.split(r'(?<=[.!?])\s+', review_text.strip())
    first = sentences[0].strip() if sentences else ""
    if 10 < len(first) < 120:
        return first
    return None


def generate_pseudo_pairs(row: pd.Series) -> list[dict]:
    """Pair each chunk's first sentence with that specific chunk."""
    query = extract_pseudo_query(row["chunk_text"])
    if query:
        return [{
            "query": query,
            "chunk_id": row["chunk_id"],
            "business_id": row["business_id"],
            "source": "pseudo",
        }]
    return []


# ---------------------------------------------------------------------------
# Strategy B: Synthetic queries via GPT-5.4-mini
# ---------------------------------------------------------------------------

SYNTHETIC_PROMPT = """\
You are helping build a restaurant search engine.
Given the following restaurant reviews, generate exactly 5 natural language search queries \
that a person might type to find this restaurant. Queries should sound like real user searches, \
not marketing copy. Return only a JSON array of 5 strings.

Restaurant: {name}
Categories: {categories}
Reviews: {reviews}

Return format: ["query 1", "query 2", "query 3", "query 4", "query 5"]"""


def generate_synthetic_pairs(restaurant_chunks: pd.DataFrame, client, max_chunks: int = 5) -> list[dict]:
    """Generate 5 queries per restaurant, paired with top-N chunks for that restaurant."""
    row = restaurant_chunks.iloc[0]  # metadata is same across all chunks
    chunks_to_pair = restaurant_chunks.head(max_chunks)
    reviews_snippet = " | ".join(restaurant_chunks["chunk_text"].tolist())[:3000]
    prompt = SYNTHETIC_PROMPT.format(
        name=row["name"],
        categories=row["categories"],
        reviews=reviews_snippet,
    )
    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini-2026-03-17",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        content = response.choices[0].message.content.strip()
        queries = json.loads(content)
        if not isinstance(queries, list):
            return []
        pairs = []
        for q in queries:
            if isinstance(q, str) and q.strip():
                # Pair each synthetic query with all chunks of this restaurant
                for _, chunk_row in chunks_to_pair.iterrows():
                    pairs.append({
                        "query": q.strip(),
                        "chunk_id": chunk_row["chunk_id"],
                        "business_id": chunk_row["business_id"],
                        "source": "synthetic",
                    })
        return pairs
    except Exception as e:
        print(f"  [warn] Synthetic query failed for {row['name']}: {e}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Set up OpenAI client if needed
    client = None
    if args.strategy in ("synthetic", "both"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for synthetic strategy. Run: pip3 install openai")
        api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Pass --openai-key or set OPENAI_API_KEY env var.")
        client = OpenAI(api_key=api_key)

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    n_restaurants = df["business_id"].nunique()
    print(f"Loaded {len(df):,} chunks from {n_restaurants:,} restaurants.")

    # Group chunks by restaurant for synthetic strategy
    grouped = df.groupby("business_id")
    business_ids = list(grouped.groups.keys())

    if args.limit:
        business_ids = business_ids[:args.limit]
        print(f"Limited to {args.limit:,} restaurants.")

    all_pairs: list[dict] = []
    skipped = 0

    for bid in tqdm(business_ids, desc="Generating pairs"):
        chunks = grouped.get_group(bid)
        pairs: list[dict] = []

        if args.strategy in ("pseudo", "both"):
            for _, row in chunks.iterrows():
                pairs.extend(generate_pseudo_pairs(row))

        if args.strategy in ("synthetic", "both"):
            pairs.extend(generate_synthetic_pairs(chunks, client, args.max_chunks_per_query))

        if len(pairs) < args.min_pairs:
            skipped += 1
            continue

        all_pairs.extend(pairs)

    print(f"Generated {len(all_pairs):,} pairs ({skipped} restaurants skipped for <{args.min_pairs} pairs).")

    # Deduplicate on (query, chunk_id)
    seen = set()
    deduped = []
    for pair in all_pairs:
        key = (pair["query"], pair["chunk_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(pair)
    print(f"After deduplication: {len(deduped):,} pairs.")

    # Shuffle
    random.seed(args.seed)
    random.shuffle(deduped)

    # Write JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in deduped:
            f.write(json.dumps(pair) + "\n")

    print(f"Saved to {args.output}")

    # Summary
    sources = {}
    for p in deduped:
        sources[p["source"]] = sources.get(p["source"], 0) + 1
    for src, count in sources.items():
        print(f"  {src}: {count:,} pairs")


if __name__ == "__main__":
    main()
