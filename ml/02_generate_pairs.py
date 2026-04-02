"""
Stage 2: Generate Training Pairs
Produces (query, business_id) pairs from restaurant documents for dual-encoder training.

Strategy A (pseudo): Extract first sentence of each review as a pseudo-query.
Strategy B (synthetic): Use GPT-4o-mini to generate 5 natural queries per restaurant.
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
    """Generate pseudo-query pairs from a restaurant's combined reviews."""
    pairs = []
    reviews = row["combined_reviews"].split(" | ")
    for review_text in reviews:
        query = extract_pseudo_query(review_text)
        if query:
            pairs.append({
                "query": query,
                "business_id": row["business_id"],
                "source": "pseudo",
            })
    return pairs


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


def generate_synthetic_pairs(row: pd.Series, client) -> list[dict]:
    """Call GPT-4o-mini to generate 5 queries for a restaurant."""
    # Truncate reviews to avoid token limits
    reviews_snippet = row["combined_reviews"][:3000]
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
        return [
            {"query": q.strip(), "business_id": row["business_id"], "source": "synthetic"}
            for q in queries
            if isinstance(q, str) and q.strip()
        ]
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
    print(f"Loaded {len(df):,} restaurants.")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df):,} restaurants.")

    all_pairs: list[dict] = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating pairs"):
        pairs: list[dict] = []

        if args.strategy in ("pseudo", "both"):
            pairs.extend(generate_pseudo_pairs(row))

        if args.strategy in ("synthetic", "both"):
            pairs.extend(generate_synthetic_pairs(row, client))

        if len(pairs) < args.min_pairs:
            skipped += 1
            continue

        all_pairs.extend(pairs)

    print(f"Generated {len(all_pairs):,} pairs ({skipped} restaurants skipped for <{args.min_pairs} pairs).")

    # Deduplicate on (query, business_id)
    seen = set()
    deduped = []
    for pair in all_pairs:
        key = (pair["query"], pair["business_id"])
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
