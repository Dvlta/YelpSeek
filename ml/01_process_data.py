"""
Stage 1: Process Raw Yelp Data
Filters Yelp businesses to restaurants in one or more target cities,
outputs one row per review chunk (chunked encoding approach).
Each restaurant produces up to --top-reviews rows, one per review.
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

RESTAURANT_CATEGORIES = {"Restaurants", "Food", "Bars"}


def parse_args():
    parser = argparse.ArgumentParser(description="Process raw Yelp data into restaurant documents.")
    parser.add_argument("--city", type=str, nargs="+", default=["Philadelphia"], help="Target city or cities (default: Philadelphia)")
    parser.add_argument("--min-reviews", type=int, default=10, help="Minimum review count (default: 10)")
    parser.add_argument("--min-stars", type=float, default=3.0, help="Minimum star rating (default: 3.0)")
    parser.add_argument("--top-reviews", type=int, default=50, help="Max review chunks per restaurant (default: 50)")
    parser.add_argument("--raw-data-dir", type=str, default="data/raw", help="Path to raw Yelp JSON files")
    parser.add_argument("--output", type=str, default="data/processed/restaurant_docs.parquet", help="Output Parquet path")
    return parser.parse_args()


def is_restaurant(categories: str) -> bool:
    """Check if a business belongs to a restaurant-related category."""
    if not categories:
        return False
    cats = {c.strip() for c in categories.split(",")}
    return bool(cats & RESTAURANT_CATEGORIES)


def load_businesses(business_path: str, cities: list[str], min_reviews: int, min_stars: float) -> dict:
    """Load and filter businesses from the business JSON file."""
    city_set = set(cities)
    businesses = {}
    print(f"Loading businesses from {business_path}...")
    with open(business_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Filtering businesses"):
            line = line.strip()
            if not line:
                continue
            try:
                biz = json.loads(line)
            except json.JSONDecodeError:
                continue

            if biz.get("city") not in city_set:
                continue
            if biz.get("review_count", 0) < min_reviews:
                continue
            if biz.get("stars", 0) < min_stars:
                continue
            if not is_restaurant(biz.get("categories", "")):
                continue

            businesses[biz["business_id"]] = biz

    print(f"Found {len(businesses):,} restaurants across {cities} after filtering.")
    return businesses


def load_reviews(review_path: str, business_ids: set, top_k: int) -> dict[str, list]:
    """
    Stream through the reviews file and collect reviews for target businesses.
    Returns a dict mapping business_id -> list of (useful_count, review_text).
    """
    reviews: dict[str, list] = defaultdict(list)
    print(f"Streaming reviews from {review_path}...")
    with open(review_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading reviews"):
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            bid = review.get("business_id")
            if bid not in business_ids:
                continue

            text = review.get("text", "").strip()
            if not text:
                continue

            useful = review.get("useful", 0)
            reviews[bid].append((useful, text))

    # Sort by useful descending and keep top_k
    for bid in reviews:
        reviews[bid].sort(key=lambda x: x[0], reverse=True)
        reviews[bid] = [text for _, text in reviews[bid][:top_k]]

    return reviews


def build_documents(businesses: dict, reviews: dict[str, list]) -> list[dict]:
    """Output one row per review chunk. Each chunk is a single review for a restaurant."""
    docs = []
    for bid, biz in businesses.items():
        review_texts = reviews.get(bid, [])
        if not review_texts:
            continue

        name = biz.get("name", "")
        categories = biz.get("categories", "")

        for i, review_text in enumerate(review_texts):
            docs.append({
                "chunk_id": f"{bid}_{i}",
                "business_id": bid,
                "chunk_index": i,
                "chunk_text": review_text,
                "name": name,
                "city": biz.get("city", ""),
                "state": biz.get("state", ""),
                "stars": biz.get("stars"),
                "review_count": biz.get("review_count"),
                "categories": categories,
                "address": biz.get("address", ""),
                "latitude": biz.get("latitude"),
                "longitude": biz.get("longitude"),
            })

    return docs


def main():
    args = parse_args()

    business_path = os.path.join(args.raw_data_dir, "yelp_academic_dataset_business.json")
    review_path = os.path.join(args.raw_data_dir, "yelp_academic_dataset_review.json")

    for path in [business_path, review_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Stage 1a: filter businesses
    businesses = load_businesses(business_path, args.city, args.min_reviews, args.min_stars)
    if not businesses:
        raise ValueError(f"No restaurants found for cities={args.city}. Check your filters or city names.")

    # Stage 1b: collect reviews for filtered businesses
    reviews = load_reviews(review_path, set(businesses.keys()), args.top_reviews)

    # Stage 1c: build document records
    print("Building document records...")
    docs = build_documents(businesses, reviews)
    print(f"Built {len(docs):,} restaurant documents.")

    # Stage 1d: save to Parquet
    df = pd.DataFrame(docs)
    df.to_parquet(args.output, index=False)
    n_restaurants = df["business_id"].nunique()
    print(f"Saved {len(df):,} chunks from {n_restaurants:,} restaurants to {args.output}")
    print(f"\nSample chunk_text:\n{df['chunk_text'].iloc[0][:300]}...")


if __name__ == "__main__":
    main()
