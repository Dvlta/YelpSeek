"""
Stage 6: End-to-End Pipeline Evaluation
Evaluates the full retrieval pipeline (encoder -> FAISS index -> aggregation)
at the restaurant level using the same validation split as training.

This measures what users actually experience - not just encoder quality,
but the combined effect of the index, aggregation strategy, and any
post-retrieval steps (re-ranking, etc.).

Usage:
    python ml/06_evaluate_pipeline.py
    python ml/06_evaluate_pipeline.py --retrieval-k 200 --top-k 50
"""

import argparse
import json
import math
import random
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation.")
    parser.add_argument("--pairs", type=str, default="data/processed/training_pairs.jsonl")
    parser.add_argument("--docs", type=str, default="data/processed/restaurant_docs.parquet")
    parser.add_argument("--index", type=str, default="backend/data/index.faiss")
    parser.add_argument("--metadata", type=str, default="backend/data/metadata.parquet")
    parser.add_argument("--model", type=str, default="models/encoder_v1/best")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-k", type=int, default=100,
                        help="Number of chunks to retrieve from FAISS per query")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of restaurants to return after aggregation (eval checks @1,5,10,50)")
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--aggregation", type=str, default="max",
                        choices=["max", "top3_mean", "top5_mean", "blend"],
                        help="Aggregation strategy for chunk scores per restaurant")
    parser.add_argument("--blend-alpha", type=float, default=0.5,
                        help="Blend weight for 'blend' aggregation: alpha*max + (1-alpha)*top3_mean")
    parser.add_argument("--rerank", action="store_true",
                        help="Enable cross-encoder re-ranking")
    parser.add_argument("--rerank-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-k", type=int, default=50,
                        help="Number of restaurants to re-rank (default: 50)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metrics (same as 03_train_encoder.py)
# ---------------------------------------------------------------------------

def recall_at_k(ranks: list[int], k: int) -> float:
    return sum(1 for r in ranks if r <= k) / len(ranks)


def mrr(ranks: list[int]) -> float:
    return sum(1.0 / r for r in ranks) / len(ranks)


def ndcg_at_k(ranks: list[int], k: int) -> float:
    scores = []
    for r in ranks:
        if r <= k:
            scores.append(1.0 / math.log2(r + 1))
        else:
            scores.append(0.0)
    ideal = 1.0 / math.log2(2)
    return (sum(scores) / len(scores)) / ideal


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

def aggregate_scores(chunk_scores: list[float], strategy: str, alpha: float = 0.5) -> float:
    """Aggregate a list of chunk similarity scores into a single restaurant score."""
    if not chunk_scores:
        return -np.inf
    sorted_scores = sorted(chunk_scores, reverse=True)

    if strategy == "max":
        return sorted_scores[0]
    elif strategy == "top3_mean":
        return float(np.mean(sorted_scores[:3]))
    elif strategy == "top5_mean":
        return float(np.mean(sorted_scores[:5]))
    elif strategy == "blend":
        max_score = sorted_scores[0]
        top3_mean = float(np.mean(sorted_scores[:3]))
        return alpha * max_score + (1 - alpha) * top3_mean
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load FAISS index
    print(f"Loading FAISS index from {args.index}...")
    index = faiss.read_index(args.index)
    index.hnsw.efSearch = args.ef_search
    print(f"Index contains {index.ntotal:,} vectors.")

    # Load metadata (aligned with FAISS index positions)
    print(f"Loading metadata from {args.metadata}...")
    metadata = pd.read_parquet(args.metadata)

    # Load model
    print(f"Loading model from {args.model}...")
    model = SentenceTransformer(args.model)

    # Load cross-encoder if re-ranking
    cross_encoder = None
    if args.rerank:
        print(f"Loading cross-encoder from {args.rerank_model}...")
        cross_encoder = CrossEncoder(args.rerank_model)

    # Load and split pairs (same logic as 03_train_encoder.py)
    print(f"Loading pairs from {args.pairs}...")
    docs = pd.read_parquet(args.docs)
    chunk_ids_set = set(docs["chunk_id"])
    pairs = []
    with open(args.pairs, "r") as f:
        for line in f:
            p = json.loads(line)
            if p["chunk_id"] in chunk_ids_set:
                pairs.append(p)
    random.shuffle(pairs)
    val_size = int(len(pairs) * args.val_split)
    val_pairs = pairs[:val_size]
    print(f"Total pairs: {len(pairs):,} | Val pairs: {len(val_pairs):,}")

    # Deduplicate queries pointing to the same business (keep one per query-business)
    seen = set()
    unique_val = []
    for p in val_pairs:
        key = (p["query"], p["business_id"])
        if key not in seen:
            seen.add(key)
            unique_val.append(p)
    print(f"Unique (query, business) pairs for eval: {len(unique_val):,}")

    # Encode all validation queries
    print("Encoding validation queries...")
    queries = [p["query"] for p in unique_val]
    prefixed = [BGE_QUERY_INSTRUCTION + q for q in queries]
    t0 = time.time()
    query_embeddings = model.encode(
        prefixed, batch_size=args.batch_size,
        show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)
    encode_time = time.time() - t0
    print(f"Encoded {len(queries):,} queries in {encode_time:.1f}s "
          f"({encode_time / len(queries) * 1000:.1f}ms/query)")

    # Run FAISS search for all queries
    print(f"Searching FAISS (retrieval_k={args.retrieval_k})...")
    t0 = time.time()
    all_scores, all_indices = index.search(query_embeddings, args.retrieval_k)
    search_time = time.time() - t0
    print(f"FAISS search complete in {search_time:.1f}s "
          f"({search_time / len(queries) * 1000:.1f}ms/query)")

    # Evaluate: for each query, aggregate chunks to restaurants, find rank of true positive
    rerank_str = f" + rerank top {args.rerank_k}" if args.rerank else ""
    print(f"Aggregating with strategy='{args.aggregation}'{rerank_str}...")
    ranks = []
    for i, pair in enumerate(unique_val):
        true_bid = pair["business_id"]
        scores = all_scores[i]
        indices = all_indices[i]

        # Group chunk scores by business_id, track best chunk_text
        restaurant_chunks: dict[str, list[float]] = {}
        restaurant_best_text: dict[str, tuple[float, str]] = {}
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            row = metadata.iloc[idx]
            bid = str(row["business_id"])
            restaurant_chunks.setdefault(bid, []).append(float(score))
            if bid not in restaurant_best_text or score > restaurant_best_text[bid][0]:
                restaurant_best_text[bid] = (float(score), str(row["chunk_text"]))

        # Aggregate scores per restaurant
        restaurant_scores = {
            bid: aggregate_scores(chunk_scores, args.aggregation, args.blend_alpha)
            for bid, chunk_scores in restaurant_chunks.items()
        }

        # Re-rank with cross-encoder if enabled
        if cross_encoder is not None:
            # Take top rerank_k by bi-encoder score
            top_bids = sorted(restaurant_scores, key=restaurant_scores.get, reverse=True)[:args.rerank_k]
            pairs_to_rank = [(queries[i], restaurant_best_text[bid][1]) for bid in top_bids]
            rerank_scores = cross_encoder.predict(pairs_to_rank)
            restaurant_scores = {bid: float(rs) for bid, rs in zip(top_bids, rerank_scores)}

        # Rank restaurants
        sorted_bids = sorted(restaurant_scores, key=restaurant_scores.get, reverse=True)

        if true_bid in sorted_bids:
            rank = sorted_bids.index(true_bid) + 1
        else:
            rank = args.top_k + 1  # not found in retrieved set

        ranks.append(rank)

    # Compute metrics
    metrics = {
        "Recall@1": recall_at_k(ranks, 1),
        "Recall@5": recall_at_k(ranks, 5),
        "Recall@10": recall_at_k(ranks, 10),
        "Recall@50": recall_at_k(ranks, 50),
        "MRR": mrr(ranks),
        "NDCG@10": ndcg_at_k(ranks, 10),
    }

    print(f"\n{'='*50}")
    print(f"Pipeline Evaluation Results")
    print(f"{'='*50}")
    print(f"  Aggregation:  {args.aggregation}")
    print(f"  Re-ranking:   {args.rerank_model if args.rerank else 'disabled'}")
    print(f"  Rerank K:     {args.rerank_k if args.rerank else 'n/a'}")
    print(f"  Retrieval K:  {args.retrieval_k}")
    print(f"  Top K:        {args.top_k}")
    print(f"  ef_search:    {args.ef_search}")
    print(f"  Val queries:  {len(unique_val):,}")
    print(f"{'='*50}")
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")
    print(f"{'='*50}")
    print(f"  Avg encode:   {encode_time / len(queries) * 1000:.1f}ms/query")
    print(f"  Avg search:   {search_time / len(queries) * 1000:.1f}ms/query")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
