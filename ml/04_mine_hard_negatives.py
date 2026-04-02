"""
Stage 4: Hard Negative Mining
Iteratively mines hard negatives from the current model and retrains to improve retrieval precision.

Each iteration:
  1. Encode all documents with the current best model
  2. For each training query, retrieve top-100 similar docs
  3. Sample 5 hard negatives from ranks 5-50 (avoids false negatives at top ranks)
  4. Retrain for 1 epoch on original pairs + hard negative triplets
  5. Evaluate and save if Recall@10 improves
"""

import argparse
import json
import math
import os
import random

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def parse_args():
    parser = argparse.ArgumentParser(description="Hard negative mining for dual-encoder improvement.")
    parser.add_argument("--model", type=str, default="models/encoder_v1/best", help="Path to base model checkpoint")
    parser.add_argument("--pairs", type=str, default="data/processed/training_pairs.jsonl")
    parser.add_argument("--docs", type=str, default="data/processed/restaurant_docs.parquet")
    parser.add_argument("--output", type=str, default="models/encoder_v2")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=100, help="Retrieve top-K docs for negative mining")
    parser.add_argument("--neg-sample-range", type=str, default="5,50", help="Sample negatives from this rank range (default: 5,50)")
    parser.add_argument("--negs-per-query", type=int, default=5, help="Hard negatives to sample per query")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5, help="Lower LR than initial training (default: 1e-5)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Encoding helpers (BGE-specific)
# ---------------------------------------------------------------------------

def encode_queries(model: SentenceTransformer, queries: list[str], batch_size: int = 64) -> np.ndarray:
    prefixed = [BGE_QUERY_INSTRUCTION + q for q in queries]
    return np.array(model.encode(prefixed, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True))


def encode_docs(model: SentenceTransformer, doc_texts: list[str], batch_size: int = 64) -> np.ndarray:
    return np.array(model.encode(doc_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True))


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def recall_at_k(ranks: list[int], k: int) -> float:
    return sum(1 for r in ranks if r <= k) / len(ranks)


def mrr(ranks: list[int]) -> float:
    return sum(1.0 / r for r in ranks) / len(ranks)


def ndcg_at_k(ranks: list[int], k: int) -> float:
    scores = [1.0 / math.log2(r + 1) if r <= k else 0.0 for r in ranks]
    ideal = 1.0 / math.log2(2)
    return (sum(scores) / len(scores)) / ideal


def evaluate(model: SentenceTransformer, val_pairs: list[dict], doc_index: dict) -> dict:
    print("  Evaluating...")
    val_bids = list({p["business_id"] for p in val_pairs})
    bid_to_idx = {bid: i for i, bid in enumerate(val_bids)}
    doc_texts = [doc_index[bid] for bid in val_bids]
    doc_embeddings = encode_docs(model, doc_texts)
    queries = [p["query"] for p in val_pairs]
    query_embeddings = encode_queries(model, queries)

    ranks = []
    for i, pair in enumerate(val_pairs):
        scores = doc_embeddings @ query_embeddings[i]
        true_idx = bid_to_idx[pair["business_id"]]
        rank = int((-scores).argsort().tolist().index(true_idx)) + 1
        ranks.append(rank)

    return {
        "Recall@1": recall_at_k(ranks, 1),
        "Recall@5": recall_at_k(ranks, 5),
        "Recall@10": recall_at_k(ranks, 10),
        "Recall@50": recall_at_k(ranks, 50),
        "MRR": mrr(ranks),
        "NDCG@10": ndcg_at_k(ranks, 10),
    }


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    model: SentenceTransformer,
    train_pairs: list[dict],
    doc_index: dict,
    bid_list: list[str],
    top_k: int,
    neg_range: tuple[int, int],
    negs_per_query: int,
) -> list[dict]:
    """
    For each training query, retrieve top_k docs, exclude the true positive,
    and sample negs_per_query hard negatives from neg_range ranks.
    Returns list of {query, positive_id, negative_id} dicts.
    """
    print("  Encoding documents for mining...")
    doc_texts = [doc_index[bid] for bid in bid_list]
    doc_embeddings = encode_docs(model, doc_texts)
    bid_to_idx = {bid: i for i, bid in enumerate(bid_list)}

    # Build exact FAISS index (IndexFlatIP for cosine sim on normalized vecs)
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings.astype(np.float32))

    print("  Mining hard negatives...")
    queries = [p["query"] for p in train_pairs]
    query_embeddings = encode_queries(model, queries)

    neg_start, neg_end = neg_range
    triplets = []

    for i, pair in enumerate(train_pairs):
        q_emb = query_embeddings[i].reshape(1, -1).astype(np.float32)
        _, top_indices = index.search(q_emb, top_k)
        top_indices = top_indices[0].tolist()

        true_idx = bid_to_idx.get(pair["business_id"], -1)
        candidates = [idx for idx in top_indices if idx != true_idx]

        # Sample from rank range [neg_start, neg_end] to avoid false negatives
        neg_pool = candidates[neg_start:neg_end]
        if not neg_pool:
            continue

        sampled = random.sample(neg_pool, min(negs_per_query, len(neg_pool)))
        for neg_idx in sampled:
            triplets.append({
                "query": pair["query"],
                "positive_id": pair["business_id"],
                "negative_id": bid_list[neg_idx],
            })

    print(f"  Mined {len(triplets):,} hard negative triplets.")
    return triplets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    neg_range = tuple(int(x) for x in args.neg_sample_range.split(","))

    # Load data
    print("Loading docs...")
    df_docs = pd.read_parquet(args.docs)
    doc_index = dict(zip(df_docs["business_id"], df_docs["doc_text"]))
    bid_list = list(doc_index.keys())
    print(f"Loaded {len(doc_index):,} documents.")

    print("Loading pairs...")
    all_pairs = []
    with open(args.pairs) as f:
        for line in f:
            p = json.loads(line)
            if p["business_id"] in doc_index:
                all_pairs.append(p)
    print(f"Loaded {len(all_pairs):,} pairs.")

    # Train/val split
    random.shuffle(all_pairs)
    val_size = int(len(all_pairs) * args.val_split)
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    os.makedirs(args.output, exist_ok=True)
    current_model_path = args.model
    best_recall_at_10 = 0.0

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*50}")
        print(f"Hard Negative Mining — Iteration {iteration}/{args.iterations}")
        print(f"{'='*50}")

        model = SentenceTransformer(current_model_path)

        # Mine hard negatives
        triplets = mine_hard_negatives(
            model, train_pairs, doc_index, bid_list,
            top_k=args.top_k,
            neg_range=neg_range,
            negs_per_query=args.negs_per_query,
        )

        # Build training examples:
        # - Original positive pairs: InputExample(texts=[query, pos_doc])
        # - Hard negative triplets: InputExample(texts=[query, pos_doc, neg_doc])
        train_examples = [
            InputExample(texts=[BGE_QUERY_INSTRUCTION + p["query"], doc_index[p["business_id"]]])
            for p in train_pairs
        ]
        for t in triplets:
            train_examples.append(InputExample(
                texts=[
                    BGE_QUERY_INSTRUCTION + t["query"],
                    doc_index[t["positive_id"]],
                    doc_index[t["negative_id"]],
                ]
            ))

        random.shuffle(train_examples)
        print(f"  Training on {len(train_examples):,} examples ({len(train_pairs):,} positives + {len(triplets):,} triplets).")

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        iter_output = os.path.join(args.output, f"iter_{iteration}")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=0,
            optimizer_params={"lr": args.lr},
            output_path=iter_output,
            show_progress_bar=True,
            save_best_model=False,
        )

        model = SentenceTransformer(iter_output)
        metrics = evaluate(model, val_pairs, doc_index)
        print(f"  Iteration {iteration} metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        if metrics["Recall@10"] > best_recall_at_10:
            best_recall_at_10 = metrics["Recall@10"]
            best_path = os.path.join(args.output, "best")
            model.save(best_path)
            print(f"  -> New best saved to {best_path} (Recall@10={best_recall_at_10:.4f})")
            current_model_path = best_path
        else:
            print(f"  -> No improvement (best Recall@10={best_recall_at_10:.4f}). Keeping previous best.")
            current_model_path = os.path.join(args.output, "best")

    print(f"\nDone. Best model at {os.path.join(args.output, 'best')} (Recall@10={best_recall_at_10:.4f})")


if __name__ == "__main__":
    main()
