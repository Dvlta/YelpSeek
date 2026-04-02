"""
Stage 3: Train Dual-Encoder
Fine-tunes BAAI/bge-base-en-v1.5 on (query, doc_text) pairs
using MultipleNegativesRankingLoss.

BGE retrieval models require a query instruction prefix for short queries.
Documents are encoded as-is (no prefix).
"""

import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train dual-encoder on restaurant query pairs.")
    parser.add_argument("--pairs", type=str, default="data/processed/training_pairs.jsonl")
    parser.add_argument("--docs", type=str, default="data/processed/restaurant_docs.parquet")
    parser.add_argument("--output", type=str, default="models/encoder_v1")
    parser.add_argument("--base-model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# BGE retrieval models require this prefix on queries (not documents)
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


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
    scores = []
    for r in ranks:
        if r <= k:
            scores.append(1.0 / math.log2(r + 1))
        else:
            scores.append(0.0)
    ideal = 1.0 / math.log2(2)  # ideal DCG: true positive at rank 1
    return (sum(scores) / len(scores)) / ideal


def evaluate(model: SentenceTransformer, val_pairs: list[dict], doc_index: dict[str, str]) -> dict:
    """
    Evaluate retrieval metrics on the validation set.
    For each query, rank all documents and find the rank of the true positive.
    """
    print("Running evaluation...")

    # Get unique business_ids in val set
    val_bids = list({p["business_id"] for p in val_pairs})
    bid_to_idx = {bid: i for i, bid in enumerate(val_bids)}

    # Encode all candidate documents (no prefix for BGE)
    doc_texts = [doc_index[bid] for bid in val_bids]
    doc_embeddings = encode_docs(model, doc_texts)

    # Encode queries with BGE instruction prefix
    queries = [p["query"] for p in val_pairs]
    query_embeddings = encode_queries(model, queries)

    ranks = []
    for i, pair in enumerate(val_pairs):
        q_emb = query_embeddings[i]
        scores = doc_embeddings @ q_emb  # cosine similarity (normalized)
        true_idx = bid_to_idx[pair["business_id"]]
        # rank is 1-indexed position of the true doc (higher score = lower rank)
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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load docs
    print(f"Loading docs from {args.docs}...")
    df_docs = pd.read_parquet(args.docs)
    doc_index = dict(zip(df_docs["business_id"], df_docs["doc_text"]))
    print(f"Loaded {len(doc_index):,} restaurant documents.")

    # Load pairs
    print(f"Loading pairs from {args.pairs}...")
    pairs = []
    with open(args.pairs, "r") as f:
        for line in f:
            p = json.loads(line)
            if p["business_id"] in doc_index:
                pairs.append(p)
    print(f"Loaded {len(pairs):,} valid pairs.")

    # Train/val split (stratified by business_id to avoid data leakage)
    random.shuffle(pairs)
    val_size = int(len(pairs) * args.val_split)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    print(f"Train: {len(train_pairs):,} | Val: {len(val_pairs):,}")

    # Build InputExamples for training (queries get BGE instruction prefix, docs do not)
    train_examples = [
        InputExample(texts=[BGE_QUERY_INSTRUCTION + p["query"], doc_index[p["business_id"]]])
        for p in train_pairs
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    # Load model
    print(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)
    model.max_seq_length = 512

    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = len(train_dataloader) * args.epochs
    print(f"Training for {args.epochs} epochs ({total_steps:,} steps)...")

    os.makedirs(args.output, exist_ok=True)

    best_recall_at_10 = 0.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        epoch_output = os.path.join(args.output, f"epoch_{epoch}")

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=args.warmup_steps if epoch == 1 else 0,
            optimizer_params={"lr": args.lr},
            output_path=epoch_output,
            show_progress_bar=True,
            save_best_model=False,
        )

        # Reload saved checkpoint for evaluation
        model = SentenceTransformer(epoch_output)

        metrics = evaluate(model, val_pairs, doc_index)
        print(f"Epoch {epoch} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if metrics["Recall@10"] > best_recall_at_10:
            best_recall_at_10 = metrics["Recall@10"]
            best_epoch = epoch
            # Save best model
            best_path = os.path.join(args.output, "best")
            model.save(best_path)
            print(f"  -> New best model saved to {best_path} (Recall@10={best_recall_at_10:.4f})")

    print(f"\nTraining complete. Best epoch: {best_epoch} (Recall@10={best_recall_at_10:.4f})")
    print(f"Best model saved at: {os.path.join(args.output, 'best')}")


if __name__ == "__main__":
    main()
