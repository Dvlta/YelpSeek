"""
Stage 5: Build FAISS Index
Encodes all restaurant documents with the final trained model and builds
an HNSW index for fast approximate nearest-neighbor search at serving time.

Outputs:
  backend/data/index.faiss     — FAISS HNSW index
  backend/data/metadata.parquet — Restaurant metadata (row i = FAISS index i)
"""

import argparse
import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS HNSW index from trained encoder.")
    parser.add_argument("--model", type=str, default="models/encoder_v2/best",
                        help="Path to trained model checkpoint (default: models/encoder_v2/best)")
    parser.add_argument("--docs", type=str, default="data/processed/restaurant_docs.parquet",
                        help="Path to restaurant_docs.parquet")
    parser.add_argument("--index-out", type=str, default="backend/data/index.faiss",
                        help="Output path for FAISS index")
    parser.add_argument("--metadata-out", type=str, default="backend/data/metadata.parquet",
                        help="Output path for metadata parquet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=32,
                        help="HNSW M parameter — number of neighbors per node (default: 32)")
    parser.add_argument("--ef-construction", type=int, default=200,
                        help="HNSW efConstruction — controls index build quality (default: 200)")
    parser.add_argument("--ef-search", type=int, default=50,
                        help="HNSW efSearch — controls query-time recall/speed tradeoff (default: 50)")
    return parser.parse_args()


def main():
    args = parse_args()

    for path in [args.model, args.docs]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_out), exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}...")
    model = SentenceTransformer(args.model)

    # Load docs (chunked — one row per review)
    print(f"Loading docs from {args.docs}...")
    df = pd.read_parquet(args.docs)
    n_restaurants = df["business_id"].nunique()
    print(f"Loaded {len(df):,} chunks from {n_restaurants:,} restaurants.")

    # Encode each chunk individually (no query instruction prefix for documents)
    print("Encoding chunks...")
    chunk_texts = df["chunk_text"].tolist()
    embeddings = model.encode(
        chunk_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build HNSW index
    dim = embeddings.shape[1]
    print(f"Building HNSW index (dim={dim}, M={args.hnsw_m}, efConstruction={args.ef_construction})...")
    index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.ef_construction
    index.hnsw.efSearch = args.ef_search
    index.add(embeddings)
    print(f"Index built: {index.ntotal:,} chunks indexed ({n_restaurants:,} restaurants).")

    # Save index
    faiss.write_index(index, args.index_out)
    print(f"FAISS index saved to {args.index_out}")

    # Save metadata — one row per chunk, row i matches FAISS index position i
    metadata_cols = ["chunk_id", "business_id", "chunk_text", "name", "city", "state",
                     "stars", "review_count", "categories", "address", "latitude", "longitude"]
    df[metadata_cols].to_parquet(args.metadata_out, index=False)
    print(f"Metadata saved to {args.metadata_out}")


if __name__ == "__main__":
    main()
