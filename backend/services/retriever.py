import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# BGE retrieval models require this prefix on queries (not documents)
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self, index_path: str, metadata_path: str, model_path: str, ef_search: int = 50):
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        self.index.hnsw.efSearch = ef_search

        print(f"Loading metadata from {metadata_path}...")
        self.metadata = pd.read_parquet(metadata_path)

        print(f"Loading model from {model_path}...")
        self.model = SentenceTransformer(model_path)
        print("Retriever ready.")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        # Encode query with BGE instruction prefix
        prefixed = BGE_QUERY_INSTRUCTION + query
        q_vec = self.model.encode([prefixed], normalize_embeddings=True).astype(np.float32)

        # Search FAISS index
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[idx]
            results.append({
                "business_id": str(row["business_id"]),
                "name": str(row["name"]),
                "city": str(row["city"]),
                "stars": float(row["stars"]),
                "review_count": int(row["review_count"]),
                "categories": str(row["categories"]),
                "address": str(row["address"]),
                "similarity_score": float(score),
                "snippet": str(row.get("combined_reviews", ""))[:200],
            })

        return results
