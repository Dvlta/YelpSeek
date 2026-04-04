import re

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

    def _extract_snippet(self, q_emb: np.ndarray, chunk_text: str) -> str:
        """Find the sentence in chunk_text most similar to the query."""
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return chunk_text[:200]
        if len(sentences) == 1:
            return sentences[0][:200]

        s_embs = self.model.encode(sentences, normalize_embeddings=True)
        scores = (s_embs @ q_emb.T).flatten()
        best_idx = int(np.argmax(scores))
        return sentences[best_idx][:200]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        # Encode query with BGE instruction prefix
        prefixed = BGE_QUERY_INSTRUCTION + query
        q_vec = self.model.encode([prefixed], normalize_embeddings=True).astype(np.float32)

        # Retrieve more chunks than needed — we'll aggregate by restaurant
        n_chunks = min(top_k * 10, self.index.ntotal)
        scores, indices = self.index.search(q_vec, n_chunks)

        # Aggregate: keep max score and best chunk_text per restaurant
        best: dict[str, dict] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[idx]
            bid = str(row["business_id"])
            if bid not in best or score > best[bid]["similarity_score"]:
                best[bid] = {
                    "business_id": bid,
                    "name": str(row["name"]),
                    "city": str(row["city"]),
                    "stars": float(row["stars"]),
                    "review_count": int(row["review_count"]),
                    "categories": str(row["categories"]),
                    "address": str(row["address"]),
                    "similarity_score": float(score),
                    "chunk_text": str(row.get("chunk_text", "")),
                }

        # Sort by score descending and keep top_k
        results = sorted(best.values(), key=lambda x: x["similarity_score"], reverse=True)[:top_k]

        # Extract snippets only for final results
        for result in results:
            result["snippet"] = self._extract_snippet(q_vec, result.pop("chunk_text"))

        return results
