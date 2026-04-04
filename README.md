# YelpSeek

Semantic restaurant search engine over 5,000+ Philadelphia restaurants. Instead of keyword matching, YelpSeek understands the *meaning* of natural language queries and retrieves restaurants whose reviews describe that experience.

> "cozy birthday dinner, not too loud" → returns intimate restaurants with great atmosphere, even if those exact words don't appear in any review.

![Search UI](docs/ui1.png)
![Search Results](docs/ui2.png)

---

## How It Works

1. **Data pipeline** — Filters the Yelp Open Dataset to Philadelphia restaurants, stores each review as an independent chunk (~195K chunks across 5,117 restaurants)
2. **Training pairs** — Generates 316K+ query-document pairs via pseudo-query extraction and LLM-synthesized queries
3. **Training** — Fine-tunes `BAAI/bge-base-en-v1.5` (dual-encoder) on query-chunk pairs using contrastive learning (MultipleNegativesRankingLoss)
4. **Indexing** — Encodes all review chunks and builds a FAISS HNSW index for sub-100ms retrieval
5. **Serving** — FastAPI backend with semantic snippet extraction + React/TypeScript frontend

### Chunked Encoding

The key architectural decision: each review is encoded as its own vector rather than concatenating reviews into a single document per restaurant. This avoids silent truncation at the model's 512-token limit and enables granular per-review matching. At query time, chunk scores are aggregated per restaurant via max-pooling. This single change improved Recall@10 from 0.61 to 0.90.

### Semantic Snippets

Search results show the most relevant sentence from the best-matching review, not just the first 200 characters. The matched review is split into sentences, each encoded and compared to the query, and the highest-similarity sentence is returned.

---

## Setup

### Prerequisites
- Python 3.12+
- Node.js 18+
- Yelp Open Dataset JSON files (download at [yelp.com/dataset](https://www.yelp.com/dataset))

### 1. Data Pipeline (run once)

```bash
cd ml
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Stage 1: Process raw Yelp data
python3 01_process_data.py --city "Philadelphia"

# Stage 2: Generate training pairs
python3 02_generate_pairs.py --strategy both --openai-key YOUR_KEY
# or free (pseudo only):
python3 02_generate_pairs.py --strategy pseudo
```

### 2. Train the Model (Google Colab recommended)

Upload `ml/03_train_encoder.py`, `data/processed/restaurant_docs.parquet`, and `data/processed/training_pairs.jsonl` to Google Colab, then:

```bash
!pip install sentence-transformers pandas pyarrow
!python3 03_train_encoder.py --batch-size 128 --lr 5e-5 --warmup-steps 300
```

Save the best checkpoint (`models/encoder_v2/best`) back to your machine.

### 3. Build the FAISS Index

```bash
python3 05_build_index.py --model models/encoder_v2/best
```

This outputs `backend/data/index.faiss` and `backend/data/metadata.parquet`.

### 4. Run the Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Create .env
echo "MODEL_PATH=../models/encoder_v2/best" > .env
echo "INDEX_PATH=data/index.faiss" >> .env
echo "METADATA_PATH=data/metadata.parquet" >> .env

uvicorn main:app --reload --port 8000
```

### 5. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

---

## Benchmarks

### Retrieval Quality

Evaluated on a 10% held-out validation set (~29K query-document pairs) after 3 epochs of fine-tuning on 5,117 Philadelphia restaurants with 195K review chunks.

#### Training Evaluation (exact search, full corpus)

| Metric | Score |
|--------|-------|
| Recall@1 | 0.700 |
| Recall@5 | 0.845 |
| Recall@10 | **0.898** |
| Recall@50 | 0.979 |
| MRR | 0.767 |
| NDCG@10 | 0.795 |

#### Pipeline Evaluation (FAISS HNSW, end-to-end)

| Metric | Score |
|--------|-------|
| Recall@1 | 0.675 |
| Recall@5 | 0.805 |
| Recall@10 | **0.853** |
| Recall@50 | 0.928 |
| MRR | 0.736 |
| NDCG@10 | 0.760 |

The gap between training and pipeline eval reflects HNSW approximation and retrieval truncation.

### Query Latency
Measured end-to-end on a MacBook (CPU inference, FAISS HNSW index).

| Step | Time |
|------|------|
| Query encoding | ~80ms |
| FAISS search (195K vectors) | <5ms |
| Snippet extraction | ~490ms |
| Total API response | ~576ms |

---

## Experiments

| Change | Result | Outcome |
|--------|--------|---------|
| Chunked encoding (vs blob) | Recall@10: 0.61 → 0.90 | Adopted |
| Max aggregation (vs top3_mean, blend) | Max outperformed all alternatives | Kept max |
| Cross-encoder re-ranking | Recall@10 dropped 0.85 → 0.80 | Rejected — generic model hurt domain-specific rankings |
| Hard negative mining | Recall@10 regressed | Rejected — false negatives in dense 5K-restaurant pool |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | PyTorch, sentence-transformers, FAISS |
| Backend | FastAPI, Uvicorn |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Data | Pandas, Parquet |
