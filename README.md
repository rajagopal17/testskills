# SAP RAG Search — pgvector + OpenAI

A production-ready Retrieval-Augmented Generation (RAG) pipeline over SAP documentation stored in PostgreSQL with pgvector. Queries are embedded with OpenAI, retrieved via vector search, reranked with a cross-encoder, and answered by GPT-4o-mini.

---

## Architecture

```
User Query
    │
    ▼
[Embedder]  OpenAI text-embedding-3-small (1536 dims)
    │
    ▼
[Searcher]  PostgreSQL pgvector — cosine similarity (flat/HNSW/IVFFlat)
    │
    ▼
[Reranker]  sentence-transformers cross-encoder (ms-marco-MiniLM-L-6-v2)
    │
    ▼
[LLM]       GPT-4o-mini — synthesizes grounded answer from top chunks
    │
    ▼
Answer + ranked chunks
```

---

## Features

- **Vector search** — pgvector with HNSW, IVFFlat, or flat (exact) scan
- **Cross-encoder reranking** — improves chunk ordering post-retrieval
- **LLM synthesis** — GPT-4o-mini produces a cited, context-grounded answer
- **Metadata filters** — filter by source, doc_category, fiscal_period, entity
- **Debug HTML server** — FastAPI page showing ranked chunks, scores, and metrics
- **Golden dataset evaluation** — P@K, R@K, F1@K, MRR, NDCG@K, Hit Rate@K
- **Answer quality scoring** — keyword coverage of expected answer keywords
- **Adversarial robustness** — detects false retrievals on unanswerable queries

---

## Project Structure

```
├── main.py                  # CLI entry point — run a query
├── config.py                # Loads and validates all .env settings
├── embedder.py              # OpenAI embedding with retry
├── searcher.py              # pgvector cosine search (flat/HNSW/IVFFlat)
├── reranker.py              # Cross-encoder reranker
├── llm.py                   # GPT-4o-mini answer synthesis
├── metrics.py               # P@K, R@K, MRR, NDCG, answer quality
├── debug_server.py          # FastAPI HTML debug viewer (--debug flag)
├── eval.py                  # Full evaluation runner against golden dataset
├── build_ground_truth.py    # Converts goldendataset.txt → ground_truth.json
├── goldendataset.txt        # 12 golden queries with expected answers & keywords
├── ground_truth.json        # Generated — maps queries to relevant chunk IDs
├── requirements.txt         # Python dependencies
└── .env                     # Config (not committed — see .env.example below)
```

---

## Setup

### 1. Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension installed
- Existing table with `content` (text) and `embedding` (vector) columns

### 2. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/sap-rag-pgvector.git
cd sap-rag-pgvector
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Configure `.env`

Create a `.env` file (see template below):

```env
# OpenAI
OPENAI_API_KEY=sk-...

# PostgreSQL
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=postgres
DB_PASSWORD=your_password
DB_SCHEMA=public

# Table columns
TABLE_NAME=your_table
TEXT_COLUMN=content
EMBEDDING_COLUMN=embedding
METADATA_COLUMNS=source,file_name,file_type,doc_category,fiscal_period,entity

# pgvector index
PGVECTOR_INDEX_TYPE=flat          # flat | hnsw | ivfflat
PGVECTOR_DISTANCE_METRIC=cosine

# OpenAI embedding — must match the model used to ingest the data
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# LLM
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1024

# Search
DEFAULT_TOP_K=7
SIMILARITY_THRESHOLD=0.2

# Reranker
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=5
```

---

## Usage

### Run a query

```bash
python main.py --query "What is the blocking indicator in SAP asset master?"
```

```bash
python main.py --query "How do you change the asset class?" --top-k 5
```

### Filter by metadata

```bash
python main.py --query "cash flow forecast" --filters '{"file_type": "pdf"}'
```

### Debug mode — HTML viewer at localhost:8000

```bash
python main.py --query "What is the blocking indicator?" --debug
```

Open `http://127.0.0.1:8000` to see:
- Synthesized answer
- Score statistics (similarity, rerank, Spearman correlation)
- Retrieval accuracy metrics (P@K, R@K, F1@K, MRR, NDCG@K, Hit Rate@K)
- Answer quality — keyword coverage bar with matched/missing keywords
- Adversarial robustness check
- Ranked chunks table with similarity and rerank scores

> Retrieval accuracy and answer quality panels populate only for queries present in `goldendataset.txt` / `ground_truth.json`.

---

## Evaluation

### Build ground truth from golden dataset

```bash
python build_ground_truth.py
```

Finds chunk IDs in the DB matching each golden query's keywords and writes `ground_truth.json`.

### Run full evaluation

```bash
python eval.py --top-k 7
```

Outputs per-query and aggregate metrics stratified by category:

```
========================================================================
  AGGREGATE METRICS BY CATEGORY
────────────────────────────────────────────────────────────────────────
  factual         n=5  |  P@K=0.086  R@K=0.003  MRR=0.400  Hit@K=0.400
  procedural      n=2  |  P@K=0.714  R@K=0.009  MRR=1.000  Hit@K=1.000
  comparative     n=2  |  P@K=0.572  R@K=0.029  MRR=0.667  Hit@K=1.000
  adversarial     n=3  |  robustness=0/3 PASS
────────────────────────────────────────────────────────────────────────
  OVERALL (non-adv)    |  P@K=0.333  R@K=0.010  MRR=0.593  Hit@K=0.667
========================================================================
```

Saves detailed results to `eval_report.json`.

---

## Golden Dataset Format

`goldendataset.txt` uses a structured Python format:

```python
GoldenItem(
    "What is the purpose of the blocking indicator in the SAP asset master record?",
    "The blocking indicator prevents future acquisition postings to the asset...",
    ["blocking indicator", "acquisitions", "asset under construction", "blocked"],
    "factual",
)
```

Categories: `factual` | `procedural` | `comparative` | `adversarial`

---

## Notes

- **Embedding model must match**: the model in `.env` must match what was used to embed the stored vectors. Mismatched models produce low similarity scores.
- **Index type**: use `flat` if unsure — it bypasses approximate indexes and does an exact scan. Switch to `hnsw` after re-embedding with the correct model for faster search.
- **Similarity threshold**: lower (0.2) to get more results; raise (0.45+) to reduce false retrievals on adversarial queries.

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | Embeddings + GPT-4o-mini synthesis |
| `psycopg2-binary` + `pgvector` | PostgreSQL vector search |
| `sentence-transformers` | Cross-encoder reranker |
| `fastapi` + `uvicorn` | Debug HTML server |
| `python-dotenv` | `.env` configuration |
| `loguru` | Structured logging |
| `scipy` | Spearman correlation in score stats |
