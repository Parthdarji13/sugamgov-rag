# SugamGov AI — RAG Backend

A high-performance, strictly grounded Retrieval-Augmented Generation (RAG) backend engineered to provide Indian citizens with verified, multilingual access to welfare schemes and government entitlements.

---

## 1. Project Overview

**SugamGov RAG** is an asynchronous Python microservice built with **FastAPI**, **PostgreSQL + pgvector**, and **Google Gemini**. It solves the problem of finding relevant welfare schemes across fragmented government datasets by combining:
- **Lexical Search:** PostgreSQL Full-Text Search (GIN-indexed `search_vector`).
- **Dense Vector Search:** Local 384-dimensional multilingual embeddings using `intfloat/multilingual-e5-small` indexed with HNSW.
- **Hybrid Fusion:** Reciprocal Rank Fusion (RRF, $k=60$) merging lexical and semantic ranking signals.
- **Deterministic Scheme Reranking:** Multi-tier chunk consolidation that prevents single schemes from monopolizing result slots.
- **Grounded Answer Generation:** Strict context-bounded answer synthesis via Gemini (`gemini-3.5-flash-lite`) with bracketed evidence citations and zero parametric hallucination.
- **Multilingual Support:** Native querying and answering across **English**, **Hindi**, and **Gujarati**.

---

## 2. Architecture

```text
Next.js / Sugamai Frontend (Port 3000)
       │
       ▼  HTTP / SSE
FastAPI Backend (Port 8000)
       │
       ▼
Query Processing & Language Detection ('auto', 'en', 'hi', 'gu')
       │
       ▼
Hybrid Retrieval Layer (Concurrent Execution)
 ├── PostgreSQL Full-Text Search (GIN search_vector + ts_rank_cd)
 └── pgvector Dense Similarity (HNSW cosine ops + local e5-small 384d)
       │
       ▼
Reciprocal Rank Fusion (RRF, k=60)
       │
       ▼
Scheme Reranker & Chunk Consolidator (Unique scheme deduplication)
       │
       ▼
Evidence Context Builder (Prompt-injection isolation & 12,000-char budget)
       │
       ▼
Gemini Generation Engine (gemini-3.5-flash-lite @ temperature 0.0)
       │
       ▼
Progressive Streaming Response (Server-Sent Events / SSE)
```

---

## 3. Technology Stack

- **Web Framework:** FastAPI `>=0.115.0`, Uvicorn `>=0.30.0`
- **Validation & Serialization:** Pydantic v2 `>=2.0.0`
- **Database & Vector Extension:** PostgreSQL 15+, pgvector extension
- **Database ORM & Driver:** SQLAlchemy `>=2.0.0`, psycopg[binary] `>=3.1.0`
- **Local Embedding Model:** `intfloat/multilingual-e5-small` (384 dimensions, `sentence-transformers>=3.0.0`, `torch>=2.0.0`)
- **Indexing & Retrieval:** PostgreSQL GIN full-text index, HNSW vector index (`vector_cosine_ops`), Reciprocal Rank Fusion (RRF)
- **Generative LLM:** Google Gemini `gemini-3.5-flash-lite` via `google-genai>=1.0.0`
- **Data Engineering:** Pandas `>=2.0.0`, PyArrow `>=14.0.0` (Parquet)
- **Configuration:** `python-dotenv>=1.0.0`

---

## 4. Repository Structure

```text
sugamgov-rag/
├── api/                                       # FastAPI web service layer
│   ├── config.py                              # Safe application settings & credential isolation
│   ├── dependencies.py                        # Singleton providers (DB engine, model, services)
│   ├── main.py                                # App factory, CORS, exception handlers, lifecycle
│   ├── schemas.py                             # Pydantic request & response validation schemas
│   └── chat/                                  # Conversational multi-turn chat subsystem
│       ├── models.py                          # Chat session dataclasses & Pydantic models
│       ├── routes.py                          # APIRouter (/api/chat, /api/chat/stream, DELETE)
│       ├── service.py                         # Chat orchestration & deterministic query rewriter
│       └── session_store.py                   # Thread-safe in-memory session manager
├── src/                                       # Core RAG domain logic
│   ├── generation/                            # Grounded answer synthesis
│   │   ├── models.py                          # RAGAnswer & EvidenceCitation dataclasses
│   │   ├── prompt.py                          # Grounding prompt, language detection, directives
│   │   └── rag_generator.py                   # Gemini API client, zero-evidence short-circuit
│   └── retrieval/                             # Retrieval, indexing, and reranking
│       ├── context_builder.py                 # Budget-bounded evidence context assembly
│       ├── hybrid_retriever.py                # Concurrent lexical + vector search with RRF
│       ├── keyword_retriever.py               # PostgreSQL FTS with GIN search_vector
│       ├── models.py                          # RetrievalResult schema
│       ├── retrieval_service.py               # Unified service facade with coverage tracking
│       ├── scheme_reranker.py                 # Deterministic scheme-level chunk consolidation
│       └── vector_retriever.py                # Local e5-small SentenceTransformer retriever
├── database/                                  # Database migrations & documentation
│   ├── README.md                              # Database architecture & parameter guide
│   └── migrations/
│       ├── 001_initial_schema.sql             # Core tables (schemes, sources, scheme_chunks)
│       └── 002_add_fts_and_hnsw_indexes.sql   # Generated tsvector, GIN index, and HNSW index
├── data/                                      # Datasets & evaluation testsets
│   ├── raw/
│   │   └── updated_data.csv                   # Raw source dataset (3,397 schemes)
│   ├── processed/
│   │   ├── clean_schemes.parquet              # Standardized clean schemes
│   │   └── enriched_schemes.parquet           # Enriched schemes with normalized categories/states
│   └── evaluation/
│       ├── retrieval_eval.json                # Standard 60-query English benchmark testset
│       ├── multilingual_eval_corrected.json   # 26-query Hindi & Gujarati verified benchmark
│       └── rag_answer_quality_testset.json    # 31-case answer quality evaluation testset
├── scripts/                                   # Data pipelines, test suites, and benchmarks
│   ├── 01_clean_data.py                       # Data pipeline: raw CSV -> clean Parquet
│   ├── 02_enrich_metadata.py                  # Data pipeline: category/state enrichment
│   ├── 03_test_database.py                    # Database connection & extension verification
│   ├── 04_ingest_schemes.py                   # Ingests enriched_schemes.parquet into PostgreSQL
│   ├── 05_verify_database_integrity.py        # Validates schemes table counts & integrity
│   ├── 06_chunk_schemes.py                    # Chunks schemes table into 20,497 scheme_chunks
│   ├── 10_test_keyword_retrieval.py           # Verification suite: Keyword FTS retrieval
│   ├── 11_test_vector_retrieval.py            # Verification suite: Local 384d vector retrieval
│   ├── 12_test_hybrid_retrieval.py            # Verification suite: Hybrid RRF retrieval
│   ├── 14_test_scheme_reranker.py             # Verification suite: Scheme deduplication
│   ├── 15_test_retrieval_service.py           # Verification suite: Unified RetrievalService
│   ├── 16_test_context_builder.py             # Verification suite: EvidenceContextBuilder
│   ├── 17_test_rag_generator.py               # Verification suite: Grounded answer generation
│   ├── 18_generate_local_embeddings.py        # Populates 384d embedding_local in scheme_chunks
│   ├── 18_test_fastapi_backend.py             # Verification suite: FastAPI core endpoints
│   ├── 19_test_chat_api.py                    # Verification suite: Multi-turn chat API
│   ├── 20_test_streaming_chat.py              # Verification suite: SSE streaming chat
│   ├── 25_apply_retrieval_optimizations.py    # Database optimization runner (migration 002)
│   ├── 26_benchmark_step19_performance.py     # Latency benchmark script
│   ├── 27_evaluate_step20_final.py            # Active 86-query production retrieval benchmark
│   ├── 28_evaluate_rag_answer_quality.py      # Active 31-case RAG answer quality evaluation
│   ├── 29_verify_multilingual_refusal.py      # Active multilingual refusal verification suite
│   ├── test_rag_integration.py                # Optional: Next.js frontend integration suite
│   └── validate_e2e_scenarios.py              # Optional: Browser end-to-end scenario suite
├── reports/                                   # Final master technical reports & benchmarks
│   ├── 26_final_retrieval_evaluation.md       # Step 20 retrieval evaluation report
│   ├── 27_final_rag_answer_quality.md         # Step 21 answer quality report
│   ├── 28_multilingual_refusal_fix.md         # Step 22 refusal verification report
│   ├── 29_final_production_readiness_audit.md # Step 23 46-point readiness audit report
│   ├── 30_sugamgov_final_technical_report.md  # Master Technical Engineering Documentation
│   ├── final_retrieval_evaluation.json        # Step 20 retrieval benchmark raw scores
│   ├── final_rag_answer_quality.json          # Step 21 answer quality evaluation scores
│   ├── multilingual_refusal_fix.json          # Step 22 refusal verification scores
│   ├── production_readiness_audit.json        # Step 23 audit verification scores
│   └── sugamgov_final_metrics.json            # Final consolidated metrics summary
├── requirements.txt                           # Project Python dependencies
└── .env.example                               # Environment configuration template
```

---

## 5. Dataset

The corpus consists of **3,397 verified Indian government schemes** across Central and State jurisdictions:
- **Total Schemes:** 3,397 rows in the `schemes` table.
- **Total Chunks:** 20,497 field-aware contextual chunks in the `scheme_chunks` table.
- **Fields Segmented:** `scheme_name`, `details`, `benefits`, `eligibility`, `application`, `documents`.
- **Pre-computed Artifacts:**
  - `data/raw/updated_data.csv`: Authentic raw dataset (12.85 MB).
  - `data/processed/clean_schemes.parquet`: Standardized clean schemes (5.11 MB).
  - `data/processed/enriched_schemes.parquet`: Enriched schemes with normalized categories and states (5.19 MB).
  - `data/evaluation/`: Ground-truth benchmark query sets for English (60), Hindi (12), Gujarati (14), and Answer Quality (31).

> [!NOTE]
> Git stores the raw and processed Parquet files. PostgreSQL database tables and pgvector embeddings are maintained separately in your local PostgreSQL instance and must be populated during setup.

---

## 6. Database Setup

The backend requires **PostgreSQL 15+** with the **`pgvector`** extension enabled.

- **Database Name:** `sugamgov`
- **User:** `postgres` (or your configured user)
- **Port:** `5432`

### Applying Migrations

1. Connect to PostgreSQL and create the database:
   ```sql
   CREATE DATABASE sugamgov;
   ```
2. Apply Migration 001 (enables `pgvector` and creates core tables):
   ```bash
   psql -U postgres -d sugamgov -f database/migrations/001_initial_schema.sql
   ```
3. Populate Data & Embeddings:
   - Ingest schemes: `python scripts/04_ingest_schemes.py`
   - Generate chunks: `python scripts/06_chunk_schemes.py`
   - Generate local embeddings: `python scripts/18_generate_local_embeddings.py`
4. Apply Migration 002 (adds generated tsvector, GIN index, and HNSW index):
   ```bash
   psql -U postgres -d sugamgov -f database/migrations/002_add_fts_and_hnsw_indexes.sql
   ```

---

## 7. Environment Variables

Copy the provided template to create your `.env` file:
```powershell
Copy-Item .env.example .env
```

Configure the following variables in `.env`:

| Variable | Description | Default / Example |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection URL with psycopg dialect | `postgresql+psycopg://postgres:<PASSWORD>@localhost:5432/sugamgov` |
| `GEMINI_API_KEY` | Google Gemini API key for answer generation | `your_gemini_api_key_here` |
| `GEMINI_GENERATION_MODEL` | Gemini generative model name | `gemini-3.5-flash-lite` |
| `API_HOST` | FastAPI bind address | `127.0.0.1` |
| `API_PORT` | FastAPI port | `8000` |
| `API_RELOAD` | Enable auto-reload during development | `true` |
| `CORS_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost:3000,http://127.0.0.1:3000` |

> [!WARNING]
> Never commit `.env` to Git. Keep your real API keys and database passwords local.

---

## 8. Installation

Execute the following commands in Windows PowerShell:

```powershell
# 1. Clone repository
git clone https://github.com/Parthdarji13/sugamgov-rag.git
cd sugamgov-rag

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
Copy-Item .env.example .env
# Edit .env and supply your local DATABASE_URL and GEMINI_API_KEY

# 5. Verify database connection
python scripts/03_test_database.py
```

*Note: On first startup, the local embedding model `intfloat/multilingual-e5-small` (~1.1 GB weights) will be automatically downloaded and cached in your HuggingFace cache directory.*

---

## 9. Running the Backend

Launch the FastAPI development server:
```powershell
uvicorn api.main:app --reload --port 8000
```

- **API Base URL:** `http://127.0.0.1:8000`
- **Service Health Check:** `http://127.0.0.1:8000/health`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **ReDoc Documentation:** `http://127.0.0.1:8000/redoc`

---

## 10. API Endpoints

### System Endpoints
- `GET /health`
  - Safe health check returning operational status without revealing internal connection strings.

### Retrieval Endpoints
- `POST /api/retrieve`
  - Performs hybrid keyword + vector retrieval and scheme deduplication.
  - Does **not** call Gemini (zero generation cost / quota used).
  - Body:
    ```json
    {
      "query": "financial assistance for small farmers",
      "top_k": 5,
      "candidate_k": 25,
      "state": "Gujarat",
      "level": "Central",
      "category": "Agriculture"
    }
    ```

### Generation Endpoints
- `POST /api/generate`
  - End-to-end grounded RAG answer generation with citations.
  - Automatically falls back to deterministic refusal messages if no evidence is found.
  - Body:
    ```json
    {
      "query": "What are the eligibility criteria for PM Kisan?",
      "language": "auto"
    }
    ```

### Conversational Chat Endpoints
- `POST /api/chat`
  - Multi-turn conversational chat with bounded session memory and deterministic query reformulation.
  - Body:
    ```json
    {
      "message": "What documents are required?",
      "session_id": "optional_session_uuid",
      "language": "auto"
    }
    ```
- `POST /api/chat/stream`
  - Progressive Server-Sent Events (SSE) streaming chat endpoint.
  - Emits events:
    - `event: metadata` (citations, schemes, coverage info)
    - `event: token` (progressive answer text chunks)
    - `event: done` (turn summary and character counts)
    - `event: error` (sanitized error message)
- `DELETE /api/chat/{session_id}`
  - Terminates and clears an in-memory chat session.

---

## 11. Testing & Evaluation

The repository includes a comprehensive automated verification and evaluation suite in `scripts/`:

### Core Regression Test Suites
Run individual regression test scripts:
```powershell
# Test database connection & schema
python scripts/03_test_database.py
python scripts/05_verify_database_integrity.py

# Test retrieval layers
python scripts/10_test_keyword_retrieval.py
python scripts/11_test_vector_retrieval.py
python scripts/12_test_hybrid_retrieval.py
python scripts/14_test_scheme_reranker.py
python scripts/15_test_retrieval_service.py
python scripts/16_test_context_builder.py

# Test generation & API layers
python scripts/17_test_rag_generator.py
python scripts/18_test_fastapi_backend.py
python scripts/19_test_chat_api.py
python scripts/20_test_streaming_chat.py
python scripts/29_verify_multilingual_refusal.py
```

### Final Evaluation Benchmarks
Evaluate the active production pipeline:
```powershell
# 1. 86-Query Retrieval Benchmark (English, Hindi, Gujarati)
python scripts/27_evaluate_step20_final.py

# 2. 31-Case End-to-End RAG Answer Quality Benchmark
python scripts/28_evaluate_rag_answer_quality.py
```

### Review / Integration Scripts
- `scripts/test_rag_integration.py`: Integration test suite for the Next.js frontend proxy (requires frontend on `localhost:3000`).
- `scripts/validate_e2e_scenarios.py`: End-to-end browser scenario validation.

---

## 12. Frontend Integration

The backend is pre-configured to integrate seamlessly with the **Sugamai** Next.js 16 frontend:
- **CORS Allowed Origins:** Default includes `http://localhost:3000` and `http://127.0.0.1:3000`.
- **Chat Routing:** The Next.js chat route forwards citizen queries to `POST /api/chat/stream` and parses incoming Server-Sent Events.
- **Session Handling:** Frontend session tokens can be mapped directly to `session_id` query parameters.

---

## 13. Production & Performance Metrics

*(Empirical project evaluation results measured on October 2, 2026)*

- **Retrieval Accuracy (Step 20 Evaluation across 86 Queries):**
  - **Overall Recall@10:** **90.70%** (78 / 86 queries)
  - **English Recall@10:** **96.67%** (58 / 60 queries)
  - **Hindi Recall@10:** **83.33%** (10 / 12 queries)
  - **Gujarati Recall@10:** **71.43%** (10 / 14 queries)
- **Retrieval Latency:**
  - **Mean Latency:** **125.40 ms** (a 95.03% latency reduction achieved via GIN and HNSW indexing)
  - **P95 Latency:** **243.07 ms**
- **Regression Verification:** **157 / 157** automated tests passing.
- **RAG Answer Quality (Step 21 Evaluation across 31 Cases):**
  - **Factual Precision:** 100% grounded in retrieved database evidence.
  - **Observed Hallucinations:** **0.00%** (zero fabricated facts, amounts, or URLs observed).
  - **Adversarial Injection Defense:** 100% pass rate (untrusted database text strictly framed as passive data).

---

## 14. Important Notes & Known Limitations

1. **Static Dataset Snapshot:** The current database is a curated snapshot of 3,397 schemes. The `sources` table currently contains 0 records; responses do not claim live official website scraping.
2. **Linguistic Variance:** Gujarati semantic retrieval (71.43% R@10) is lower than English (96.67% R@10) due to asymmetric token density in government terminology.
3. **Generation Latency:** While retrieval executes locally in ~125 ms, total answer generation time depends on Google Gemini API network latency and streaming token rates.
4. **Local Database Requirement:** PostgreSQL tables and pgvector embeddings reside locally on your database server and are not packaged inside Git.

---

## 15. Handover Notes for New Developers

Before running the project on a new workstation, ensure you have:
1. **PostgreSQL 15+** installed with the `vector` extension (`CREATE EXTENSION vector;`).
2. A `.env` file containing a valid `DATABASE_URL` pointing to your local PostgreSQL instance and a valid `GEMINI_API_KEY`.
3. If starting from an empty database, execute migrations `001` and `002`, then run ingestion scripts `scripts/04_ingest_schemes.py`, `scripts/06_chunk_schemes.py`, and `scripts/18_generate_local_embeddings.py`.
4. Ensure your system has sufficient RAM (~2 GB) to load `intfloat/multilingual-e5-small` in memory.

---

## 16. Security Guidelines

- **Credential Isolation:** Never commit `.env` or paste database connection strings into documentation.
- **Prompt Injection Defense:** All database chunk content is wrapped in strict delimiters and system instructions prohibiting instruction execution.
- **API Error Masking:** The global exception handler in `api/main.py` intercepts unhandled exceptions to ensure database passwords, hostnames, or internal API keys are never leaked to clients.
