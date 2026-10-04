# SugamGov Database Architecture & Setup

This directory contains the database migration files and architecture documentation for the **sugamgov** PostgreSQL database.

---

## 1. Database Overview (Beginner's Guide)

1. **PostgreSQL as the Relational Foundation**:
   PostgreSQL serves as the primary system of record for all government scheme metadata, relational links, source tracking, and version history.
2. **pgvector for Vector Search**:
   The `pgvector` extension is enabled inside PostgreSQL, allowing dense vector embeddings to be stored alongside text and queried using vector similarity distance operators (e.g., cosine similarity, inner product, L2 distance).
3. **`schemes` Table**:
   Stores the primary, structured information for each government scheme (title, eligibility, benefits, application process, required documents, normalized categories, tags, and state attribution).
4. **`sources` Table**:
   Tracks the provenance and authenticity of scheme data (e.g., official ministry websites, scheme portal URLs, verification status, and check timestamps).
5. **`scheme_versions` Table**:
   Preserves historical snapshots (using PostgreSQL `JSONB`) whenever scheme details or guidelines change over time.
6. **`scheme_chunks` Table**:
   Stores modular, RAG-ready text chunks derived from individual scheme fields (e.g., `details`, `eligibility`, `benefits`, `application`, `documents`) along with their future vector embeddings.
7. **Empty Schema in Step 3**:
   Step 3 establishes the database schema, indexes, and triggers only. No scheme data is inserted in this step.
8. **Data Ingestion Schedule**:
   The 3,397 enriched schemes from Step 2 will be ingested in a dedicated upcoming ingestion phase.
9. **Embeddings Schedule**:
   Dense vector embeddings will be generated and stored in `scheme_chunks.embedding` after selecting and evaluating the embedding model.

---

## 2. Local Connection Parameters

To connect to the local PostgreSQL database using CLI tools or database management software (such as pgAdmin or DBeaver):

| Parameter | Value |
|---|---|
| **Host** | `localhost` (or `127.0.0.1`) |
| **Port** | `5432` |
| **Database** | `sugamgov` |
| **User** | `postgres` |
| **Password** | *(Your private administrator password set during installation)* |

> **Security Note**: Never commit or hardcode actual database passwords into source code or repository files. Use the `.env` file (which is git-ignored) for local authentication credentials.

---

## 3. Directory Structure

```text
database/
├── README.md                          # Database architecture guide
└── migrations/
    └── 001_initial_schema.sql         # Initial DDL migration script
```
