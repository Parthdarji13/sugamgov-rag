# Step 20: Final Post-Optimization Retrieval Evaluation Report

**Date:** 2026-10-02 12:48:00 UTC  
**Status:** COMPLETE (Evaluation-Only, Zero Code/DB Mutations)  
**System Evaluated:** SugamGov AI Production Retrieval Subsystem (`KeywordRetriever`, `VectorRetriever`, `HybridRetriever`, `SchemeReranker`, `RetrievalService`, PostgreSQL `sugamgov`)  

---

## 1. Executive Summary

Following the retrieval performance optimization executed in Step 19 (persisted tsvector GIN index, pgvector HNSW index, and parallel hybrid execution), Step 20 conducted the comprehensive final empirical evaluation of the SugamGov retrieval subsystem.

Key findings:
1. **Quality Preservation on Standard Benchmark:** On the established 60-query English benchmark (`data/evaluation/retrieval_eval.json`), retrieval accuracy was completely preserved with **zero regression**:
   - Scheme Reranked Recall@5 = **93.33%** (56/60)
   - Scheme Reranked Recall@10 = **96.67%** (58/60)
   - Scheme Reranked MRR@10 = **0.8449**
   - Rank #1 hits: **76.7%** (46/60)
   - Candidate pool hit rate: **100.0%** (60/60)
2. **Corrected Multilingual Diagnostic Resolution:** Using the verified real scheme targets in PostgreSQL (resolving the testset misalignments identified in Step 18), multilingual cross-lingual retrieval capability was accurately evaluated:
   - Hindi ($N=12$): Recall@5 = **75.00%**, Recall@10 = **83.33%**, Hits = **11/12 (91.7%)**
   - Gujarati ($N=14$): Recall@5 = **50.00%**, Recall@10 = **71.43%**, Hits = **11/14 (78.6%)**, Rank #1 hits = **5/14 (35.7%)**
3. **Overall System Metrics ($N=86$):**
   - Scheme Reranked Recall@5 = **83.72%** (72/86)
   - Scheme Reranked Recall@10 = **90.70%** (78/86)
   - Scheme Reranked MRR@10 = **0.7390**
   - Candidate Pool Hit Rate: **95.35%** (82/86)
4. **Latency Verification Across All 86 Queries:**
   - Full production end-to-end retrieval (`RetrievalService.retrieve()`) averaged **125.40 ms** (median **105.68 ms**, p95 **243.07 ms**, min **93.14 ms**), confirming a **95.03% latency reduction (20.1x speedup)** compared to the Step 18 baseline of **2,524.76 ms**.
5. **PostgreSQL Query Plans Verified:**
   - GIN index (`idx_scheme_chunks_search_vector`): `Bitmap Index Scan` confirmed via `EXPLAIN (ANALYZE, BUFFERS)`.
   - HNSW index (`idx_scheme_chunks_embedding_local_hnsw`): `Index Scan` confirmed via `EXPLAIN (ANALYZE, BUFFERS)`.
6. **Safety & Database Integrity:**
   - All 157 automated regression tests passed (157/157).
   - Zero database rows, chunks, embeddings, or indexes were modified or deleted during evaluation.

---

## 2. Evaluation Dataset

The evaluation suite encompasses **86 authentic test queries** covering central and state government schemes across three languages:

| Dataset Segment | Source File / Definition | Query Count | Target Corpus Language | Query Language |
| :--- | :--- | :---: | :---: | :---: |
| **English Benchmark** | `data/evaluation/retrieval_eval.json` | 60 | English | English |
| **Hindi Benchmark** | `data/evaluation/multilingual_eval_corrected.json` | 12 | English | Hindi (Devanagari) |
| **Gujarati Benchmark**| `data/evaluation/multilingual_eval_corrected.json` | 14 | English | Gujarati (Gujarati script) |
| **Total Scope** | | **86** | | |

*Corpus Note:* 100% of the 20,497 scheme chunks in the database are stored in English text. All Hindi and Gujarati queries operate in zero-shot cross-lingual retrieval mode through `intfloat/multilingual-e5-small`.

---

## 3. Ground-Truth Corrections

In Step 18, diagnostic investigation revealed that the low multilingual Recall reported in Step 17 was primarily an evaluation artifact caused by label mismatches in the testset (where query definitions expected scheme IDs corresponding to unrelated out-of-state schemes). Per instructions in Part 1, the original benchmark file was left untouched, and a separate verified copy was created at `data/evaluation/multilingual_eval_corrected.json`.

### A. Gujarati Corrections ($N=14$)

| Query ID | Gujarati Query Text | Original Target ID | Original Scheme in DB | Corrected Target ID | Corrected Scheme in DB | Reason for Correction |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| `GU_01` | મુખ્યમંત્રી અમૃતમ મા યોજના હેઠળ કેશલેસ સારવારની પાત્રતા | `S1923` | Modernisation And Removal Of Obsolescence (Central AICTE) | `S1976` | Mukhyamantri Amrutum Yojana (Gujarat) | `S1923` is an engineering lab grant; `S1976` is the actual Gujarat health card scheme. |
| `GU_02` | ગુજરાતમાં કુંવરબાઈનું મામેરું યોજના માટે કોણ અરજી કરી શકે? | `S1726` | Livelihood Support to Marine Fishermen (Odisha) | `S1686` | Kunwar Bai Nu Mameru Yojana (Gujarat) | `S1726` is an Odisha fishing ban relief scheme; `S1686` is the Gujarat marriage grant. |
| `GU_03` | ગુજરાત માનવ ગરિમા યોજના હેઠળ સ્વરોજગાર માટે સાધન સહાય | `S1765` | Mahila Kisan Yojana (Maharashtra) | `S1793` | Manav Garima Yojana (Gujarat) | `S1765` is a Maharashtra agriculture scheme; `S1793` is the Gujarat self-employment tools scheme. |
| `GU_04` | ગંગા સ્વરૂપા વિધવા આર્થિક સહાય યોજના ગુજરાત પેન્શન નિયમો | `S1072` | Financial Assistance to Widower (Haryana) | `S1166`, `S1167`| Ganga Swarupa Pension Scheme (Gujarat) | `S1072` is a Haryana widower scheme; `S1166`/`S1167` are the Gujarat widow assistance schemes. |
| `GU_05` | ગુજરાત બિનઅનામત શૈક્ષણિક લોન યોજના ઉચ્ચ શિક્ષણ | `S0967` | Fishermen Gill Net (Goa) | `S0880`, `S0798`| Educational Study Scheme / Foreign Study Loan (Gujarat) | `S0967` is a Goa fishing scheme; `S0880` is higher education in Gujarat. |
| `GU_06` | ગુજરાતના ખેડૂતો માટે સનેડો કૃષિ સાધન ખરીદી પર સહાય | `S2848` | Sanedo Agricultural Equipment (Gujarat) | `S2848` | Sanedo Agricultural Equipment (Gujarat) | Verified accurate (no change). |
| `GU_07` | ગુજરાત અનુસૂચિત જાતિ ખેડૂતો માટે જમીન ખરીદી સહાય યોજના | `S0959` | Land Purchase SC Farmers (Gujarat) | `S0959` | Land Purchase SC Farmers (Gujarat) | Verified accurate (no change). |
| `GU_08` | ગુજરાત સરદાર પટેલ આવાસ યોજના ગ્રામીણ મકાન સહાય | `S2677` | Sardar Patel Awas Yojana (Gujarat) | `S2677` | Sardar Patel Awas Yojana (Gujarat) | Verified accurate (no change). |
| `GU_09` | બાંધકામ શ્રમિકો માટે શ્રમિક અન્નપૂર્ણા યોજના 5 રૂપિયામાં ભોજન | `S2889` | Seed Testing Laboratory (Meghalaya) | `S2931` | Shramik Annapurna Yojana (GBOCWWB) (Gujarat) | `S2889` is a Meghalaya seed lab; `S2931` is the Gujarat construction meal scheme. |
| `GU_10` | ગુજરાત દિવ્યાંગ વિદ્યાર્થીઓ માટે સાધન સહાય યોજના | `S0156` | Aatmanirbhar MSME ICT Assistance (Gujarat) | `S2510` | Prosthetic Aid & Appliances for PwD (Gujarat) | `S0156` is an MSME software grant; `S2510` is disability prosthetic aid. |
| `GU_11` | સરસ્વતી સાધના યોજના ગુજરાત વિદ્યાર્થિનીઓને મફત સાઇકલ | `S2680` | Sarvajan Pension (Jharkhand) | `S1111` | Free Cycles to SC Girl Students (Sarasvati Sadhna) (Gujarat) | `S2680` is a Jharkhand pension; `S1111` is the Gujarat Saraswati Sadhna bicycle scheme. |
| `GU_12` | ગુજરાત પશુપાલકો માટે 1 થી 20 દુધાળા પશુ યુનિટ વ્યાજ સહાય | `S2845` | Processing Units in Market Committees (Gujarat) | `S2708`, `S0895`| Subsidy on Interest for 1 to 20 Milch Animal Farm (Gujarat) | `S2845` is market committee infrastructure; `S2708`/`S0895` are the milch animal schemes. |
| `GU_13` | વહાલી દીકરી યોજના પ્રથમ હપ્તો અને પાત્રતા શરતો | `S3290` | Day Care for Disabled Children (MP) | `S3268` | Vahli Dikri Yojana (Gujarat) | `S3290` is day care in MP; `S3268` is the Gujarat girl-child incentive scheme. |
| `GU_14` | ગુજરાત વનબંધુ કલ્યાણ યોજના આદિજાતિ સર્વાંગી વિકાસ | `S3292` | Vikramaditya Yojna (MP) | `S3270` | Vanbandhu Kalyan Yojana (State) | `S3292` is education in MP; `S3270` is the Vanbandhu Kalyan tribal development scheme. |

### B. Hindi Corrections ($N=12$)
- `HI_10`: Original `S2648` (*Samagra Samajik Suraksha Vridhavastha Pension Yojana* in MP) was corrected to `S0112` (*AICTE – Saksham Scholarship Scheme For Specially-Abled Student (Degree)*, Central).
- `HI_11`: Original `S1873` (*Medical Assistance in Arunachal Pradesh*) was corrected to `S0528` (*Chief Minister Kanya Utthan Yojana* in Bihar).
- `HI_12`: Original `S2441` (*Pratibha Kiran Scholarship* in MP) was corrected to `S2341` / `S2084` (*Pension for Construction Workers* / *Sambal 2.0*).
- `HI_01` to `HI_09`: Verified accurate as previously defined.

---

## 4. Overall Retrieval Metrics ($N=86$)

Evaluated across all 86 benchmark queries using the four retrieval modes:

| Retrieval Modality | Recall@5 | Recall@10 | MRR@5 | MRR@10 | Hits (Any Rank) | Rank #1 Hits | Misses (>Top-10) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keyword (FTS GIN)** | 0.5116 | 0.5349 | 0.4310 | 0.4349 | 53 / 86 | 33 / 86 (38.4%) | 40 / 86 |
| **Local Vector (HNSW, 384d)** | **0.8721** | **0.9302** | **0.7579** | **0.7663** | **82 / 86** | **59 / 86 (68.6%)** | **6 / 86** |
| **Hybrid RRF ($k=60$)** | 0.8488 | 0.9070 | 0.7421 | 0.7500 | 82 / 86 | 57 / 86 (66.3%) | 8 / 86 |
| **Scheme Reranked (Production)** | **0.8372** | **0.9070** | **0.7291** | **0.7390** | **82 / 86** | **56 / 86 (65.1%)** | **8 / 86** |

---

## 5. English Benchmark Metrics ($N=60$)

Evaluated on the standard 60-query English testset (`retrieval_eval.json`):

| Retrieval Modality | Recall@5 | Recall@10 | MRR@5 | MRR@10 | Hits (Top-40) | Rank #1 Hits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Keyword (FTS GIN) | 0.7333 | 0.7667 | 0.6178 | 0.6233 | 53 / 60 | 33 / 60 (55.0%) |
| Local Vector (HNSW) | 0.9833 | 1.0000 | 0.8817 | 0.8840 | 60 / 60 | 49 / 60 (81.7%) |
| Hybrid RRF ($k=60$) | 0.9500 | 0.9667 | 0.8589 | 0.8607 | 60 / 60 | 46 / 60 (76.7%) |
| **Scheme Reranked (Production)** | **0.9333** | **0.9667** | **0.8403** | **0.8449** | **60 / 60** | **46 / 60 (76.7%)** |

---

## 6. Hindi Benchmark Metrics ($N=12$)

Evaluated on 12 native Devanagari Hindi queries:

| Retrieval Modality | Recall@5 | Recall@10 | MRR@5 | MRR@10 | Hits (Top-40) | Rank #1 Hits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Keyword (FTS GIN) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 / 12 | 0 / 12 |
| Local Vector (HNSW) | 0.7500 | 0.8333 | 0.5583 | 0.5722 | 11 / 12 | 5 / 12 (41.7%) |
| Hybrid RRF ($k=60$) | 0.7500 | 0.8333 | 0.5583 | 0.5722 | 11 / 12 | 5 / 12 (41.7%) |
| **Scheme Reranked (Production)** | **0.7500** | **0.8333** | **0.5583** | **0.5722** | **11 / 12** | **5 / 12 (41.7%)** |

*Analysis:* Keyword FTS scores 0.0000 because all chunks are stored in English text. Vector semantic search successfully bridges the cross-lingual divide, achieving 83.33% Recall@10 with 5 Rank #1 hits.

---

## 7. Gujarati Benchmark Metrics ($N=14$)

Evaluated on 14 native Gujarati script queries:

| Retrieval Modality | Recall@5 | Recall@10 | MRR@5 | MRR@10 | Hits (Top-40) | Rank #1 Hits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Keyword (FTS GIN) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 / 14 | 0 / 14 |
| Local Vector (HNSW) | 0.5000 | 0.7143 | 0.3988 | 0.4281 | 11 / 14 | 5 / 14 (35.7%) |
| Hybrid RRF ($k=60$) | 0.5000 | 0.7143 | 0.3988 | 0.4281 | 11 / 14 | 5 / 14 (35.7%) |
| **Scheme Reranked (Production)** | **0.5000** | **0.7143** | **0.3988** | **0.4281** | **11 / 14** | **5 / 14 (35.7%)** |

---

## 8. Query-by-Query Gujarati Diagnostic Breakdown ($N=14$)

To avoid relying solely on aggregate metrics, every Gujarati query was individually analyzed through the production retrieval pipeline:

| QID | Gujarati Query Text | Target SID | Target Scheme Name | Vec Rank | Hyb Rank | Rerank Rank | Top-5 | Top-10 | Top-40 | Top Retrieved Scheme (#1) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `GU_01` | મુખ્યમંત્રી અમૃતમ મા યોજના હેઠળ કેશલેસ સારવારની પાત્રતા | `S1976` | Mukhyamantri Amrutum Yojana | 26 | 26 | 26 | No | No | Yes | `[S1642]` Kanyashree Prakalpa |
| `GU_02` | ગુજરાતમાં કુંવરબાઈનું મામેરું યોજના માટે કોણ અરજી કરી શકે? | `S1686` | Kunwar Bai Nu Mameru Yojana | 10 | 10 | 10 | No | Yes | Yes | `[S0204]` Agricultural Marketing Infrastructure |
| `GU_03` | ગુજરાત માનવ ગરિમા યોજના હેઠળ સ્વરોજગાર માટે સાધન સહાય | `S1793` | Manav Garima Yojana | None | None | None | No | No | No | `[S2898]` Self-Employment: Vahan Loan Sahay |
| `GU_04` | ગંગા સ્વરૂપા વિધવા આર્થિક સહાય યોજના ગુજરાત પેન્શન નિયમો | `S1166` | Ganga Swarupa Pension Scheme | 3 | 3 | 3 | Yes | Yes | Yes | `[S0729]` Disability Pension - West Bengal |
| `GU_05` | ગુજરાત બિનઅનામત શૈક્ષણિક લોન યોજના ઉચ્ચ શિક્ષણ | `S0880` | Educational Study Scheme | 4 | 4 | 4 | Yes | Yes | Yes | `[S2627]` Saansad Adarsh Gram Yojana |
| `GU_06` | ગુજરાતના ખેડૂતો માટે સનેડો કૃષિ સાધન ખરીદી પર સહાય | `S2848` | Sanedo Agricultural Equipment | 1 | 1 | 1 | Yes | Yes | Yes | `[S2848]` Sanedo Agricultural Equipment |
| `GU_07` | ગુજરાત અનુસૂચિત જાતિ ખેડૂતો માટે જમીન ખરીદી સહાય યોજના | `S0959` | Land Purchase SC Farmers | 1 | 1 | 1 | Yes | Yes | Yes | `[S0959]` Land Purchase SC Farmers |
| `GU_08` | ગુજરાત સરદાર પટેલ આવાસ યોજના ગ્રામીણ મકાન સહાય | `S2677` | Sardar Patel Awas Yojana | 1 | 1 | 1 | Yes | Yes | Yes | `[S2677]` Sardar Patel Awas Yojana |
| `GU_09` | બાંધકામ શ્રમિકો માટે શ્રમિક અન્નપૂર્ણા યોજના 5 રૂપિયામાં ભોજન | `S2931` | Shramik Annapurna Yojana (GBOCWWB) | 1 | 1 | 1 | Yes | Yes | Yes | `[S2931]` Shramik Annapurna Yojana (GBOCWWB) |
| `GU_10` | ગુજરાત દિવ્યાંગ વિદ્યાર્થીઓ માટે સાધન સહાય યોજના | `S2510` | Prosthetic Aid & Appliances for PwD | None | None | None | No | No | No | `[S2675]` Saraswati Vidya Yojana |
| `GU_11` | સરસ્વતી સાધના યોજના ગુજરાત વિદ્યાર્થિનીઓને મફત સાઇકલ | `S1111` | Sarasvati Sadhna Yojana (Free Cycles) | 7 | 7 | 7 | No | Yes | Yes | `[S3316]` Walking Device - Tamil Nadu |
| `GU_12` | ગુજરાત પશુપાલકો માટે 1 થી 20 દુધાળા પશુ યુનિટ વ્યાજ સહાય | `S2708` | Subsidy on Interest for 1 to 20 Milch Animal | 1 | 1 | 1 | Yes | Yes | Yes | `[S0895]` Establishment of 1 to 20 Milch Animal |
| `GU_13` | વહાલી દીકરી યોજના પ્રથમ હપ્તો અને પાત્રતા શરતો | `S3268` | Vahli Dikri Yojana | None | None | None | No | No | No | `[S1642]` Kanyashree Prakalpa |
| `GU_14` | ગુજરાત વનબંધુ કલ્યાણ યોજના આદિજાતિ સર્વાંગી વિકાસ | `S3270` | Vanbandhu Kalyan Yojana | 6 | 6 | 6 | No | Yes | Yes | `[S2074]` Mukhyamantri Vriksharopan Protsahan |

### Observations on Gujarati Retrieval:
- **5 Queries are Rank #1 Hits:** `GU_06` (Sanedo), `GU_07` (SC Land), `GU_08` (Sardar Patel Awas), `GU_09` (Shramik Annapurna), `GU_12` (Milch Animal Farm).
- **5 Queries are Ranked in Top 2–10:** `GU_04` (Rank #3), `GU_05` (Rank #4), `GU_14` (Rank #6), `GU_11` (Rank #7), `GU_02` (Rank #10).
- **1 Query is in Candidate Pool (Rank #26):** `GU_01` (*Mukhyamantri Amrutum*).
- **3 Queries Missed the Top-40 Candidate Pool:** `GU_03` (*Manav Garima*), `GU_10` (*Divyang Student Appliances*), and `GU_13` (*Vahali Dikri*). In these three cases, competition from out-of-state schemes sharing identical semantic concepts pushed the Gujarat-specific scheme past rank 40.

---

## 9. Quality Comparison: Step 17 Baseline vs Step 20 Final

| Language / Metric | Step 17 Metric | Step 20 Metric | Delta / Status |
| :--- | :---: | :---: | :---: |
| **English Benchmark ($N=60$)** | | | |
| Vector Recall@5 | 0.9833 | 0.9833 | **+0.0000 (Identical)** |
| Vector Recall@10 | 1.0000 | 1.0000 | **+0.0000 (Identical)** |
| Hybrid Recall@5 | 0.9500 | 0.9500 | **+0.0000 (Identical)** |
| Hybrid Recall@10 | 0.9667 | 0.9667 | **+0.0000 (Identical)** |
| Scheme Reranked Recall@5 | 0.9333 | 0.9333 | **+0.0000 (Identical)** |
| Scheme Reranked Recall@10 | 0.9667 | 0.9667 | **+0.0000 (Identical)** |
| **Hindi Benchmark ($N=12$)** | | | |
| Vector Recall@5 | 0.5000 | 0.7500 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Vector Recall@10 | 0.5833 | 0.8333 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Hybrid Recall@5 | 0.5000 | 0.7500 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Hybrid Recall@10 | 0.5833 | 0.8333 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Scheme Reranked Recall@5 | 0.5000 | 0.7500 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Scheme Reranked Recall@10 | 0.5833 | 0.8333 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| **Gujarati Benchmark ($N=14$)** | | | |
| Vector Recall@5 | 0.2143 | 0.5000 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Vector Recall@10 | 0.2143 | 0.7143 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Hybrid Recall@5 | 0.2143 | 0.5000 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Hybrid Recall@10 | 0.2143 | 0.7143 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Scheme Reranked Recall@5 | 0.2143 | 0.5000 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| Scheme Reranked Recall@10 | 0.2143 | 0.7143 | **NOT DIRECTLY COMPARABLE** (Corrected ground truth) |
| **Overall Dataset ($N=86$)** | | | |
| Scheme Reranked Recall@5 | 0.7558 | 0.8372 | **NOT DIRECTLY COMPARABLE** (Multilingual labels corrected) |
| Scheme Reranked Recall@10 | 0.7907 | 0.9070 | **NOT DIRECTLY COMPARABLE** (Multilingual labels corrected) |

---

## 10. Latency Comparison: Step 18 Baseline vs Step 20 Final

Measured across the complete production retrieval path:

| Retrieval Stage | Step 18 Baseline (Avg) | Step 20 Final (Avg) | Step 20 Median | Step 20 p95 | Step 20 Min | Measured Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Query Embedding** | 20.62 ms | 24.97 ms | 22.75 ms | 30.12 ms | 16.49 ms | -21.1% (CPU inference) |
| **Vector SQL (HNSW)** | 134.89 ms | 84.45 ms | 84.70 ms | 94.97 ms | 71.47 ms | **+ 37.4% reduction** |
| **Vector Total Leg** | 155.51 ms | 109.42 ms | 107.23 ms | 122.91 ms | 92.59 ms | **+ 29.6% reduction** |
| **Keyword SQL (GIN)** | 1,795.14 ms | 42.32 ms | 2.56 ms | 212.49 ms | 1.22 ms | **+ 97.6% reduction (99.9% median)** |
| **Hybrid Leg (Parallel)** | 1,925.14 ms | 108.59 ms | 83.42 ms | 226.82 ms | 71.04 ms | **+ 94.4% reduction** |
| **Scheme Reranking** | 0.20 ms | 0.24 ms | 0.24 ms | 0.33 ms | 0.15 ms | Negligible |
| **Full Prod Retrieval E2E** | **2,524.76 ms** | **125.40 ms** | **105.68 ms** | **243.07 ms** | **93.14 ms** | **+ 95.03% latency reduction (20.1x speedup)** |

---

## 11. Query Plan & Performance Index Verification

Execution plans inspected directly in PostgreSQL using `EXPLAIN (ANALYZE, BUFFERS)`:

### A. GIN Full-Text Search Plan
```text
Bitmap Heap Scan on scheme_chunks f (cost=30.03..34.05 rows=1 width=28) (actual time=1.956..11.191 rows=1340 loops=1)
  Recheck Cond: (search_vector @@ '''scholarship'' & ''student'''::tsquery)
  -> Bitmap Index Scan on idx_scheme_chunks_search_vector (cost=0.00..30.03 rows=1 width=0) (actual time=1.056..1.056 rows=1340 loops=1)
Planning Time: 6.550 ms
Execution Time: 11.611 ms
```
*Verification:* The query planner chooses `idx_scheme_chunks_search_vector` with a sub-2ms bitmap index scan.

### B. HNSW Vector Cosine Plan
```text
Index Scan using idx_scheme_chunks_embedding_local_hnsw on scheme_chunks c (cost=590.22..36632.43 rows=20497 width=40) (actual time=7.821..8.114 rows=40 loops=1)
  Order By: (embedding_local <=> '[...]'::vector)
  Filter: (embedding_local IS NOT NULL)
Planning Time: 0.985 ms
Execution Time: 8.368 ms
```
*Verification:* The query planner chooses `idx_scheme_chunks_embedding_local_hnsw` directly, completing the cosine ranking in ~8 ms.

---

## 12. Regression Tests Status

| Test Suite / Script | Functional Focus | Checks | Result |
| :--- | :--- | :---: | :---: |
| `scripts/10_test_keyword_retrieval.py` | GIN FTS search, ranking, filtering | 14 / 14 | **PASS** |
| `scripts/11_test_vector_retrieval.py` | Local vector cosine similarity, HNSW, dimensions | 26 / 26 | **PASS** |
| `scripts/12_test_hybrid_retrieval.py` | Concurrent RRF fusion, score math, deduplication | 17 / 17 | **PASS** |
| `scripts/15_test_retrieval_service.py` | End-to-end service, pre-filtering, response formats | 47 / 47 | **PASS** |
| `scripts/18_test_fastapi_backend.py` | FastAPI `/health`, `/api/retrieve`, `/api/generate` | 28 / 28 | **PASS** |
| `scripts/19_test_chat_api.py` | Multi-turn chat, session store, query rewriting | 25 / 25 | **PASS** |
| **Total Automated Regression Tests** | | **157 / 157** | **100% PASS** |

---

## 13. Database Integrity Audit

| Metric | Pre-Step 20 | Post-Step 20 | Parity Status |
| :--- | :---: | :---: | :---: |
| `schemes` count | 3,397 | 3,397 | **Exact Match (+0)** |
| `scheme_chunks` count | 20,497 | 20,497 | **Exact Match (+0)** |
| `scheme_chunks` distinct chunk IDs | 20,497 | 20,497 | **Zero Duplicates** |
| Gemini embeddings populated (768d) | 1,260 | 1,260 | **Untouched (+0)** |
| Gemini embeddings NULL | 19,237 | 19,237 | **Untouched (+0)** |
| Local embeddings populated (384d) | 20,497 | 20,497 | **Untouched (+0)** |
| Local embeddings NULL | 0 | 0 | **Zero NULL (+0)** |
| Generated column `search_vector` | 20,497 | 20,497 | **100% Populated** |
| GIN index `idx_scheme_chunks_search_vector` | Present | Present | **Active** |
| HNSW index `idx_scheme_chunks_embedding_local_hnsw` | Present | Present | **Active** |
| Row mutations / deletions / insertions | 0 | 0 | **Zero Modifications** |

---

## 14. Evaluation Limitations

1. **Multilingual Sample Size:** While the English benchmark contains 60 queries across central and state domains, the Hindi ($N=12$) and Gujarati ($N=14$) test sets are modest in size. Aggregate percentages should be interpreted in the context of these sample sizes.
2. **Corpus Language Limitation:** Because the scheme text corpus is 100% English, Indic script queries must rely entirely on cross-lingual representation alignment in `intfloat/multilingual-e5-small`. In queries where multiple states offer semantically similar programs (e.g. self-employment or disability aids), state filtering is necessary to prevent out-of-state schemes from outranking the intended local scheme.
3. **Keyword FTS Limitation on Non-English Scripts:** PostgreSQL full-text search with an English dictionary does not stem or index Indic script tokens. Hybrid retrieval gracefully handles this by allowing the dense vector leg to supply the candidates.

---

## 15. Final Retrieval Status

- **Quality:**
  - English Retrieval: **96.67% Recall@10** (58/60)
  - Hindi Retrieval: **83.33% Recall@10** (10/12)
  - Gujarati Retrieval: **71.43% Recall@10** (10/14)
  - Overall Benchmark: **90.70% Recall@10** (78/86)
- **Performance:**
  - Production retrieval latency: **~125 ms average** (down from ~2,525 ms).
- **Stability:**
  - 157 / 157 regression checks passing.
  - Zero database integrity violations.

---

## 16. Recommended Next Step

With database performance optimization, index verification, and full empirical evaluation complete, the retrieval tier is verified and robust.

The recommended next step is:
**Step 21 — Conversational Quality & Citation Grounding Evaluation**, evaluating generation fidelity, LLM citation accuracy, and end-to-end response generation latency on the integrated Next.js frontend.
