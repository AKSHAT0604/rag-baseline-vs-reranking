# RAG: Baseline Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

LMA assignment: two retrieval pipelines over the same chunked PDF, compared on failure-mode queries,
answer quality, latency and scaling.

Everything is in one notebook: **[`RAG_Baseline_vs_ReRanking.ipynb`](RAG_Baseline_vs_ReRanking.ipynb)**
(committed with its outputs).

## Setup

| Component | Choice |
|---|---|
| Document | *FIDE Laws of Chess* (in force from 1 Jan 2023), 45-page PDF in [`data/`](data/). Printed from the official FIDE Handbook page <https://handbook.fide.com/chapter/E012023> |
| Chunking | Clause-aware: split on article numbers (e.g. `9.6.2`, `A.5.2`), pack clauses of the same Article/Appendix into ≤ ~700-char chunks, each prefixed with its section title → 141 chunks |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2`, normalised embeddings, `faiss.IndexFlatIP` (cosine) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `Qwen/Qwen2.5-3B-Instruct`, run locally with Hugging Face `transformers`, greedy decoding, no API key needed |

**Pipeline 1:** bi-encoder top-10 → top-3 → LLM → Answer 1
**Pipeline 2:** same top-10 → cross-encoder scores 10 (query, chunk) pairs → re-sort → top-3 → same LLM and prompt → Answer 2

## Running

```bash
pip install -r requirements.txt
jupyter notebook RAG_Baseline_vs_ReRanking.ipynb   # Run All
```

The models download from Hugging Face on first run (~6 GB, mostly the LLM). The notebook uses CUDA if available,
otherwise CPU; on CPU a full run takes about 15–20 minutes, almost all of it LLM generation.

## Headline results (CPU run, see notebook for full tables)

| | Pipeline 1 (bi-encoder) | Pipeline 2 (re-ranked) |
|---|---|---|
| Hit@1 (61 labelled queries) | 0.525 | 0.639 |
| Hit@3 | 0.836 | 0.852 |
| MRR@10 | 0.688 | 0.771 |
| Retrieval latency (mean) | 7.6 ms | 104.2 ms |

Stress-test rank shifts (gold chunk, bi-encoder rank → cross-encoder rank):
two-hands castling #4→#1, scoresheet when short of time #7→#1, rapid unnoticed illegal move #4→#1,
rapid incorrect draw-claim penalty #6→#2, automatic 75-move draw #4→#1, time before pressing clock #4→#1.
Counter-examples where re-ranking hurt: bare king on time #3→#5, arbiter announcing move count #2→#7.

## Notebook contents

1. Setup & config
2. PDF loading + clause-aware chunking
3. Pipeline 1: bi-encoder + FAISS
4. Pipeline 2: cross-encoder re-ranking
5. Shared LLM prompt
6. **Failure-mode analysis**: keyword trap, synonyms, exceptions, negation, buried chunks, plus 2 counter-examples where re-ranking hurts. Each case has a top-10 rank table and Answer 1 vs Answer 2
7. Aggregate evaluation on 61 labelled questions (Hit@1, Hit@3, MRR@10, Recall@10)
8. Latency profiling (per stage, mean / median / p95)
9. Scaling experiment: cross-encoding the full corpus vs two-stage, and re-rank depth vs accuracy
10. Results saved to [`results/`](results/) (CSV/JSON)

## Repository layout

```
├── RAG_Baseline_vs_ReRanking.ipynb   # all code + outputs
├── data/FIDE_Laws_of_Chess_2023.pdf  # the source document
├── results/                          # tables exported by the notebook
├── requirements.txt
└── README.md
```
