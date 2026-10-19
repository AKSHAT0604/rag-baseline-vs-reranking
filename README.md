# RAG: Baseline Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

LMA assignment: two retrieval pipelines over the same chunked PDF, compared on failure-mode queries,
answer quality, latency and scaling.

Everything is in one notebook: **[`RAG_Baseline_vs_ReRanking.ipynb`](RAG_Baseline_vs_ReRanking.ipynb)**
(committed with its outputs).

## Setup

| Component | Choice |
|---|---|
| Document | *IIIT-Delhi Regulations for B.Tech. Programs*, Version 13 (May 2024), 21-page PDF in [`data/`](data/). Source: <https://iiitd.ac.in/sites/default/files/docs/education/2024/2024-May-UG%20Regulations.pdf> |
| Chunking | Structure-aware: track numbered sections (`5.8 Academic Load`), start paragraphs at list markers (`(1)`, `a)`, `•`), pack paragraphs of the same section into ≤ ~600-char chunks, each prefixed with `[Section x.y: title]` → 98 chunks |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2`, normalised embeddings, `faiss.IndexFlatIP` (cosine) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLMs (local Qwen) | `Qwen2.5-1.5B-Instruct` via Hugging Face `transformers` (CPU) and `Qwen3.8-27B` (GGUF Q3_K_XL) via `llama.cpp`'s `llama-server` (GPU). Greedy decoding, no API keys |

**Pipeline 1:** bi-encoder top-10 → top-3 → LLM → Answer 1
**Pipeline 2:** same top-10 → cross-encoder scores 10 (query, chunk) pairs → re-sort → top-3 → same LLM and prompt → Answer 2

### Why this document

A popular rulebook (the earlier version of this repo used the FIDE Laws of Chess) is part of every LLM's training
data, so the model can answer correctly even when retrieval fails, which hides the difference between Answer 1 and
Answer 2. An institute's own regulations are niche. Asked **without context**, both Qwen models invent the rules:

| Question (closed-book) | Regulations say | Qwen2.5-1.5B | Qwen3.8-27B |
|---|---|---|---|
| Credits to graduate | 156 | ~300 | 160 |
| Grade points for an F | 2 | 0.5 | 0 |
| CGPA for a 2nd-year overload | > 8.0 | 7.5 | 8.00 |
| Semester leaves allowed | at most four | "two per year" | two |
| SGPA for academic warning | ≤ 4.5 | 3.0 | < 6.0 |

IIIT-Hyderabad's own regulations are on intranet.iiit.ac.in, which is only reachable from the campus network, so the
public IIIT-Delhi B.Tech. regulations are used. The code is document-agnostic: change `PDF_PATH` and re-label the queries.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU torch is enough: --extra-index-url https://download.pytorch.org/whl/cpu
# large Qwen model (optional; skipped automatically if the server is not running)
llama-server -m Qwen3.8-27B-UD-Q3_K_XL.gguf --alias qwen3.8-27b --port 8081 -ngl 99 -c 32768 --jinja
jupyter nbconvert --to notebook --execute --inplace RAG_Baseline_vs_ReRanking.ipynb   # or open it and Run All
```

The encoders and the 1.5B model download from Hugging Face on first run (~3.5 GB). Set `LLAMA_SERVER` to point at a
different server URL. A full run takes about 7 minutes (RTX 5070 Ti for the 27B model, everything else on CPU).

## Headline results (see notebook for full tables)

Retrieval on 151 labelled questions (gold = verbatim phrase of the answering clause):

| | Pipeline 1 (bi-encoder) | Pipeline 2 (re-ranked) |
|---|---|---|
| Hit@1 | 0.656 | 0.854 |
| Hit@3 | 0.854 | 0.907 |
| MRR@10 | 0.768 | 0.890 |
| Recall@10 (ceiling for re-ranking) | 0.974 | 0.974 |
| Retrieval latency (mean, CPU) | 4.8 ms | 49.6 ms |

By query style (Hit@3, P1 → P2): rule-phrased 0.950 → 0.988, student-phrased 0.720 → 0.760,
exception questions 0.810 → 0.952. Re-ranking rescued 11 queries into the top 3 and pushed 3 out.

Stress-test rank shifts (gold chunk, bi-encoder rank → cross-encoder rank):

| Category | Query | Shift |
|---|---|---|
| Keyword trap | My CGPA is 7.5. Can I take an extra course in my 5th semester? | #9 → #1 |
| Keyword trap | After finishing 156 credits, for how many extra semesters can I stay to improve my CGPA? | #7 → #1 |
| Synonyms | Can I take a gap semester to work on my startup? | #4 → #2 |
| Synonyms | Can I spend a year studying at a foreign university? | #6 → #2 |
| Exception | Does the six-year maximum duration include semester leaves? | #6 → #1 |
| Exception | Does an online course count as an in-class course during my internship semester? | #4 → #1 |
| Negation | How many semester leaves are allowed in the whole program? | #4 → #1 |
| Negation | Is a student expelled from another institute on disciplinary grounds eligible for admission? | #9 → #1 |
| Buried chunk | When is a student's program terminated for poor academic performance? | #6 → #3 |
| Counter-example | Does a failing grade stay on my grade sheet after I clear the course? | #2 → #4 |
| Counter-example | How long can I stay enrolled before I run out of time to finish my degree? | #3 → #4 |
| Limitation | How many credits do I need in total to get my degree? | #6 → #7 |

Answer quality on the 12 stress-test questions: each answer is graded against a reference answer written from the
regulation text (graded by the local 27B model, verdicts checked by hand):

| Correct answers (out of 12) | Qwen2.5-1.5B | Qwen3.8-27B |
|---|---|---|
| Closed-book (no context) | 0 | 1 |
| Answer 1 (baseline top-3) | 2 | 3 |
| Answer 2 (re-ranked top-3) | 9 | 9 |

When the gold clause was missing from the baseline top 3, the 27B model mostly said "the provided context does not
answer this", while the 1.5B model usually answered anyway from the wrong clause (e.g. citing the extra-credit rule
of Sec 6.9 for an overload question). In the two counter-examples, re-ranking turned a correct 27B answer into a wrong one.

Latency per answer: re-ranking adds ~45 ms, versus ~3.9 s (1.5B, CPU) or ~1.4 s (27B, GPU) of generation.
Cross-encoding all 98 chunks costs ~0.5 s per query; extrapolated linearly that is ~50 s at 10k chunks and
~1.4 h at 1M chunks, while the two-stage pipeline stays at ~0.05 s.

## Notebook contents

1. Setup & config
2. PDF loading + structure-aware chunking
3. Pipeline 1: bi-encoder + FAISS
4. Pipeline 2: cross-encoder re-ranking
5. Two local Qwen LLMs, shared prompt, closed-book check
6. **Failure-mode analysis**: keyword traps, synonyms, exceptions, negation, buried chunks, 2 counter-examples and
   1 limitation case. Each has a top-10 rank table and Answer 1 vs Answer 2 from both models, graded against a
   reference answer
7. Aggregate evaluation on 151 labelled questions (Hit@1, Hit@3, MRR@10, Recall@10), split by query style
8. Latency profiling (per stage, mean / median / p95; LLM time per model; end-to-end)
9. Scaling experiment: cross-encoding the full corpus vs two-stage, and re-rank depth vs accuracy
10. Results saved to [`results/`](results/) (CSV/JSON)

## Repository layout

```
├── RAG_Baseline_vs_ReRanking.ipynb          # all code + outputs
├── data/IIITD_BTech_Regulations_2024.pdf    # the source document
├── results/                                 # tables exported by the notebook
├── requirements.txt
└── README.md
```
