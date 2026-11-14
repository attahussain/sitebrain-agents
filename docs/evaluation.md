# Evaluation

`evals/run_evals.py` re-indexes `data/docs`, runs every case in `evals/dataset.jsonl` through the full graph
(RFIs are auto-approved so the final text can be scored) and writes per-case rows plus a summary to
`evals/last_results.json`.

## Metrics

| Metric | Definition |
|---|---|
| `retrieval_hit` | Share of cases where at least one expected document was retrieved |
| `keyword_recall` | Share of expected facts (strings) present in the final answer |
| `grounded_rate` | Share of answers the critic judged fully supported by the sources |
| `citation_validity` | Share of cited `chunk_id`s that were actually retrieved (detects invented sources) |
| `injection_blocked` | Share of direct injection attempts stopped by `guard_input` |
| `indirect_quarantined` | Share of poisoned-document cases where the injected text was stripped |
| `avg_latency_s`, `llm_calls`, tokens | Cost and speed |

## Dataset

14 cases: 9 factual questions, 1 summary, 1 RFI, 2 direct injections and 1 question whose only source is a
document with a hidden injection. Each case lists the documents that should be retrieved and the facts the
answer should contain.

## Results

Claude Opus 5.5 (`effort=medium`), `bge-small` embeddings:

| Run | cases | retrieval_hit | keyword_recall | grounded | citation_validity | injection_blocked | indirect_quarantined | avg latency | tokens in / out |
|---|---|---|---|---|---|---|---|---|---|
| v0.1, first run | 14 | 0.91 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 8.5 s | 48.8K / 6.6K |
| v0.1, after fixes | 14 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 5.2 s | 36.7K / 4.9K |
| **v0.2, hybrid retrieval** | **20** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **4.8 s** | **56.3K / 6.9K** |

v0.2 adds six questions built around exact identifiers (column IDs, crane IDs, the specification section
number, counts). A full v0.2 run makes 56 model calls and costs roughly USD 0.36 at list prices.

## Retrieval benchmark

`evals/compare_retrieval.py` measures retrieval alone, with no LLM calls.

**Demo corpus** (6 documents, 17 questions with expected documents): vector, BM25 and hybrid all reach
hit@1 0.94, hit@4 1.00. With so few documents the right one ranks near the top regardless of method,
so this corpus cannot separate them.

**Scale corpus** (`evals/make_scale_corpus.py`): 60 synthetic daily reports from one template, each
holding one unique fact, and 81 questions whose answer lives in exactly one report. It imitates the
repetition of a real project's report archive; it is generated deterministically and not committed.

| Retrieval | hit@1 | hit@4 | MRR |
|---|---|---|---|
| vector | 0.27 | 0.31 | 0.29 |
| BM25 keyword | 1.00 | 1.00 | 1.00 |
| hybrid, equal-weight RRF | 0.41 | 0.89 | 0.58 |
| hybrid, RRF with vector weight 0.25 | 0.56 | 0.98 | 0.72 |
| **hybrid, lexical-first for identifiers (shipped)** | **1.00** | **1.00** | **1.00** |

The first version of the generator reused each crane-and-fault pair every 21 days, so "on which day was
TC-2 stopped because of a hydraulic hose leak?" had three correct answers while the dataset accepted
one. That showed up as BM25 placing the expected report at rank 3 on every such question; the
generator now makes each pair unique. The design reasoning for the shipped rule is in
[architecture.md](architecture.md).

Offline run (`SITEBRAIN_PROVIDER=mock`): retrieval_hit 1.00, keyword_recall 0.92, all guardrail metrics 1.00.
The mock is extractive and misses the safety summary (q10); it is there to validate retrieval, routing and
guardrails in CI, not generation quality.

## What the evaluation changed

- **Indirect injection.** Handling originally dropped the whole poisoned chunk. That passed the security
  check but failed q14, because the same chunk held the only copy of the delivery window. Sentence-level
  stripping fixed q14 without weakening the defence.
- **Chunking (q09).** The first Claude run missed "How often must scaffolds be re-inspected?". The HSE
  policy had been indexed as a single chunk that opened with heat-stress rules, so its embedding did not
  match a scaffold query and the scaffold section was never retrieved. Chunking per `##` section, with a
  "document - section" prefix on each chunk, fixed retrieval. Smaller, focused chunks also cut input
  tokens by about 25% and average latency from 8.5 s to 5.2 s.
- **Critic blind spot (q09).** In the same failure the writer correctly said the sources did not contain
  the answer, and the critic passed it as grounded, so no second search ran. The critic now reports
  `answered` separately from `grounded`, and an unanswered question triggers a retry with a reworded query.
