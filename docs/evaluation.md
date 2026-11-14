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
| v0.2, hybrid retrieval | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 4.8 s | 56.3K / 6.9K |
| **v0.3, direct API, run 1** | **20** | **1.00** | **0.94** | **1.00** | **1.00** | **1.00** | **1.00** | **6.9 s** | **56.4K / 6.8K** |
| **v0.3, direct API, run 2** | **20** | **1.00** | **1.00** | **0.94** | **1.00** | **1.00** | **1.00** | **7.1 s** | **56.4K / 7.1K** |

The v0.1 and v0.2 rows went through a local proxy that compresses requests (see "Measurement notes"); the
v0.3 rows call the Anthropic API directly and are the reference numbers. Between the two direct runs, one
fact was scored as missing because of formatting (q03: the scorer compared "32 C" literally; it now ignores
unit formatting, after checking the model's answer was correct), and in the other run the critic judged one
correct, cited answer (q17) as not fully supported. LLM-judged metrics move by a case or two between runs,
so both runs are reported rather than the better one.

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


## Visual retrieval (v0.3)

`evals/make_visual_pages.py` draws 10 fictional pages: four near-identical structural drawings (Levels
13-16, each with a title block, revision and one enlarged column), a bar bending schedule, three
near-identical pour checklists (ticked boxes, slump, inspector), a site logistics plan and a progress
chart. `evals/visual_queries.jsonl` has 12 questions whose answers exist only in those images.

**Retrieval only** (`evals/compare_visual.py`, RTX 4070 Laptop 8 GB, i7-12700H):

| Configuration | hit@1 | hit@3 | MRR | index time per page | query time |
|---|---|---|---|---|---|
| CLIP ViT-B/32, single vector (CPU) | 0.33 | 0.58 | 0.44 | 0.03 s | 35 ms |
| **ColSmol-256M (CPU)** | **0.83** | **1.00** | **0.89** | 8.1 s | 105 ms |
| **ColSmol-256M (GPU)** | **0.83** | **1.00** | **0.89** | 0.38 s | 113 ms |
| ColQwen2.5-v0.2 (GPU, 4 pages per batch) | 0.75 | 1.00 | 0.88 | 16.6 s | 105 ms |
| ColQwen2.5-v0.2 (GPU, 1 page per batch) | 0.92 | 1.00 | 0.94 | 11.3 s | 132 ms |

- Late-interaction models find the right page in the top 3 every time; single-vector CLIP mostly returns
  the site plan or the chart, because one vector cannot hold the text on a drawing.
- **CPU users lose nothing in accuracy** with ColSmol, only indexing speed (8 s per page, once per page).
- **ColQwen2.5 does not fit an 8 GB card well.** VRAM peaked at 7.9 of 8 GB even one page at a time;
  the driver spills into shared memory and indexing runs 30x slower than ColSmol on the same GPU. The
  `auto` setting therefore picks ColQwen2.5 only on GPUs with at least 12 GB.
- Accuracy changed with batch size (0.75 vs 0.92). Batched inference can change outputs slightly; the
  cause was not isolated, so it is reported as observed.
- Hardest case for every model: "which checklist shows cube samples were **not** taken?" (picking out a
  missing tick among look-alike forms).

**End to end with Claude** (`evals/run_evals.py --visual`, ColSmol-256M on GPU, direct API):

| routed to pages | right page retrieved | expected facts | grounded | valid citations | avg latency | cost |
|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9.8 s | $0.92 |

The writer and critic receive the retrieved page images, so the answer is read from the drawing,
schedule or form itself (for example "32 mm" from the schedule table, "QA-01" from a checklist).

## Curated knowledge (v0.4, OKF)

`evals/okf_dataset.jsonl` has 8 questions written against the OKF bundle in `data/okf/`: an approved
decision that changes a specification limit (k01, k02), a procedure that exists only in the bundle (k03),
a value whose superseded revision is in the bundle as `deprecated` (k04, which also fails if the
superseded 35 C appears in the answer), an unverified machine-drafted lesson that the answer must flag
as unverified (k05), a fact reached through a link (k06), a provenance question (k07), and one question
answerable from the raw specification alone (k08). The questions were written for this bundle, so the
raw-only numbers show what the bundle adds, not how a general system would score.

| Sources | retrieval hit | expected facts | grounded | valid citations | no superseded values | avg latency | cost |
|---|---|---|---|---|---|---|---|
| Raw documents only | 0.25 | 0.56 | 1.00 | 1.00 | 1.00 | 12.5 s | $0.27 |
| Documents + OKF bundle | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | 10.2 s | $0.23 |

- **Grounded is not correct.** Raw-only answers were all judged grounded: for k01 the system correctly
  quoted the 90-minute rule and rejected a load that RFI-021 allows. The critic can only check answers
  against what was retrieved; knowledge the documents never captured is invisible to it.
- **First run, before the provenance fix:** documents + OKF scored facts 0.88, grounded 0.75. The model
  could not answer who confirmed RFI-021 because the row carried only a trust label, and the critic
  rejected answers that relied on it. Adding a provenance line to each concept and telling the critic
  it is reliable metadata gave the row above. One of the two critic rejections in that run was
  unrelated to OKF (the writer added an unsupported cube-count example to k08); it did not recur.
- **No regression:** the original 20-case set with the bundle indexed scored 1.00 on every metric
  (7.7 s, $0.46). The offline mock run is unchanged with and without the bundle (same three
  extractive misses).
- Retrieval only (mock model, local embeddings): raw documents 0.25, bundle only 0.88 (it cannot answer
  the cube-sampling question, which lives only in the specification), both 1.00.

## Measurement notes

- **Proxy.** This machine routes Anthropic traffic through a local proxy that compresses requests and
  replays responses to identical requests. A repeated visual run finished in 0.16 s per question with
  token counts identical to the first run, which exposed it. All v0.3 numbers bypass the proxy.
- **Token accounting.** Through the proxy, most input tokens were reported as cache writes, and the harness
  only counted `input_tokens`, under-reporting cost. It now counts uncached, cache-write and cache-read
  tokens and prices them separately (`est_cost_usd`).
