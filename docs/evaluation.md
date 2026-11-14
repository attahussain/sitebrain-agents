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

Offline run (`SITEBRAIN_PROVIDER=mock`, real `bge-small` embeddings):

| retrieval_hit | keyword_recall | grounded | citation_validity | injection_blocked | indirect_quarantined |
|---|---|---|---|---|---|
| 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 |

The single miss is the safety summary (q10): the extractive mock picked three sentences that did not
mention the scaffold incident. This run validates retrieval, routing and guardrails; it does not measure
generation quality.

## History

- Indirect-injection handling originally dropped the whole poisoned chunk. That passed the security check
  but failed q14, because the same chunk held the only copy of the delivery window. Switching to
  sentence-level stripping fixed q14 without weakening the defence (both are covered by tests).
