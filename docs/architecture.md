# Architecture and design decisions

## Request lifecycle

1. **guard_input** masks PII in the question, then checks it for prompt-injection patterns. A flagged
   request ends the run before any model is called, so a hostile prompt costs nothing and reaches nothing.
2. **planner** returns a `Plan` (route + search query). Routing is a model decision, but the graph edges
   are fixed in code, so the planner can choose a path, not invent one.
3. **researcher** runs a cosine search over LanceDB and strips injected sentences from each chunk.
4. **writer** produces an `Answer` or an `RFIDraft` from the retrieved chunks only, citing `chunk_id`s.
5. **critic** returns a `Critique`. If the draft is not grounded and the critic suggests a better query,
   the graph loops back to the researcher, up to `SITEBRAIN_MAX_ATTEMPTS` research passes.
6. **human_review** (RFIs only) calls `interrupt()`. The run stops with its state checkpointed and resumes
   when the caller sends `Command(resume={"approved": ..., "note": ...})`.
7. **finalize** and **guard_output** assemble the result and mask any PII that came from the documents.

## Decisions and trade-offs

**Fixed graph, model-chosen route.** A free-form agent loop could decide its own next step, but every
path here is reviewable in `graph.py`, which matters when the output is a formal RFI to a consultant.
The cost is flexibility: a new capability means a new node and edge, not a new prompt.

**Bounded critic loop.** An unbounded generate/critique loop can spin on a question the documents cannot
answer. After the last attempt the answer is returned with a visible caution instead of looping again.

**Strip injected sentences instead of dropping the chunk.** The first version discarded any chunk that
contained an injection. The evaluation showed the cost: the poisoned supplier note was also the only
source of the delivery window, so the system could no longer answer that question. Removing only the
offending sentences keeps the legitimate facts and still keeps the instruction away from every model.

**Regex guardrails, deliberately simple.** Pattern matching is fast, free and testable, and it catches the
common phrasings. It will not catch paraphrased or encoded attacks. Production would add a classifier
(or an LLM judge) behind the regex and keep the regex as the cheap first layer.

**PII masked on both sides.** Input masking keeps personal data out of prompts; output masking covers
personal data that lives in the documents themselves (the site engineer's contact details in the
12 Oct report).

**Structured outputs everywhere.** Each agent returns a Pydantic model enforced by the API's
structured-output support, so the graph never parses free text to find a route or a verdict.

**Local embeddings.** `BAAI/bge-small-en-v1.5` via fastembed runs on CPU, costs nothing per query and keeps
document text inside the machine during indexing. LanceDB stores vectors and metadata in one embedded
table, with no server to run.

**Mock provider.** `MockLLM` is extractive and deterministic. It lets the tests, the CI pipeline and the
retrieval/guardrail metrics run without an API key. It says nothing about answer quality; that is what
the Claude evaluation run is for.

## Known limitations

- `InMemorySaver` loses paused RFIs when the process exits. A deployment would use a persistent
  checkpointer (SQLite or Postgres) so approvals can happen hours later.
- Chunking is paragraph-based. Long tables and drawings (the original use case) need layout-aware parsing.
- The evaluation set is 14 cases over 6 short documents: enough to catch regressions, not to claim accuracy.
- The critic is the same model family as the writer, so it can share the writer's blind spots.
