# stock-desk

stock-desk answers questions about US public companies from their SEC filings, and measures how
well it does against the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark.

What exists today:

- **RAG over 5 filings.** The 10-Ks listed in `data_manifest.txt` are split into 1000-character
  chunks, embedded with OpenAI `text-embedding-3-large`, and stored in PostgreSQL with pgvector.
  A question is answered by vector search over one filing (top 10 chunks, attached exhibits
  excluded), the full text of the first 5 distinct pages, and an LLM prompted to answer only
  from those pages: list the figures it uses with their pages, show any arithmetic, and end with
  an "Answer:" line, or reply "Not found in the filing." This is the best config so far
  (experiment 012) and the default; see [Results](#results).
- **FinanceBench eval.** The 30 FinanceBench questions about those filings are answered by the
  same code the CLI runs, scored for retrieval (did it find the gold evidence page and text?),
  and graded by an LLM judge against the gold answer. See [Results](#results).

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), Docker and an OpenAI API key.

```bash
git clone <this repo> stock-desk && cd stock-desk
cp .env.example .env   # set OPENAI_API_KEY; LLM_MODEL defaults to gpt-5.4-mini
make install           # uv sync + pre-commit hooks (ruff check, ruff format)
make db-up             # Postgres + pgvector in Docker, waits until healthy
make fetch             # FinanceBench questions + the 5 PDFs in data_manifest.txt, into data/
make ingest            # chunk, embed and store the PDFs in naive_1000_large (OpenAI embeddings API)
make test
```

Then ask a question about one filing:

```bash
make ask DOC=AMD_2022_10K Q="What was total revenue in 2022?"
```

Expected output (the wording can vary between runs, even at temperature 0):

```text
Figures:
- Total net revenue / total sales to external customers in 2022: $23,601 million (pages 47, 65)
- Consolidated net revenue summary for 2022: $23,601 million (page 47)

Answer: Total revenue in 2022 was $23,601 million.

Retrieved pages (AMD_2022_10K): 47, 66, 49, 48, 67
```

`DOC` is any `doc_name` in `data_manifest.txt`. Pages are zero-based PDF page indexes, the same
numbering FinanceBench uses for its evidence pages.

Add `ANSWER_MODE=agent` to answer with the LangGraph agent instead (`src/stockdesk/agent/`). It
calls `search_filings` (the retrieval pipeline above, on `DOC` only) and `calculate` (safe
arithmetic) up to `MAX_AGENT_STEPS=6` times, and the output also lists its tool calls.

## Data

Questions, document metadata and filing PDFs come from
[FinanceBench](https://github.com/patronus-ai/financebench) by Patronus AI, licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (noncommercial use only).
Files are downloaded into `data/`, which is gitignored. The PDFs to fetch are listed in
`data_manifest.txt`. If that file is missing, the script creates it from the documents with the
most questions.

`make fetch` runs the fetch script with no options. To change what it fetches:

```bash
uv run python scripts/fetch_data.py            # fetch what data_manifest.txt lists (= make fetch)
uv run python scripts/fetch_data.py --top 10   # rebuild the manifest from the 10 most-asked docs
uv run python scripts/fetch_data.py --force    # re-download everything
```

## Evaluation

```bash
make eval NAME=001_naive_1000          # 3 runs by default
make eval NAME=002_my_change RUNS=1    # a quicker single run
```

**Splits.** `dev` (default) is the 30 questions on the 5 filings in `data_manifest.txt`; every
experiment above was tuned on it. `test` is the other 120 FinanceBench questions, on 79 filings
listed in `data_manifest_test.txt` (10-Ks, 10-Qs, 8-Ks, earnings releases), held out for a final
check. The split is recorded in each results file.

```bash
make fetch SPLIT=test && make ingest SPLIT=test   # ~10,800 pages, ~50,000 chunks
make eval NAME=013_test_final SPLIT=test
```

Each run answers all 30 questions with `stockdesk.rag.answer`, then:

- **Accuracy:** an LLM judge (`JUDGE_MODEL`, default `gpt-5.4`, stronger than the answering
  model) compares the answer to the gold answer. Numbers may differ by rounding (up to 1%),
  every part of a multi-part question must be right, and an answer that only says it was not
  found is wrong. A hedged answer is graded on its substantive part.
- **Page hit / evidence hit:** whether any gold evidence page is among the retrieved pages, and
  whether any gold evidence text fuzzy-matches a retrieved chunk.
- **Refusal rate:** how often the answer is only "Not found in the filing."
- **Groundedness** (`evals/groundedness.py`; does not change the metrics above): every number in
  the answer must be on a retrieved page, in a `calculate` result from the trace, or in the
  question (rounding and million/billion scaling allowed; years, page numbers and list numbering
  ignored); cited pages must be retrieved and hold at least one of the figures; a claimed
  `calculate` result must match a call that ran. An answer failing any check is flagged. Reported
  separately for correct and wrong answers. For results files written before these checks:
  `make grounded FILES="results/016_agent.json ..."` (re-reads the PDFs; no RAG, agent or judge).

The defaults in `config.py` are the best config so far. Each results file records the config
it ran with; set those values as environment variables to reproduce an earlier experiment
(its table must be ingested with the same `TABLE`, `EMBEDDING_MODEL` and chunking settings).

Results go to `results/<NAME>.json` with the full config, the git commit, a dirty flag, every
answer and the judge's reason. The runner refuses to overwrite an existing file unless you pass
`--force` (`uv run python -m stockdesk.evals.runner --name ... --force`).

## Results

### Held-out test (120 questions, 79 filings)

3 runs, mean ± std. Day-1 config (001) vs final config (012), both run once on the test split.

| Run | Config | Accuracy | Page hit | Evidence hit | Refusal | Context chars |
|---|---|---|---|---|---|---|
| [014_test_baseline](results/014_test_baseline.json) | 001: naive, k=5, 3-small, prompt v1 | 38.1% ± 1.3 | 53.3% | 47.5% | 40.0% ± 1.7 | 4.5k |
| [013_test_final](results/013_test_final.json) | 012: expand, k=10, 5 pages, no exhibits, 3-large, prompt v2 | 71.9% ± 1.7 | 79.2% | 79.2% | 5.0% ± 0.8 | 15.5k |

+33.8 points on unseen questions, matching the dev gain (+34.4), so the tuning generalizes.
Latency is left out: the two runs ran in parallel, which inflated it. Retrieval is filtered to
the question's filing (single-document setting), so these numbers are not directly comparable
to FinanceBench results over a shared store of all filings.

### Experiments (dev: 30 questions, 5 filings)

3 runs. Values are mean ± std across runs.

| Experiment | Retriever | Chunking | k | Accuracy | Page hit | Evidence hit | Refusal | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| [001_naive_1000](results/001_naive_1000.json) | naive | 1000 / 200 overlap | 5 | 38.9% ± 5.1 | 50.0% ± 0.0 | 40.0% ± 0.0 | 34.4% ± 5.1 | 1.38 s / 2.31 s |
| [002_k10](results/002_k10.json) | naive | 1000 / 200 overlap | 10 | 50.0% ± 3.3 | 63.3% ± 0.0 | 50.0% ± 0.0 | 21.1% ± 1.9 | 1.43 s / 2.39 s |
| [003_dedup_k5](results/003_dedup_k5.json) | dedup (1 chunk/page) | 1000 / 200 overlap | 5 | 33.3% ± 3.3 | 53.3% ± 0.0 | 40.0% ± 0.0 | 35.6% ± 1.9 | 1.33 s / 2.26 s |
| [004_dedup_k10](results/004_dedup_k10.json) | dedup (1 chunk/page) | 1000 / 200 overlap | 10 | 45.6% ± 5.1 | 66.7% ± 0.0 | 50.0% ± 0.0 | 26.7% ± 0.0 | 1.48 s / 2.25 s |
| [005_expand_k10](results/005_expand_k10.json) | expand (full pages, max 5) | 1000 / 200 overlap | 10 | 54.4% ± 5.1 | 53.3% ± 0.0 | 56.7% ± 0.0 | 20.0% ± 0.0 | 1.55 s / 4.41 s* |
| [005b_expand_k10_repeat](results/005b_expand_k10_repeat.json) | repeat of 005 (same code and config) | 1000 / 200 overlap | 10 | 48.9% ± 1.9 | 53.3% ± 0.0 | 56.7% ± 0.0 | 20.0% ± 3.3 | 1.50 s / 4.59 s* |
| [006_expand_k10_p8](results/006_expand_k10_p8.json) | expand (full pages, max 8) | 1000 / 200 overlap | 10 | 56.7% ± 3.3 | 63.3% ± 0.0 | 63.3% ± 0.0 | 20.0% ± 3.3 | 1.57 s / 4.23 s* |
| [007_expand_p5_noexhibits](results/007_expand_p5_noexhibits.json) | expand (full pages, max 5), exhibits excluded | 1000 / 200 overlap | 10 | 55.6% ± 1.9 | 60.0% ± 0.0 | 63.3% ± 0.0 | 15.6% ± 1.9 | 1.51 s / 4.33 s* |
| [008_rewrite_multi](results/008_rewrite_multi.json) | 007 + multi-query rewriting (3 rewrites, RRF) | 1000 / 200 overlap | 10 per query | 54.4% ± 13.5 | 63.3% ± 5.8 | 64.4% ± 6.9 | 11.1% ± 5.1 | 4.48 s / 8.41 s* |
| [009_section_header](results/009_section_header.json) | 007 on section-header chunks | 1000 / 200 overlap, `[section]` header | 10 | 48.9% ± 1.9 | 60.0% ± 0.0 | 63.3% ± 0.0 | 26.7% ± 3.3 | 1.10 s / 3.82 s* |
| [010_rerank](results/010_rerank.json) | 007 + cross-encoder re-ranking (bge-reranker-base, 30 candidates) | 1000 / 200 overlap | 30 → re-ranked | 45.6% ± 1.9 | 43.3% ± 0.0 | 43.3% ± 0.0 | 34.4% ± 3.8 | 1.91 s / 5.36 s* |
| [011_large_embedding](results/011_large_embedding.json) | 007 on `text-embedding-3-large` (3072 dims) | 1000 / 200 overlap | 10 | 67.8% ± 5.1 | 66.7% ± 0.0 | 66.7% ± 0.0 | 7.8% ± 1.9 | 1.21 s / 3.78 s* |
| [012_prompt_v2](results/012_prompt_v2.json) | 011 + answer prompt v2 (figures, steps, `Answer:` line) | 1000 / 200 overlap | 10 | 73.3% ± 5.8 | 66.7% ± 0.0 | 66.7% ± 0.0 | 4.4% ± 1.9 | 1.87 s / 5.07 s* |
| [015_hybrid_bm25](results/015_hybrid_bm25.json) | 012 + BM25 hybrid (BM25 k=10, fused with RRF) | 1000 / 200 overlap | 10 + 10 BM25 | 61.1% ± 3.8 | 56.7% ± 0.0 | 56.7% ± 0.0 | 6.7% ± 0.0 | 2.86 s / 7.07 s* |
| [016_agent](results/016_agent.json) | 012's retrieval as a LangGraph agent tool (`search_filings`, `calculate`, ≤ 6 calls) | 1000 / 200 overlap | 10 per search | 83.3% ± 3.3 | 87.8% ± 3.8 | 87.8% ± 3.8 | 0.0% ± 0.0 | 5.52 s / 10.59 s* |
| [017_agent_prefetch](results/017_agent_prefetch.json) | 016 + prefetch: the question verbatim searched before the first model call | 1000 / 200 overlap | 10 per search | 70.0% ± 3.3 | 76.7% ± 0.0 | 76.7% ± 0.0 | 0.0% ± 0.0 | 3.28 s / 7.91 s* |
| [018_agent_checklist](results/018_agent_checklist.json) | 017 + evidence checklist (missing items must be searched) | 1000 / 200 overlap | 10 per search | 70.0% ± 8.8 | 75.6% ± 5.1 | 75.6% ± 5.1 | 0.0% ± 0.0 | 4.12 s / 8.99 s* |
| [019_agent_v2](results/019_agent_v2.json) | 016 + list tools (≤ 3 queries / 10 expressions per call), arithmetic discipline, short answer | 1000 / 200 overlap | 10 per query | 75.6% ± 3.8 | 85.6% ± 1.9 | 85.6% ± 1.9 | 1.1% ± 1.9 | 4.57 s / 8.17 s* |

All experiments answer with `gpt-5.4-mini` and are judged by `gpt-5.4`; 001–010 embed with
`text-embedding-3-small`, 011 with `text-embedding-3-large`. \*p95 includes a cold start (the
first question per filing parses its PDF).
Hypotheses and example questions: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

**Metrics v3:** an answer is a refusal if it is "Not found in the filing" alone, or if its final
"Answer:" line is (v3); hedged answers go to the judge. All results are rescored to v3 (v2 changed
9 of 720 records, v3 changed 4, all in 012).

- **002 (k=10):** +11 points accuracy, +3 evidence questions over 001. Adopted.
- **003 / 004 (one chunk per page):** worse; same-page chunks are one split passage. Rejected.
- **005 (page expansion, 5 pages):** evidence hit 56.7%, best so far, but the cap cuts 3 gold
  pages. Adopted.
- **005b (repeat of 005):** same retrieval, accuracy −5.5. Accuracy noise is about ±7 points, so
  retrieval is judged by evidence hit.
- **006 (8 pages):** evidence hit 63.3%, at +47% context. Deferred.
- **007 (exhibits excluded):** 006's evidence hit at 005's context. Adopted; best config.
- **008 (multi-query rewriting):** no measurable gain; adds retrieval variance and 3x latency.
  Rejected.
- **009 (section-header chunking):** same evidence hit, refusals +11 points; headers dilute
  number-dense statement chunks. Rejected.
- **010 (cross-encoder re-ranking):** evidence hit 43.3% vs 63.3% (−6 questions); confident on
  statement tables, near-random on prose questions. Rejected; code removed.
- **011 (text-embedding-3-large):** accuracy 67.8% vs 55.6% (+12.2, beyond the noise floor),
  refusals halved, no latency cost. Adopted.
- **012 (answer prompt v2):** accuracy 73.3% vs 67.8% (+5.5, within noise but consistent: 6
  questions up, 2 down, identical retrieval); refusals 4.4%. Adopted; now the default config.
- **015 (BM25 + vector hybrid):** evidence hit 56.7% vs 66.7% (−3 questions), accuracy 61.1% vs
  73.3% (−12.2, beyond noise); BM25-only pages displace vector's 4th–5th pages. Rejected.
- **016 (LangGraph agent):** accuracy 83.3% vs 73.3% (+10.0, beyond noise), evidence hit 87.8%;
  ~$0.0127 vs ~$0.0041 per question at list price (3.1x, before prompt caching), p50 5.5 s vs
  1.9 s. No invented figures, but copying errors into `calculate` and cited calls that never ran.
  Stepping stone, not final.
- **017 / 018 (prefetch the verbatim question; + evidence checklist):** accuracy 70.0% vs 83.3%
  (−13.3, beyond noise), evidence hit 76.7% / 75.6% vs 87.8%. The agent's own first query beats
  the verbatim one. Latency is decided by model calls (~2 s each), not retrieval. Rejected.
- **019 (agent v2: list tools, arithmetic discipline, short answer):** accuracy 75.6% vs 83.3%
  (−7.7, at the noise edge); p50 4.6 s vs 5.5 s, ~$0.0069 vs ~$0.0127 per question. No copying
  errors (vs 3); the short format likely cuts reasoning on judgment questions. Rejected.

### Groundedness (dev, pooled over 3 runs; correct / wrong answers)

| Run | Grounded numbers | Failed citation check | False `calculate` claim | Flagged |
|---|---|---|---|---|
| 012 (RAG) | 80.6% / 86.7% | 12.1% / 4.2% | n/a | 45.5% / 33.3% |
| 016 (agent) | 87.7% / 85.4% | 17.3% / 13.3% | 63.6% / 80.0% (of 16 claiming) | 50.7% / 73.3% |
| 020 (agent, tools v2) | 88.8% / 94.9% | 21.4% / 10.0% | n/a (none claimed) | 55.7% / 35.0% |

Most untraceable numbers are the model's own arithmetic, done without `calculate` (in 020 about
3 in 4 are a single +, −, ×, ÷ of grounded figures). Invented filing figures are rare but
present (012 00720: $54,578 million, in no page of the filing). Citation failures are mostly a
cited page that was never retrieved. Flags do not predict wrong answers.

## Roadmap

- **Phase 1: Ingestion + baseline RAG** (done)
- **Phase 2: Retrieval and eval** (done): eval runner, metrics, LLM judge, experiments 001–012,
  and the held-out evaluation (accuracy 38.1% → 71.9% on 120 unseen questions).
  - Optional last retrieval experiment: BM25 + vector hybrid with RRF, on the dev split only.
- **Phase 3: LangGraph agent** (in progress): `search_filings` as a tool, `calculate`, multi-step
  retrieval. Developed on dev, measured once on test.
  - Step 1 (done, 016): minimal ReAct agent, `ANSWER_MODE=agent`.
- **Phase 4: Eval quality** (in progress): groundedness checks (numbers, citations, `calculate`
  claims; done), judge validation against human labels, eval in CI.
- **Phase 5: Observability and UI** (planned): tracing, cost per question, Gradio UI.

## License

The code is released under the [MIT License](LICENSE). The FinanceBench data (questions, document
metadata and filing PDFs) is licensed separately under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) by Patronus AI and is not covered
by the code license. It is not included in this repository.
