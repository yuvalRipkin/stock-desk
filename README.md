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

The defaults in `config.py` are the best config so far. Each results file records the config
it ran with; set those values as environment variables to reproduce an earlier experiment
(its table must be ingested with the same `TABLE`, `EMBEDDING_MODEL` and chunking settings).

Results go to `results/<NAME>.json` with the full config, the git commit, a dirty flag, every
answer and the judge's reason. The runner refuses to overwrite an existing file unless you pass
`--force` (`uv run python -m stockdesk.evals.runner --name ... --force`).

## Results

30 FinanceBench questions over 5 filings, 3 runs. Values are mean ± std across runs.

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

## Roadmap

- **Phase 1: Ingestion + baseline RAG** (done)
- **Phase 2: FinanceBench evals** (done: runner, metrics, LLM judge, naive baseline)
- **Phase 3: Hybrid search + re-ranking** (planned): BM25 + vector retrieval
- **Phase 4: LangGraph agent + tools + routing** (planned): including price data and technical
  indicators
- **Phase 5: CI regression subset** (planned)
- **Phase 6: Tracing, caching, cost/latency** (planned): OpenTelemetry / Arize Phoenix

## License

The code is released under the [MIT License](LICENSE). The FinanceBench data (questions, document
metadata and filing PDFs) is licensed separately under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) by Patronus AI and is not covered
by the code license. It is not included in this repository.
