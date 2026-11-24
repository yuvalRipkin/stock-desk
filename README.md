# stock-desk

stock-desk answers questions about US public companies from their SEC filings, and measures how
well it does against the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark.

What exists today:

- **Naive RAG over 5 filings.** The 10-Ks listed in `data_manifest.txt` are split into
  1000-character chunks, embedded with OpenAI `text-embedding-3-small`, and stored in PostgreSQL
  with pgvector. A question is answered by vector search over one filing (top 5 chunks) and an
  LLM prompted to answer only from those excerpts, cite page numbers, or reply
  "Not found in the filing."
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
make ingest            # chunk, embed and store the PDFs (calls the OpenAI embeddings API)
make test
```

Then ask a question about one filing:

```bash
make ask DOC=AMD_2022_10K Q="What was total revenue in 2022?"
```

Expected output (the wording can vary between runs, even at temperature 0):

```text
Total net revenue in 2022 was $23.601 billion (page 47).

Retrieved pages (AMD_2022_10K): 47, 67, 66, 48, 63
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

Each run answers all 30 questions with `stockdesk.rag.answer`, then:

- **Accuracy:** an LLM judge (`JUDGE_MODEL`, default `gpt-5.4`, stronger than the answering
  model) compares the answer to the gold answer. Numbers may differ by rounding (up to 1%),
  every part of a multi-part question must be right, and a refusal is always wrong.
- **Page hit / evidence hit:** whether any gold evidence page is among the retrieved pages, and
  whether any gold evidence text fuzzy-matches a retrieved chunk.
- **Refusal rate:** how often the answer is "Not found in the filing."

Results go to `results/<NAME>.json` with the full config, the git commit, a dirty flag, every
answer and the judge's reason. The runner refuses to overwrite an existing file unless you pass
`--force` (`uv run python -m stockdesk.evals.runner --name ... --force`).

## Results

30 FinanceBench questions over 5 filings, 3 runs. Values are mean ± std across runs.

| Experiment | Retriever | Chunking | k | Accuracy | Page hit | Evidence hit | Refusal | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| [001_naive_1000](results/001_naive_1000.json) | naive | 1000 / 200 overlap | 5 | 38.9% ± 5.1 | 50.0% ± 0.0 | 40.0% ± 0.0 | 34.4% ± 5.1 | 1.38 s / 2.31 s |
| [002_k10](results/002_k10.json) | naive | 1000 / 200 overlap | 10 | 50.0% ± 3.3 | 63.3% ± 0.0 | 50.0% ± 0.0 | 21.1% ± 1.9 | 1.43 s / 2.39 s |
| [003_dedup_k5](results/003_dedup_k5.json) | dedup (1 chunk/page) | 1000 / 200 overlap | 5 | 33.3% ± 3.3 | 53.3% ± 0.0 | 40.0% ± 0.0 | 37.8% ± 3.8 | 1.33 s / 2.26 s |
| [004_dedup_k10](results/004_dedup_k10.json) | dedup (1 chunk/page) | 1000 / 200 overlap | 10 | 45.6% ± 5.1 | 66.7% ± 0.0 | 50.0% ± 0.0 | 26.7% ± 0.0 | 1.48 s / 2.25 s |

All experiments answer with `gpt-5.4-mini`, embed with `text-embedding-3-small`, and are judged
by `gpt-5.4`.

**002 (k=10):** retrieving 10 chunks instead of 5 finds the gold evidence for 3 more questions
and lifts accuracy by 11 points, more than twice the baseline's run-to-run spread. Refusals drop
from 34% to 21%. Retrieval still matters most: with the gold passage retrieved, 64% of answers
are correct; without it, 35%. Even with the passage, about a third of answers are wrong, so
generation is a second bottleneck.

**Cost:** k=10 doubles the retrieved context, so the answering model's input tokens per question
roughly double. Latency barely changes. The embedding call and the judge (which never sees the
context) cost the same.

**003 (dedup, k=5):** hurts. Same-page chunks are complementary (split passages), not
duplicates. Keeping one chunk per page reaches 5 distinct pages (vs 4.4), but accuracy is 5.6
points below the baseline and refusals rise. That drop is only about one baseline std; the
stronger evidence is per question: two questions that were right 3/3 became wrong 3/3 because
dedup dropped the second chunk of their gold page. See
[the write-up](docs/notes/2026-11-24-dedup-negative.md).

**004 (dedup, k=10):** confirms 003 at k=10: more distinct pages (10 vs 8.1), no extra evidence.
Evidence hit stays at 50%, and accuracy (45.6%) is within noise of 002's 50.0%.

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
