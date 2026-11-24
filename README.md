# stock-desk

stock-desk answers questions about US public companies from their SEC filings, and measures how
well it does against the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark.

What exists today:

- **Naive RAG over 5 filings.** The 10-Ks listed in `data_manifest.txt` are split into
  1000-character chunks, embedded with OpenAI `text-embedding-3-small`, and stored in PostgreSQL
  with pgvector. A question is answered by vector search over one filing (top 5 chunks) and an
  LLM prompted to answer only from those excerpts, cite page numbers, or reply
  "Not found in the filing."
- **FinanceBench eval (in progress).** A loader for the 30 FinanceBench questions about those
  filings, plus retrieval and answer metrics. The eval runner and LLM judge are not written yet.

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), Docker and an OpenAI API key.

```bash
git clone <this repo> stock-desk && cd stock-desk
cp .env.example .env   # set OPENAI_API_KEY; LLM_MODEL defaults to gpt-5.4-mini
make install           # uv sync
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

## Roadmap

- **Phase 1: Ingestion + baseline RAG** (done)
- **Phase 2: FinanceBench evals** (in progress: dataset loader and metrics done; runner next)
- **Phase 3: Hybrid search + re-ranking** (planned): BM25 + vector retrieval
- **Phase 4: LangGraph agent + tools + routing** (planned): including price data and technical
  indicators
- **Phase 5: LLM judge + CI regression subset** (planned)
- **Phase 6: Tracing, caching, cost/latency** (planned): OpenTelemetry / Arize Phoenix

## License

The code is released under the [MIT License](LICENSE). The FinanceBench data (questions, document
metadata and filing PDFs) is licensed separately under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) by Patronus AI and is not covered
by the code license. It is not included in this repository.
