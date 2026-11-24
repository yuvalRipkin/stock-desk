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

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env   # then fill in your keys
make install
make db-up
make test
```

## Data

Questions, document metadata and filing PDFs come from
[FinanceBench](https://github.com/patronus-ai/financebench) by Patronus AI, licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (noncommercial use only).
Files are downloaded into `data/`, which is gitignored. The PDFs to fetch are listed in
`data_manifest.txt`. If that file is missing, the script creates it from the documents with the
most questions.

```bash
uv run python scripts/fetch_data.py            # fetch what data_manifest.txt lists
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
