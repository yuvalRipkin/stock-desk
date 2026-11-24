# stock-desk

stock-desk is an AI research agent that answers questions about US public companies by
grounding its answers in SEC filings (10-K/10-Q), historical price data and technical
indicators. Filings are ingested and indexed in PostgreSQL with pgvector, retrieved with a
hybrid BM25 + vector search, and reasoned over by a LangGraph agent, with answer quality
measured against the FinanceBench benchmark and runs traced via OpenTelemetry / Arize Phoenix.

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

- **Phase 1:** Ingestion + baseline RAG
- **Phase 2:** FinanceBench evals
- **Phase 3:** Hybrid search + re-ranking
- **Phase 4:** LangGraph agent + tools + routing
- **Phase 5:** LLM judge + CI regression subset
- **Phase 6:** Tracing, caching, cost/latency

## License

The code is released under the [MIT License](LICENSE). The FinanceBench data (questions, document
metadata and filing PDFs) is licensed separately under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) by Patronus AI and is not covered
by the code license. It is not included in this repository.
