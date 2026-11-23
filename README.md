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

## Roadmap

- **Phase 1:** Ingestion + baseline RAG
- **Phase 2:** FinanceBench evals
- **Phase 3:** Hybrid search + re-ranking
- **Phase 4:** LangGraph agent + tools + routing
- **Phase 5:** LLM judge + CI regression subset
- **Phase 6:** Tracing, caching, cost/latency
