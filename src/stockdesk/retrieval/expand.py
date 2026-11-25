"""Page-expansion retriever: vector search for chunks, then send their whole pages.

Searches for k chunks, keeps the first MAX_PAGES distinct pages in rank order, and returns one
Document per page holding that page's full text, so a passage the splitter cut across chunks
reaches the model intact.

With QUERY_REWRITE=multi, it searches k chunks for the question and for each LLM rewrite of it,
and merges the ranked lists with Reciprocal Rank Fusion before picking pages.

With RERANK=cross_encoder, it fetches RERANK_CANDIDATES chunks instead of k and sorts them by
cross-encoder score before picking pages.
"""

import time

from langchain_core.documents import Document

from stockdesk.config import get_settings
from stockdesk.retrieval import Retrieval, naive, rerank, rewrite
from stockdesk.retrieval.pages import dedup_by_page, page_texts, rrf_merge


def search_queries(question: str) -> list[str]:
    """The queries to search: the question, then its rewrites if QUERY_REWRITE=multi."""
    if get_settings().query_rewrite == "multi":
        return [question, *rewrite.rewrite_queries(question)]
    return [question]


def search(question: str, doc_name: str | None = None, k: int | None = None) -> Retrieval:
    """Full text of up to MAX_PAGES pages whose chunks rank best, and the queries searched."""
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k
    queries = search_queries(question)
    rerank_ms = None
    if settings.rerank == "cross_encoder":
        hits = naive.retrieve(question, doc_name, settings.rerank_candidates)
        start = time.perf_counter()
        hits = rerank.rerank(question, hits)
        rerank_ms = (time.perf_counter() - start) * 1000
    elif len(queries) == 1:
        hits = naive.retrieve(question, doc_name, k)
    else:
        hits = rrf_merge([naive.retrieve(q, doc_name, k) for q in queries])
    pages = []
    for hit in dedup_by_page(hits, settings.max_pages):
        name, page = hit.metadata["doc_name"], hit.metadata["page"]
        text = page_texts(name).get(page, hit.page_content)
        pages.append(Document(page_content=text, metadata={"doc_name": name, "page": page}))
    return Retrieval(docs=pages, queries=queries, rerank_ms=rerank_ms)


def retrieve(question: str, doc_name: str | None = None, k: int | None = None) -> list[Document]:
    """The pages from search(), without the queries."""
    return search(question, doc_name, k).docs
