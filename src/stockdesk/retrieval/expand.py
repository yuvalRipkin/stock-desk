"""Page-expansion retriever: vector search for chunks, then send their whole pages.

Searches for k chunks, keeps the first MAX_PAGES distinct pages in rank order, and returns one
Document per page holding that page's full text, so a passage the splitter cut across chunks
reaches the model intact.

With QUERY_REWRITE=multi, it searches k chunks for the question and for each LLM rewrite of it,
and merges the ranked lists with Reciprocal Rank Fusion before picking pages. With HYBRID=bm25,
a BM25 keyword ranking of the question (BM25_K chunks) is fused in the same way.
"""

from langchain_core.documents import Document

from stockdesk.config import get_settings
from stockdesk.retrieval import Retrieval, bm25, naive, rewrite
from stockdesk.retrieval.pages import dedup_by_page, page_texts, rrf_merge
from stockdesk.timing import timed


def search_queries(question: str) -> list[str]:
    """The queries to search: the question, then its rewrites if QUERY_REWRITE=multi."""
    if get_settings().query_rewrite == "multi":
        with timed("rewrite_ms"):
            return [question, *rewrite.rewrite_queries(question)]
    return [question]


def search(question: str, doc_name: str | None = None, k: int | None = None) -> Retrieval:
    """Full text of up to MAX_PAGES pages whose chunks rank best, and the queries searched."""
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k
    queries = search_queries(question)
    ranked = [naive.retrieve(q, doc_name, k) for q in queries]
    if settings.hybrid == "bm25":
        with timed("bm25_ms"):
            ranked.append(bm25.retrieve(question, doc_name, settings.bm25_k))
    hits = ranked[0] if len(ranked) == 1 else rrf_merge(ranked)
    pages = []
    with timed("pages_ms"):
        for hit in dedup_by_page(hits, settings.max_pages):
            name, page = hit.metadata["doc_name"], hit.metadata["page"]
            text = page_texts(name).get(page, hit.page_content)
            pages.append(Document(page_content=text, metadata={"doc_name": name, "page": page}))
    return Retrieval(docs=pages, queries=queries)


def retrieve(question: str, doc_name: str | None = None, k: int | None = None) -> list[Document]:
    """The pages from search(), without the queries."""
    return search(question, doc_name, k).docs
