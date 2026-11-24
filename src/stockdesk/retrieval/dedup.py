"""Page-dedup retriever: vector search, then at most one chunk per page.

Fetches k * FETCH_MULTIPLIER candidates with the naive vector search and keeps the best-ranked
chunk of each (doc_name, page), so the k chunks returned cover k distinct pages when possible.
"""

from langchain_core.documents import Document

from stockdesk.config import get_settings
from stockdesk.retrieval import naive


def dedup_by_page(docs: list[Document], k: int) -> list[Document]:
    """Keep the first (best-ranked) chunk per (doc_name, page), in rank order, up to k.

    Returns fewer than k chunks if there are fewer distinct pages.
    """
    seen: set[tuple[str | None, int]] = set()
    kept: list[Document] = []
    for doc in docs:
        if len(kept) == k:
            break
        key = (doc.metadata.get("doc_name"), doc.metadata["page"])
        if key not in seen:
            seen.add(key)
            kept.append(doc)
    return kept


def retrieve(question: str, doc_name: str | None = None, k: int | None = None) -> list[Document]:
    """Return up to k chunks from distinct pages, most similar first, from doc_name if given."""
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k
    candidates = naive.retrieve(question, doc_name, k * settings.fetch_multiplier)
    return dedup_by_page(candidates, k)
