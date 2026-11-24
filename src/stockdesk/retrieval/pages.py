"""Page-level helpers for retrievers: full page text, and one chunk per page."""

from functools import lru_cache

from langchain_core.documents import Document

from stockdesk.ingestion.ingestion import load_pdf


@lru_cache(maxsize=32)  # one entry per filing
def page_texts(doc_name: str) -> dict[int, str]:
    """Full text of each non-empty page of a filing, keyed by zero-based page number.

    Read from the PDF and cached per process. Production would store page text in the DB
    (e.g. a pages table written at ingestion) instead of re-reading PDFs.
    """
    return {d.metadata["page"]: d.page_content for d in load_pdf(doc_name)}


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
