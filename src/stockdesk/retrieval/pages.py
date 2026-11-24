"""Page-level helpers for retrievers: full page text, one chunk per page, rank fusion."""

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


def chunk_key(doc: Document) -> tuple[str | None, int, str]:
    """Identity of a chunk across searches: (doc_name, page, chunk text)."""
    return (doc.metadata.get("doc_name"), doc.metadata["page"], doc.page_content)


def rrf_merge(ranked_lists: list[list[Document]], k_rrf: int = 60) -> list[Document]:
    """Merge ranked chunk lists with Reciprocal Rank Fusion, best first.

    A chunk scores sum(1 / (k_rrf + rank)) over the lists it appears in (rank starts at 1; a
    repeat within one list counts once, at its best rank). Ties keep first-seen order, scanning
    the lists in order, so the earlier list (the original question) wins a tie.
    """
    scores: dict[tuple[str | None, int, str], float] = {}
    first_seen: dict[tuple[str | None, int, str], Document] = {}
    for ranked in ranked_lists:
        counted: set[tuple[str | None, int, str]] = set()
        for rank, doc in enumerate(ranked, 1):
            key = chunk_key(doc)
            if key in counted:
                continue
            counted.add(key)
            first_seen.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1 / (k_rrf + rank)
    order = sorted(first_seen, key=lambda key: -scores[key])  # stable: ties keep first-seen
    return [first_seen[key] for key in order]
