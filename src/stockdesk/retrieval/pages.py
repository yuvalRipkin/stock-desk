"""Page-level helpers for retrievers."""

from langchain_core.documents import Document


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
