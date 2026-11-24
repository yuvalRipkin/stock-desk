"""Naive retriever: pure vector similarity search, optionally scoped to one filing."""

from langchain_core.documents import Document

from stockdesk.config import get_settings
from stockdesk.store import get_store


def retrieve(question: str, doc_name: str | None = None, k: int | None = None) -> list[Document]:
    """Return the k chunks most similar to the question, from doc_name if given."""
    if k is None:
        k = get_settings().retrieval_k
    filter_ = {"doc_name": {"$eq": doc_name}} if doc_name else None
    return get_store().similarity_search(question, k=k, filter=filter_)
