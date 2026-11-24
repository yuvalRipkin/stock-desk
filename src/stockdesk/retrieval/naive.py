"""Naive retriever: pure vector similarity search, optionally scoped to one filing."""

from langchain_core.documents import Document

from stockdesk.config import get_settings
from stockdesk.data import read_manifest
from stockdesk.retrieval import Retrieval
from stockdesk.retrieval.exhibits import exhibit_cutoff
from stockdesk.store import get_store


def doc_filter(doc_name: str) -> dict:
    """Metadata filter for one filing, excluding its exhibit pages if EXCLUDE_EXHIBITS is set."""
    same_doc = {"doc_name": {"$eq": doc_name}}
    cutoff = exhibit_cutoff(doc_name) if get_settings().exclude_exhibits else None
    if cutoff is None:
        return same_doc
    return {"$and": [same_doc, {"page": {"$lt": cutoff}}]}


def search_filter(doc_name: str | None) -> dict | None:
    """The filter for a search scoped to doc_name, or across all manifest filings."""
    if doc_name:
        return doc_filter(doc_name)
    if get_settings().exclude_exhibits:
        return {"$or": [doc_filter(name) for name in read_manifest()]}
    return None


def retrieve(question: str, doc_name: str | None = None, k: int | None = None) -> list[Document]:
    """Return the k chunks most similar to the question, from doc_name if given."""
    if k is None:
        k = get_settings().retrieval_k
    return get_store().similarity_search(question, k=k, filter=search_filter(doc_name))


def search(question: str, doc_name: str | None = None, k: int | None = None) -> Retrieval:
    """retrieve(), with the question as the only query searched."""
    return Retrieval(docs=retrieve(question, doc_name, k), queries=[question])
