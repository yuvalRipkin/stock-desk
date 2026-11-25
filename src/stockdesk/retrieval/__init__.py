"""Retrieval: find the chunks of a filing that are relevant to a question."""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class Retrieval:
    docs: list[Document]
    queries: list[str]  # what was searched: the question first, then any rewrites
    rerank_ms: float | None = None  # time spent re-ranking; None if not re-ranked


Retriever = Callable[[str, str | None, int | None], Retrieval]


def get_retriever(name: str) -> Retriever:
    """Return the search function for a RETRIEVER setting value ("naive" or "expand")."""
    from stockdesk.retrieval import expand, naive

    retrievers: dict[str, Retriever] = {"naive": naive.search, "expand": expand.search}
    if name not in retrievers:
        raise ValueError(f"unknown retriever {name!r}; expected one of {sorted(retrievers)}")
    return retrievers[name]
