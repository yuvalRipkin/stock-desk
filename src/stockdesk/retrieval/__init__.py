"""Retrieval: find the chunks of a filing that are relevant to a question."""

from collections.abc import Callable

from langchain_core.documents import Document

Retriever = Callable[[str, str | None, int | None], list[Document]]


def get_retriever(name: str) -> Retriever:
    """Return the retrieve function for a RETRIEVER setting value ("naive" or "expand")."""
    from stockdesk.retrieval import expand, naive

    retrievers: dict[str, Retriever] = {"naive": naive.retrieve, "expand": expand.retrieve}
    if name not in retrievers:
        raise ValueError(f"unknown retriever {name!r}; expected one of {sorted(retrievers)}")
    return retrievers[name]
