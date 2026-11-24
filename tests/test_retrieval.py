"""Tests for page helpers and retriever dispatch. No DB: dedup_by_page is pure."""

import pytest
from langchain_core.documents import Document

from stockdesk.retrieval import get_retriever, naive
from stockdesk.retrieval.pages import dedup_by_page


def chunk(page: int, text: str = "", doc_name: str = "DOC") -> Document:
    return Document(page_content=text or f"p{page}", metadata={"doc_name": doc_name, "page": page})


def pages(docs: list[Document]) -> list[int]:
    return [d.metadata["page"] for d in docs]


# dedup_by_page


def test_keeps_first_chunk_per_page():
    docs = [chunk(3, "best"), chunk(3, "worse"), chunk(7, "only")]

    kept = dedup_by_page(docs, k=5)

    assert [d.page_content for d in kept] == ["best", "only"]


def test_preserves_rank_order():
    docs = [chunk(9), chunk(2), chunk(9), chunk(5), chunk(2), chunk(1)]

    assert pages(dedup_by_page(docs, k=10)) == [9, 2, 5, 1]


def test_respects_k():
    docs = [chunk(1), chunk(1), chunk(2), chunk(3), chunk(4)]

    assert pages(dedup_by_page(docs, k=2)) == [1, 2]


def test_fewer_distinct_pages_than_k():
    docs = [chunk(4), chunk(4), chunk(8)]

    assert pages(dedup_by_page(docs, k=5)) == [4, 8]


def test_empty_list():
    assert dedup_by_page([], k=5) == []


def test_same_page_number_in_different_docs_is_kept():
    docs = [chunk(3, doc_name="A"), chunk(3, doc_name="B")]

    assert len(dedup_by_page(docs, k=5)) == 2


# get_retriever


def test_dispatch_returns_the_right_retriever():
    assert get_retriever("naive") is naive.retrieve


@pytest.mark.parametrize("name", ["bm25", "dedup"])
def test_dispatch_rejects_unknown_name(name: str):
    with pytest.raises(ValueError, match=f"unknown retriever '{name}'"):
        get_retriever(name)
