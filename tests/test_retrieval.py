"""Tests for page helpers, page expansion and retriever dispatch. No DB or PDFs: mocked."""

import pytest
from langchain_core.documents import Document

from stockdesk.retrieval import expand, get_retriever, naive
from stockdesk.retrieval import pages as pages_module
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


# expand.retrieve

FULL_PAGES = {
    "DOC": {3: "full page three", 7: "full page seven", 9: "full page nine", 12: "page twelve"}
}


@pytest.fixture
def mock_expand(monkeypatch: pytest.MonkeyPatch):
    """Mock the vector search and the PDF reader; return a setter for the search hits."""
    hits: list[Document] = []
    pdf_reads: list[str] = []

    def fake_load_pdf(doc_name: str) -> list[Document]:
        pdf_reads.append(doc_name)
        return [chunk(p, text, doc_name) for p, text in FULL_PAGES[doc_name].items()]

    monkeypatch.setattr(expand.naive, "retrieve", lambda q, doc, k: hits[:k])
    monkeypatch.setattr(pages_module, "load_pdf", fake_load_pdf)
    pages_module.page_texts.cache_clear()
    yield hits, pdf_reads
    pages_module.page_texts.cache_clear()


def set_max_pages(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    settings = expand.get_settings().model_copy(update={"max_pages": n})
    monkeypatch.setattr(expand, "get_settings", lambda: settings)


def test_expand_returns_full_page_text_in_rank_order(mock_expand, monkeypatch):
    hits, pdf_reads = mock_expand
    hits += [chunk(7, "chunk a"), chunk(3, "chunk b"), chunk(7, "chunk c"), chunk(9, "chunk d")]
    set_max_pages(monkeypatch, 5)

    docs = expand.retrieve("q", "DOC", k=4)

    assert [d.page_content for d in docs] == [
        "full page seven",
        "full page three",
        "full page nine",
    ]
    assert [d.metadata for d in docs] == [
        {"doc_name": "DOC", "page": 7},
        {"doc_name": "DOC", "page": 3},
        {"doc_name": "DOC", "page": 9},
    ]
    assert pdf_reads == ["DOC"]  # read once, then cached


def test_expand_respects_max_pages(mock_expand, monkeypatch):
    hits, _ = mock_expand
    hits += [chunk(12), chunk(3), chunk(9), chunk(7)]
    set_max_pages(monkeypatch, 2)

    docs = expand.retrieve("q", "DOC", k=4)

    assert pages(docs) == [12, 3]


# get_retriever


def test_dispatch_returns_the_right_retriever():
    assert get_retriever("naive") is naive.retrieve
    assert get_retriever("expand") is expand.retrieve


@pytest.mark.parametrize("name", ["bm25", "dedup"])
def test_dispatch_rejects_unknown_name(name: str):
    with pytest.raises(ValueError, match=f"unknown retriever '{name}'"):
        get_retriever(name)
