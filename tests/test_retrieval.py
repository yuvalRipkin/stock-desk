"""Tests for page helpers, rank fusion, page expansion, query rewriting and retriever dispatch.

No DB, PDFs or LLM: all mocked.
"""

import pytest
from langchain_core.documents import Document

from stockdesk.retrieval import expand, get_retriever, naive
from stockdesk.retrieval import pages as pages_module
from stockdesk.retrieval.pages import dedup_by_page, rrf_merge


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


# rrf_merge


def texts(docs: list[Document]) -> list[str]:
    return [d.page_content for d in docs]


def test_rrf_single_list_keeps_its_order():
    ranked = [chunk(1, "a"), chunk(2, "b"), chunk(3, "c")]

    assert texts(rrf_merge([ranked])) == ["a", "b", "c"]


def test_rrf_orders_by_summed_reciprocal_rank():
    # a: 1/61 + 1/63, b: 1/62 + 1/61, c: 1/63, d: 1/62
    first = [chunk(1, "a"), chunk(2, "b"), chunk(3, "c")]
    second = [chunk(2, "b"), chunk(4, "d"), chunk(1, "a")]

    assert texts(rrf_merge([first, second])) == ["b", "a", "d", "c"]


def test_rrf_chunk_found_by_several_queries_ranks_higher():
    # "shared" is third in every list, but beats each list's single top hit
    lists = [[chunk(i, f"top{i}"), chunk(9, "x" * i), chunk(5, "shared")] for i in range(3)]

    assert rrf_merge(lists)[0].page_content == "shared"


def test_rrf_ties_keep_first_seen_order():
    # a and b both score 1/61; a comes from the earlier list (the original question)
    assert texts(rrf_merge([[chunk(1, "a")], [chunk(2, "b")]])) == ["a", "b"]
    assert texts(rrf_merge([[chunk(2, "b")], [chunk(1, "a")]])) == ["b", "a"]


def test_rrf_identifies_chunks_by_doc_page_and_text():
    same_page_other_text = [chunk(1, "a"), chunk(1, "b")]
    other_doc = [chunk(1, "a", doc_name="OTHER")]

    merged = rrf_merge([same_page_other_text, other_doc])

    assert [(d.metadata["doc_name"], d.page_content) for d in merged] == [
        ("DOC", "a"),
        ("OTHER", "a"),
        ("DOC", "b"),
    ]


def test_rrf_counts_a_repeat_within_one_list_once():
    # a: 1/61 (its repeats ignored) loses to b: 1/61 + 1/61; counting repeats, a would win
    lists = [[chunk(1, "a"), chunk(1, "a"), chunk(1, "a")], [chunk(2, "b")], [chunk(2, "b")]]

    assert texts(rrf_merge(lists)) == ["b", "a"]


def test_rrf_empty_lists():
    assert rrf_merge([]) == []
    assert rrf_merge([[], []]) == []
    assert texts(rrf_merge([[], [chunk(1, "a")]])) == ["a"]


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


def set_settings(monkeypatch: pytest.MonkeyPatch, **update) -> None:
    settings = expand.get_settings().model_copy(update={"query_rewrite": "none", **update})
    monkeypatch.setattr(expand, "get_settings", lambda: settings)


def set_max_pages(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    set_settings(monkeypatch, max_pages=n)


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


# expand with QUERY_REWRITE=multi


class FakeStore:
    """Vector store returning canned hits per query, recording every search."""

    def __init__(self, hits_by_query: dict[str, list[Document]]):
        self.hits_by_query = hits_by_query
        self.searches: list[tuple[str, int, dict | None]] = []

    def similarity_search(self, query: str, k: int, filter: dict | None = None):
        self.searches.append((query, k, filter))
        return self.hits_by_query[query][:k]


@pytest.fixture
def multi_query(monkeypatch: pytest.MonkeyPatch):
    """Rewriting on, with the rewriter, store and PDF reader mocked."""
    settings = expand.get_settings().model_copy(
        update={"query_rewrite": "multi", "max_pages": 5, "exclude_exhibits": False}
    )
    monkeypatch.setattr(expand, "get_settings", lambda: settings)
    monkeypatch.setattr(naive, "get_settings", lambda: settings)
    monkeypatch.setattr(expand.rewrite, "rewrite_queries", lambda q: ["balance sheet", "cash"])
    store = FakeStore(
        {
            "q": [chunk(7, "a"), chunk(3, "b")],
            "balance sheet": [chunk(12, "c"), chunk(3, "b")],
            "cash": [chunk(3, "b"), chunk(9, "d")],
        }
    )
    monkeypatch.setattr(naive, "get_store", lambda: store)
    monkeypatch.setattr(pages_module, "load_pdf", lambda name: [])
    pages_module.page_texts.cache_clear()
    yield store
    pages_module.page_texts.cache_clear()


def test_multi_searches_the_question_and_each_rewrite(multi_query):
    result = expand.search("q", "DOC", k=4)

    assert result.queries == ["q", "balance sheet", "cash"]
    assert [(query, k) for query, k, _ in multi_query.searches] == [
        ("q", 4),
        ("balance sheet", 4),
        ("cash", 4),
    ]
    assert all(f == {"doc_name": {"$eq": "DOC"}} for _, _, f in multi_query.searches)


def test_multi_fuses_the_ranked_lists(multi_query):
    # b is found by all three queries, so it ranks first; then the top hits in query order
    assert pages(expand.retrieve("q", "DOC", k=4)) == [3, 7, 12, 9]


def test_multi_caps_pages_at_max_pages(multi_query, monkeypatch):
    settings = expand.get_settings().model_copy(update={"max_pages": 2})
    monkeypatch.setattr(expand, "get_settings", lambda: settings)

    assert pages(expand.retrieve("q", "DOC", k=4)) == [3, 7]


def test_no_rewrite_searches_only_the_question(mock_expand, monkeypatch):
    hits, _ = mock_expand
    hits += [chunk(3)]
    set_settings(monkeypatch, max_pages=5)
    monkeypatch.setattr(expand.rewrite, "rewrite_queries", lambda q: pytest.fail("rewrote"))

    result = expand.search("q", "DOC", k=4)

    assert result.queries == ["q"]
    assert pages(result.docs) == [3]


def test_naive_search_reports_the_question_as_its_only_query(monkeypatch):
    monkeypatch.setattr(naive, "retrieve", lambda q, doc, k: [chunk(1)])

    result = naive.search("q", "DOC", 5)

    assert result.queries == ["q"]
    assert pages(result.docs) == [1]


# get_retriever


def test_dispatch_returns_the_right_retriever():
    assert get_retriever("naive") is naive.search
    assert get_retriever("expand") is expand.search


@pytest.mark.parametrize("name", ["bm25", "dedup"])
def test_dispatch_rejects_unknown_name(name: str):
    with pytest.raises(ValueError, match=f"unknown retriever '{name}'"):
        get_retriever(name)
