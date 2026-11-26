"""Tests for BM25 keyword search and hybrid merging. No DB, PDFs or network: all mocked."""

import pytest
from langchain_core.documents import Document

from stockdesk.retrieval import bm25, expand
from stockdesk.retrieval import pages as pages_module
from stockdesk.retrieval.bm25 import build_index, tokenize


def chunk(page: int, text: str, doc_name: str = "DOC") -> Document:
    return Document(page_content=text, metadata={"doc_name": doc_name, "page": page})


# tokenize


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Total Revenue (Net); EMEA/APAC") == ["total", "revenue", "net", "emea", "apac"]


def test_tokenize_keeps_numbers_and_hyphenated_terms():
    assert tokenize("Form 10-K: $23,601 million, up 4.5% in 2022.") == [
        "form",
        "10-k",
        "23601",
        "million",
        "up",
        "4.5",
        "2022",
    ]


def test_tokenize_thousands_separator_does_not_matter():
    assert tokenize("23,601") == tokenize("23601") == ["23601"]


def test_tokenize_drops_stopwords():
    assert tokenize("What is the net income of the company?") == ["net", "income", "company"]


def test_tokenize_empty():
    assert tokenize("") == tokenize("of the") == []


# BM25 ranking

CORPUS = [
    chunk(1, "Revenue grew in all segments, driven by strong demand."),
    chunk(2, "Management discussion of revenue, margins and operating results."),
    chunk(3, "Goodwill impairment charge recorded in the EMEA segment."),
    chunk(4, "Liquidity and capital resources; revenue from customers."),
]


def test_bm25_ranks_the_exact_term_chunk_first():
    assert build_index(CORPUS).search("goodwill impairment", k=3) == [CORPUS[2]]


def test_bm25_leaves_out_chunks_without_query_terms_and_respects_k():
    results = build_index(CORPUS).search("revenue", k=2)

    assert len(results) == 2
    assert all("revenue" in d.page_content.lower() for d in results)


def test_bm25_empty_corpus_and_no_match():
    assert build_index([]).search("revenue", k=5) == []
    assert build_index(CORPUS).search("dividend", k=5) == []


# retrieve: per-filing index, exhibit exclusion, cache


@pytest.fixture
def filing(monkeypatch: pytest.MonkeyPatch):
    """A filing with body pages 2-7 and exhibits from page 10; both mention indemnification.

    Four body chunks, so the term is in under half of them and BM25 gives it a positive IDF.
    """
    loads: list[str] = []
    chunks = [
        chunk(2, "Legal proceedings and indemnification obligations."),
        chunk(5, "Revenue by segment."),
        chunk(6, "Liquidity and capital resources."),
        chunk(7, "Risk factors."),
        chunk(10, "Exhibit 10.1 Indemnification agreement indemnification indemnification."),
        chunk(11, "Exhibit 31 certification of indemnification."),
    ]

    def fake_load(table: str, doc_name: str) -> list[Document]:
        loads.append(doc_name)
        return chunks

    monkeypatch.setattr(bm25, "load_chunks", fake_load)
    monkeypatch.setattr(bm25, "exhibit_cutoff", lambda doc_name: 10)
    bm25.filing_index.cache_clear()
    yield loads
    bm25.filing_index.cache_clear()


def use_settings(monkeypatch: pytest.MonkeyPatch, *modules, **update) -> None:
    settings = bm25.get_settings().model_copy(update=update)
    for module in modules:
        monkeypatch.setattr(module, "get_settings", lambda: settings)


def test_exhibit_pages_are_never_returned(filing, monkeypatch):
    use_settings(monkeypatch, bm25, exclude_exhibits=True)

    results = bm25.retrieve("indemnification agreement", "DOC", k=10)

    assert [d.metadata["page"] for d in results] == [2]


def test_exhibits_are_searched_when_not_excluded(filing, monkeypatch):
    use_settings(monkeypatch, bm25, exclude_exhibits=False)

    assert bm25.retrieve("indemnification agreement", "DOC", k=1)[0].metadata["page"] == 10


def test_index_is_built_once_per_filing(filing, monkeypatch):
    use_settings(monkeypatch, bm25, exclude_exhibits=True)

    bm25.retrieve("revenue", "DOC", k=5)
    bm25.retrieve("legal", "DOC", k=5)

    assert filing == ["DOC"]


def test_retrieve_needs_a_filing():
    with pytest.raises(ValueError, match="needs a doc_name"):
        bm25.retrieve("revenue", None, k=5)


# hybrid merging in the expand retriever


@pytest.fixture
def hybrid(monkeypatch: pytest.MonkeyPatch):
    """HYBRID=bm25 with vector search, BM25 and rrf_merge recorded; no PDFs."""
    calls: dict[str, list] = {"bm25": [], "rrf": []}
    vector_hits = [chunk(7, "vector a"), chunk(3, "shared")]
    keyword_hits = [chunk(3, "shared"), chunk(9, "keyword b")]

    def fake_bm25(question, doc_name, k):
        calls["bm25"].append((question, doc_name, k))
        return keyword_hits

    real_rrf = expand.rrf_merge

    def spy_rrf(ranked_lists):
        calls["rrf"].append(ranked_lists)
        return real_rrf(ranked_lists)

    use_settings(monkeypatch, expand, hybrid="bm25", bm25_k=7, query_rewrite="none", max_pages=5)
    monkeypatch.setattr(expand.naive, "retrieve", lambda q, doc, k: vector_hits)
    monkeypatch.setattr(expand.bm25, "retrieve", fake_bm25)
    monkeypatch.setattr(expand, "rrf_merge", spy_rrf)
    monkeypatch.setattr(pages_module, "load_pdf", lambda name: [])
    pages_module.page_texts.cache_clear()
    yield calls, vector_hits, keyword_hits
    pages_module.page_texts.cache_clear()


def test_hybrid_merges_vector_and_bm25_lists_with_rrf(hybrid):
    calls, vector_hits, keyword_hits = hybrid

    result = expand.search("q", "DOC", k=10)

    assert calls["bm25"] == [("q", "DOC", 7)]
    assert calls["rrf"] == [[vector_hits, keyword_hits]]  # vector first: it wins ties
    assert [d.metadata["page"] for d in result.docs] == [3, 7, 9]  # "shared" is in both
    assert result.queries == ["q"]


def test_no_hybrid_does_not_run_bm25(hybrid, monkeypatch):
    calls, _, _ = hybrid
    use_settings(monkeypatch, expand, hybrid="none", query_rewrite="none", max_pages=5)

    result = expand.search("q", "DOC", k=10)

    assert calls == {"bm25": [], "rrf": []}
    assert [d.metadata["page"] for d in result.docs] == [7, 3]
