"""Tests for exhibit-page detection and exclusion. No PDFs or DB: texts and the store are fakes."""

import pytest
from langchain_core.documents import Document

from stockdesk.retrieval import naive
from stockdesk.retrieval.exhibits import find_exhibit_cutoff

SIGNATURES = "SIGNATURES Pursuant to the requirements of Section 13 or 15(d) of the Act..."


# find_exhibit_cutoff


def test_cutoff_is_first_exhibit_after_signatures():
    pages = {
        0: "Table of Contents PART I Item 1. Business",
        1: "Table of Contents Item 15. Exhibits\n10.1 Plan (Exhibit 10.1 to Form 8-K)",
        2: f"Table of Contents {SIGNATURES}",
        3: "Table of Contents  Pursuant to the requirements ... signed below by the following",
        4: "Exhibit 21\nLIST OF SUBSIDIARIES",
        5: "more subsidiaries",
    }
    assert find_exhibit_cutoff(pages) == 4


def test_content_between_signatures_and_exhibits_is_kept():
    """Amcor's Schedule II and AmEx's appendix come after SIGNATURES but are 10-K content."""
    pages = {
        0: "Business",
        1: SIGNATURES,
        2: "Schedule II - Valuation and Qualifying Accounts",
        3: "Appendix STATISTICAL DISCLOSURE BY BANK HOLDING COMPANIES",
        4: "THIS DOCUMENT CONSTITUTES PART OF A PROSPECTUS. Exhibit 10.13 THE COMPANY PLAN",
    }
    assert find_exhibit_cutoff(pages) == 4


def test_exhibit_mentions_before_signatures_do_not_count():
    pages = {
        0: "Exhibit 10.1 incorporated by reference",
        1: SIGNATURES,
        2: "EXHIBIT 4.14 SECURITIES",
    }
    assert find_exhibit_cutoff(pages) == 2


def test_exhibit_word_deep_in_a_page_is_not_a_heading():
    pages = {0: SIGNATURES, 1: "x" * 300 + " see Exhibit 4.1", 2: "Exhibit 99 Press release"}
    assert find_exhibit_cutoff(pages) == 2


def test_whitespace_in_pdf_text_is_normalized():
    pages = {
        0: "Table of\nContents\n SIGNATURES\nPursuant to the requirements of Section 13",
        1: "Exhibit\n18.1 Letter",
    }
    assert find_exhibit_cutoff(pages) == 1


def test_no_signatures_or_no_exhibits_means_no_cutoff():
    assert find_exhibit_cutoff({0: "Business", 1: "Exhibit 21"}) is None
    assert find_exhibit_cutoff({0: SIGNATURES, 1: "Schedule II"}) is None
    assert find_exhibit_cutoff({}) is None


# exclusion in the vector search


def matches(filter_: dict | None, metadata: dict) -> bool:
    """Evaluate the PGVectorStore filter operators the retriever uses."""
    if filter_ is None:
        return True
    if "$and" in filter_:
        return all(matches(f, metadata) for f in filter_["$and"])
    if "$or" in filter_:
        return any(matches(f, metadata) for f in filter_["$or"])
    ops = {"$eq": lambda a, b: a == b, "$lt": lambda a, b: a < b}
    return all(
        ops[op](metadata[field], value)
        for field, cond in filter_.items()
        for op, value in cond.items()
    )


class FakeStore:
    """Returns stored chunks in their stored (rank) order, honoring the metadata filter."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.filters: list[dict | None] = []

    def similarity_search(self, question: str, k: int, filter: dict | None) -> list[Document]:
        self.filters.append(filter)
        return [d for d in self.docs if matches(filter, d.metadata)][:k]


CUTOFFS = {"PEP": 130, "AMD": 108}


def chunk(doc_name: str, page: int) -> Document:
    return Document(
        page_content=f"{doc_name} p{page}", metadata={"doc_name": doc_name, "page": page}
    )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    # Exhibit chunks rank highest, so an unfiltered search would return them first.
    docs = [chunk("PEP", p) for p in (488, 489, 31, 477, 3, 130, 4, 129, 70)]
    docs += [chunk("AMD", p) for p in (109, 42, 108, 4)]
    fake = FakeStore(docs)
    monkeypatch.setattr(naive, "get_store", lambda: fake)
    monkeypatch.setattr(naive, "exhibit_cutoff", CUTOFFS.get)
    monkeypatch.setattr(naive, "read_manifest", lambda: list(CUTOFFS))
    return fake


def set_exclude(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    settings = naive.get_settings().model_copy(update={"exclude_exhibits": value})
    monkeypatch.setattr(naive, "get_settings", lambda: settings)


def test_excluded_search_never_returns_exhibit_pages_and_still_returns_k(store, monkeypatch):
    set_exclude(monkeypatch, True)

    hits = naive.retrieve("q", "PEP", k=5)

    assert [h.metadata["page"] for h in hits] == [31, 3, 4, 129, 70]
    assert all(h.metadata["page"] < CUTOFFS["PEP"] for h in hits)


def test_excluded_search_across_all_filings(store, monkeypatch):
    set_exclude(monkeypatch, True)

    hits = naive.retrieve("q", None, k=20)

    assert hits
    assert all(h.metadata["page"] < CUTOFFS[h.metadata["doc_name"]] for h in hits)


def test_disabled_by_default_keeps_exhibit_pages(store, monkeypatch):
    set_exclude(monkeypatch, False)

    hits = naive.retrieve("q", "PEP", k=3)

    assert [h.metadata["page"] for h in hits] == [488, 489, 31]
    assert store.filters == [{"doc_name": {"$eq": "PEP"}}]


def test_no_detected_cutoff_means_no_page_filter(store, monkeypatch):
    set_exclude(monkeypatch, True)
    monkeypatch.setattr(naive, "exhibit_cutoff", lambda name: None)

    naive.retrieve("q", "PEP", k=3)

    assert store.filters == [{"doc_name": {"$eq": "PEP"}}]
