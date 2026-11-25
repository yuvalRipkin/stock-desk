"""Tests for 10-K section detection and section-header chunking, on fake page text."""

import pytest

from stockdesk.config import get_settings
from stockdesk.ingestion import ingestion
from stockdesk.ingestion.sections import COVER, EXHIBITS, parse_item, split_sections
from stockdesk.store import SECTION_COLUMN, metadata_columns

ITEM_7 = (
    "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations"
)
ITEM_8 = "Item 8. Financial Statements and Supplementary Data"
ITEMS = ["1", "1A", "1B", "2", "3", "4", "5", "7", "7A", "8"]

TOC = "Table of Contents\n" + "\n".join(
    f"Item {k}. Some Title {10 + i}" for i, k in enumerate(ITEMS)
)


def labels(pages: dict[int, str], cutoff: int | None = None) -> dict[int, list[str]]:
    return {p: [lab for lab, _ in segs] for p, segs in split_sections(pages, cutoff).items()}


# parse_item


@pytest.mark.parametrize(
    "line, label",
    [
        ("ITEM 1A. RISK FACTORS", "Item 1A. Risk Factors"),
        ("Item 1. - Business", "Item 1. Business"),
        ("PART II Item 6. [Reserved]", "Item 6. [Reserved]"),
        ("ITEM 7.    MANAGEMENT’S DISCUSSION AND ANALYSIS", ITEM_7),  # standard name, not printed
    ],
)
def test_parse_item_uses_standard_sec_names(line: str, label: str):
    item = parse_item(line)

    assert item is not None and item.label == label


def test_parse_item_combined_heading():
    item = parse_item("ITEMS 10, 11, 12 and 13. DIRECTORS, EXECUTIVE OFFICERS")

    assert item is not None
    assert item.key == "13"
    assert item.label.startswith("Items 10–13. Directors, Executive Officers")


@pytest.mark.parametrize(
    "line",
    [
        "Item 8 of this Annual Report on Form 10-K.",  # no period after the number
        "see Item 7. Management's Discussion and Analysis",  # not at line start
        "ITEM 15(a)(3)",
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk 50",  # TOC page no.
        "Item 17. Not an Item",
    ],
)
def test_parse_item_rejects_non_headings(line: str):
    assert parse_item(line) is None


# split_sections: Items


def test_table_of_contents_is_not_a_heading():
    pages = {0: "Cover", 1: TOC, 2: "Item 1. Business\nWe make chips."}

    assert labels(pages) == {0: [COVER], 1: [COVER], 2: ["Item 1. Business"]}


def test_references_to_items_are_not_headings():
    pages = {
        0: "Item 1. Business\nWe make chips.",
        1: "see Item 7. Management's Discussion for more.\n"
        "Our results are described in\nItem 8 of this Annual Report on Form 10-K.",
        2: "Item 7. MD&A\nRevenue grew.",
        3: "Item 1A. Risk Factors\nan out-of-order reference to Item 1A, not a heading",
    }

    assert labels(pages) == {
        0: ["Item 1. Business"],
        1: ["Item 1. Business"],
        2: [ITEM_7],
        3: [ITEM_7],
    }


def test_page_splits_at_headings_and_keeps_all_text():
    page = "Item 2. Properties\nWe own offices.\nItem 3. Legal Proceedings\nWe are sued.\n"
    pages = {0: "Item 1. Business\nchips", 1: "tail of business\n" + page}

    segments = split_sections(pages)[1]

    assert [lab for lab, _ in segments] == [
        "Item 1. Business",
        "Item 2. Properties",
        "Item 3. Legal Proceedings",
    ]
    assert segments[2][1] == "Item 3. Legal Proceedings\nWe are sued.\n"
    assert "".join(text for _, text in segments) == pages[1]


def test_short_section_is_kept_separate():
    pages = {0: "Item 9. Changes in Accountants\nNone.\nItem 9A. Controls and Procedures\nOk."}

    assert labels(pages)[0] == [
        "Item 9. Changes in and Disagreements with Accountants on Accounting and Financial "
        "Disclosure",
        "Item 9A. Controls and Procedures",
    ]


def test_cover_page_and_exhibits():
    pages = {0: "Annual report cover", 1: "Item 1. Business\nchips", 2: "EXHIBIT 10.1\nPlan"}

    assert labels(pages, cutoff=2) == {0: [COVER], 1: ["Item 1. Business"], 2: [EXHIBITS]}


# split_sections: statements inside Items 7 and 8


def statements_filing() -> dict[int, str]:
    return {
        0: "Item 7. MD&A\nWe sold more.",
        1: f"{ITEM_8.upper()}\nAcme Inc.\nConsolidated Statements of Operations\nRevenue 10",
        2: "Table of Contents\nAcme Inc.\nCONSOLIDATED BALANCE SHEETS\nCustomer deposits 110",
        3: "Acme Inc.\nConsolidated Balance Sheets (continued)\nEquity 5",
        4: "Notes to Consolidated Financial Statements\nNote 1. Basis of presentation",
        5: "Note 22. Subsidiaries\nConsolidated Statements of Income\nSubsidiary revenue 3",
        6: "end of notes\nReport of Independent Registered Public Accounting Firm\nOpinion",
        7: "Item 9. Changes in Accountants\nNone.",
    }


def test_statement_titles_become_sub_sections():
    got = labels(statements_filing())

    assert got[1] == [f"{ITEM_8} > Consolidated Statements of Operations"]
    assert got[2] == [f"{ITEM_8} > Consolidated Balance Sheets"]
    assert got[3] == [f"{ITEM_8} > Consolidated Balance Sheets"]  # (continued): no new section
    assert got[4] == [f"{ITEM_8} > Notes to Consolidated Financial Statements"]


def test_statement_titles_inside_the_notes_are_ignored():
    assert labels(statements_filing())[5] == [
        f"{ITEM_8} > Notes to Consolidated Financial Statements"
    ]


def test_auditor_report_after_the_notes_is_a_section():
    assert labels(statements_filing())[6] == [
        f"{ITEM_8} > Notes to Consolidated Financial Statements",
        f"{ITEM_8} > Report of Independent Registered Public Accounting Firm",
    ]


def test_new_item_ends_the_statement_sub_section():
    assert labels(statements_filing())[7][0].startswith("Item 9. Changes")


def test_caption_and_bare_item_heading_join_the_statement():
    # "ITEM 8" and the company name have nothing of their own: one segment, the statement's
    segments = split_sections(statements_filing())[1]

    assert len(segments) == 1
    assert segments[0][1].startswith("ITEM 8.")


def test_statement_title_must_be_near_the_top():
    body = "\n".join(f"line {i}" for i in range(20))
    pages = {0: f"{ITEM_8}\n{body}\nConsolidated Balance Sheets\nmore"}

    assert labels(pages)[0] == [ITEM_8]


def test_statement_titles_outside_items_7_and_8_are_ignored():
    pages = {0: "Item 1. Business\nConsolidated Balance Sheets\nchips"}

    assert labels(pages)[0] == ["Item 1. Business"]


def test_statements_inside_item_7_stay_there():
    # PepsiCo prints its statements inside Item 7
    pages = {
        0: "Item 7. MD&A\nRevenue grew.",
        1: "PepsiCo, Inc.\nConsolidated Balance Sheet\nCash 5",
    }

    assert labels(pages)[1] == [f"{ITEM_7} > Consolidated Balance Sheet"]


# chunking


@pytest.fixture
def small_chunks(monkeypatch: pytest.MonkeyPatch):
    settings = get_settings().model_copy(
        update={"chunk_size": 40, "chunk_overlap": 0, "chunking": "section_header"}
    )
    monkeypatch.setattr(ingestion, "get_settings", lambda: settings)
    return settings


def page_docs(pages: dict[int, str]):
    from langchain_core.documents import Document

    return [
        Document(page_content=t, metadata={"doc_name": "DOC", "page": p}) for p, t in pages.items()
    ]


def test_section_header_chunks(small_chunks):
    pages = {
        0: "Item 1. Business\nWe make chips and sell them worldwide.",
        1: "More about chips.\nItem 1A. Risk Factors\nDemand may fall.",
    }

    chunks = ingestion.split_with_section_headers(page_docs(pages))

    for c in chunks:
        assert c.page_content.startswith(f"[{c.metadata['section']}]\n")
        body = c.page_content.split("\n", 1)[1]
        assert body in pages[c.metadata["page"]]  # page metadata stays exact
    assert [(c.metadata["page"], c.metadata["section"]) for c in chunks] == [
        (0, "Item 1. Business"),
        (0, "Item 1. Business"),
        (1, "Item 1. Business"),
        (1, "Item 1A. Risk Factors"),
    ]


def test_no_chunk_crosses_a_section(small_chunks):
    pages = {0: "Item 2. Properties\nOffices.\nItem 3. Legal Proceedings\nLawsuits."}

    chunks = ingestion.split_with_section_headers(page_docs(pages))

    assert [c.page_content for c in chunks] == [
        "[Item 2. Properties]\nItem 2. Properties\nOffices.",
        "[Item 3. Legal Proceedings]\nItem 3. Legal Proceedings\nLawsuits.",
    ]


def test_exhibit_pages_are_labeled(small_chunks):
    pages = {0: "Item 1. Business\nchips", 1: "EXHIBIT 21\nSubsidiaries"}

    chunks = ingestion.split_with_section_headers(page_docs(pages), exhibit_cutoff=1)

    assert chunks[-1].metadata["section"] == EXHIBITS
    assert chunks[-1].page_content.startswith("[Exhibits]\n")


# store columns


@pytest.mark.parametrize("chunking, has_section", [("plain", False), ("section_header", True)])
def test_section_column_only_for_section_tables(monkeypatch, chunking: str, has_section: bool):
    import stockdesk.store as store

    settings = get_settings().model_copy(update={"chunking": chunking})
    monkeypatch.setattr(store, "get_settings", lambda: settings)

    assert (SECTION_COLUMN in metadata_columns()) is has_section
