"""Tests for parsing the evidence checklist of AGENT_PREFETCH_PROMPT=v2 (pure: no model)."""

from langchain_core.messages import AIMessage

from stockdesk.agent.checklist import Item, parse_checklist, summarize
from stockdesk.prompts import AGENT_PREFETCH_PROMPT_V2


def test_parses_items_with_pages_and_missing_items():
    text = """Checklist:
- Net sales FY2022: page 54
- Cost of sales FY2022: MISSING
- Gross profit FY2021: pages 54, 55

Figures: ..."""

    assert parse_checklist(text) == [
        Item("Net sales FY2022: page 54", True),
        Item("Cost of sales FY2022: MISSING", False),
        Item("Gross profit FY2021: pages 54, 55", True),
    ]


def test_tolerates_markdown_and_numbering():
    text = """Some preamble.

**Checklist:**

1. **Customer deposits 2022**: page 97
2) Long-term debt 2022: missing
* Total liabilities: (page 97)
Answer: x"""

    items = parse_checklist(text)

    assert [i.has_page for i in items] == [True, False, True]
    assert items[0].text.startswith("Customer deposits 2022")


def test_no_header_means_no_checklist():
    assert parse_checklist("Figures:\n- Net sales: page 54\nAnswer: 1") is None


def test_header_without_items_is_an_empty_checklist():
    assert parse_checklist("Checklist:\nAll figures are on page 54.") == []


def test_the_prompt_format_example_parses_as_one_found_and_one_missing():
    block = AGENT_PREFETCH_PROMPT_V2.split("format:\n", 1)[1]
    # the format lines as the prompt shows them, with a real page number for "<n>"
    example = "\n".join(line.strip() for line in block.splitlines()[:3]).replace("<n>", "12")

    items = parse_checklist(example)

    assert [i.has_page for i in items] == [True, False]


def search(query: str) -> dict:
    return {"name": "search_filings", "args": {"query": query}, "id": query}


def test_summarize_missing_items_searched_in_the_same_step():
    replies = [
        AIMessage(
            content="Checklist:\n- Revenue 2022: page 4\n- Debt 2022: MISSING\n- Cash: MISSING",
            tool_calls=[search("long-term debt"), search("cash and cash equivalents")],
        ),
        AIMessage(
            content="Checklist:\n- Revenue 2022: page 4\n- Debt 2022: page 9\n- Cash: MISSING"
        ),
    ]

    assert summarize(replies) == {
        "written": True,
        "items": 3,
        "missing": ["Debt 2022: MISSING", "Cash: MISSING"],
        "searches_same_step": 2,
        "searched_after": True,
        "final_missing": ["Cash: MISSING"],
    }


def test_summarize_missing_items_with_no_search_afterwards():
    replies = [AIMessage(content="Checklist:\n- Debt 2022: MISSING\n\nAnswer: about $40 billion")]

    s = summarize(replies)

    assert s["missing"] == ["Debt 2022: MISSING"]
    assert (s["searched_after"], s["searches_same_step"]) == (False, 0)
    assert s["final_missing"] == ["Debt 2022: MISSING"]


def test_summarize_a_search_in_a_later_step_counts_as_searched_after():
    replies = [
        AIMessage(content="Checklist:\n- Debt 2022: MISSING"),
        AIMessage(content="", tool_calls=[search("long-term debt")]),
        AIMessage(content="Answer: x"),
    ]

    s = summarize(replies)

    assert (s["searches_same_step"], s["searched_after"]) == (0, True)


def test_summarize_without_any_checklist():
    s = summarize([AIMessage(content="Answer: 42 (page 3)")])

    assert s["written"] is False
    assert (s["items"], s["missing"], s["searched_after"]) == (0, [], False)
