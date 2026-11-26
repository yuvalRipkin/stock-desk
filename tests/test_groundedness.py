"""Tests for the groundedness checks. Pure: page text is passed in, no PDFs or LLM calls."""

import pytest

from stockdesk.evals.groundedness import (
    answer_figures,
    check_record,
    cited_pages,
    summarize,
)
from stockdesk.evals.runner import QuestionRecord, summarize_run

PAGES = {
    47: "Net revenue $ 23,601 $ 16,434\nOperating income 1,264 3,648\n(In millions)",
    55: "Cash and cash equivalents $ 4,835\nShort-term investments 1,020\nTotal current "
    "liabilities 6,369\nOther income (expense), net (88)",
    90: "Not retrieved: goodwill 24,262",
}


def record(answer: str, tool_calls: list[dict] | None = None, **overrides) -> dict:
    return {
        "id": "q1",
        "doc_name": "AMD_2022_10K",
        "question": "What was AMD's FY2022 operating margin? Is it above 3%?",
        "answer": answer,
        "retrieved_pages": [47, 55],
        "correct": True,
        "error": None,
        "tool_calls": tool_calls,
        **overrides,
    }


def calc(expression: str, result: str) -> dict:
    return {"name": "calculate", "args": {"expressions": [expression]}, "pages": [],
            "result": f"{expression} = {result}"}  # fmt: skip


def check(answer: str, tool_calls: list[dict] | None = None, **overrides) -> dict:
    result = check_record(record(answer, tool_calls, **overrides), PAGES)
    assert result is not None
    return result


# number grounding


def test_grounded_number_on_retrieved_page():
    g = check("Net revenue FY2022: $23,601 million (page 47).\nAnswer: $23,601 million.")

    assert g["numbers"] == 2
    assert g["untraceable_numbers"] == []
    assert not g["flagged"]


def test_scaled_number_matches_page_in_millions():
    g = check("Net revenue was $23.6 billion and cash was $4.8bn (pages 47, 55).")

    assert g["untraceable_numbers"] == []


def test_rounded_number_matches_calculate_result_to_stated_precision():
    g = check(
        "Operating margin = 5.36% (page 47).",
        [calc("1264 / 23601 * 100", "5.3557052667")],
    )

    assert g["untraceable_numbers"] == []


def test_rounding_beyond_stated_precision_is_not_grounded():
    g = check("Operating margin = 5.37% (page 47).", [calc("1264 / 23601 * 100", "5.3557")])

    assert g["untraceable_numbers"] == ["5.37%"]


def test_percentage_matches_ratio_from_calculate():
    g = check("Margin: 5.36% (page 47).", [calc("1264 / 23601", "0.0535570527")])

    assert g["untraceable_numbers"] == []


def test_invented_number_is_flagged():
    g = check("Net revenue was $24,912 million (page 47).")

    assert g["untraceable_numbers"] == ["$24,912 million"]
    assert g["flagged"]


def test_number_from_unretrieved_page_is_not_grounded():
    g = check("Goodwill was $24,262 million (page 47).")

    assert g["untraceable_numbers"] == ["$24,262 million"]


def test_self_computed_number_without_calculate_is_not_grounded():
    g = check("Quick assets = 4,835 + 1,020 = $5,855 million (page 55).")

    assert g["untraceable_numbers"] == ["$5,855 million"]


def test_question_numbers_and_negatives_in_parentheses_are_grounded():
    g = check("Other income was $(88) million, a loss, and the margin is above 3% (page 55).")

    assert g["untraceable_numbers"] == []


def test_source_percentage_matches_its_ratio():
    g = check("The 3% threshold is 0.03 of revenue (page 47).")

    assert g["untraceable_numbers"] == []


def test_years_page_numbers_and_list_numbering_are_ignored():
    figures = answer_figures(
        "1. Figures:\n- FY2022 net revenue in 2022: $23,601 million (pages 47, 55)\n"
        "2. Calculation: 1264 / 23601 * 100 per the 10-K"
    )

    assert [f.text for f in figures] == ["$23,601 million", "1264", "23601"]


# citations


def test_cited_page_not_retrieved_is_invalid():
    g = check("Net revenue: $23,601 million (pages 47, 31).")

    assert g["cited_pages"] == [47, 31]
    assert g["invalid_citations"] == [31]
    assert g["untraceable_numbers"] == []
    assert g["flagged"]


def test_citation_whose_page_holds_none_of_the_figures_fails():
    g = check("Net revenue: $23,601 million (page 55).")

    assert g["invalid_citations"] == []
    assert g["figure_on_cited_page"] is False
    assert g["flagged"]


def test_citation_without_figures_only_needs_a_retrieved_page():
    g = check("AMD sells CPUs and GPUs (page 47).")

    assert g["figure_on_cited_page"] is None
    assert not g["flagged"]


def test_cited_pages_parses_ranges_and_lists():
    pages, _ = cited_pages("(page 3) and (pages 47-49, 52 and 60); see p. 7")

    assert pages == [3, 47, 48, 49, 52, 60, 7]


# tool claims


def test_false_tool_claim_without_a_calculate_call():
    g = check("Share = 1,264 / 23,601 = 0.0536 (calculate: 1,264 / 23,601) (page 47).")

    assert g["tool_claims"] == 1
    assert len(g["false_tool_claims"]) == 1
    assert g["flagged"]


def test_false_tool_claim_with_a_result_calculate_never_returned():
    g = check(
        "Margin: calculate(1264 / 23601 * 100) = 5.81% (page 47).",
        [calc("1264 / 23601 * 100", "5.3557052667")],
    )

    assert len(g["false_tool_claims"]) == 1


def test_valid_tool_claim_matches_a_calculate_result():
    g = check(
        "Margin: calculate(1264 / 23601 * 100) = 5.3557052667% (page 47).",
        [calc("1264 / 23601 * 100", "5.3557052667")],
    )

    assert g["tool_claims"] == 1
    assert g["false_tool_claims"] == []
    assert not g["flagged"]


def test_prose_about_calculation_is_not_a_tool_claim():
    g = check("2. Calculation: none needed; the margin is calculated on page 47 as 5.4%.")

    assert g["tool_claims"] == 0


def test_errored_record_is_not_checked():
    assert check_record(record(None, error="answer: TimeoutError"), PAGES) is None


# summaries


def test_summarize_splits_correct_and_wrong():
    grounded = {**record("x"), "groundedness": check("Revenue: $23,601 million (page 47).")}
    invented = {
        **record("x", correct=False),
        "groundedness": check("Revenue: $24,912 million (page 31)."),
    }

    s = summarize([grounded, invented])

    assert s["grounded_number_rate_correct"] == 1.0
    assert s["grounded_number_rate_wrong"] == 0.0
    assert s["invalid_citation_rate_wrong"] == 1.0
    assert s["false_tool_claim_rate_correct"] is None
    assert (s["flagged_rate_correct"], s["flagged_rate_wrong"]) == (0.0, 1.0)


def test_summarize_run_includes_groundedness_and_tolerates_old_records():
    base = dict(
        id="q1", doc_name="D", question="q", gold_answer="g", answer="a", retrieved_pages=[1],
        correct=True, judge_reason=None, page_hit=True, evidence_hit=True, refusal=False,
        distinct_pages=1, context_chars=10, latency_ms=1.0, error=None,
    )  # fmt: skip
    checked = QuestionRecord(**base, groundedness=check("Revenue: $23,601 million (page 47)."))

    assert summarize_run([checked])["flagged_rate_correct"] == 0.0
    assert summarize_run([QuestionRecord(**base)])["flagged_rate_correct"] is None


@pytest.mark.parametrize("text", ["10-K", "FY22", "Q4", "COVID-19"])
def test_labels_glued_to_letters_are_not_numbers(text):
    assert answer_figures(f"See the {text} filing.") == []
