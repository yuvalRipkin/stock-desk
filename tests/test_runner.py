"""Tests for the runner's pure aggregation functions, with fake records (no LLM or DB)."""

import pytest

from stockdesk.evals.runner import QuestionRecord, aggregate, percentile, summarize_run


def record(**overrides) -> QuestionRecord:
    fields = dict(
        id="q",
        doc_name="DOC",
        question="?",
        gold_answer="1",
        answer="1",
        retrieved_pages=[1, 2, 2],
        correct=True,
        judge_reason="ok",
        page_hit=True,
        evidence_hit=True,
        refusal=False,
        distinct_pages=2,
        context_chars=1000,
        latency_ms=100.0,
        error=None,
    )
    fields.update(overrides)
    return QuestionRecord(**fields)


def error_record() -> QuestionRecord:
    return record(
        answer=None,
        retrieved_pages=[],
        correct=False,
        judge_reason=None,
        page_hit=False,
        evidence_hit=False,
        distinct_pages=0,
        context_chars=0,
        latency_ms=None,
        error="answer: RagError: boom",
    )


# percentile


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([10, 20, 30, 40], 95) == pytest.approx(38.5)


def test_percentile_single_and_empty():
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) is None


# summarize_run


def test_summarize_run_rates_and_latency():
    records = [
        record(latency_ms=100.0, context_chars=2000),
        record(correct=False, page_hit=False, latency_ms=200.0, distinct_pages=4),
        record(correct=False, evidence_hit=False, refusal=True, latency_ms=300.0, distinct_pages=3),
        record(latency_ms=400.0, distinct_pages=3),
    ]
    s = summarize_run(records)

    assert s["n"] == 4
    assert s["accuracy"] == 0.5
    assert s["page_hit_rate"] == 0.75
    assert s["evidence_hit_rate"] == 0.75
    assert s["refusal_rate"] == 0.25
    assert s["avg_distinct_pages"] == 3.0
    assert s["avg_context_chars"] == 1250
    assert s["latency_p50_ms"] == 250.0
    assert s["latency_p95_ms"] == pytest.approx(385.0)
    assert s["error_count"] == 0


def test_summarize_run_errors_count_as_wrong_but_skip_latency_and_pages():
    s = summarize_run([record(latency_ms=100.0, distinct_pages=4), error_record()])

    assert s["accuracy"] == 0.5
    assert s["page_hit_rate"] == 0.5
    assert s["error_count"] == 1
    assert s["avg_distinct_pages"] == 4.0
    assert s["avg_context_chars"] == 1000
    assert s["latency_p50_ms"] == 100.0


def test_summarize_run_all_errors():
    s = summarize_run([error_record(), error_record()])

    assert s["accuracy"] == 0.0
    assert s["error_count"] == 2
    assert s["avg_distinct_pages"] is None
    assert s["avg_context_chars"] is None
    assert s["latency_p50_ms"] is None


# aggregate


def test_aggregate_mean_and_sample_std():
    summaries = [
        {"accuracy": 0.4, "error_count": 0},
        {"accuracy": 0.5, "error_count": 1},
        {"accuracy": 0.6, "error_count": 2},
    ]
    agg = aggregate(summaries)

    assert agg["accuracy"]["mean"] == pytest.approx(0.5)
    assert agg["accuracy"]["std"] == pytest.approx(0.1)
    assert agg["error_count"] == {"mean": 1, "std": 1.0}


def test_aggregate_single_run_has_zero_std():
    agg = aggregate([summarize_run([record()])])

    assert agg["accuracy"] == {"mean": 1.0, "std": 0.0}


def test_aggregate_skips_none_values():
    agg = aggregate([{"latency_p50_ms": None}, {"latency_p50_ms": 120.0}])

    assert agg["latency_p50_ms"] == {"mean": 120.0, "std": 0.0}
    assert agg["accuracy"] == {"mean": None, "std": None}


def test_summarize_run_averages_answer_chars_over_answered_questions():
    records = [record(answer_chars=100), record(answer_chars=300), error_record()]

    assert summarize_run(records)["avg_answer_chars"] == 200


def test_summarize_run_answer_chars_missing_in_old_records():
    assert summarize_run([record()])["avg_answer_chars"] is None


def timing(**overrides) -> dict:
    t = dict(
        model_calls=2,
        model_ms=2000.0,
        model_call_ms=[800.0, 1200.0],
        searches=2,
        search_ms=500.0,
        embed_ms=300.0,
        db_ms=100.0,
        pages_ms=60.0,
        pdf_parses=0,
        pdf_parse_ms=0.0,
        calculate_calls=1,
        calculate_ms=0.5,
    )
    t.update(overrides)
    return t


def test_summarize_run_averages_agent_timing():
    records = [
        record(timing=timing(), input_tokens=1000, output_tokens=100, cache_read_tokens=0),
        record(
            timing=timing(model_calls=4, model_ms=6000.0, searches=1, embed_ms=200.0, pdf_parses=1),
            input_tokens=3000,
            output_tokens=300,
            cache_read_tokens=2048,
        ),
        error_record(),
    ]
    s = summarize_run(records)

    assert s["avg_model_calls"] == 3
    assert s["avg_model_ms"] == 4000
    assert s["avg_model_ms_per_call"] == pytest.approx(8000 / 6)
    assert s["avg_searches"] == 1.5
    assert s["avg_search_embed_ms"] == pytest.approx(500 / 3)  # per search, not per question
    assert s["avg_search_db_ms"] == pytest.approx(200 / 3)
    assert s["avg_search_pages_ms"] == pytest.approx(120 / 3)
    assert s["pdf_parses"] == 1
    assert s["avg_calculate_ms"] == 0.5
    assert (s["avg_input_tokens"], s["avg_output_tokens"]) == (2000, 200)
    assert s["avg_cache_read_tokens"] == 1024


def test_summarize_run_timing_is_none_without_agent_records():
    s = summarize_run([record()])

    assert s["avg_model_calls"] is None
    assert s["avg_search_embed_ms"] is None
    assert s["pdf_parses"] is None
    assert s["avg_cache_read_tokens"] is None


def test_summarize_run_calculate_ms_is_none_without_calculate_calls():
    s = summarize_run([record(timing=timing(calculate_calls=0, calculate_ms=0.0))])

    assert s["avg_calculate_ms"] is None
    assert s["avg_model_calls"] == 2
