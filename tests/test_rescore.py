"""Tests for rescoring saved results. The judge is mocked: no LLM calls."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stockdesk.evals import rescore as rescore_module
from stockdesk.evals.judge import Verdict
from stockdesk.evals.metrics import METRICS_VERSION
from stockdesk.evals.rescore import AlreadyRescored, SummaryMismatch, as_record, rescore
from stockdesk.evals.runner import QuestionRecord, aggregate, summarize_run

HEDGED = "Customer deposits were the largest liability. Not found in the filing."


def record(id_: str, answer: str, refusal: bool, correct: bool) -> dict:
    return asdict(
        QuestionRecord(
            id=id_,
            doc_name="DOC",
            question=f"question {id_}",
            gold_answer="Customer deposits",
            answer=answer,
            retrieved_pages=[1],
            correct=correct,
            judge_reason="v1",
            page_hit=True,
            evidence_hit=True,
            refusal=refusal,
            distinct_pages=1,
            context_chars=100,
            latency_ms=10.0,
            error=None,
        )
    )


def v1_results(run_records: list[list[dict]]) -> dict:
    """A results file as the v1 runner wrote it (refusal = phrase anywhere)."""
    runs = [
        {"summary": summarize_run([as_record(r) for r in recs]), "records": recs}
        for recs in run_records
    ]
    return {
        "name": "test",
        "config": {"judge_model": "gpt-5.4"},
        "aggregate": aggregate([r["summary"] for r in runs]),
        "runs": runs,
    }


@pytest.fixture
def results() -> dict:
    run = [
        record("a", "Customer deposits (page 97).", refusal=False, correct=True),
        record("b", HEDGED, refusal=True, correct=False),  # v1 false positive
        record("c", "Not found in the filing.", refusal=True, correct=False),  # true refusal
        record("d", "Long-term debt (page 138).", refusal=False, correct=False),
    ]
    return v1_results([run, [dict(r) for r in run]])


def test_rejudges_only_records_whose_refusal_changed(results: dict) -> None:
    judge = MagicMock(return_value=Verdict(correct=True, reason="Same liability."))

    out = rescore(results, judge)

    assert judge.call_count == 2  # record "b" in each of the two runs
    judge.assert_called_with("question b", "Customer deposits", HEDGED)
    assert [(r["run"], r["id"]) for r in out["rescore"]["rejudged"]] == [(1, "b"), (2, "b")]
    b = out["runs"][0]["records"][1]
    assert (b["refusal"], b["correct"], b["judge_reason"]) == (False, True, "Same liability.")
    c = out["runs"][0]["records"][2]
    assert (c["refusal"], c["correct"]) == (True, False)  # pure refusal untouched


def test_recomputes_summaries_and_aggregate(results: dict) -> None:
    out = rescore(results, MagicMock(return_value=Verdict(correct=True, reason="ok")))

    assert results["aggregate"]["accuracy"]["mean"] == 0.25
    assert results["aggregate"]["refusal_rate"]["mean"] == 0.5
    assert out["runs"][0]["summary"]["accuracy"] == 0.5
    assert out["runs"][0]["summary"]["refusal_rate"] == 0.25
    assert out["aggregate"]["accuracy"] == {"mean": 0.5, "std": 0.0}
    assert out["rescore"]["previous_aggregate"]["accuracy"]["mean"] == 0.25


def test_sets_metrics_version_and_keeps_input_unchanged(results: dict) -> None:
    before = json.dumps(results, sort_keys=True)

    out = rescore(results, MagicMock(return_value=Verdict(correct=False, reason="no")))

    assert out["metrics_version"] == METRICS_VERSION
    assert out["rescore"]["from_metrics_version"] == 1
    assert list(out).index("metrics_version") == list(out).index("config") + 1
    assert json.dumps(results, sort_keys=True) == before


def test_refuses_files_already_at_current_version(results: dict) -> None:
    results["metrics_version"] = METRICS_VERSION
    judge = MagicMock()

    with pytest.raises(AlreadyRescored):
        rescore(results, judge)
    judge.assert_not_called()


def test_refuses_when_stored_summary_does_not_reproduce(results: dict) -> None:
    results["runs"][0]["summary"]["accuracy"] = 0.9

    with pytest.raises(SummaryMismatch):
        rescore(results, MagicMock())


def test_old_files_without_context_chars_keep_their_summary_shape() -> None:
    run = [record("a", HEDGED, refusal=True, correct=False)]
    for r in run:
        del r["context_chars"]
    results = v1_results([run])
    for key in ("avg_context_chars",):
        del results["runs"][0]["summary"][key]
        del results["aggregate"][key]

    out = rescore(results, MagicMock(return_value=Verdict(correct=True, reason="ok")))

    assert "avg_context_chars" not in out["runs"][0]["summary"]
    assert "avg_context_chars" not in out["aggregate"]
    assert out["aggregate"]["accuracy"]["mean"] == 1.0


def test_cli_writes_file_and_refuses_second_rescore(
    results: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps(results))
    fake_judge = MagicMock(return_value=Verdict(correct=True, reason="ok"))
    monkeypatch.setattr(rescore_module, "judge", fake_judge)

    assert rescore_module.main([str(path)]) == 0
    assert fake_judge.call_count == 2  # the mock, not the real judge, was used
    assert json.loads(path.read_text())["metrics_version"] == METRICS_VERSION
    assert list(tmp_path.iterdir()) == [path]  # no leftover .part file

    assert rescore_module.main([str(path)]) == 1


def test_as_record_ignores_fields_of_removed_features():
    raw = {**record("a", "Customer deposits", refusal=False, correct=True), "rerank_ms": 812.5}

    assert as_record(raw) == as_record(record("a", "Customer deposits", False, True))
