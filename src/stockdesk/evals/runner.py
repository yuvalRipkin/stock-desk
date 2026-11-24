"""Run the FinanceBench eval and save the results to results/<name>.json.

Usage:
    uv run python -m stockdesk.evals.runner --name 001_naive_1000 [--runs 3] [--force]

Questions run sequentially. A failing question is recorded with its error and the run continues.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from stockdesk.config import get_settings
from stockdesk.data import ROOT
from stockdesk.evals.dataset import EvalCase, load_cases
from stockdesk.evals.judge import judge
from stockdesk.evals.metrics import (
    METRICS_VERSION,
    distinct_pages,
    evidence_hit,
    is_refusal,
    page_hit,
)
from stockdesk.rag import answer, build_context

RESULTS_DIR = ROOT / "results"
SUMMARY_METRICS = [
    "accuracy",
    "page_hit_rate",
    "evidence_hit_rate",
    "refusal_rate",
    "avg_distinct_pages",
    "avg_context_chars",
    "latency_p50_ms",
    "latency_p95_ms",
    "error_count",
]


@dataclass
class QuestionRecord:
    id: str
    doc_name: str
    question: str
    gold_answer: str
    answer: str | None
    retrieved_pages: list[int]  # in retrieval order
    correct: bool
    judge_reason: str | None
    page_hit: bool
    evidence_hit: bool
    refusal: bool
    distinct_pages: int
    context_chars: int | None  # characters of context sent to the LLM; None in pre-005 files
    latency_ms: float | None  # latency of rag.answer() only
    error: str | None


# --- aggregation (pure) -----------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    """The q-th percentile (0-100) with linear interpolation between ranks; None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mean_or_none(values: list[float | int | None]) -> float | None:
    """Mean of the non-None values, or None if there are none."""
    present = [v for v in values if v is not None]
    return statistics.mean(present) if present else None


def summarize_run(records: list[QuestionRecord]) -> dict[str, float | int | None]:
    """Summary metrics for one run.

    Rates (accuracy, hit rates, refusal rate) are over ALL questions, so an errored question
    counts as incorrect and as a miss. Averages (distinct pages, context chars) and latency
    percentiles are over questions that got an answer, since errored ones have no retrieval
    or latency.
    """
    n = len(records)
    ok = [r for r in records if r.error is None]

    def rate(flag: str) -> float | None:
        return sum(getattr(r, flag) for r in records) / n if n else None

    latencies = [r.latency_ms for r in ok if r.latency_ms is not None]
    return {
        "n": n,
        "accuracy": rate("correct"),
        "page_hit_rate": rate("page_hit"),
        "evidence_hit_rate": rate("evidence_hit"),
        "refusal_rate": rate("refusal"),
        "avg_distinct_pages": statistics.mean(r.distinct_pages for r in ok) if ok else None,
        "avg_context_chars": mean_or_none([r.context_chars for r in ok]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "error_count": n - len(ok),
    }


def aggregate(summaries: list[dict]) -> dict[str, dict[str, float | None]]:
    """Mean and sample standard deviation of each summary metric across runs.

    std is 0.0 for a single run. Runs where a metric is None are left out of that metric.
    """
    result = {}
    for metric in SUMMARY_METRICS:
        values = [s[metric] for s in summaries if s.get(metric) is not None]
        result[metric] = {
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        }
    return result


# --- running ----------------------------------------------------------------------------------


def evaluate_case(case: EvalCase) -> QuestionRecord:
    """Answer, score and judge one case. Never raises: errors are recorded on the record."""
    record = QuestionRecord(
        id=case.id,
        doc_name=case.doc_name,
        question=case.question,
        gold_answer=case.gold_answer,
        answer=None,
        retrieved_pages=[],
        correct=False,
        judge_reason=None,
        page_hit=False,
        evidence_hit=False,
        refusal=False,
        distinct_pages=0,
        context_chars=0,
        latency_ms=None,
        error=None,
    )
    try:
        start = time.perf_counter()
        result = answer(case.question, case.doc_name)
        record.latency_ms = (time.perf_counter() - start) * 1000
    except Exception as e:  # noqa: BLE001 - one bad question must not stop the run
        record.error = f"answer: {type(e).__name__}: {e}"
        return record

    pages = [d.metadata["page"] for d in result.docs]
    record.answer = result.answer
    record.retrieved_pages = pages
    record.page_hit = page_hit(pages, case.evidence_pages)
    record.evidence_hit = evidence_hit([d.page_content for d in result.docs], case.evidence_texts)
    record.refusal = is_refusal(result.answer)
    record.distinct_pages = distinct_pages(pages)
    record.context_chars = len(build_context(result.docs))

    try:
        verdict = judge(case.question, case.gold_answer, result.answer)
        record.correct = verdict.correct
        record.judge_reason = verdict.reason
    except Exception as e:  # noqa: BLE001
        record.error = f"judge: {type(e).__name__}: {e}"
    return record


def git_state() -> dict[str, str | bool | None]:
    """The current commit, and whether tracked code differs from it (results/ ignored)."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain", "--", ".", ":(exclude)results")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def format_metric(stats: dict[str, float | None]) -> str:
    if stats["mean"] is None:
        return "n/a"
    return f"{stats['mean']:.3f} ± {stats['std']:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FinanceBench eval.")
    parser.add_argument("--name", required=True, help="experiment name, e.g. 001_naive_1000")
    parser.add_argument("--runs", type=int, default=3, help="number of runs (default 3)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing results file")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be >= 1")

    out_path: Path = RESULTS_DIR / f"{args.name}.json"
    if out_path.exists() and not args.force:
        print(
            f"error: {display_path(out_path)} exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    started = datetime.now(UTC)
    git = git_state()
    cases = load_cases()
    runs = []
    for run_idx in range(1, args.runs + 1):
        records = []
        for i, case in enumerate(cases, 1):
            record = evaluate_case(case)
            records.append(record)
            status = "ERROR" if record.error else ("correct" if record.correct else "wrong")
            latency = f"{record.latency_ms:.0f}ms" if record.latency_ms is not None else "-"
            print(
                f"[run {run_idx}/{args.runs}] {i:>2}/{len(cases)} {case.id}  {status:<7} {latency}"
            )
        runs.append({"summary": summarize_run(records), "records": [asdict(r) for r in records]})

    summaries = [r["summary"] for r in runs]
    aggregate_stats = aggregate(summaries)
    results = {
        "name": args.name,
        "timestamp": started.isoformat(),
        "config": get_settings().experiment_config(),
        "metrics_version": METRICS_VERSION,
        "git": git,
        "n_cases": len(cases),
        "n_runs": args.runs,
        "aggregate": aggregate_stats,
        "runs": runs,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    tmp = out_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_path)

    print(f"\n{args.name}: {len(cases)} cases x {args.runs} runs (mean ± std)")
    for metric in SUMMARY_METRICS:
        print(f"  {metric:<20} {format_metric(aggregate_stats[metric])}")
    if git["dirty"]:
        print("  warning: uncommitted changes; results may not match the recorded commit")
    print(f"wrote {display_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
