"""Run the FinanceBench eval and save the results to results/<name>.json.

Usage:
    uv run python -m stockdesk.evals.runner --name 001_naive_1000 [--runs 3] [--split dev] [--force]

ANSWER_MODE=agent answers with the LangGraph agent instead of the RAG pipeline; its records also
hold the tool calls and steps. Page and evidence hit then use every page the agent's searches
returned.

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

from stockdesk import agent, rag
from stockdesk.config import get_settings
from stockdesk.data import ROOT, SPLITS
from stockdesk.evals.dataset import EvalCase, load_cases
from stockdesk.evals.groundedness import GROUNDEDNESS_METRICS, check_record
from stockdesk.evals.groundedness import summarize as summarize_groundedness
from stockdesk.evals.judge import judge
from stockdesk.evals.metrics import (
    METRICS_VERSION,
    distinct_pages,
    evidence_hit,
    is_refusal,
    page_hit,
)
from stockdesk.rag import build_context

RESULTS_DIR = ROOT / "results"
SUMMARY_METRICS = [
    "accuracy",
    "page_hit_rate",
    "evidence_hit_rate",
    "refusal_rate",
    "avg_distinct_pages",
    "avg_context_chars",
    "avg_answer_chars",
    "latency_p50_ms",
    "latency_p95_ms",
    "error_count",
    # tokens: both modes from 016 on (cache reads from 017); None for older results
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_cache_read_tokens",
]
# ANSWER_MODE=agent timing breakdown (from 017): per question unless the comment says otherwise
TIMING_METRICS = [
    "avg_model_calls",
    "avg_model_ms",
    "avg_model_ms_per_call",
    "avg_searches",
    "avg_search_embed_ms",  # per search
    "avg_search_db_ms",  # per search
    "avg_search_pages_ms",  # per search
    "pdf_parses",  # first-time PDF parses in the run (a cold start per filing)
    "avg_calculate_ms",  # per calculate call
]
SUMMARY_METRICS += TIMING_METRICS
# numbers, citations and calculate claims traced to retrieved pages and tool calls
# (evals.groundedness); None in files written before it (see its command for those)
SUMMARY_METRICS += GROUNDEDNESS_METRICS


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
    latency_ms: float | None  # latency of rag.answer() / agent.answer() only
    error: str | None
    queries: list[str] | None = None  # searched: the question, then any rewrites; None pre-008
    answer_chars: int | None = None  # length of the answer; None pre-012
    input_tokens: int | None = None  # summed over the question's LLM calls; None pre-016
    output_tokens: int | None = None
    cache_read_tokens: int | None = None  # part of input_tokens served from cache; None pre-017
    # ANSWER_MODE=agent only: [{name, args, pages, result}] in order, and tool calls run
    tool_calls: list[dict] | None = None
    steps: int | None = None
    llm_usage: list[dict] | None = None  # agent: tokens per LLM call; None pre-017
    timing: dict | None = None  # agent: graph.timing_summary() breakdown; None pre-017
    checklist: dict | None = None  # AGENT_PREFETCH_PROMPT=v2: agent.checklist.summarize()
    groundedness: dict | None = None  # evals.groundedness.check_record(); None in older files


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
    counts as incorrect and as a miss. Averages (distinct pages, context and answer chars) and
    latency percentiles are over questions that got an answer, since errored ones have no
    retrieval or latency.
    """
    n = len(records)
    ok = [r for r in records if r.error is None]

    def rate(flag: str) -> float | None:
        return sum(getattr(r, flag) for r in records) / n if n else None

    latencies = [r.latency_ms for r in ok if r.latency_ms is not None]
    return {
        **summarize_tokens_and_timing(ok),
        **summarize_groundedness([vars(r) for r in records]),
        "n": n,
        "accuracy": rate("correct"),
        "page_hit_rate": rate("page_hit"),
        "evidence_hit_rate": rate("evidence_hit"),
        "refusal_rate": rate("refusal"),
        "avg_distinct_pages": statistics.mean(r.distinct_pages for r in ok) if ok else None,
        "avg_context_chars": mean_or_none([r.context_chars for r in ok]),
        "avg_answer_chars": mean_or_none([r.answer_chars for r in ok]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "error_count": n - len(ok),
    }


def ratio_or_none(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def summarize_tokens_and_timing(ok: list[QuestionRecord]) -> dict[str, float | int | None]:
    """Token and agent-timing averages over answered questions that recorded them."""
    timed = [r.timing for r in ok if r.timing is not None]

    def total(key: str) -> float:
        return sum(t[key] for t in timed)

    result: dict[str, float | int | None] = {
        "avg_input_tokens": mean_or_none([r.input_tokens for r in ok]),
        "avg_output_tokens": mean_or_none([r.output_tokens for r in ok]),
        "avg_cache_read_tokens": mean_or_none([r.cache_read_tokens for r in ok]),
    }
    if not timed:
        return result | dict.fromkeys(TIMING_METRICS)
    return result | {
        "avg_model_calls": total("model_calls") / len(timed),
        "avg_model_ms": total("model_ms") / len(timed),
        "avg_model_ms_per_call": ratio_or_none(total("model_ms"), total("model_calls")),
        "avg_searches": total("searches") / len(timed),
        "avg_search_embed_ms": ratio_or_none(total("embed_ms"), total("searches")),
        "avg_search_db_ms": ratio_or_none(total("db_ms"), total("searches")),
        "avg_search_pages_ms": ratio_or_none(total("pages_ms"), total("searches")),
        "pdf_parses": int(total("pdf_parses")),
        "avg_calculate_ms": ratio_or_none(total("calculate_ms"), total("calculate_calls")),
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


def retrieved_text(docs: list) -> dict[int, str]:
    """The text the model saw, per page (a page's chunks joined), for the groundedness check."""
    pages: dict[int, str] = {}
    for d in docs:
        pages[d.metadata["page"]] = pages.get(d.metadata["page"], "") + "\n" + d.page_content
    return pages


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
        if get_settings().answer_mode == "agent":
            result = agent.answer(case.question, case.doc_name)
        else:
            result = rag.answer(case.question, case.doc_name)
        record.latency_ms = (time.perf_counter() - start) * 1000
    except Exception as e:  # noqa: BLE001 - one bad question must not stop the run
        record.error = f"answer: {type(e).__name__}: {e}"
        return record

    pages = [d.metadata["page"] for d in result.docs]
    record.answer = result.answer
    record.answer_chars = len(result.answer)
    record.queries = result.queries
    record.retrieved_pages = pages
    record.page_hit = page_hit(pages, case.evidence_pages)
    record.evidence_hit = evidence_hit([d.page_content for d in result.docs], case.evidence_texts)
    record.refusal = is_refusal(result.answer)
    record.distinct_pages = distinct_pages(pages)
    record.input_tokens = result.input_tokens
    record.output_tokens = result.output_tokens
    record.cache_read_tokens = result.cache_read_tokens
    if isinstance(result, agent.AgentResult):
        record.tool_calls = [asdict(call) for call in result.tool_calls]
        record.steps = result.steps
        record.llm_usage = result.llm_usage
        record.timing = result.timing
        record.checklist = result.checklist
        record.context_chars = result.context_chars  # all tool replies the model read
    else:
        record.context_chars = len(build_context(result.docs))
    record.groundedness = check_record(asdict(record), retrieved_text(result.docs))

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
    parser.add_argument(
        "--split", choices=SPLITS, default="dev", help="dev (tuning, default), test (held out), all"
    )
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

    config = get_settings().experiment_config()
    print(f"config: {json.dumps(config, indent=2)}")
    started = datetime.now(UTC)
    git = git_state()
    cases = load_cases(split=args.split)
    print(f"split: {args.split} ({len(cases)} questions)")
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
        "config": config,
        "metrics_version": METRICS_VERSION,
        "git": git,
        "split": args.split,
        "n_cases": len(cases),
        "n_runs": args.runs,
        "aggregate": aggregate_stats,
        "runs": runs,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    tmp = out_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_path)

    print(f"\n{args.name}: {args.split}, {len(cases)} cases x {args.runs} runs (mean ± std)")
    for metric in SUMMARY_METRICS:
        print(f"  {metric:<20} {format_metric(aggregate_stats[metric])}")
    if git["dirty"]:
        print("  warning: uncommitted changes; results may not match the recorded commit")
    print(f"wrote {display_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
