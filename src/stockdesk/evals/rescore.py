"""Rescore saved results with the current metrics, without rerunning RAG.

Usage:
    uv run python -m stockdesk.evals.rescore results/<name>.json [more.json ...] [--dry-run]

Recomputes each record's refusal flag with the current rule and re-judges only the records
whose refusal status changed, using their saved answers. Then recomputes per-run summaries and
the aggregate, sets metrics_version, and records what was re-judged. Files already at the
current metrics version are refused.
"""

import argparse
import json
import math
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from stockdesk.config import get_settings
from stockdesk.evals.judge import Verdict, judge
from stockdesk.evals.metrics import METRICS_VERSION, is_refusal
from stockdesk.evals.runner import QuestionRecord, aggregate, display_path, summarize_run

Judge = Callable[[str, str, str], Verdict]


class AlreadyRescored(Exception):
    """The file is already scored with the current metrics version."""


class SummaryMismatch(Exception):
    """Recomputing a stored summary did not reproduce it, so a rescore would not compare."""


def as_record(raw: dict) -> QuestionRecord:
    return QuestionRecord(**{"context_chars": None, **raw})


def same_value(a: object, b: object) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def summarize_like(records: list[dict], keys: list[str]) -> dict:
    """summarize_run, restricted to the keys of the summary it replaces (keeps the file's shape)."""
    full = summarize_run([as_record(r) for r in records])
    return {k: full.get(k) for k in keys}


def check_summary(records: list[dict], stored: dict, where: str) -> None:
    recomputed = summarize_like(records, list(stored))
    bad = [k for k in stored if not same_value(recomputed[k], stored[k])]
    if bad:
        raise SummaryMismatch(f"{where}: recomputed {bad} differ from the stored summary")


def rescore(results: dict, judge_fn: Judge = judge) -> dict:
    """Return a rescored copy of a results dict. Raises AlreadyRescored or SummaryMismatch."""
    version = results.get("metrics_version", 1)
    if version >= METRICS_VERSION:
        raise AlreadyRescored(f"already at metrics_version {version}")

    runs, rejudged = [], []
    for run_idx, run in enumerate(results["runs"], 1):
        check_summary(run["records"], run["summary"], f"run {run_idx}")
        records = [dict(r) for r in run["records"]]
        for r in records:
            if r["answer"] is None or r["error"] is not None:
                continue
            refusal = is_refusal(r["answer"])
            if refusal == r["refusal"]:
                continue
            verdict = judge_fn(r["question"], r["gold_answer"], r["answer"])
            rejudged.append(
                {
                    "run": run_idx,
                    "id": r["id"],
                    "refusal": [r["refusal"], refusal],
                    "correct": [r["correct"], verdict.correct],
                    "judge_reason": verdict.reason,
                }
            )
            r.update(refusal=refusal, correct=verdict.correct, judge_reason=verdict.reason)
        runs.append(
            {**run, "summary": summarize_like(records, list(run["summary"])), "records": records}
        )

    new_aggregate = aggregate([r["summary"] for r in runs])
    out: dict = {}
    for key, value in results.items():
        out[key] = value
        if key == "config":
            out["metrics_version"] = METRICS_VERSION
    out["metrics_version"] = METRICS_VERSION
    out["aggregate"] = {k: new_aggregate[k] for k in results["aggregate"]}
    out["runs"] = runs
    out["rescore"] = {
        "from_metrics_version": version,
        "to_metrics_version": METRICS_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "judge_model": get_settings().judge_model,
        "previous_aggregate": {k: results["aggregate"][k] for k in ("accuracy", "refusal_rate")},
        "rejudged": rejudged,
    }
    return out


def changed_records(results: dict) -> list[tuple[int, str]]:
    """(run, id) of records whose refusal flag the current rule would change; no judge calls."""
    return [
        (run_idx, r["id"])
        for run_idx, run in enumerate(results["runs"], 1)
        for r in run["records"]
        if r["answer"] is not None
        and r["error"] is None
        and is_refusal(r["answer"]) != r["refusal"]
    ]


def pct(stats: dict) -> str:
    return f"{stats['mean'] * 100:.1f}% ± {stats['std'] * 100:.1f}"


def main(argv: list[str] | None = None) -> int:
    """Rescore each results file in place; return 1 if any file was refused or failed."""
    parser = argparse.ArgumentParser(description="Rescore results files with current metrics.")
    parser.add_argument("paths", nargs="+", type=Path, help="results/<name>.json files")
    parser.add_argument("--dry-run", action="store_true", help="only list records that change")
    args = parser.parse_args(argv)

    failed = False
    for path in args.paths:
        results = json.loads(path.read_text(encoding="utf-8"))
        name = display_path(path.resolve())
        if args.dry_run:
            changes = changed_records(results)
            print(f"{name}: {len(changes)} records would be re-judged {changes}")
            continue
        try:
            new = rescore(results)
        except (AlreadyRescored, SummaryMismatch) as e:
            print(f"error: {name}: {e}", file=sys.stderr)
            failed = True
            continue
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        before, after = results["aggregate"], new["aggregate"]
        print(
            f"{name}: re-judged {len(new['rescore']['rejudged'])} | "
            f"accuracy {pct(before['accuracy'])} -> {pct(after['accuracy'])} | "
            f"refusal {pct(before['refusal_rate'])} -> {pct(after['refusal_rate'])}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
