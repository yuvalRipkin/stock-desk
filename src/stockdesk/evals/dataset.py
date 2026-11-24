"""Load FinanceBench questions as eval cases."""

import json
from dataclasses import dataclass
from pathlib import Path

from stockdesk.ingestion.ingestion import read_manifest

DATASET = Path("data/financebench_open_source.jsonl")


@dataclass(frozen=True)
class EvalCase:
    id: str  # financebench_id
    doc_name: str
    question: str
    gold_answer: str
    evidence_pages: list[int]  # zero-based
    evidence_texts: list[str]


def load_cases(doc_names: list[str] | None = None) -> list[EvalCase]:
    """Return cases for `doc_names` (default: filings in data_manifest.txt), sorted by id."""
    wanted = set(doc_names if doc_names is not None else read_manifest())
    cases = []
    for line in DATASET.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["doc_name"] not in wanted:
            continue
        cases.append(
            EvalCase(
                id=row["financebench_id"],
                doc_name=row["doc_name"],
                question=row["question"],
                gold_answer=row["answer"],
                evidence_pages=[e["evidence_page_num"] for e in row["evidence"]],
                evidence_texts=[e["evidence_text"] for e in row["evidence"]],
            )
        )
    return sorted(cases, key=lambda c: c.id)
