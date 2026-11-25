"""Load FinanceBench questions as eval cases."""

import json
from dataclasses import dataclass

from stockdesk.data import QUESTIONS_FILE, split_doc_names


@dataclass(frozen=True)
class EvalCase:
    id: str  # financebench_id
    doc_name: str
    question: str
    gold_answer: str
    evidence_pages: list[int]  # zero-based
    evidence_texts: list[str]


def load_cases(doc_names: list[str] | None = None, split: str = "dev") -> list[EvalCase]:
    """Return cases for `doc_names`, or else for the split's filings, sorted by id.

    Splits are by filing: dev is the 30 questions on data_manifest.txt's filings (used for
    tuning), test the other 120 (held out).
    """
    wanted = set(doc_names if doc_names is not None else split_doc_names(split))
    cases = []
    for line in QUESTIONS_FILE.read_text().splitlines():
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
