import pytest

from stockdesk.data import QUESTIONS_FILE
from stockdesk.evals.dataset import load_cases

pytestmark = pytest.mark.skipif(
    not QUESTIONS_FILE.exists(), reason="FinanceBench data not fetched (run scripts/fetch_data.py)"
)


def test_load_cases():
    cases = load_cases()
    assert len(cases) == 30
    for case in cases:
        assert case.evidence_pages, case.id
        assert all(p >= 0 for p in case.evidence_pages), case.id
