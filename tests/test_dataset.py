import pytest

from stockdesk.data import QUESTIONS_FILE, TEST_MANIFEST, split_doc_names
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


@pytest.mark.skipif(not TEST_MANIFEST.exists(), reason="test manifest not built")
def test_splits_are_disjoint_and_cover_all_questions():
    dev, test, all_ = (load_cases(split=s) for s in ("dev", "test", "all"))

    assert (len(dev), len(test), len(all_)) == (30, 120, 150)
    assert not {c.doc_name for c in dev} & {c.doc_name for c in test}
    assert [c.id for c in all_] == sorted(c.id for c in dev + test)
    assert load_cases(split="test") == test  # deterministic


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        split_doc_names("train")
