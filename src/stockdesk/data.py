"""Data locations and the filing manifest, shared by the fetch script, ingestion and evals.

Paths are resolved from the repository root, so commands work from any directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_FILE = DATA_DIR / "financebench_open_source.jsonl"
MANIFEST = ROOT / "data_manifest.txt"  # dev split: the filings used for tuning
TEST_MANIFEST = ROOT / "data_manifest_test.txt"  # test split: every other FinanceBench filing
SPLITS = ("dev", "test", "all")


def read_manifest(path: Path = MANIFEST) -> list[str]:
    """Return the doc_names listed in the manifest, ignoring blank lines and # comments."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.append(name)
    return names


def split_doc_names(split: str) -> list[str]:
    """The filings of a split: dev (data_manifest.txt), test (data_manifest_test.txt) or all."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split != "dev" and not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"{TEST_MANIFEST.name} not found; run `make fetch SPLIT=test` to create it"
        )
    dev = read_manifest() if split in ("dev", "all") else []
    test = read_manifest(TEST_MANIFEST) if split in ("test", "all") else []
    return [*dev, *test]
