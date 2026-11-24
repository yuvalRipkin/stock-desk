"""Data locations and the filing manifest, shared by the fetch script, ingestion and evals.

Paths are resolved from the repository root, so commands work from any directory.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_FILE = DATA_DIR / "financebench_open_source.jsonl"
MANIFEST = ROOT / "data_manifest.txt"


def read_manifest(path: Path = MANIFEST) -> list[str]:
    """Return the doc_names listed in the manifest, ignoring blank lines and # comments."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.append(name)
    return names
