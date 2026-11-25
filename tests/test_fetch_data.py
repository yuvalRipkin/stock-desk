"""Tests for scripts/fetch_data.py."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import requests

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_data.py"


@pytest.fixture(scope="module")
def fetch_data() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_data", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_response(status: int, body: bytes = b"") -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp.url = "https://example.test/file.pdf"
    resp._content = body
    resp._content_consumed = True
    return resp


def test_retries_5xx_with_backoff_then_saves(
    fetch_data: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"%PDF-1.7 fake pdf"
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [make_response(503), make_response(503), make_response(200, body)]
    sleep = MagicMock()
    monkeypatch.setattr(fetch_data.time, "sleep", sleep)
    dest = tmp_path / "doc.pdf"

    status = fetch_data.download(
        "https://example.test/file.pdf", dest, fetch_data.is_valid_pdf, session, force=False
    )

    assert status == "downloaded"
    assert dest.read_bytes() == body
    assert session.get.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [1.0, 2.0]
    assert list(tmp_path.iterdir()) == [dest]  # no leftover temp file


def test_build_test_manifest_holds_out_every_non_dev_doc(fetch_data, tmp_path: Path):
    questions = tmp_path / "q.jsonl"
    rows = ["DEV_A", "TEST_B", "TEST_B", "DEV_A", "TEST_A"]
    questions.write_text("".join(f'{{"doc_name": "{d}"}}\n' for d in rows))
    out = tmp_path / "data_manifest_test.txt"

    names = fetch_data.build_test_manifest(questions, ["DEV_A"], out)

    assert names == ["TEST_A", "TEST_B"]  # sorted, deterministic, no dev docs
    text = out.read_text()
    assert "TEST_B  # 2 questions" in text
    assert "DEV_A" not in text
