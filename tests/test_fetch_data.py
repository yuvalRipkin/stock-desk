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
