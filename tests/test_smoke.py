"""Smoke test: the package imports and settings load."""

import stockdesk
from stockdesk.config import Settings


def test_import_and_settings() -> None:
    assert stockdesk.__doc__
    settings = Settings()
    assert settings.database_url
    assert settings.log_level
