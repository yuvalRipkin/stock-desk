"""Tests for settings validation. No .env: values are passed in."""

import pytest
from pydantic import ValidationError

from stockdesk.config import Settings


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_rewrite_defaults_to_none():
    s = settings(retriever="naive")

    assert (s.query_rewrite, s.n_rewrites, s.rewrite_model) == ("none", 3, "gpt-5.4-mini")


def test_multi_rewrite_with_expand_is_allowed():
    assert settings(retriever="expand", query_rewrite="multi").query_rewrite == "multi"


def test_multi_rewrite_with_another_retriever_is_rejected():
    with pytest.raises(ValidationError, match="QUERY_REWRITE=multi requires RETRIEVER=expand"):
        settings(retriever="naive", query_rewrite="multi")


def test_rewrite_settings_are_in_the_experiment_config():
    config = settings(retriever="expand", query_rewrite="multi", n_rewrites=4).experiment_config()

    assert config["query_rewrite"] == "multi"
    assert config["n_rewrites"] == 4
    assert config["rewrite_model"] == "gpt-5.4-mini"


def test_chunking_defaults_to_plain_and_is_in_the_experiment_config():
    assert settings().chunking == "plain"
    assert settings(chunking="section_header").experiment_config()["chunking"] == "section_header"


def test_large_embedding_model_needs_3072_dimensions():
    assert (
        settings(embedding_model="text-embedding-3-large", embedding_dim=3072).embedding_dim == 3072
    )
    with pytest.raises(ValidationError, match="produces 3072-dimensional vectors"):
        settings(embedding_model="text-embedding-3-large")  # default EMBEDDING_DIM=1536
