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
        settings(embedding_model="text-embedding-3-large", embedding_dim=1536)
    with pytest.raises(ValidationError, match="produces 1536-dimensional vectors"):
        settings(embedding_model="text-embedding-3-small")  # default EMBEDDING_DIM is 3072


def test_defaults_are_experiment_012():
    config = settings().experiment_config()

    assert {k: config[k] for k in DEFAULTS_012} == DEFAULTS_012


DEFAULTS_012 = {
    "table": "naive_1000_large",
    "embedding_model": "text-embedding-3-large",
    "embedding_dim": 3072,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "chunking": "plain",
    "retriever": "expand",
    "retrieval_k": 10,
    "max_pages": 5,
    "exclude_exhibits": True,
    "query_rewrite": "none",
    "answer_prompt": "v2",
}


def test_answer_prompt_defaults_to_v2_and_is_in_the_experiment_config():
    assert settings().answer_prompt == "v2"
    assert settings(answer_prompt="v1").experiment_config()["answer_prompt"] == "v1"


def test_hybrid_defaults_to_none_and_is_in_the_experiment_config():
    assert (settings().hybrid, settings().bm25_k) == ("none", 10)
    config = settings(hybrid="bm25", bm25_k=20).experiment_config()
    assert (config["hybrid"], config["bm25_k"]) == ("bm25", 20)


def test_hybrid_with_another_retriever_is_rejected():
    with pytest.raises(ValidationError, match="HYBRID=bm25 requires RETRIEVER=expand"):
        settings(retriever="naive", hybrid="bm25")
