"""Application settings loaded from environment variables and .env.

Everything that defines an experiment (table, models, chunking, retrieval) lives here, so a
run's config can be saved alongside its results with `get_settings().experiment_config()`.
"""

from functools import cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings; each field is read from the env var of the same name in upper case."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Infrastructure / secrets
    database_url: str = "postgresql+psycopg://stockdesk:change-me@localhost:5432/stockdesk"
    openai_api_key: SecretStr | None = None
    log_level: str = "INFO"

    # Experiment config
    table: str = "naive_1000"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # must match embedding_model
    llm_model: str | None = None
    judge_model: str = "gpt-5.4"  # grades answers; stronger than llm_model on purpose
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    retriever: Literal["naive", "expand"] = "naive"
    max_pages: int = Field(default=5, ge=1)  # expand: full pages sent to the LLM
    exclude_exhibits: bool = False  # skip pages from the first attached exhibit on
    query_rewrite: Literal["none", "multi"] = "none"  # multi: LLM rewrites, fused with RRF
    n_rewrites: int = Field(default=3, ge=1)  # multi: rewritten queries per question
    rewrite_model: str = "gpt-5.4-mini"  # multi: writes the search queries

    @model_validator(mode="after")
    def check_rewrite_retriever(self) -> "Settings":
        if self.query_rewrite == "multi" and self.retriever != "expand":
            raise ValueError(
                f"QUERY_REWRITE=multi requires RETRIEVER=expand (got RETRIEVER={self.retriever})"
            )
        return self

    def experiment_config(self) -> dict[str, object]:
        """The settings that define an experiment, safe to write into results files."""
        return self.model_dump(exclude={"database_url", "openai_api_key", "log_level"})


@cache
def get_settings() -> Settings:
    """Return the settings, loaded once per process."""
    return Settings()
