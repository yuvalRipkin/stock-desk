"""Application settings loaded from environment variables and .env.

Everything that defines an experiment (table, models, chunking, retrieval) lives here, so a
run's config can be saved alongside its results with `get_settings().experiment_config()`.
"""

from functools import cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Output dimension of each known OpenAI embedding model (we never request shortened vectors).
EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class Settings(BaseSettings):
    """All settings; each field is read from the env var of the same name in upper case."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Infrastructure / secrets
    database_url: str = "postgresql+psycopg://stockdesk:change-me@localhost:5432/stockdesk"
    openai_api_key: SecretStr | None = None
    log_level: str = "INFO"

    # Experiment config. Defaults are the best config so far (experiment 012).
    table: str = "naive_1000_large"
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072  # must match embedding_model and the table
    llm_model: str | None = None
    judge_model: str = "gpt-5.4"  # grades answers; stronger than llm_model on purpose
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking: Literal["plain", "section_header"] = "plain"  # section_header: see ingestion
    retrieval_k: int = 10
    retriever: Literal["naive", "expand"] = "expand"
    max_pages: int = Field(default=5, ge=1)  # expand: full pages sent to the LLM
    exclude_exhibits: bool = True  # skip pages from the first attached exhibit on
    query_rewrite: Literal["none", "multi"] = "none"  # multi: LLM rewrites, fused with RRF
    n_rewrites: int = Field(default=3, ge=1)  # multi: rewritten queries per question
    rewrite_model: str = "gpt-5.4-mini"  # multi: writes the search queries
    hybrid: Literal["none", "bm25"] = "none"  # bm25: BM25 keyword ranking fused in with RRF
    bm25_k: int = Field(default=10, ge=1)  # bm25: keyword-ranked chunks per question
    answer_prompt: Literal["v1", "v2"] = "v2"  # prompts.ANSWER_PROMPTS; v2: figures, steps, Answer:
    answer_mode: Literal["rag", "agent"] = "rag"  # agent: LangGraph tool loop (stockdesk.agent)
    max_agent_steps: int = Field(default=6, ge=1)  # agent: tool calls per question
    # agent: v2 = list-taking tools, arithmetic discipline, short answer (prompts.AGENT_PROMPTS)
    agent_version: Literal["v1", "v2"] = "v1"
    agent_prefetch: bool = False  # agent: search the question verbatim before the first model call
    agent_prefetch_prompt: Literal["v1", "v2"] = (
        "v1"  # prompts.AGENT_PREFETCH_PROMPTS; v2: checklist
    )

    @model_validator(mode="after")
    def check_consistency(self) -> "Settings":
        expected = EMBEDDING_DIMS.get(self.embedding_model)
        if expected is not None and self.embedding_dim != expected:
            raise ValueError(
                f"EMBEDDING_MODEL={self.embedding_model} produces {expected}-dimensional vectors, "
                f"but EMBEDDING_DIM={self.embedding_dim}"
            )
        if self.query_rewrite == "multi" and self.retriever != "expand":
            raise ValueError(
                f"QUERY_REWRITE=multi requires RETRIEVER=expand (got RETRIEVER={self.retriever})"
            )
        if self.hybrid != "none" and self.retriever != "expand":
            raise ValueError(
                f"HYBRID={self.hybrid} requires RETRIEVER=expand (got RETRIEVER={self.retriever})"
            )
        if self.agent_version != "v1" and self.agent_prefetch:
            raise ValueError(
                f"AGENT_VERSION={self.agent_version} does not support AGENT_PREFETCH=true"
            )
        if self.agent_prefetch_prompt != "v1" and not self.agent_prefetch:
            raise ValueError(
                f"AGENT_PREFETCH_PROMPT={self.agent_prefetch_prompt} requires AGENT_PREFETCH=true"
            )
        return self

    def experiment_config(self) -> dict[str, object]:
        """The settings that define an experiment, safe to write into results files."""
        return self.model_dump(exclude={"database_url", "openai_api_key", "log_level"})


@cache
def get_settings() -> Settings:
    """Return the settings, loaded once per process."""
    return Settings()
