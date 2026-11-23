"""Application settings loaded from environment variables and .env.

Everything that defines an experiment (table, models, chunking, retrieval) lives here, so a
run's config can be saved alongside its results with `get_settings().experiment_config()`.
"""

from functools import cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Infrastructure / secrets
    database_url: str = "postgresql://stockdesk:stockdesk@localhost:5432/stockdesk"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    log_level: str = "INFO"

    # Experiment config
    table: str = "naive_1000"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # must match embedding_model
    llm_model: str | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5

    def experiment_config(self) -> dict[str, object]:
        """The settings that define an experiment, safe to write into results files."""
        return self.model_dump(
            exclude={"database_url", "openai_api_key", "anthropic_api_key", "log_level"}
        )


@cache
def get_settings() -> Settings:
    return Settings()
