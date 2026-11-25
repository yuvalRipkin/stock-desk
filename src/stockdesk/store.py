"""PGVector store shared by ingestion, retrieval and evals."""

import re
from functools import cache

import sqlalchemy as sa
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import Column, PGEngine, PGVectorStore

from stockdesk.config import get_settings

METADATA_COLUMNS = [Column("doc_name", "TEXT"), Column("page", "INTEGER")]
SECTION_COLUMN = Column("section", "TEXT")  # only in tables built with section chunking


def metadata_columns() -> list[Column]:
    """The metadata columns of the configured table, which depend on CHUNKING."""
    if get_settings().chunking == "plain":
        return METADATA_COLUMNS
    return [*METADATA_COLUMNS, SECTION_COLUMN]


TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")  # interpolated into SQL, so kept strict


def filter_index_sql(table: str) -> str:
    """DDL for the btree index that filters by filing and page use.

    Search is exact (no vector index), so without it every query computes distances for every
    row in the table; with it, only the filing's rows.
    """
    if not TABLE_NAME.match(table):
        raise ValueError(f"invalid TABLE name {table!r}: use lowercase letters, digits and _")
    return f'CREATE INDEX IF NOT EXISTS "{table}_doc_name_page_idx" ON "{table}" (doc_name, page)'


def ensure_filter_index(database_url: str, table: str) -> None:
    engine = sa.create_engine(database_url)
    try:
        with engine.begin() as conn:
            conn.execute(sa.text(filter_index_sql(table)))
    finally:
        engine.dispose()


class DimensionMismatch(RuntimeError):
    """The table's vectors have a different dimension than the configured embedding model."""


def table_dimension(database_url: str, table: str) -> int | None:
    """Dimension of the table's embedding column, or None if the table does not exist."""
    query = sa.text(
        "SELECT a.atttypmod FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass(:table) AND a.attname = 'embedding'"
    )
    engine = sa.create_engine(database_url)
    try:
        with engine.connect() as conn:
            return conn.execute(query, {"table": table}).scalar()
    finally:
        engine.dispose()


def check_dimension(table: str, table_dim: int | None, model: str, embedding_dim: int) -> None:
    """Raise DimensionMismatch if an existing table can't hold this model's vectors."""
    if table_dim is not None and table_dim != embedding_dim:
        raise DimensionMismatch(
            f"table {table} stores {table_dim}-dimensional vectors, but EMBEDDING_MODEL={model} "
            f"(EMBEDDING_DIM={embedding_dim}) does not match. Use the model the table was built "
            "with, or another TABLE."
        )


@cache
def get_store() -> PGVectorStore:
    """Return the vector store for the configured table, creating the table if needed."""
    settings = get_settings()
    check_dimension(
        settings.table,
        table_dimension(settings.database_url, settings.table),
        settings.embedding_model,
        settings.embedding_dim,
    )
    engine = PGEngine.from_connection_string(url=settings.database_url)
    try:
        engine.init_vectorstore_table(
            table_name=settings.table,
            vector_size=settings.embedding_dim,
            metadata_columns=metadata_columns(),
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise
    ensure_filter_index(settings.database_url, settings.table)
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=settings.table,
        embedding_service=OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        ),
        metadata_columns=[c.name for c in metadata_columns()],
    )
