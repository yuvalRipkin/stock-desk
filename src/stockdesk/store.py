"""PGVector store shared by ingestion, retrieval and evals."""

from functools import cache

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


@cache
def get_store() -> PGVectorStore:
    """Return the vector store for the configured table, creating the table if needed."""
    settings = get_settings()
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
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=settings.table,
        embedding_service=OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        ),
        metadata_columns=[c.name for c in metadata_columns()],
    )
