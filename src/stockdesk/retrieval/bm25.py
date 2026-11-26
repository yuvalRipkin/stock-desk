"""BM25 keyword search over one filing's chunks, for hybrid retrieval (HYBRID=bm25).

The index is built in memory from the filing's chunks in the configured table (the same chunks
vector search ranks, so both lists can be fused with RRF), with exhibit pages left out when
EXCLUDE_EXHIBITS is set, and cached per filing like the page text.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import sqlalchemy as sa
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from stockdesk.config import get_settings
from stockdesk.retrieval.exhibits import exhibit_cutoff
from stockdesk.store import TABLE_NAME

# Short function words that match nearly every chunk; financial terms ("net", "total") stay.
STOPWORDS = frozenset(
    "a an and are as at be been by did do does for from had has have how in is it its of on or "
    "that the their this to was were what when which who will with".split()
)
# A number with thousands separators or decimals ("23,601", "4.5"), or a run of letters and
# digits joined by hyphens ("10-k", "emea", "2022").
TOKEN = re.compile(r"\d+(?:[.,]\d+)+|[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercased tokens of text, stopwords dropped; "23,601" and "23601" match."""
    tokens = (t.replace(",", "") for t in TOKEN.findall(text.lower()))
    return [t for t in tokens if t not in STOPWORDS]


@dataclass(frozen=True)
class Bm25Index:
    chunks: list[Document]
    bm25: BM25Okapi | None  # None for a filing with no chunks

    def search(self, query: str, k: int) -> list[Document]:
        """The k best-scoring chunks, best first; chunks sharing no query term are left out."""
        if self.bm25 is None or k < 1:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])  # stable: ties by order
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]


def build_index(chunks: list[Document]) -> Bm25Index:
    """A BM25 index over the chunks' text (pure: no DB)."""
    if not chunks:
        return Bm25Index(chunks=[], bm25=None)
    return Bm25Index(chunks=chunks, bm25=BM25Okapi([tokenize(c.page_content) for c in chunks]))


def load_chunks(table: str, doc_name: str) -> list[Document]:
    """All of a filing's chunks in the table, in a fixed order (page, then text)."""
    if not TABLE_NAME.match(table):
        raise ValueError(f"invalid TABLE name {table!r}")
    query = sa.text(
        f'SELECT content, page FROM "{table}" WHERE doc_name = :doc ORDER BY page, content'
    )
    engine = sa.create_engine(get_settings().database_url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(query, {"doc": doc_name}).all()
    finally:
        engine.dispose()
    return [Document(page_content=c, metadata={"doc_name": doc_name, "page": p}) for c, p in rows]


@lru_cache(maxsize=32)  # one entry per filing (and table / exhibit cutoff)
def filing_index(table: str, doc_name: str, cutoff: int | None) -> Bm25Index:
    """BM25 index of a filing's chunks; pages at or after cutoff (exhibits) are left out."""
    chunks = load_chunks(table, doc_name)
    return build_index([c for c in chunks if cutoff is None or c.metadata["page"] < cutoff])


def retrieve(question: str, doc_name: str | None, k: int | None = None) -> list[Document]:
    """The k chunks of doc_name that BM25 ranks best for the question."""
    if not doc_name:
        raise ValueError("HYBRID=bm25 needs a doc_name: BM25 indexes are built per filing")
    settings = get_settings()
    cutoff = exhibit_cutoff(doc_name) if settings.exclude_exhibits else None
    index = filing_index(settings.table, doc_name, cutoff)
    return index.search(question, settings.bm25_k if k is None else k)
