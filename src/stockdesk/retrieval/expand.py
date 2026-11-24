"""Page-expansion retriever: vector search for chunks, then send their whole pages.

Searches for k chunks, keeps the first MAX_PAGES distinct pages in rank order, and returns one
Document per page holding that page's full text, so a passage the splitter cut across chunks
reaches the model intact.
"""

from functools import lru_cache

from langchain_core.documents import Document

from stockdesk.config import get_settings
from stockdesk.ingestion.ingestion import load_pdf
from stockdesk.retrieval import naive
from stockdesk.retrieval.pages import dedup_by_page


@lru_cache(maxsize=32)  # one entry per filing
def page_texts(doc_name: str) -> dict[int, str]:
    """Full text of each non-empty page of a filing, keyed by zero-based page number.

    Read from the PDF and cached per process. Production would store page text in the DB
    (e.g. a pages table written at ingestion) instead of re-reading PDFs.
    """
    return {d.metadata["page"]: d.page_content for d in load_pdf(doc_name)}


def retrieve(question: str, doc_name: str | None = None, k: int | None = None) -> list[Document]:
    """Return the full text of up to MAX_PAGES pages whose chunks rank in the top k."""
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k
    hits = naive.retrieve(question, doc_name, k)
    pages = []
    for hit in dedup_by_page(hits, settings.max_pages):
        name, page = hit.metadata["doc_name"], hit.metadata["page"]
        text = page_texts(name).get(page, hit.page_content)
        pages.append(Document(page_content=text, metadata={"doc_name": name, "page": page}))
    return pages
