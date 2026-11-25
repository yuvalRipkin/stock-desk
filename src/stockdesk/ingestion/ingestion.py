"""Load manifest PDFs, split them into chunks and write them to the vector store.

CHUNKING=plain splits each page on its own. CHUNKING=section_header also splits a page at 10-K
section headings, so no chunk crosses a section, stores the section in a "section" column, and
prepends "[<section>]" to each chunk's text before embedding. That header only affects search:
the expand retriever sends the model full page text from the PDF.
"""

import argparse
import uuid
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from stockdesk.config import get_settings
from stockdesk.data import PDF_DIR, SPLITS, split_doc_names
from stockdesk.ingestion.sections import split_sections
from stockdesk.store import get_store


def load_pdf(doc_name: str) -> list[Document]:
    """Return one Document per non-empty page of the filing's PDF, with zero-based page numbers."""
    reader = PdfReader(PDF_DIR / f"{doc_name}.pdf")
    docs = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").replace("\x00", "")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"doc_name": doc_name, "page": i}))
    return docs


def splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )


def split(docs: list[Document]) -> list[Document]:
    """Split pages into chunks of CHUNK_SIZE characters with CHUNK_OVERLAP, keeping metadata."""
    return splitter().split_documents(docs)


def split_with_section_headers(
    docs: list[Document], exhibit_cutoff: int | None = None
) -> list[Document]:
    """Split one filing's pages at section headings, then into chunks; prepend "[<section>]".

    Chunks are CHUNK_SIZE characters before the header is added.
    """
    if not docs:
        return []
    doc_name = docs[0].metadata["doc_name"]
    pages = {d.metadata["page"]: d.page_content for d in docs}
    chunks = []
    for page, segments in split_sections(pages, exhibit_cutoff).items():
        for section, text in segments:
            for piece in splitter().split_text(text):
                chunks.append(
                    Document(
                        page_content=f"[{section}]\n{piece}",
                        metadata={"doc_name": doc_name, "page": page, "section": section},
                    )
                )
    return chunks


def chunk_filing(doc_name: str) -> list[Document]:
    """Load and chunk one filing according to CHUNKING."""
    docs = load_pdf(doc_name)
    if get_settings().chunking == "plain":
        return split(docs)
    # imported here: retrieval.pages imports this module
    from stockdesk.retrieval.exhibits import find_exhibit_cutoff

    cutoff = find_exhibit_cutoff({d.metadata["page"]: d.page_content for d in docs})
    return split_with_section_headers(docs, cutoff)


def chunk_ids(chunks: list[Document]) -> list[str]:
    """Stable ids (uuid5 of doc_name:page:index), so re-ingesting upserts instead of duplicating."""
    counters: defaultdict[tuple[str, int], int] = defaultdict(int)
    ids = []
    for c in chunks:
        key = (c.metadata["doc_name"], c.metadata["page"])
        stable = f"{key[0]}:{key[1]}:{counters[key]}"
        counters[key] += 1
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, stable)))
    return ids


def main(argv: list[str] | None = None) -> None:
    """Ingest every filing of a split into the configured vector store table."""
    parser = argparse.ArgumentParser(description="Chunk, embed and store a split's filings.")
    parser.add_argument("--split", choices=SPLITS, default="dev", help="default: dev")
    args = parser.parse_args(argv)
    settings = get_settings()
    names = split_doc_names(args.split)
    print(
        f"table {settings.table}, chunking {settings.chunking}, {args.split}: {len(names)} filings"
    )
    store = get_store()
    for name in names:
        chunks = chunk_filing(name)
        store.add_documents(chunks, ids=chunk_ids(chunks))
        print(f"{name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
