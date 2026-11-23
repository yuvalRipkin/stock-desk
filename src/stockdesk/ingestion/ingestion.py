"""Load manifest PDFs, split them into chunks and write them to the vector store."""

import uuid
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from stockdesk.config import get_settings
from stockdesk.store import get_store

PDF_DIR = Path("data/pdfs")
MANIFEST = Path("data_manifest.txt")


def read_manifest() -> list[str]:
    names = []
    for line in MANIFEST.read_text().splitlines():
        name = line.split("#")[0].strip()
        if name:
            names.append(name)
    return names


def load_pdf(doc_name: str) -> list[Document]:
    reader = PdfReader(PDF_DIR / f"{doc_name}.pdf")
    docs = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").replace("\x00", "")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"doc_name": doc_name, "page": i}))
    return docs


def split(docs: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_documents(docs)


def chunk_ids(chunks: list[Document]) -> list[str]:
    counters: defaultdict[tuple[str, int], int] = defaultdict(int)
    ids = []
    for c in chunks:
        key = (c.metadata["doc_name"], c.metadata["page"])
        stable = f"{key[0]}:{key[1]}:{counters[key]}"
        counters[key] += 1
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, stable)))
    return ids


def main() -> None:
    store = get_store()
    for name in read_manifest():
        chunks = split(load_pdf(name))
        store.add_documents(chunks, ids=chunk_ids(chunks))
        print(f"{name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
