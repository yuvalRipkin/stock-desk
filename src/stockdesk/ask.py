"""CLI: answer a question about one filing using naive retrieval + an LLM.

Usage:
    uv run python -m stockdesk.ask --doc AMD_2022_10K "What was total revenue in 2022?"
"""

import argparse
import sys

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from stockdesk.config import get_settings
from stockdesk.retrieval.naive import retrieve

SYSTEM_PROMPT = """You are a financial analyst answering questions about a company's SEC filing.
Answer using only the context below, which consists of excerpts from the filing.
If the answer is not in the context, reply exactly: "Not found in the filing."
Cite the page numbers you used, e.g. (page 12) or (pages 12, 14)."""


def build_context(chunks: list[Document]) -> str:
    return "\n\n".join(f"[page {c.metadata['page']}]\n{c.page_content}" for c in chunks)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ask a question about one SEC filing.")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--doc", required=True, help="doc_name, e.g. AMD_2022_10K")
    parser.add_argument(
        "-k",
        type=int,
        default=settings.retrieval_k,
        help="chunks to retrieve (default %(default)s)",
    )
    args = parser.parse_args(argv)

    if not settings.llm_model:
        print("error: LLM_MODEL is not set. Add LLM_MODEL=<model name> to .env.", file=sys.stderr)
        return 1

    chunks = retrieve(args.question, args.doc, args.k)
    if not chunks:
        msg = f"error: no chunks found for doc '{args.doc}'. Has it been ingested?"
        print(msg, file=sys.stderr)
        return 1

    llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)
    response = llm.invoke(
        [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(f"Context:\n{build_context(chunks)}\n\nQuestion: {args.question}"),
        ]
    )

    print(response.text)
    pages = list(dict.fromkeys(c.metadata["page"] for c in chunks))
    print(f"\nRetrieved pages ({args.doc}): {', '.join(map(str, pages))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
