"""CLI: answer a question about one filing with the configured retriever + an LLM.

Usage:
    uv run python -m stockdesk.ask --doc AMD_2022_10K "What was total revenue in 2022?"
"""

import argparse
import sys

from stockdesk.rag import RagError, answer


def main(argv: list[str] | None = None) -> int:
    """Answer one question from the command line; return 1 on a setup error."""
    parser = argparse.ArgumentParser(description="Ask a question about one SEC filing.")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--doc", required=True, help="doc_name, e.g. AMD_2022_10K")
    parser.add_argument(
        "-k", type=int, default=None, help="chunks to retrieve (default: RETRIEVAL_K setting)"
    )
    args = parser.parse_args(argv)

    try:
        result = answer(args.question, args.doc, args.k)
    except RagError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(result.answer)
    pages = list(dict.fromkeys(d.metadata["page"] for d in result.docs))
    print(f"\nRetrieved pages ({args.doc}): {', '.join(map(str, pages))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
