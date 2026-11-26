"""CLI: answer a question about one filing with the configured retriever + an LLM.

Usage:
    uv run python -m stockdesk.ask --doc AMD_2022_10K "What was total revenue in 2022?"

With ANSWER_MODE=agent, the LangGraph agent answers and its tool calls are printed too.
"""

import argparse
import sys

from stockdesk import agent, rag
from stockdesk.config import get_settings
from stockdesk.rag import RagError


def format_tool_call(i: int, call: agent.ToolCall) -> str:
    """One line per tool call, e.g. '2. search_filings(query='net sales') -> pages 48, 49'."""
    args = ", ".join(f"{key}={value!r}" for key, value in call.args.items())
    if call.name == "search_filings":
        outcome = f"pages {', '.join(map(str, call.pages))}" if call.pages else "no pages"
    else:
        outcome = call.result
    return f"{i}. {call.name}({args}) -> {outcome}"


def main(argv: list[str] | None = None) -> int:
    """Answer one question from the command line; return 1 on a setup error."""
    parser = argparse.ArgumentParser(description="Ask a question about one SEC filing.")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--doc", required=True, help="doc_name, e.g. AMD_2022_10K")
    parser.add_argument(
        "-k", type=int, default=None, help="chunks to retrieve (default: RETRIEVAL_K setting)"
    )
    args = parser.parse_args(argv)

    mode = get_settings().answer_mode
    if mode == "agent" and args.k is not None:
        parser.error("-k is not used with ANSWER_MODE=agent; set RETRIEVAL_K instead")
    try:
        if mode == "agent":
            result = agent.answer(args.question, args.doc)
        else:
            result = rag.answer(args.question, args.doc, args.k)
    except RagError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(result.answer)
    pages = list(dict.fromkeys(d.metadata["page"] for d in result.docs))
    print(f"\nRetrieved pages ({args.doc}): {', '.join(map(str, pages))}")
    if isinstance(result, agent.AgentResult):
        print(f"\nTool calls ({result.steps}):")
        for i, call in enumerate(result.tool_calls, 1):
            print(f"  {format_tool_call(i, call)}")
        print(
            f"LLM calls: {result.llm_calls}, tokens: {result.input_tokens} in / "
            f"{result.output_tokens} out"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
