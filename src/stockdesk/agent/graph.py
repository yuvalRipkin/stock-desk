"""A minimal ReAct-style LangGraph agent that answers a question about one filing.

The graph loops between two nodes: "agent" (the LLM, with the tools bound) and "tools" (runs
the tool calls the LLM asked for). It ends when the LLM replies without tool calls. After
MAX_AGENT_STEPS tool calls, the LLM is told to stop and called once more with tool use turned
off, so it answers with what it has.

The filing is bound per question: search_filings takes only a query, and every search runs
on the question's doc_name, whatever arguments the model sends.
"""

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from stockdesk.agent.calculate import calculate_tool
from stockdesk.config import get_settings
from stockdesk.prompts import AGENT_LIMIT_MESSAGE, AGENT_PROMPT
from stockdesk.rag import RagError, build_context
from stockdesk.retrieval import get_retriever

# OpenAI tool schemas. search_filings has no doc_name parameter: the filing is not the model's
# choice.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": (
                "Search the filing for pages relevant to a query. Returns the full text of the "
                "best-matching pages, each headed by its page number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, phrased the way a 10-K phrases it.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate an arithmetic expression: numbers, + - * / **, parentheses, round() "
                "and abs(). Write numbers without thousands separators or units."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": 'e.g. "round((14694 - 13520) / 13520 * 100, 2)"',
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]
LIMIT_REACHED = "Not run: tool call limit reached."


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]  # as the model sent them
    pages: list[int]  # pages returned, in rank order; empty unless a search
    result: str | None = None  # calculate's reply; None for searches (see pages)


@dataclass
class AgentResult:
    answer: str
    docs: list[Document]  # every page any search returned, once each, in first-seen order
    queries: list[str]  # searched, across all search_filings calls (incl. any rewrites)
    tool_calls: list[ToolCall]  # the calls that ran, in order
    steps: int  # tool calls run (at most MAX_AGENT_STEPS)
    llm_calls: int
    input_tokens: int  # incl. cache_read_tokens
    output_tokens: int
    cache_read_tokens: int  # input tokens served from OpenAI's prompt cache (billed at 10%)
    llm_usage: list[dict[str, int]]  # per LLM call: input_tokens, output_tokens, cache_read_tokens
    context_chars: int  # characters of tool results sent to the model


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_calls: Annotated[list[ToolCall], operator.add]
    docs: Annotated[list[Document], operator.add]
    queries: Annotated[list[str], operator.add]


def search_filings(query: str, doc_name: str) -> tuple[str, list[Document], list[str]]:
    """Run the configured retrieval on doc_name; return (tool reply, pages, queries searched)."""
    settings = get_settings()
    retrieval = get_retriever(settings.retriever)(query, doc_name, settings.retrieval_k)
    if not retrieval.docs:
        raise RagError(f"no chunks found for doc '{doc_name}'. Has it been ingested?")
    return build_context(retrieval.docs), retrieval.docs, retrieval.queries


def run_tool(call: dict, doc_name: str) -> tuple[str, ToolCall, list[Document], list[str]]:
    """Run one tool call; return (reply for the model, record, pages found, queries searched).

    Bad arguments and unknown tools become an "Error: ..." reply, so the model can retry.
    """
    name, args = call["name"], call.get("args") or {}
    record = ToolCall(name=name, args=args, pages=[])
    if name == "search_filings":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            reply = "Error: search_filings needs a non-empty query string."
            return reply, record, [], []
        reply, docs, queries = search_filings(query, doc_name)  # never args["doc_name"]
        record.pages = [d.metadata["page"] for d in docs]
        return reply, record, docs, queries
    if name == "calculate":
        expression = args.get("expression")
        if not isinstance(expression, str):
            reply = "Error: calculate needs an expression string."
        else:
            reply = calculate_tool(expression)
        record.result = reply
        return reply, record, [], []
    return f"Error: unknown tool {name!r}; use search_filings or calculate.", record, [], []


def build_graph(llm: Any, doc_name: str, max_steps: int):
    """The compiled agent graph for one filing. llm is a chat model supporting bind_tools."""
    with_tools = llm.bind_tools(TOOLS)
    without_tools = llm.bind_tools(TOOLS, tool_choice="none")  # the limit turn: answer only

    def agent(state: AgentState) -> dict:
        if len(state["tool_calls"]) >= max_steps:
            stop = HumanMessage(AGENT_LIMIT_MESSAGE)
            return {"messages": [stop, without_tools.invoke([*state["messages"], stop])]}
        return {"messages": [with_tools.invoke(state["messages"])]}

    def tools(state: AgentState) -> dict:
        remaining = max_steps - len(state["tool_calls"])
        update: dict[str, list] = {"messages": [], "tool_calls": [], "docs": [], "queries": []}
        for i, call in enumerate(state["messages"][-1].tool_calls):
            if i >= remaining:  # every tool call still needs a reply
                reply = LIMIT_REACHED
            else:
                reply, record, docs, queries = run_tool(call, doc_name)
                update["tool_calls"].append(record)
                update["docs"].extend(docs)
                update["queries"].extend(queries)
            update["messages"].append(
                ToolMessage(reply, tool_call_id=call["id"], name=call["name"])
            )
        return update

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        return "tools" if wants_tools and len(state["tool_calls"]) < max_steps else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile()


def call_usage(message: AIMessage) -> dict[str, int]:
    """Token usage of one LLM call; zeros where the API reported nothing."""
    usage = message.usage_metadata or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_tokens": (usage.get("input_token_details") or {}).get("cache_read", 0),
    }


def unique_pages(docs: list[Document]) -> list[Document]:
    """One Document per (doc_name, page), keeping the first seen."""
    seen: dict[tuple[str | None, int], Document] = {}
    for doc in docs:
        seen.setdefault((doc.metadata.get("doc_name"), doc.metadata["page"]), doc)
    return list(seen.values())


def answer(question: str, doc_name: str) -> AgentResult:
    """Answer a question about doc_name with the agent."""
    settings = get_settings()
    if not settings.llm_model:
        raise RagError("LLM_MODEL is not set. Add LLM_MODEL=<model name> to .env.")
    if not doc_name:  # an empty doc_name would search every filing
        raise RagError("the agent needs a doc_name: it only searches the question's filing")

    llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)
    max_steps = settings.max_agent_steps
    graph = build_graph(llm, doc_name, max_steps)
    state = graph.invoke(
        {
            "messages": [SystemMessage(AGENT_PROMPT), HumanMessage(f"Question: {question}")],
            "tool_calls": [],
            "docs": [],
            "queries": [],
        },
        # each step is an agent turn then a tools turn; plus the final answer and some slack
        {"recursion_limit": 2 * max_steps + 5},
    )

    messages = state["messages"]
    replies = [m for m in messages if isinstance(m, AIMessage)]
    usage = [call_usage(m) for m in replies]
    return AgentResult(
        answer=replies[-1].text,
        docs=unique_pages(state["docs"]),
        queries=state["queries"],
        tool_calls=state["tool_calls"],
        steps=len(state["tool_calls"]),
        llm_calls=len(replies),
        input_tokens=sum(u["input_tokens"] for u in usage),
        output_tokens=sum(u["output_tokens"] for u in usage),
        cache_read_tokens=sum(u["cache_read_tokens"] for u in usage),
        llm_usage=usage,
        context_chars=sum(len(m.text) for m in messages if isinstance(m, ToolMessage)),
    )
