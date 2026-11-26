"""A minimal ReAct-style LangGraph agent that answers a question about one filing.

The graph loops between two nodes: "agent" (the LLM, with the tools bound) and "tools" (runs
the tool calls the LLM asked for). It ends when the LLM replies without tool calls. After
MAX_AGENT_STEPS tool calls, the LLM is told to stop and called once more with tool use turned
off, so it answers with what it has.

The filing is bound per question: search_filings takes only a query, and every search runs
on the question's doc_name, whatever arguments the model sends.

With AGENT_PREFETCH=true, search_filings runs on the question verbatim before the loop starts
and its pages go into the first message. That search is recorded like the model's own (with
prefetch=True) but does not count against MAX_AGENT_STEPS. AGENT_PREFETCH_PROMPT picks its
system prompt; v2 asks for an evidence checklist, which is parsed into the result.

With AGENT_VERSION=v2, both tools take lists (several queries or expressions per call, one step
each) and the prompt asks for strict arithmetic discipline and a short answer.

Every run records a timing breakdown: each model call, each search (embedding, DB query, page
loading, first-time PDF parse) and each calculate call.
"""

import operator
import time
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

from stockdesk.agent import checklist
from stockdesk.agent.calculate import calculate_tool
from stockdesk.config import get_settings
from stockdesk.prompts import (
    AGENT_LIMIT_MESSAGE,
    AGENT_PREFETCH_MESSAGE,
    AGENT_PREFETCH_PROMPTS,
    AGENT_PROMPTS,
)
from stockdesk.rag import RagError, build_context
from stockdesk.retrieval import get_retriever
from stockdesk.timing import elapsed_ms, recording

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

# AGENT_VERSION=v2: the same tools, taking lists, so one step can hold several searches or sums.
MAX_QUERIES_PER_SEARCH = 3
MAX_EXPRESSIONS_PER_CALCULATE = 10
TOOLS_V2 = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": (
                f"Search the filing with up to {MAX_QUERIES_PER_SEARCH} queries at once. Returns "
                "the full text of the best-matching pages for all the queries, each page once, "
                "headed by its page number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_QUERIES_PER_SEARCH,
                        "description": "What to look for, each phrased the way a 10-K phrases it.",
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                f"Evaluate up to {MAX_EXPRESSIONS_PER_CALCULATE} arithmetic expressions at once: "
                "numbers, + - * / **, parentheses, round() and abs(). Write numbers without "
                'thousands separators or units. Returns one line per expression: "expression = '
                'result".'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expressions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": MAX_EXPRESSIONS_PER_CALCULATE,
                        "description": 'e.g. ["23601 - 16434", "(23601 - 16434) / 16434 * 100"]',
                    }
                },
                "required": ["expressions"],
                "additionalProperties": False,
            },
        },
    },
]
TOOLS_BY_VERSION = {"v1": TOOLS, "v2": TOOLS_V2}  # selected by the AGENT_VERSION setting
LIMIT_REACHED = "Not run: tool call limit reached."


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]  # as the model sent them
    pages: list[int]  # pages returned, in rank order; empty unless a search
    result: str | None = None  # calculate's reply; None for searches (see pages)
    ms: float | None = None  # wall time of the call
    # searches: embed_ms, db_ms, pages_ms, plus pdf_parse / pdf_parse_ms on a first-time PDF
    # parse and rewrite_ms / bm25_ms when QUERY_REWRITE / HYBRID are on
    timing: dict[str, Any] | None = None
    prefetch: bool = False  # run in code before the loop (AGENT_PREFETCH), not by the model
    query_pages: list[list[int]] | None = None  # v2 searches: pages returned for each query


@dataclass
class AgentResult:
    answer: str
    docs: list[Document]  # every page any search returned, once each, in first-seen order
    queries: list[str]  # searched, across all search_filings calls (incl. any rewrites)
    tool_calls: list[ToolCall]  # the calls that ran, in order, the prefetch search first
    steps: int  # tool calls the model made (at most MAX_AGENT_STEPS; excludes the prefetch)
    llm_calls: int
    input_tokens: int  # incl. cache_read_tokens
    output_tokens: int
    cache_read_tokens: int  # input tokens served from OpenAI's prompt cache (billed at 10%)
    llm_usage: list[dict[str, int]]  # per LLM call: input_tokens, output_tokens, cache_read_tokens
    context_chars: int  # characters of search pages and tool results sent to the model
    timing: dict[str, Any]  # per-question breakdown, see timing_summary()
    checklist: dict | None = None  # AGENT_PREFETCH_PROMPT=v2: checklist.summarize(); else None


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_calls: Annotated[list[ToolCall], operator.add]
    docs: Annotated[list[Document], operator.add]
    queries: Annotated[list[str], operator.add]
    model_ms: Annotated[list[float], operator.add]  # wall time of each model call


def steps_used(tool_calls: list[ToolCall]) -> int:
    """Tool calls that count against MAX_AGENT_STEPS: all but the prefetch search."""
    return sum(not call.prefetch for call in tool_calls)


def search_filings(query: str, doc_name: str) -> tuple[str, list[Document], list[str], dict]:
    """Run the configured retrieval on doc_name.

    Returns (tool reply, pages, queries searched, timing breakdown).
    """
    settings = get_settings()
    with recording() as timing:
        retrieval = get_retriever(settings.retriever)(query, doc_name, settings.retrieval_k)
    if not retrieval.docs:
        raise RagError(f"no chunks found for doc '{doc_name}'. Has it been ingested?")
    return build_context(retrieval.docs), retrieval.docs, retrieval.queries, timing


def prefetch(question: str, doc_name: str) -> tuple[str, ToolCall, list[Document], list[str]]:
    """The AGENT_PREFETCH search: the question verbatim. Returns (pages text, record, ...)."""
    start = time.perf_counter()
    context, docs, queries, timing = search_filings(question, doc_name)
    record = ToolCall(
        name="search_filings",
        args={"query": question},
        pages=[d.metadata["page"] for d in docs],
        ms=elapsed_ms(start),
        timing=timing,
        prefetch=True,
    )
    return context, record, docs, queries


def run_tool(
    call: dict, doc_name: str, version: str = "v1"
) -> tuple[str, ToolCall, list[Document], list[str]]:
    """Run one tool call; return (reply for the model, record, pages found, queries searched).

    version is AGENT_VERSION: v1 tools take one query or expression, v2 tools take lists. Bad
    arguments and unknown tools become an "Error: ..." reply, so the model can retry.
    """
    start = time.perf_counter()
    if version == "v2":
        reply, record, docs, queries = _run_tool_v2(call, doc_name)
    else:
        reply, record, docs, queries = _run_tool(call, doc_name)
    record.ms = elapsed_ms(start)
    return reply, record, docs, queries


def _run_tool(call: dict, doc_name: str) -> tuple[str, ToolCall, list[Document], list[str]]:
    name, args = call["name"], call.get("args") or {}
    record = ToolCall(name=name, args=args, pages=[])
    if name == "search_filings":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            reply = "Error: search_filings needs a non-empty query string."
            return reply, record, [], []
        reply, docs, queries, timing = search_filings(query, doc_name)  # never args["doc_name"]
        record.pages = [d.metadata["page"] for d in docs]
        record.timing = timing
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


def strings(value: Any) -> list[str]:
    """The non-empty strings of a list argument (anything else is dropped)."""
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def _run_tool_v2(call: dict, doc_name: str) -> tuple[str, ToolCall, list[Document], list[str]]:
    name, args = call["name"], call.get("args") or {}
    record = ToolCall(name=name, args=args, pages=[])
    if name == "search_filings":
        queries = strings(args.get("queries"))
        if not queries:
            return "Error: search_filings needs a list of query strings.", record, [], []
        return search_many(queries, doc_name, record)
    if name == "calculate":
        expressions = args.get("expressions")
        if not isinstance(expressions, list) or not expressions:
            reply = "Error: calculate needs a list of expression strings."
        else:
            reply = calculate_many(expressions)
        record.result = reply
        return reply, record, [], []
    return f"Error: unknown tool {name!r}; use search_filings or calculate.", record, [], []


def search_many(
    queries: list[str], doc_name: str, record: ToolCall
) -> tuple[str, ToolCall, list[Document], list[str]]:
    """Run each query (up to MAX_QUERIES_PER_SEARCH) and return their pages, each page once."""
    run, skipped = queries[:MAX_QUERIES_PER_SEARCH], queries[MAX_QUERIES_PER_SEARCH:]
    docs: list[Document] = []
    searched: list[str] = []
    timing: dict[str, Any] = {"queries": len(run)}
    record.query_pages = []
    for query in run:
        _, found, query_searched, query_timing = search_filings(query, doc_name)
        docs.extend(found)
        searched.extend(query_searched)
        record.query_pages.append([d.metadata["page"] for d in found])
        for key, value in query_timing.items():
            if isinstance(value, bool):
                timing[key] = timing.get(key, False) or value
            else:
                timing[key] = timing.get(key, 0.0) + value
    pages = unique_pages(docs)
    record.pages = [d.metadata["page"] for d in pages]
    record.timing = timing
    header = [
        f'Query "{q}": pages {", ".join(map(str, p))}'
        for q, p in zip(run, record.query_pages, strict=True)
    ]
    if skipped:
        header.append(f"Not run (at most {MAX_QUERIES_PER_SEARCH} queries per call): {skipped}")
    return "\n".join(header) + "\n\n" + build_context(pages), record, docs, searched


def calculate_many(expressions: list[Any]) -> str:
    """One line per expression: "expression = result" (or "= Error: ...")."""
    run = expressions[:MAX_EXPRESSIONS_PER_CALCULATE]
    lines = [
        f"{e} = {calculate_tool(e)}" if isinstance(e, str) else f"{e!r} = Error: not a string"
        for e in run
    ]
    if len(expressions) > len(run):
        lines.append(
            f"Not run: {len(expressions) - len(run)} more "
            f"(at most {MAX_EXPRESSIONS_PER_CALCULATE} expressions per call)"
        )
    return "\n".join(lines)


def build_graph(llm: Any, doc_name: str, max_steps: int, version: str = "v1"):
    """The compiled agent graph for one filing. llm is a chat model supporting bind_tools."""
    tools_schema = TOOLS_BY_VERSION[version]
    with_tools = llm.bind_tools(tools_schema)
    without_tools = llm.bind_tools(tools_schema, tool_choice="none")  # the limit turn: answer only

    def agent(state: AgentState) -> dict:
        start = time.perf_counter()
        if steps_used(state["tool_calls"]) >= max_steps:
            stop = HumanMessage(AGENT_LIMIT_MESSAGE)
            reply = without_tools.invoke([*state["messages"], stop])
            return {"messages": [stop, reply], "model_ms": [elapsed_ms(start)]}
        reply = with_tools.invoke(state["messages"])
        return {"messages": [reply], "model_ms": [elapsed_ms(start)]}

    def tools(state: AgentState) -> dict:
        remaining = max_steps - steps_used(state["tool_calls"])
        update: dict[str, list] = {"messages": [], "tool_calls": [], "docs": [], "queries": []}
        for i, call in enumerate(state["messages"][-1].tool_calls):
            if i >= remaining:  # every tool call still needs a reply
                reply = LIMIT_REACHED
            else:
                reply, record, docs, queries = run_tool(call, doc_name, version)
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
        return "tools" if wants_tools and steps_used(state["tool_calls"]) < max_steps else END

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


def expression_count(args: dict) -> int:
    """Expressions in a calculate call: a v2 list, or one for v1."""
    expressions = args.get("expressions")
    return len(expressions) if isinstance(expressions, list) else 1


def timing_summary(model_ms: list[float], tool_calls: list[ToolCall]) -> dict[str, Any]:
    """Per-question timing: model calls, searches (summed breakdown) and calculate calls."""
    searches = [c for c in tool_calls if c.name == "search_filings" and c.timing is not None]
    calcs = [c for c in tool_calls if c.name == "calculate"]

    def total(key: str) -> float:
        return sum(c.timing.get(key, 0.0) for c in searches)

    return {
        "model_calls": len(model_ms),
        "model_ms": sum(model_ms),
        "model_call_ms": model_ms,
        "searches": sum(c.timing.get("queries", 1) for c in searches),  # queries run
        "search_calls": len(searches),
        "search_ms": sum(c.ms or 0.0 for c in searches),
        "embed_ms": total("embed_ms"),
        "db_ms": total("db_ms"),
        "pages_ms": total("pages_ms"),
        "pdf_parses": sum(bool(c.timing.get("pdf_parse")) for c in searches),
        "pdf_parse_ms": total("pdf_parse_ms"),
        "calculate_calls": len(calcs),
        "calculate_expressions": sum(expression_count(c.args) for c in calcs),
        "calculate_ms": sum(c.ms or 0.0 for c in calcs),
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
    graph = build_graph(llm, doc_name, max_steps, settings.agent_version)
    if settings.agent_prefetch:
        context, record, docs, queries = prefetch(question, doc_name)
        messages = [
            SystemMessage(AGENT_PREFETCH_PROMPTS[settings.agent_prefetch_prompt]),
            HumanMessage(AGENT_PREFETCH_MESSAGE.format(question=question, context=context)),
        ]
        initial = {"tool_calls": [record], "docs": docs, "queries": queries}
    else:
        context = ""
        system = AGENT_PROMPTS[settings.agent_version]
        messages = [SystemMessage(system), HumanMessage(f"Question: {question}")]
        initial = {"tool_calls": [], "docs": [], "queries": []}
    state = graph.invoke(
        {"messages": messages, **initial, "model_ms": []},
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
        steps=steps_used(state["tool_calls"]),
        llm_calls=len(replies),
        input_tokens=sum(u["input_tokens"] for u in usage),
        output_tokens=sum(u["output_tokens"] for u in usage),
        cache_read_tokens=sum(u["cache_read_tokens"] for u in usage),
        llm_usage=usage,
        context_chars=len(context)
        + sum(len(m.text) for m in messages if isinstance(m, ToolMessage)),
        timing=timing_summary(state["model_ms"], state["tool_calls"]),
        checklist=checklist.summarize(replies) if uses_checklist(settings) else None,
    )


def uses_checklist(settings: Any) -> bool:
    """True if the system prompt asks for an evidence checklist (prefetch prompt v2)."""
    return settings.agent_prefetch and settings.agent_prefetch_prompt == "v2"
