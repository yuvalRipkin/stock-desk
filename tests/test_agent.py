"""Tests for the LangGraph agent. The model and the retriever are fakes: no network, no DB."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from stockdesk import agent
from stockdesk.agent import graph
from stockdesk.evals import runner
from stockdesk.evals.dataset import EvalCase
from stockdesk.evals.judge import Verdict
from stockdesk.prompts import (
    AGENT_LIMIT_MESSAGE,
    AGENT_PREFETCH_PROMPT,
    AGENT_PREFETCH_PROMPT_V2,
    AGENT_PROMPT,
)
from stockdesk.rag import RagError
from stockdesk.retrieval import Retrieval


def reply(
    text: str = "", *calls: tuple[str, dict], tokens: tuple[int, int] = (10, 2), cached: int = 0
):
    """A scripted model reply with optional tool calls and token usage (cached: cache reads)."""
    return AIMessage(
        content=text,
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{name}_{i}"}
            for i, (name, args) in enumerate(calls)
        ],
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
            "input_token_details": {"cache_read": cached},
        },
    )


class ScriptedLLM:
    """Returns the scripted replies in order; records (tool_choice, messages) for each call."""

    def __init__(self, replies: list[AIMessage], events: list | None = None):
        self.replies = list(replies)
        self.calls: list[tuple[str | None, list]] = []
        self.events = events if events is not None else []  # shared with the fake search

    def bind_tools(self, tools, tool_choice=None):
        assert [t["function"]["name"] for t in tools] == ["search_filings", "calculate"]
        llm = self

        class Bound:
            def invoke(self, messages):
                llm.calls.append((tool_choice, list(messages)))
                llm.events.append("model")
                if not llm.replies:
                    raise AssertionError("the model was called more often than scripted")
                return llm.replies.pop(0)

        return Bound()


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch):
    """run(replies, max_steps=6, prefetch=False) -> (AgentResult, model, searches).

    searches are (query, doc_name, k); model.events is the order of "model" calls and
    ("search", query) calls.
    """
    searches: list[tuple[str, str | None, int | None]] = []
    events: list = []

    def fake_search(query: str, doc_name: str | None, k: int | None) -> Retrieval:
        searches.append((query, doc_name, k))
        events.append(("search", query))
        page = 10 * len(searches)  # pages 10, 11 for the 1st search, 20, 21 for the 2nd, ...
        docs = [
            Document(page_content=f"text of page {p}", metadata={"doc_name": doc_name, "page": p})
            for p in (page, page + 1)
        ]
        return Retrieval(docs, [query])

    monkeypatch.setattr(graph, "get_retriever", lambda name: fake_search)

    def _run(
        replies: list[AIMessage],
        max_steps: int = 6,
        doc_name: str = "DOC",
        prefetch: bool = False,
        prefetch_prompt: str = "v1",
    ):
        llm = ScriptedLLM(replies, events)
        settings = graph.get_settings().model_copy(
            update={
                "llm_model": "fake-model",
                "max_agent_steps": max_steps,
                "retrieval_k": 7,
                "agent_prefetch": prefetch,
                "agent_prefetch_prompt": prefetch_prompt,
            }
        )
        monkeypatch.setattr(graph, "get_settings", lambda: settings)
        monkeypatch.setattr(graph, "ChatOpenAI", lambda **kw: llm)
        result = agent.answer("What was revenue?", doc_name)
        assert not llm.replies, "the agent stopped before using every scripted reply"
        return result, llm, searches

    return _run


def tool_messages(messages: list) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def test_search_calculate_then_answer(run):
    result, llm, searches = run(
        [
            reply("", ("search_filings", {"query": "net sales"}), tokens=(100, 5)),
            reply(
                "",
                ("calculate", {"expression": "(120 - 100) / 100 * 100"}),
                tokens=(200, 6),
                cached=0,
            ),
            reply("Figures...\nAnswer: 20% (page 10)", tokens=(300, 40), cached=128),
        ]
    )

    assert result.answer == "Figures...\nAnswer: 20% (page 10)"
    assert searches == [("net sales", "DOC", 7)]
    assert [(c.name, c.args, c.pages, c.result) for c in result.tool_calls] == [
        ("search_filings", {"query": "net sales"}, [10, 11], None),
        ("calculate", {"expression": "(120 - 100) / 100 * 100"}, [], "20.0"),
    ]
    assert result.steps == 2
    assert result.llm_calls == 3
    assert (result.input_tokens, result.output_tokens) == (600, 51)
    assert result.cache_read_tokens == 128
    assert result.llm_usage == [
        {"input_tokens": 100, "output_tokens": 5, "cache_read_tokens": 0},
        {"input_tokens": 200, "output_tokens": 6, "cache_read_tokens": 0},
        {"input_tokens": 300, "output_tokens": 40, "cache_read_tokens": 128},
    ]
    assert [d.metadata["page"] for d in result.docs] == [10, 11]
    assert result.queries == ["net sales"]

    system, question = llm.calls[0][1]
    assert system.content == AGENT_PROMPT
    assert question.content == "Question: What was revenue?"
    # the model saw the search's pages with their numbers, then calculate's result
    replies = tool_messages(llm.calls[2][1])
    assert replies[0].content == "[page 10]\ntext of page 10\n\n[page 11]\ntext of page 11"
    assert replies[1].content == "20.0"
    assert result.context_chars == len(replies[0].content) + len(replies[1].content)
    assert all(choice is None for choice, _ in llm.calls)  # tools available on every turn


def test_pages_from_every_search_are_kept_once_each(run):
    result, _, searches = run(
        [
            reply("", ("search_filings", {"query": "balance sheet"})),
            reply("", ("search_filings", {"query": "total liabilities"})),
            reply("Answer: x"),
        ]
    )

    assert len(searches) == 2
    assert [d.metadata["page"] for d in result.docs] == [10, 11, 20, 21]
    assert result.queries == ["balance sheet", "total liabilities"]


# --- doc_name binding ---------------------------------------------------------------------------


def test_the_model_cannot_search_another_filing(run):
    result, _, searches = run(
        [
            reply(
                "",
                ("search_filings", {"query": "revenue", "doc_name": "OTHER_2022_10K"}),
                ("search_filings", {"query": "revenue", "doc": "OTHER_2022_10K", "k": 100}),
            ),
            reply("Answer: x"),
        ]
    )

    assert [(doc, k) for _, doc, k in searches] == [("DOC", 7), ("DOC", 7)]
    assert {d.metadata["doc_name"] for d in result.docs} == {"DOC"}
    # the attempt is still visible in the trace
    assert result.tool_calls[0].args["doc_name"] == "OTHER_2022_10K"


def test_search_tool_schema_has_no_doc_parameter():
    search_schema = graph.TOOLS[0]["function"]
    assert search_schema["name"] == "search_filings"
    assert list(search_schema["parameters"]["properties"]) == ["query"]
    assert search_schema["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize("doc_name", ["", None])
def test_an_empty_doc_name_is_refused_rather_than_searching_every_filing(run, doc_name):
    with pytest.raises(RagError, match="doc_name"):
        run([], doc_name=doc_name)


# --- step limit ---------------------------------------------------------------------------------


def test_step_limit_forces_an_answer_without_tools(run):
    search = ("search_filings", {"query": "revenue"})
    result, llm, searches = run(
        [reply("", search), reply("", search), reply("Answer: best effort")], max_steps=2
    )

    assert len(searches) == 2
    assert result.steps == 2
    assert result.answer == "Answer: best effort"
    final_choice, final_messages = llm.calls[-1]
    assert final_choice == "none"  # tool use turned off for the last turn
    assert isinstance(final_messages[-1], HumanMessage)
    assert final_messages[-1].content == AGENT_LIMIT_MESSAGE


def test_step_limit_counts_parallel_tool_calls(run):
    result, llm, searches = run(
        [
            reply(
                "",
                ("search_filings", {"query": "a"}),
                ("calculate", {"expression": "1 + 1"}),
                ("search_filings", {"query": "c"}),
            ),
            reply("Answer: x"),
        ],
        max_steps=2,
    )

    assert [q for q, _, _ in searches] == ["a"]
    assert [c.name for c in result.tool_calls] == ["search_filings", "calculate"]
    assert result.steps == 2
    # every tool call got a reply; the one over the limit was not run
    replies = tool_messages(llm.calls[-1][1])
    assert [m.tool_call_id for m in replies] == [
        "call_search_filings_0",
        "call_calculate_1",
        "call_search_filings_2",
    ]
    assert replies[2].content == graph.LIMIT_REACHED
    assert llm.calls[-1][0] == "none"


def test_answer_without_tools_is_allowed_under_the_limit(run):
    result, llm, searches = run([reply("Not found in the filing.")])

    assert searches == []
    assert result.steps == 0
    assert result.docs == []
    assert len(llm.calls) == 1


# --- bad tool calls -----------------------------------------------------------------------------


def test_bad_tool_calls_get_an_error_reply_not_a_crash(run):
    result, llm, searches = run(
        [
            reply(
                "",
                ("search_filings", {}),
                ("calculate", {"expression": "__import__('os').system('echo pwned')"}),
                ("delete_files", {"path": "/"}),
            ),
            reply("Answer: x"),
        ]
    )

    assert searches == []
    contents = [m.content for m in tool_messages(llm.calls[-1][1])]
    assert all(c.startswith("Error: ") for c in contents)
    assert "unknown tool 'delete_files'" in contents[2]
    assert result.steps == 3


def test_a_filing_with_no_chunks_raises(run, monkeypatch):
    monkeypatch.setattr(graph, "get_retriever", lambda name: lambda q, d, k: Retrieval([], [q]))
    with pytest.raises(RagError, match="no chunks"):
        run([reply("", ("search_filings", {"query": "revenue"}))])


# --- prefetch -----------------------------------------------------------------------------------


def test_prefetch_searches_the_question_verbatim_before_the_first_model_call(run):
    result, llm, searches = run([reply("Answer: 42 (page 10)")], prefetch=True)

    assert searches == [("What was revenue?", "DOC", 7)]
    assert llm.events == [("search", "What was revenue?"), "model"]
    system, first = llm.calls[0][1]
    assert system.content == AGENT_PREFETCH_PROMPT
    assert first.content.startswith("Question: What was revenue?\n\nInitial search results")
    assert first.content.endswith("[page 10]\ntext of page 10\n\n[page 11]\ntext of page 11")
    # recorded like any other search, marked as the prefetch
    [call] = result.tool_calls
    assert (call.name, call.args, call.pages, call.prefetch) == (
        "search_filings",
        {"query": "What was revenue?"},
        [10, 11],
        True,
    )
    assert result.steps == 0
    assert [d.metadata["page"] for d in result.docs] == [10, 11]
    assert result.queries == ["What was revenue?"]
    assert result.context_chars == len(first.content.split("returned:\n\n", 1)[1])
    assert result.timing["searches"] == 1


def test_prefetch_does_not_count_toward_the_step_limit(run):
    search = ("search_filings", {"query": "balance sheet"})
    result, llm, searches = run(
        [reply("", search), reply("", search), reply("Answer: x")], max_steps=2, prefetch=True
    )

    assert len(searches) == 3  # the prefetch plus the model's 2
    assert result.steps == 2
    assert [c.prefetch for c in result.tool_calls] == [True, False, False]
    assert [choice for choice, _ in llm.calls] == [None, None, "none"]  # limit after 2, not 1


def test_without_prefetch_the_first_model_call_is_unchanged(run):
    result, llm, searches = run(
        [reply("", ("search_filings", {"query": "net sales"})), reply("Answer: x")],
        prefetch=False,
    )

    assert llm.events[0] == "model"
    system, first = llm.calls[0][1]
    assert (system.content, first.content) == (AGENT_PROMPT, "Question: What was revenue?")
    assert len(llm.calls[0][1]) == 2
    assert searches == [("net sales", "DOC", 7)]
    assert not any(c.prefetch for c in result.tool_calls)


# --- evidence checklist (AGENT_PREFETCH_PROMPT=v2) ---------------------------------------------


@pytest.mark.parametrize(
    "version, prompt", [("v1", AGENT_PREFETCH_PROMPT), ("v2", AGENT_PREFETCH_PROMPT_V2)]
)
def test_prefetch_prompt_setting_selects_the_system_prompt(run, version, prompt):
    _, llm, _ = run([reply("Answer: x")], prefetch=True, prefetch_prompt=version)

    assert llm.calls[0][1][0].content == prompt


def test_v1_prompt_is_unchanged_and_v2_asks_for_a_checklist():
    assert "Checklist:" not in AGENT_PREFETCH_PROMPT
    assert "Checklist:" in AGENT_PREFETCH_PROMPT_V2
    assert "one query per missing" in AGENT_PREFETCH_PROMPT_V2
    assert "all issued together in one step" in AGENT_PREFETCH_PROMPT_V2
    assert "general knowledge" in AGENT_PREFETCH_PROMPT_V2


def test_checklist_with_missing_items_then_searches_is_recorded(run):
    result, _, searches = run(
        [
            reply(
                "Checklist:\n- Revenue 2022: page 10\n- Debt 2022: MISSING\n- Cash 2022: MISSING",
                ("search_filings", {"query": "long-term debt"}),
                ("search_filings", {"query": "cash and cash equivalents"}),
            ),
            reply(
                "Checklist:\n- Revenue 2022: page 10\n- Debt 2022: page 20\n- Cash: page 30\n\n"
                "Answer: x"
            ),
        ],
        prefetch=True,
        prefetch_prompt="v2",
    )

    assert [q for q, _, _ in searches] == [
        "What was revenue?",
        "long-term debt",
        "cash and cash equivalents",
    ]
    assert result.checklist == {
        "written": True,
        "items": 3,
        "missing": ["Debt 2022: MISSING", "Cash 2022: MISSING"],
        "searches_same_step": 2,
        "searched_after": True,
        "final_missing": [],
    }


def test_checklist_with_missing_items_but_no_search_is_recorded(run):
    result, _, searches = run(
        [reply("Checklist:\n- Debt 2022: MISSING\n\nAnswer: about $40 billion")],
        prefetch=True,
        prefetch_prompt="v2",
    )

    assert len(searches) == 1  # the prefetch only
    assert result.checklist["missing"] == ["Debt 2022: MISSING"]
    assert result.checklist["searched_after"] is False


def test_checklist_is_not_recorded_with_the_v1_prompt(run):
    result, _, _ = run(
        [reply("Checklist:\n- Debt 2022: MISSING\n\nAnswer: x")],
        prefetch=True,
        prefetch_prompt="v1",
    )

    assert result.checklist is None


# --- timing -------------------------------------------------------------------------------------


def test_timing_fields_are_recorded(run):
    result, _, _ = run(
        [
            reply("", ("search_filings", {"query": "net sales"})),
            reply("", ("calculate", {"expression": "1 + 1"})),
            reply("Answer: 2"),
        ],
        prefetch=True,
    )

    t = result.timing
    assert t["model_calls"] == 3
    assert len(t["model_call_ms"]) == 3
    assert t["model_ms"] == pytest.approx(sum(t["model_call_ms"]))
    assert (t["searches"], t["calculate_calls"]) == (2, 1)
    for key in ("search_ms", "embed_ms", "db_ms", "pages_ms", "pdf_parse_ms", "calculate_ms"):
        assert t[key] >= 0
    assert t["pdf_parses"] == 0
    assert all(c.ms is not None and c.ms >= 0 for c in result.tool_calls)
    searches = [c for c in result.tool_calls if c.name == "search_filings"]
    assert all(isinstance(c.timing, dict) for c in searches)


def test_search_timing_breaks_down_embedding_db_pages_and_first_pdf_parse(monkeypatch):
    """Through the real expand/naive retrievers, with the store and PDF reader faked."""
    from stockdesk.retrieval import expand, naive
    from stockdesk.retrieval import pages as pages_module

    hit = Document(page_content="chunk", metadata={"doc_name": "DOC", "page": 4})
    store = type(
        "FakeStore",
        (),
        {
            "embeddings": type("E", (), {"embed_query": lambda self, q: [0.0]})(),
            "similarity_search_by_vector": lambda self, e, k, filter: [hit],
        },
    )()
    settings = graph.get_settings().model_copy(
        update={
            "retriever": "expand",
            "exclude_exhibits": False,
            "query_rewrite": "none",
            "hybrid": "none",
        }
    )
    for module in (graph, expand, naive):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(naive, "get_store", lambda: store)
    monkeypatch.setattr(pages_module, "load_pdf", lambda name: [hit])
    pages_module.page_texts.cache_clear()
    try:
        first = graph.search_filings("q", "DOC")[3]
        second = graph.search_filings("q", "DOC")[3]
    finally:
        pages_module.page_texts.cache_clear()

    for timing in (first, second):
        assert {"embed_ms", "db_ms", "pages_ms"} <= timing.keys()
    assert first["pdf_parse"] is True and first["pdf_parse_ms"] >= 0
    assert "pdf_parse" not in second  # cached: no parse the second time


# --- runner -------------------------------------------------------------------------------------


TIMING = {
    "model_calls": 3,
    "model_ms": 3000.0,
    "model_call_ms": [1000.0, 1000.0, 1000.0],
    "searches": 2,
    "search_ms": 400.0,
    "embed_ms": 300.0,
    "db_ms": 60.0,
    "pages_ms": 40.0,
    "pdf_parses": 1,
    "pdf_parse_ms": 35.0,
    "calculate_calls": 0,
    "calculate_ms": 0.0,
}


def test_runner_records_the_agent_trace_and_scores_all_searched_pages(monkeypatch):
    docs = [
        Document(page_content="Revenue was 42.", metadata={"doc_name": "DOC", "page": 3}),
        Document(page_content="Costs were 7.", metadata={"doc_name": "DOC", "page": 9}),
    ]
    result = agent.AgentResult(
        answer="Answer: 42 (page 9)",
        docs=docs,
        queries=["revenue", "costs"],
        tool_calls=[
            agent.ToolCall("search_filings", {"query": "revenue"}, [3]),
            agent.ToolCall("search_filings", {"query": "costs"}, [9]),
        ],
        steps=2,
        llm_calls=3,
        input_tokens=900,
        output_tokens=80,
        cache_read_tokens=512,
        llm_usage=[
            {"input_tokens": 200, "output_tokens": 20, "cache_read_tokens": 0},
            {"input_tokens": 700, "output_tokens": 60, "cache_read_tokens": 512},
        ],
        context_chars=123,
        timing=TIMING,
        checklist={
            "written": True,
            "items": 1,
            "missing": [],
            "searches_same_step": 0,
            "searched_after": False,
            "final_missing": [],
        },
    )
    settings = runner.get_settings().model_copy(update={"answer_mode": "agent"})
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(runner.agent, "answer", lambda q, d: result)
    monkeypatch.setattr(runner.rag, "answer", lambda q, d: pytest.fail("used the RAG path"))
    monkeypatch.setattr(runner, "judge", lambda q, g, a: Verdict(correct=True, reason="ok"))
    case = EvalCase(
        id="q1",
        doc_name="DOC",
        question="Revenue?",
        gold_answer="42",
        evidence_pages=[9],  # found by the second search only
        evidence_texts=["Costs were 7."],
    )

    record = runner.evaluate_case(case)

    assert record.error is None
    assert record.retrieved_pages == [3, 9]
    assert record.page_hit and record.evidence_hit
    assert record.steps == 2
    no_timing = {"result": None, "ms": None, "timing": None, "prefetch": False}
    assert record.tool_calls == [
        {"name": "search_filings", "args": {"query": "revenue"}, "pages": [3], **no_timing},
        {"name": "search_filings", "args": {"query": "costs"}, "pages": [9], **no_timing},
    ]
    assert (record.input_tokens, record.output_tokens) == (900, 80)
    assert record.cache_read_tokens == 512
    assert record.llm_usage == result.llm_usage
    assert record.timing == TIMING
    assert record.checklist == result.checklist
    assert record.context_chars == 123
    assert record.queries == ["revenue", "costs"]
