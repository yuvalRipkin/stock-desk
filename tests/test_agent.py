"""Tests for the LangGraph agent. The model and the retriever are fakes: no network, no DB."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from stockdesk import agent
from stockdesk.agent import graph
from stockdesk.evals import runner
from stockdesk.evals.dataset import EvalCase
from stockdesk.evals.judge import Verdict
from stockdesk.prompts import AGENT_LIMIT_MESSAGE, AGENT_PROMPT
from stockdesk.rag import RagError
from stockdesk.retrieval import Retrieval


def reply(text: str = "", *calls: tuple[str, dict], tokens: tuple[int, int] = (10, 2)):
    """A scripted model reply with optional tool calls and token usage."""
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
        },
    )


class ScriptedLLM:
    """Returns the scripted replies in order; records (tool_choice, messages) for each call."""

    def __init__(self, replies: list[AIMessage]):
        self.replies = list(replies)
        self.calls: list[tuple[str | None, list]] = []

    def bind_tools(self, tools, tool_choice=None):
        assert [t["function"]["name"] for t in tools] == ["search_filings", "calculate"]
        llm = self

        class Bound:
            def invoke(self, messages):
                llm.calls.append((tool_choice, list(messages)))
                if not llm.replies:
                    raise AssertionError("the model was called more often than scripted")
                return llm.replies.pop(0)

        return Bound()


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch):
    """run(replies, max_steps=6) -> (AgentResult, model, searches as (query, doc_name, k))."""
    searches: list[tuple[str, str | None, int | None]] = []

    def fake_search(query: str, doc_name: str | None, k: int | None) -> Retrieval:
        searches.append((query, doc_name, k))
        page = 10 * len(searches)  # pages 10, 11 for the 1st search, 20, 21 for the 2nd, ...
        docs = [
            Document(page_content=f"text of page {p}", metadata={"doc_name": doc_name, "page": p})
            for p in (page, page + 1)
        ]
        return Retrieval(docs, [query])

    monkeypatch.setattr(graph, "get_retriever", lambda name: fake_search)

    def _run(replies: list[AIMessage], max_steps: int = 6, doc_name: str = "DOC"):
        llm = ScriptedLLM(replies)
        settings = graph.get_settings().model_copy(
            update={"llm_model": "fake-model", "max_agent_steps": max_steps, "retrieval_k": 7}
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
            reply("", ("calculate", {"expression": "(120 - 100) / 100 * 100"}), tokens=(200, 6)),
            reply("Figures...\nAnswer: 20% (page 10)", tokens=(300, 40)),
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


# --- runner -------------------------------------------------------------------------------------


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
        context_chars=123,
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
    assert record.tool_calls == [
        {"name": "search_filings", "args": {"query": "revenue"}, "pages": [3], "result": None},
        {"name": "search_filings", "args": {"query": "costs"}, "pages": [9], "result": None},
    ]
    assert (record.input_tokens, record.output_tokens) == (900, 80)
    assert record.context_chars == 123
    assert record.queries == ["revenue", "costs"]
