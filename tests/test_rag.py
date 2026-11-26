"""Tests for answer-prompt selection in rag.answer. Retriever and LLM are mocked: no network."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from stockdesk import rag
from stockdesk.evals.metrics import is_refusal
from stockdesk.prompts import ANSWER_PROMPTS, PROMPT, PROMPT_V2, REFUSAL_PHRASE
from stockdesk.retrieval import Retrieval


class FakeLLM:
    """Records the messages it is sent and replies with a fixed text."""

    def __init__(self, sent: list, **_kwargs):
        self.sent = sent

    def invoke(self, messages):
        self.sent.append(messages)
        return AIMessage(
            content="Answer: 42 (page 1)",
            usage_metadata={
                "input_tokens": 2000,
                "output_tokens": 50,
                "total_tokens": 2050,
                "input_token_details": {"cache_read": 1024},
            },
        )


@pytest.fixture
def system_prompt(monkeypatch: pytest.MonkeyPatch):
    """Run rag.answer with ANSWER_PROMPT set; return the system prompt the LLM received."""
    sent: list = []
    doc = Document(page_content="Revenue was 42.", metadata={"doc_name": "DOC", "page": 1})
    monkeypatch.setattr(rag, "get_retriever", lambda name: lambda q, d, k: Retrieval([doc], [q]))
    monkeypatch.setattr(rag, "ChatOpenAI", lambda **kw: FakeLLM(sent, **kw))

    def run(answer_prompt: str) -> str:
        settings = rag.get_settings().model_copy(
            update={"answer_prompt": answer_prompt, "llm_model": "fake-model"}
        )
        monkeypatch.setattr(rag, "get_settings", lambda: settings)
        sent.clear()
        rag.answer("What was revenue?", "DOC")
        return sent[0][0].content

    return run


@pytest.mark.parametrize("name, prompt", [("v1", PROMPT), ("v2", PROMPT_V2)])
def test_setting_selects_the_prompt(system_prompt, name: str, prompt: str):
    assert system_prompt(name) == prompt


def test_prompts_are_distinct_and_v1_is_unchanged():
    assert ANSWER_PROMPTS == {"v1": PROMPT, "v2": PROMPT_V2}
    assert PROMPT != PROMPT_V2
    assert PROMPT.startswith("You are a financial analyst answering questions")


def test_v2_asks_for_the_exact_refusal_reply():
    assert f'reply with exactly "{REFUSAL_PHRASE}."' in PROMPT_V2
    assert is_refusal(f"{REFUSAL_PHRASE}.")  # the reply v2 asks for counts as a refusal


def test_v2_asks_for_a_final_answer_line():
    assert 'starts with "Answer:"' in PROMPT_V2


def test_answer_records_token_usage_including_cache_reads(system_prompt, monkeypatch):
    system_prompt("v2")  # sets up the fakes and settings
    result = rag.answer("What was revenue?", "DOC")
    assert (result.input_tokens, result.output_tokens, result.cache_read_tokens) == (2000, 50, 1024)
