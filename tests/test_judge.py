"""Tests for the LLM judge. The chat model is mocked: no real LLM calls."""

from unittest.mock import MagicMock

import pytest

from stockdesk.evals import judge as judge_module
from stockdesk.evals.judge import Verdict, judge
from stockdesk.prompts import REFUSAL_PHRASE

QUESTION = "What was AMD's total revenue in FY2022?"
GOLD = "$23,601 million"
ANSWER = "Total net revenue in 2022 was $23.601 billion (page 47)."


@pytest.fixture
def chat_model(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ChatOpenAI in the judge module; returns the structured-output runnable mock."""
    chat_cls = MagicMock()
    structured = chat_cls.return_value.with_structured_output.return_value
    structured.invoke.return_value = Verdict(correct=True, reason="Same number.")
    monkeypatch.setattr(judge_module, "ChatOpenAI", chat_cls)
    return structured


def prompt_text(structured: MagicMock) -> str:
    messages = structured.invoke.call_args.args[0]
    return "\n".join(m.content for m in messages)


def test_prompt_contains_question_gold_and_answer(chat_model: MagicMock) -> None:
    judge(QUESTION, GOLD, ANSWER)

    text = prompt_text(chat_model)
    assert QUESTION in text
    assert GOLD in text
    assert ANSWER in text


def test_returns_mocked_verdict_unchanged(chat_model: MagicMock) -> None:
    expected = Verdict(correct=False, reason="Different number.")
    chat_model.invoke.return_value = expected

    assert judge(QUESTION, GOLD, ANSWER) is expected


def test_uses_structured_output(chat_model: MagicMock) -> None:
    judge(QUESTION, GOLD, ANSWER)

    chat_cls = judge_module.ChatOpenAI
    chat_cls.return_value.with_structured_output.assert_called_once_with(Verdict)


def test_refusal_short_circuits_without_llm(chat_model: MagicMock) -> None:
    verdict = judge(QUESTION, GOLD, f"{REFUSAL_PHRASE}.")

    assert verdict.correct is False
    chat_model.invoke.assert_not_called()
    judge_module.ChatOpenAI.assert_not_called()


def test_hedged_answer_is_judged_not_short_circuited(chat_model: MagicMock) -> None:
    hedged = f"Customer deposits were the largest liability. {REFUSAL_PHRASE}."

    judge(QUESTION, GOLD, hedged)

    chat_model.invoke.assert_called_once()
    assert hedged in prompt_text(chat_model)


def test_judge_prompt_grades_the_substantive_part_of_hedged_answers() -> None:
    from stockdesk.prompts import JUDGE_PROMPT

    assert "only says it was not found is incorrect" in JUDGE_PROMPT
    assert "judge the substantive answer" in JUDGE_PROMPT
