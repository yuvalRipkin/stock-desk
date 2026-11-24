"""LLM judge: is a RAG answer correct against the FinanceBench gold answer?"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from stockdesk.config import get_settings
from stockdesk.evals.metrics import is_refusal
from stockdesk.prompts import JUDGE_PROMPT


class Verdict(BaseModel):
    correct: bool = Field(description="Whether the candidate answer is correct.")
    reason: str = Field(description="One sentence explaining the verdict.")


REFUSAL_VERDICT = Verdict(correct=False, reason="The answer is a refusal.")


def judge(question: str, gold_answer: str, answer: str) -> Verdict:
    """Grade `answer` against `gold_answer` with JUDGE_MODEL.

    Refusals are graded incorrect without calling the LLM, so they are free and deterministic.
    """
    if is_refusal(answer):
        return REFUSAL_VERDICT

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.judge_model, temperature=0, api_key=settings.openai_api_key
    ).with_structured_output(Verdict)
    return llm.invoke(
        [
            SystemMessage(JUDGE_PROMPT),
            HumanMessage(
                f"Question: {question}\n\nGold answer: {gold_answer}\n\nCandidate answer: {answer}"
            ),
        ]
    )
