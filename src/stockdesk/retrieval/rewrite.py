"""Query rewriting: turn a question into search queries phrased the way a 10-K is written."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from stockdesk.config import get_settings
from stockdesk.prompts import REWRITE_PROMPT


class SearchQueries(BaseModel):
    queries: list[str] = Field(description="Search queries for the filing. Never an answer.")


def rewrite_queries(question: str) -> list[str]:
    """Up to N_REWRITES search queries for the question, written by REWRITE_MODEL."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.rewrite_model, temperature=0, api_key=settings.openai_api_key
    ).with_structured_output(SearchQueries)
    result = llm.invoke(
        [
            SystemMessage(REWRITE_PROMPT.format(n=settings.n_rewrites)),
            HumanMessage(f"Question: {question}"),
        ]
    )
    queries = [q.strip() for q in result.queries if q.strip()]
    return queries[: settings.n_rewrites]
