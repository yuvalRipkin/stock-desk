"""Answer a question about one filing: retrieve chunks, build context, call the LLM.

This is the single answering path shared by the CLI (stockdesk.ask) and the evals.
"""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from stockdesk.config import get_settings
from stockdesk.prompts import PROMPT
from stockdesk.retrieval import get_retriever


class RagError(RuntimeError):
    """Raised when a question can't be answered for a setup reason (config, missing data)."""


@dataclass
class RagResult:
    answer: str
    docs: list[Document]
    queries: list[str]  # searched: the question, then any rewrites


def build_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[page {d.metadata['page']}]\n{d.page_content}" for d in docs)


def answer(question: str, doc_name: str, k: int | None = None) -> RagResult:
    settings = get_settings()
    if not settings.llm_model:
        raise RagError("LLM_MODEL is not set. Add LLM_MODEL=<model name> to .env.")

    retrieve = get_retriever(settings.retriever)
    retrieval = retrieve(question, doc_name, k if k is not None else settings.retrieval_k)
    docs = retrieval.docs
    if not docs:
        raise RagError(f"no chunks found for doc '{doc_name}'. Has it been ingested?")

    llm = ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)
    response = llm.invoke(
        [
            SystemMessage(PROMPT),
            HumanMessage(f"Context:\n{build_context(docs)}\n\nQuestion: {question}"),
        ]
    )
    return RagResult(answer=response.text, docs=docs, queries=retrieval.queries)
