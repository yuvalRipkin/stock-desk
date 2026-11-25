"""Cross-encoder re-ranking: score each (question, chunk) pair jointly and sort by that score.

A cross-encoder reads the question and the chunk together, so it ranks more precisely than
vector similarity but costs one model pass per candidate. The model runs locally (MPS on Apple
Silicon, else CPU) and is loaded once per process.
"""

from functools import cache
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from stockdesk.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@cache
def load_model(name: str) -> "CrossEncoder":
    """The cross-encoder for RERANK_MODEL, downloaded on first use and cached per process."""
    from sentence_transformers import CrossEncoder  # imported here: slow, and only for RERANK

    return CrossEncoder(name, device=device())


def rerank(question: str, docs: list[Document]) -> list[Document]:
    """docs sorted by cross-encoder score, best first; ties keep their vector-search order."""
    if not docs:
        return []
    model = load_model(get_settings().rerank_model)
    scores = model.predict([(question, d.page_content) for d in docs])
    order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
    return [docs[i] for i in order]
