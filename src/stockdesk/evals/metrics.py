"""Retrieval and answer metrics for evals. Pure functions: no LLM calls, no I/O.

Some cases have several gold evidence pages/texts. The hit metrics count a case as a hit
when ANY of them is retrieved: they measure "did retrieval find some gold evidence", not
"did it find all of it".
"""

from rapidfuzz import fuzz

from stockdesk.prompts import REFUSAL_PHRASE

EVIDENCE_MATCH_THRESHOLD = 85
REFUSAL = REFUSAL_PHRASE.lower()


def page_hit(retrieved_pages: list[int], evidence_pages: list[int]) -> bool:
    """True if any gold evidence page was retrieved. False if either list is empty."""
    return bool(set(retrieved_pages) & set(evidence_pages))


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def evidence_hit(
    chunks: list[str], evidence_texts: list[str], threshold: int = EVIDENCE_MATCH_THRESHOLD
) -> bool:
    """True if any gold evidence text fuzzy-matches any retrieved chunk.

    Both sides are lowercased and whitespace-collapsed, then compared with
    fuzz.partial_ratio with the shorter string first, so it works whether the evidence
    is longer or shorter than the chunk. Empty evidence texts are skipped.
    """
    norm_chunks = [_normalize(c) for c in chunks]
    for evidence in evidence_texts:
        ev = _normalize(evidence)
        if not ev:
            continue
        for chunk in norm_chunks:
            short, long = (ev, chunk) if len(ev) <= len(chunk) else (chunk, ev)
            if fuzz.partial_ratio(short, long) >= threshold:
                return True
    return False


def is_refusal(answer: str) -> bool:
    """True if the answer contains the prompt's refusal phrase (case-insensitive)."""
    return REFUSAL in answer.lower()


def distinct_pages(retrieved_pages: list[int]) -> int:
    return len(set(retrieved_pages))
