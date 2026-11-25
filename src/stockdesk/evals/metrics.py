"""Retrieval and answer metrics for evals. Pure functions: no LLM calls, no I/O.

Some cases have several gold evidence pages/texts. The hit metrics count a case as a hit
when ANY of them is retrieved: they measure "did retrieval find some gold evidence", not
"did it find all of it".
"""

import re
import string

from rapidfuzz import fuzz

from stockdesk.prompts import REFUSAL_PHRASE

# Version of the scoring rules, recorded in every results file.
#   1: an answer was a refusal if REFUSAL_PHRASE appeared anywhere in it, so a hedged answer
#      ("Customer deposits... Not found in the filing.") was marked wrong without being judged.
#   2: an answer is a refusal only if it is REFUSAL_PHRASE alone (after normalizing); anything
#      with other content goes to the judge, which grades the substantive answer.
#   3: also a refusal if the answer's final line is an "Answer:" line holding only
#      REFUSAL_PHRASE (answer prompt v2 sometimes lists its figures, then concludes that).
METRICS_VERSION = 3

EVIDENCE_MATCH_THRESHOLD = 85
REFUSAL = REFUSAL_PHRASE.lower()
# A final "Answer:" line, also in markdown ("**Answer:**", "- Answer:"); group 1 is its text.
FINAL_ANSWER_LINE = re.compile(r"^[\s*_#>-]*answer[\s*_]*:[\s*_]*(.*)$", re.I)


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


def _is_refusal_phrase(text: str) -> bool:
    """True if text is only the refusal phrase (case, whitespace, end punctuation ignored)."""
    return " ".join(text.lower().split()).rstrip(string.punctuation + " ") == REFUSAL


def is_refusal(answer: str) -> bool:
    """True if the answer is only the refusal phrase, or ends with an "Answer:" line that is.

    A hedged answer (a real answer that also says "Not found in the filing") is not a refusal:
    the judge grades its substantive part.
    """
    if _is_refusal_phrase(answer):
        return True
    lines = [line for line in answer.splitlines() if line.strip()]
    final = FINAL_ANSWER_LINE.match(lines[-1]) if lines else None
    return final is not None and _is_refusal_phrase(final.group(1))


def distinct_pages(retrieved_pages: list[int]) -> int:
    return len(set(retrieved_pages))
