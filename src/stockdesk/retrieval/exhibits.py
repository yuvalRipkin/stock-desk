"""Find where a 10-K's attached exhibits begin, so retrieval can skip them.

A 10-K ends with a SIGNATURES page; the attached exhibits (plans, agreements, certifications)
follow. Some filings put real 10-K content between the two (Amcor's Schedule II, American
Express's statistical appendix), so the cutoff is the first page after the signature page that
opens with "Exhibit N", not the page after the signatures.
"""

import re
from functools import lru_cache

from stockdesk.retrieval.pages import page_texts

SIGNATURES = re.compile(
    r"^(Table of Contents )?SIGNATURES? Pursuant to the requirements of Section 13", re.I
)
EXHIBIT_HEADING = re.compile(r"\bEXHIBIT\s+\d+(\.\d+)?\b", re.I)
HEADING_CHARS = 200  # an exhibit names itself near the top of its first page


def find_exhibit_cutoff(pages: dict[int, str]) -> int | None:
    """First exhibit page after the SIGNATURES page, or None if either is not found."""
    flat = {i: " ".join(text.split()) for i, text in sorted(pages.items())}
    signatures = next((i for i, text in flat.items() if SIGNATURES.search(text)), None)
    if signatures is None:
        return None
    return next(
        (
            i
            for i, text in flat.items()
            if i > signatures and EXHIBIT_HEADING.search(text[:HEADING_CHARS])
        ),
        None,
    )


@lru_cache(maxsize=32)  # one entry per filing
def exhibit_cutoff(doc_name: str) -> int | None:
    """Pages at or after this number are exhibits; None means nothing is excluded.

    Detected from the PDF text and cached per process. Production would store an is_exhibit
    flag per page at ingestion instead.
    """
    return find_exhibit_cutoff(page_texts(doc_name))
