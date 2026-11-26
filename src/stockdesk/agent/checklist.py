"""Read the evidence checklist that AGENT_PREFETCH_PROMPT=v2 asks the model to write.

The prompt asks for:

    Checklist:
    - <figure or fact>: page <n>
    - <figure or fact>: MISSING

An item is missing if it names no page. The parser tolerates markdown ("**Checklist:**",
"* item", "1. item") since models add it.
"""

import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage

HEADER = re.compile(r"^[\s*#_>-]*checklist[\s*_]*:?[\s*_]*$", re.I)
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
PAGE = re.compile(r"\bpages?\s*\(?\s*\d", re.I)
MAX_ITEM_CHARS = 200


@dataclass
class Item:
    text: str  # the line without its bullet, truncated
    has_page: bool


def parse_checklist(text: str) -> list[Item] | None:
    """The items of the first checklist in text, or None if it has no checklist header."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if HEADER.match(line)), None)
    if start is None:
        return None
    items: list[Item] = []
    for line in lines[start + 1 :]:
        bullet = BULLET.match(line)
        if bullet:
            item = bullet.group(1).strip("*_ ")
            items.append(Item(text=item[:MAX_ITEM_CHARS], has_page=bool(PAGE.search(item))))
        elif items or line.strip():  # a blank line may sit between header and items
            break
    return items


def summarize(replies: list[AIMessage]) -> dict:
    """What the checklists in the model's replies show, for the eval records.

    written: a checklist was found in some reply.
    items / missing: the first checklist's item count and its items without a page.
    searches_same_step: search_filings calls in the reply that holds the first checklist.
    searched_after: the model called search_filings in that reply or a later one.
    final_missing: items without a page in the last checklist written (after any searches).
    """
    parsed = [(i, parse_checklist(m.text)) for i, m in enumerate(replies)]
    found = [(i, items) for i, items in parsed if items is not None]
    if not found:
        return {
            "written": False,
            "items": 0,
            "missing": [],
            "searches_same_step": 0,
            "searched_after": False,
            "final_missing": [],
        }

    def searches(message: AIMessage) -> int:
        return sum(call["name"] == "search_filings" for call in message.tool_calls)

    first_index, first = found[0]
    last = found[-1][1]
    return {
        "written": True,
        "items": len(first),
        "missing": [item.text for item in first if not item.has_page],
        "searches_same_step": searches(replies[first_index]),
        "searched_after": any(searches(m) for m in replies[first_index:]),
        "final_missing": [item.text for item in last if not item.has_page],
    }
