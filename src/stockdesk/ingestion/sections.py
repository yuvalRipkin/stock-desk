"""Find 10-K section headings in page text and split pages at them.

Sections are the 10-K Items (named with their standard SEC names, whatever the filing prints)
and, inside Items 7 and 8, the financial statements, the notes and the auditor's report. A
page's text before its first heading belongs to the section in force from the previous page.
Pages before Item 1 are "Cover page"; pages from the exhibit cutoff on are "Exhibits".

False positives this avoids:
- Table of contents: a page listing TOC_MIN_ITEMS or more Items is skipped, and so is any Item
  line ending in a page number.
- References: an Item heading needs "Item N." followed by a title, so "Item 8 of this Annual
  Report" does not match, and an Item that does not come after the current one is ignored.
- Statement names in text: a statement title must be a whole line near the top of the page, and
  only counts before the notes start (Amcor's Note 22 has its own "Consolidated Statements").

Known quirk: PepsiCo prints its financial statements inside Item 7 (its Item 8 points back to
them), so its statements are labeled "Item 7. ... > Consolidated Balance Sheet", as filed.
"""

import re
from dataclasses import dataclass

SEC_ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer "
    "Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related "
    "Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
COVER = "Cover page"
EXHIBITS = "Exhibits"

ITEM_HEADING = re.compile(
    r"^\s*(?:PART\s+[IV]+\s+)?ITEMS?\s+(\d{1,2}[A-C]?)((?:\s*(?:,|and)\s*\d{1,2}[A-C]?)*)\s*\."
    r"\s*[-–—]?\s*[A-Z\[]",
    re.I,
)
TRAILING_PAGE_NUMBER = re.compile(r"\s\d{1,3}\s*$")
TOC_MIN_ITEMS = 8
SUB_HEADING = re.compile(
    r"^(consolidated\s+(balance\s+sheets?|statements?\s+of\s+(operations|income|earnings|"
    r"comprehensive\s+(income|loss)|cash\s+flows|financial\s+position|"
    r"(changes\s+in\s+)?(shareholders|stockholders)['’]?\s+equity|equity))"
    r"|notes\s+to\s+(the\s+)?consolidated\s+financial\s+statements"
    r"|report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm)"
    r"(\s*\(continued\))?\s*$",
    re.I,
)
SUB_HEADING_ITEMS = {"7", "8"}  # where financial statements are printed
TOP_LINES = 8  # a sub-heading must be among the first lines of its page
RUNNING_HEADER = re.compile(r"^table of contents$", re.I)
NOT_CONTENT = re.compile(r"^\s*(PART\s+[IV]+|table of contents)?\s*$", re.I)  # blank, "PART II"
CAPTION_MAX_LINES, CAPTION_MAX_CHARS = 2, 80
SMALL_WORDS = {"of", "to", "the", "and", "in", "for"}


@dataclass(frozen=True)
class Item:
    key: str  # e.g. "7A"; for a combined heading, the last Item ("13")
    label: str  # e.g. "Item 7A. Quantitative and ...", "Items 10–13. ..."


def item_order(key: str) -> tuple[int, str]:
    m = re.fullmatch(r"(\d+)([A-C]?)", key.upper())
    assert m, key
    return int(m.group(1)), m.group(2)


def parse_item(line: str) -> Item | None:
    """The Item a heading line opens, or None if the line is not an Item heading."""
    m = ITEM_HEADING.match(line)
    if not m or TRAILING_PAGE_NUMBER.search(line):
        return None
    keys = [m.group(1).upper(), *(k.upper() for k in re.findall(r"\d{1,2}[A-C]?", m.group(2)))]
    if any(k not in SEC_ITEMS for k in keys):
        return None
    if len(keys) == 1:
        return Item(keys[0], f"Item {keys[0]}. {SEC_ITEMS[keys[0]]}")
    names = "; ".join(SEC_ITEMS[k] for k in keys)
    return Item(keys[-1], f"Items {keys[0]}–{keys[-1]}. {names}")


def sub_heading_title(line: str) -> str | None:
    """The normalized title if the line is a statement, notes or auditor's report heading."""
    text = " ".join(line.split())
    if not SUB_HEADING.match(text):
        return None
    text = re.sub(r"\s*\(continued\)", "", text, flags=re.I)
    words = text.lower().split()
    return " ".join(w if i and w in SMALL_WORDS else w.capitalize() for i, w in enumerate(words))


def is_caption(content: list[str]) -> bool:
    """Whether a page's text before a sub-heading is only a short caption, like a company name."""
    return len(content) <= CAPTION_MAX_LINES and all(len(c) < CAPTION_MAX_CHARS for c in content)


def is_toc(lines: list[str]) -> bool:
    return sum(1 for line in lines if ITEM_HEADING.match(line)) >= TOC_MIN_ITEMS


def split_sections(
    pages: dict[int, str], exhibit_cutoff: int | None = None
) -> dict[int, list[tuple[str, str]]]:
    """Each page's text as (section, text) segments in page order, split at section headings.

    The section is "<Item> > <sub-heading>" inside a statement, the notes or the auditor's
    report, else "<Item>". A heading with nothing under it before the next heading (e.g. "Item 8"
    directly above "Consolidated Statements of Operations") is merged into the next segment.
    """
    item: Item | None = None
    sub: str | None = None
    notes_started = False
    out: dict[int, list[tuple[str, str]]] = {}

    def label() -> str:
        if item is None:
            return COVER
        return f"{item.label} > {sub}" if sub else item.label

    for page in sorted(pages):
        text = pages[page]
        if exhibit_cutoff is not None and page >= exhibit_cutoff:
            out[page] = [(EXHIBITS, text)]
            continue
        lines = text.splitlines(keepends=True)
        if is_toc(lines):
            out[page] = [(label(), text)]
            continue
        top = [s for s in (line.strip() for line in lines) if s and not RUNNING_HEADER.match(s)]
        top = top[:TOP_LINES]

        segments: list[tuple[str, list[str]]] = [(label(), [])]
        content: list[str] = []  # lines of the current segment besides its heading lines
        for line in lines:
            new_item = parse_item(line)
            if new_item and item is not None and item_order(new_item.key) <= item_order(item.key):
                new_item = None  # a reference back, not a heading
            title = None
            if new_item is None and line.strip() in top and item and item.key in SUB_HEADING_ITEMS:
                title = sub_heading_title(line)
                is_notes_or_report = title is not None and title.startswith(("Notes", "Report"))
                if title == sub or (notes_started and not is_notes_or_report):
                    title = None
            if new_item is None and title is None:
                segments[-1][1].append(line)
                if not NOT_CONTENT.match(line):
                    content.append(line.strip())
                continue

            # Relabel instead of splitting when nothing is under the previous heading yet, or
            # when a statement or the notes open the page under a short caption ("Amcor plc
            # and Subsidiaries"), which belongs to them. After the notes start, text above a
            # heading is the end of the notes.
            caption = (
                title is not None
                and not notes_started
                and len(segments) == 1
                and is_caption(content)
            )
            if new_item:
                item, sub, notes_started = new_item, None, False
            else:
                sub = title
                notes_started = notes_started or title.startswith("Notes")
            if content and not caption:
                segments.append((label(), [line]))
            else:
                segments[-1] = (label(), [*segments[-1][1], line])
            content = []

        out[page] = [(lab, "".join(seg)) for lab, seg in segments if "".join(seg).strip()]
    return out
