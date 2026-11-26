"""Groundedness checks: can every number, citation and tool claim in an answer be traced?

Usage (existing results files; reads the filing PDFs for page text, no RAG/agent/judge calls):
    uv run python -m stockdesk.evals.groundedness results/<name>.json [more.json ...] [--examples 5]

Per answer (check_record, pure):
1. Number grounding: each number the answer states must be traceable to the text of a page the
   system retrieved, a calculate result in the record's trace, or the question. Numbers are
   compared by magnitude (a parenthesized or minus-signed negative matches its positive), with
   commas, $ and % stripped and rounding allowed to the answer's stated precision
   (1.57 matches 1.5671). Scaling: "$4.8 billion" matches 4,835 on a page in millions (a page's
   units are unknown, so a scaled answer is tried against units, thousands, millions and
   billions); a bare number with 3+ significant digits may restate a page figure in other units
   (42.573 for 42,573); a percentage also matches a ratio (38.81% for 0.3881). Not checked:
   years, page numbers in citations, list numbering, labels glued to letters (FY22, 10-K, Q4),
   and the 100 of a "* 100" percentage conversion.
2. Citation validity: every cited page ("page 12", "pages 12, 14", "pages 47-48") must have been
   retrieved, and at least one of the answer's numbers must be on a cited page (when it states
   any).
3. Tool-claim validity: a line that claims a calculate result ("calculate(...) = x",
   "(calculate: ...)", "calculate result", "from calculate") needs a calculate call in the
   trace, and each number in the claimed result (after the line's last "=", else the whole
   line) that is on no page and not in the question must match a calculate result.
   "calculated" and "calculation" are not claims about the tool.
4. flagged: any of the three fails.

summarize() pools these over records, separately for correct and wrong answers. The checks
are new metrics only: accuracy, refusal and the hit rates are unchanged. The runner checks new
records against the text the model saw; this command re-reads the retrieved pages from the PDFs
(the same text for full-page retrieval, as in 012 and the agent).
"""

import argparse
import bisect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCALES = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}
# A number with thousands separators (taken whole) or plain digits, with optional decimals.
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+(?:\.\d+)?")
# Directly after a number: a scale suffix ("4.8bn", "$5M") or a multiple ("1.57x").
GLUED_SUFFIX = re.compile(r"(thousand|million|billion|trillion|mn|bn|tn|[kmbx])(?![A-Za-z])", re.I)
# After optional spaces: a percent sign or word, then an optional scale word.
PERCENT = re.compile(r"\s*(\\?%|percent(age points?)?\b|pp\b)", re.I)
SCALE_WORD = re.compile(r"\s*(thousand|million|billion|trillion)s?\b", re.I)
# Page citations: "page 12", "pages 12, 14", "pages 12 and 14", "pp. 47-48", "page 5; page 9".
CITATION = re.compile(
    r"\b(?:pages?|pp?\.)\s*(\d{1,4}(?:\s*(?:,|and|&|-|–|to)\s*\d{1,4}(?![\d,.]\d))*)", re.I
)
LIST_ITEM = re.compile(r"^[\s>#*-]*$")  # what may precede a list number on its line
# A claim about the calculate tool (the verb "calculate" in prose is not one).
TOOL_CLAIM = re.compile(
    r"\bcalculate\s*(?:\(|:|\)|`|result|returned|tool|expression|output|gave|gives)"
    r"|\b(?:using|from|via|with|per|by)\s+(?:the\s+)?`?calculate\b",
    re.I,
)
PERCENT_MULTIPLIER = re.compile(r"(?:[*×]|\\times|\bx)\s*$")


@dataclass(frozen=True)
class Number:
    text: str  # as written, e.g. "$4.8 billion"
    value: float  # magnitude with any scale word applied (sign dropped)
    tolerance: float  # half a unit of the last stated digit, scaled
    scaled: bool  # had a scale word or suffix
    percent: bool
    sig_digits: int
    start: int
    end: int


def _significant_digits(digits: str) -> int:
    return len(digits.replace(",", "").replace(".", "").lstrip("0")) or 1


def extract_numbers(text: str) -> list[Number]:
    """Every number in text, with its scale, percent sign and stated precision.

    Labels glued to letters (FY2022, Q4, 10-K, COVID-19) are skipped.
    """
    numbers = []
    for m in NUMBER.finditer(text):
        start, end, digits = m.start(), m.end(), m.group()
        before = text[max(0, start - 2) : start]
        if before[-1:].isalpha() or before[-1:] in {"_", "."} or re.fullmatch(r"[A-Za-z]-", before):
            continue
        after = text[end:]
        if re.match(r"-[A-Za-z]", after):
            continue
        scale, percent, text_end = 1.0, False, end
        if after[:1].isalpha():
            glued = GLUED_SUFFIX.match(after)
            if not glued:
                continue
            suffix = glued.group(1).lower()
            scale = 1.0 if suffix == "x" else SCALES[suffix]
            text_end += glued.end()
        else:
            if p := PERCENT.match(after):
                percent, text_end = True, end + p.end()
            if s := SCALE_WORD.match(text, text_end):
                scale, text_end = SCALES[s.group(1).lower()], s.end()
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        numbers.append(
            Number(
                text=text[start - (before[-1:] == "$") : text_end].strip(),
                value=float(digits.replace(",", "")) * scale,
                tolerance=0.5 * 10**-decimals * scale,
                scaled=scale != 1.0,
                percent=percent,
                sig_digits=_significant_digits(digits),
                start=start,
                end=text_end,
            )
        )
    return numbers


def cited_pages(answer: str) -> tuple[list[int], list[tuple[int, int]]]:
    """The pages the answer cites (in order, once each) and the character spans of the citations."""
    pages: list[int] = []
    spans = []
    for m in CITATION.finditer(answer):
        spans.append(m.span(1))
        for part in re.split(r"\s*(?:,|and|&)\s*", m.group(1)):
            bounds = [int(x) for x in re.split(r"\s*(?:-|–|to)\s*", part) if x]
            if len(bounds) == 2 and 0 <= bounds[1] - bounds[0] <= 10:
                bounds = list(range(bounds[0], bounds[1] + 1))
            pages += [p for p in bounds if p not in pages]
    return pages, spans


def _is_year(n: Number, answer: str) -> bool:
    raw = answer[n.start : n.end]
    return (
        raw.isdigit()
        and 1900 <= n.value <= 2100
        and not n.percent
        and answer[max(0, n.start - 1) : n.start] != "$"
    )


def _is_list_number(n: Number, answer: str) -> bool:
    line_start = answer.rfind("\n", 0, n.start) + 1
    return bool(
        LIST_ITEM.match(answer[line_start : n.start])
        and re.match(r"[.)]\s", answer[n.end : n.end + 2])
    )


def answer_figures(answer: str) -> list[Number]:
    """The numbers to check: all of the answer's numbers except years, cited page numbers, list
    numbering and the 100 of a "* 100" percentage conversion."""
    _, spans = cited_pages(answer)
    figures = []
    for n in extract_numbers(answer):
        if any(s <= n.start < e for s, e in spans) or _is_year(n, answer):
            continue
        if _is_list_number(n, answer):
            continue
        if n.value == 100 and PERCENT_MULTIPLIER.search(answer[max(0, n.start - 8) : n.start]):
            continue
        figures.append(n)
    return figures


class Source:
    """The numbers of one text (a page, the calculate results, the question), for lookup."""

    def __init__(self, values: list[float]):
        self.values = sorted({abs(v) for v in values})

    @classmethod
    def from_text(cls, text: str) -> "Source":
        # Raw digits, no unit handling: the answer side decides which units to try. A
        # percentage is also offered as its ratio (20% as 0.20).
        values = []
        for m in NUMBER.finditer(text):
            value = float(m.group().replace(",", ""))
            values.append(value)
            if PERCENT.match(text, m.end()):
                values.append(value / 100)
        return cls(values)

    def _has(self, lo: float, hi: float) -> bool:
        i = bisect.bisect_left(self.values, lo)
        return i < len(self.values) and self.values[i] <= hi

    def contains(self, n: Number) -> bool:
        """True if some number here equals n after rounding to n's precision, in some units."""
        tol = n.tolerance + 1e-9 * max(1.0, n.value)
        if n.percent:  # a percentage or its ratio, as written
            units = [1.0]
        elif n.scaled:  # the page may be in units, thousands, millions or billions
            units = [1.0, 1e3, 1e6, 1e9]
        elif n.sig_digits >= 3:  # may restate a figure in other units (42.573 for 42,573)
            units = [10.0 ** (3 * k) for k in range(-3, 4)]
        else:  # a 1-2 digit bare number must match as written
            units = [1.0]
        targets = [n.value] + ([n.value / 100] if n.percent else [])
        for target in targets:
            t_tol = tol if target == n.value else tol / 100
            for u in units:
                if self._has((target - t_tol) / u, (target + t_tol) / u):
                    return True
        return False


def calculate_results(tool_calls: list[dict] | None) -> list[float]:
    """Every number calculate returned in the trace (v1: "1.766"; v2: "expr = 1.766" per line)."""
    results = []
    for call in tool_calls or []:
        if call.get("name") != "calculate" or not call.get("result"):
            continue
        for line in call["result"].splitlines():
            value = line.rsplit(" = ", 1)[-1].strip()
            try:
                results.append(float(value.replace(",", "")))
            except ValueError:
                continue  # an error message
    return results


def tool_claim_lines(answer: str) -> list[str]:
    return [line.strip() for line in answer.splitlines() if TOOL_CLAIM.search(line)]


def check_record(record: dict, pages: dict[int, str]) -> dict | None:
    """Groundedness of one saved record; None if it has no answer (an errored question).

    pages: page number -> text for the record's filing (at least its retrieved pages). Only the
    record's retrieved_pages are used as evidence.
    """
    answer = record.get("answer")
    if answer is None or record.get("error") is not None:
        return None
    retrieved = set(record["retrieved_pages"])
    page_sources = {p: Source.from_text(pages.get(p, "")) for p in retrieved}
    calc_values = calculate_results(record.get("tool_calls"))
    calc = Source(calc_values)
    question = Source.from_text(record["question"])

    figures = answer_figures(answer)
    untraceable, on_page = [], {}
    for n in figures:
        found = [p for p, src in page_sources.items() if src.contains(n)]
        on_page[n] = found
        if not found and not calc.contains(n) and not question.contains(n):
            untraceable.append(n.text)

    cited, _ = cited_pages(answer)
    invalid_citations = [p for p in cited if p not in retrieved]
    valid_cited = set(cited) & retrieved
    figure_on_cited_page = (
        any(set(found) & valid_cited for found in on_page.values()) if cited and figures else None
    )
    citations_ok = not invalid_citations and figure_on_cited_page is not False

    claims = tool_claim_lines(answer)
    false_claims = []
    for line in claims:
        result_part = line.rsplit("=", 1)[-1]  # the claimed result, when the line has one
        derived = [
            n
            for n in answer_figures(result_part)
            if not any(src.contains(n) for src in page_sources.values())
            and not question.contains(n)
        ]
        if not calc_values or any(not calc.contains(n) for n in derived):
            false_claims.append(line)

    return {
        "numbers": len(figures),
        "untraceable_numbers": untraceable,
        "cited_pages": cited,
        "invalid_citations": invalid_citations,
        "figure_on_cited_page": figure_on_cited_page,
        "tool_claims": len(claims),
        "false_tool_claims": false_claims,
        "flagged": bool(untraceable) or not citations_ok or bool(false_claims),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _citation_failed(check: dict) -> bool:
    return bool(check["invalid_citations"]) or check["figure_on_cited_page"] is False


def summarize(records: list[dict]) -> dict[str, float | int | None]:
    """Pooled groundedness metrics over records that hold a "groundedness" check, for correct
    and for wrong answers. Rates: grounded numbers / numbers; answers failing the citation check
    / answers citing a page; answers with a false calculate claim / answers claiming one;
    flagged answers / answers."""
    out: dict[str, float | int | None] = {}
    for group, want in (("correct", True), ("wrong", False)):
        checks = [
            r["groundedness"] for r in records if r.get("groundedness") and r["correct"] == want
        ]
        numbers = sum(c["numbers"] for c in checks)
        citing = [c for c in checks if c["cited_pages"]]
        claiming = [c for c in checks if c["tool_claims"]]
        out |= {
            f"answers_{group}": len(checks),
            f"grounded_number_rate_{group}": _rate(
                numbers - sum(len(c["untraceable_numbers"]) for c in checks), numbers
            ),
            f"invalid_citation_rate_{group}": _rate(
                sum(_citation_failed(c) for c in citing), len(citing)
            ),
            f"false_tool_claim_rate_{group}": _rate(
                sum(bool(c["false_tool_claims"]) for c in claiming), len(claiming)
            ),
            f"flagged_rate_{group}": _rate(sum(c["flagged"] for c in checks), len(checks)),
        }
    return out


RATES = ["grounded_number_rate", "invalid_citation_rate", "false_tool_claim_rate", "flagged_rate"]
GROUNDEDNESS_METRICS = [f"{m}_{g}" for m in RATES for g in ("correct", "wrong")]


# --- command: existing results files ----------------------------------------------------------


def check_results(results: dict) -> list[dict]:
    """Every record of a results file with its "groundedness" check added (copies)."""
    from stockdesk.retrieval.pages import page_texts  # reads the PDFs; keeps the core pure

    checked = []
    for run_idx, run in enumerate(results["runs"], 1):
        for r in run["records"]:
            pages = page_texts(r["doc_name"]) if r.get("answer") is not None else {}
            checked.append({**r, "run": run_idx, "groundedness": check_record(r, pages)})
    return checked


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def counts(records: list[dict], group: bool) -> str:
    checks = [r["groundedness"] for r in records if r["groundedness"] and r["correct"] == group]
    numbers = sum(c["numbers"] for c in checks)
    untraceable = sum(len(c["untraceable_numbers"]) for c in checks)
    citing = sum(bool(c["cited_pages"]) for c in checks)
    claiming = sum(bool(c["tool_claims"]) for c in checks)
    return f"{numbers} numbers ({untraceable} untraceable), {citing} citing, {claiming} claiming"


def print_example(r: dict) -> None:
    g = r["groundedness"]
    print(
        f"  - {r['id']} run {r['run']} ({'correct' if r['correct'] else 'wrong'}), "
        f"retrieved {r['retrieved_pages']}"
    )
    if g["untraceable_numbers"]:
        print(f"    untraceable: {g['untraceable_numbers']}")
    if g["invalid_citations"]:
        print(f"    cited but not retrieved: {g['invalid_citations']}")
    if g["figure_on_cited_page"] is False:
        print(f"    no figure on cited pages {g['cited_pages']}")
    for line in g["false_tool_claims"]:
        print(f"    false calculate claim: {line[:160]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Groundedness metrics for results files.")
    parser.add_argument("paths", nargs="+", type=Path, help="results/<name>.json files")
    parser.add_argument("--examples", type=int, default=0, help="print N flagged records per file")
    args = parser.parse_args(argv)

    for path in args.paths:
        records = check_results(json.loads(path.read_text(encoding="utf-8")))
        summary = summarize(records)
        print(f"{path.name}: pooled over runs")
        print(f"  {'':<24}{'correct':>10}{'wrong':>10}")
        print(f"  {'answers':<24}{summary['answers_correct']:>10}{summary['answers_wrong']:>10}")
        for metric in RATES:
            c, w = summary[f"{metric}_correct"], summary[f"{metric}_wrong"]
            print(f"  {metric:<24}{_fmt(c):>10}{_fmt(w):>10}")
        print(f"  correct: {counts(records, True)}")
        print(f"  wrong:   {counts(records, False)}")
        flagged = [r for r in records if r["groundedness"] and r["groundedness"]["flagged"]]
        for r in flagged[: args.examples]:
            print_example(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
