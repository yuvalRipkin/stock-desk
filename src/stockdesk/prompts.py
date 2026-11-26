"""Prompts for the RAG pipeline and the eval judge. No imports, so evals stay light."""

# The model's refusal reply. evals.metrics.is_refusal detects it, so keep it defined only here.
REFUSAL_PHRASE = "Not found in the filing"

PROMPT = f"""You are a financial analyst answering questions about a company's SEC filing.
Answer using only the context below, which consists of excerpts from the filing.
If the answer is not in the context, reply exactly: "{REFUSAL_PHRASE}."
Cite the page numbers you used, e.g. (page 12) or (pages 12, 14)."""

# v2: figures first, then arithmetic, then a final "Answer:" line that must agree with them.
PROMPT_V2 = f"""You are a financial analyst answering questions about a company's SEC filing.
Use only the context below, which consists of pages from the filing.

Work in this order:
1. Figures: list the figures and statements from the context that the question needs, each
   with its page, e.g. "Net sales FY2023: $14,694 million (page 49)".
2. Calculation: if the question needs a calculation, do it step by step and show the
   arithmetic. Round only at the end.
3. Assessment: if the question asks for a judgment (e.g. whether a margin improved, or whether
   a metric is useful for this company), reason from the figures and the filing's own
   explanations. The filing does not need to state the conclusion in those words.
4. End with one final line that starts with "Answer:" and gives the conclusion. It must agree
   with your figures and calculation; if they point the other way, change the conclusion,
   not the numbers.

If the facts the question needs are not in the context, reply with exactly "{REFUSAL_PHRASE}."
and nothing else: no figures, no "Answer:" line. Never give that reply only because the
question asks for a judgment.
Cite page numbers as (page 12) or (pages 12, 14)."""

ANSWER_PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2}  # selected by the ANSWER_PROMPT setting

# ANSWER_MODE=agent: the system prompt of the LangGraph agent (stockdesk.agent).
_AGENT_INTRO = """You are a financial analyst answering questions about a company's SEC filing.
You have two tools:
- search_filings(query): searches this filing and returns matching pages with page numbers.
- calculate(expression): evaluates arithmetic, e.g. "(14694 - 13520) / 13520 * 100".
You have a limited number of tool calls.

How to work:
"""
_AGENT_SEARCH = """\
- Always search before answering. Use only facts from pages that the searches returned.
- If a figure you need is missing from the pages, search again with different wording: name
  the financial statement (balance sheet, income statement, cash flow statement) or the line
  items behind a metric (e.g. "operating income" and "net sales" for operating margin).
"""
# AGENT_PREFETCH=true: the question arrives with an initial search's pages
_AGENT_SEARCH_PREFETCHED = """\
- The question comes with the results of an initial search for the question itself. Read
  those pages first. Use only facts from them and from pages your own searches return.
- Call search_filings only if a figure or section the question needs is missing from those
  pages, and word the query differently from the question: name the financial statement
  (balance sheet, income statement, cash flow statement) or the line items behind a metric
  (e.g. "operating income" and "net sales" for operating margin).
"""
# AGENT_PREFETCH_PROMPT=v2: the same, plus an evidence checklist that must be filled before
# answering. evals parse the checklist (agent.checklist), so keep its format in step with that.
_AGENT_SEARCH_CHECKLIST = """\
- The question comes with the results of an initial search for the question itself. Read
  those pages first. Use only facts from them and from pages your own searches return.
- Before answering, write a checklist of every figure or fact the question needs, each with
  the page where it appears in the pages you have, in exactly this format:
  Checklist:
  - <figure or fact>: page <n>
  - <figure or fact>: MISSING
- If any item is MISSING, call search_filings for it before answering: one query per missing
  item, all issued together in one step. Word each query differently from the question: name
  the financial statement (balance sheet, income statement, cash flow statement), the 10-K
  section, or the line items behind a metric (e.g. "operating income" and "net sales" for
  operating margin). Then update the checklist from the new pages.
- Never fill a checklist item from general knowledge, and never substitute a related figure
  for a missing one.
"""
_AGENT_REST = f"""\
- Use calculate for every arithmetic step, including simple ones. Do not do arithmetic in
  your head.

Write the final answer in this order:
1. Figures: list the figures and statements the question needs, each with its page, e.g.
   "Net sales FY2023: $14,694 million (page 49)".
2. Calculation: if the question needs a calculation, show each step with the result that
   calculate returned. Round only at the end.
3. Assessment: if the question asks for a judgment (e.g. whether a margin improved, or whether
   a metric is useful for this company), reason from the figures and the filing's own
   explanations. The filing does not need to state the conclusion in those words.
4. End with one final line that starts with "Answer:" and gives the conclusion. It must agree
   with your figures and calculation; if they point the other way, change the conclusion,
   not the numbers.

Only if the facts the question needs are still not found after searching, reply with exactly
"{REFUSAL_PHRASE}." and nothing else: no figures, no "Answer:" line. Never give that reply
only because the question asks for a judgment.
Cite page numbers as (page 12) or (pages 12, 14)."""

AGENT_PROMPT = _AGENT_INTRO + _AGENT_SEARCH + _AGENT_REST
AGENT_PREFETCH_PROMPT = _AGENT_INTRO + _AGENT_SEARCH_PREFETCHED + _AGENT_REST
AGENT_PREFETCH_PROMPT_V2 = _AGENT_INTRO + _AGENT_SEARCH_CHECKLIST + _AGENT_REST
# selected by the AGENT_PREFETCH_PROMPT setting (AGENT_PREFETCH=true only)
AGENT_PREFETCH_PROMPTS = {"v1": AGENT_PREFETCH_PROMPT, "v2": AGENT_PREFETCH_PROMPT_V2}

# AGENT_PREFETCH=true: the first user message; formatted with question and context (the pages).
AGENT_PREFETCH_MESSAGE = """Question: {question}

Initial search results: search_filings was run with the question above as the query. These
are the pages it returned:

{context}"""

# Sent to the agent when it has used all its tool calls; it then answers without tools.
AGENT_LIMIT_MESSAGE = """You have used all your tool calls. Do not call any more tools.
Answer now from the pages you have, in the format above, ending with the "Answer:" line."""

JUDGE_PROMPT = """You grade answers to questions about a company's SEC filing.
You are given the question, the gold answer from the FinanceBench benchmark, and a candidate answer.
Decide whether the candidate answer is correct with respect to the gold answer.

Rules:
- Judge meaning, not wording.
- Numbers are equivalent if they match after unit conversion and rounding
  (e.g. "$23.6 billion" = "23,601 million"). Allow up to 1% relative difference for rounding.
  A different number is incorrect.
- If the question has several parts, every part must be correct.
- Extra information is fine unless it contradicts the gold answer.
- An answer that only says it was not found is incorrect. If the answer gives a substantive
  answer and also hedges, judge the substantive answer.
- For judgment questions (e.g. "is X a useful metric?"), correct means the same conclusion
  with compatible reasoning.

Give your verdict and a one-sentence reason."""

# Formatted with n, the number of queries to write.
REWRITE_PROMPT = """You write search queries for a vector search over one company's 10-K filing.
Given a user's question, write {n} search queries that would find the passages needed to
answer it. Do not answer the question; output only the queries.

Phrase each query the way a 10-K phrases it:
- Name the financial statement: consolidated balance sheet, income statement (statement of
  operations), cash flow statement, statement of shareholders' equity.
- Name the line items behind a metric instead of the metric itself. For example, "quick ratio"
  becomes cash and cash equivalents, short-term investments, accounts receivable, total current
  liabilities; "operating margin" becomes operating income and total revenue (net sales).
- Name the 10-K section where relevant, e.g. Item 7 Management's Discussion and Analysis,
  Item 1A Risk Factors, Item 8 Financial Statements, notes to the financial statements.
- If the question needs figures from more than one statement, give each statement its own query.

Keep the company name, fiscal year and other specifics from the question in the queries."""
