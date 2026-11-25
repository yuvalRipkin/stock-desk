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
