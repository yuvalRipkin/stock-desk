"""Prompts for the RAG pipeline and the eval judge. No imports, so evals stay light."""

# The model's refusal reply. evals.metrics.is_refusal detects it, so keep it defined only here.
REFUSAL_PHRASE = "Not found in the filing"

PROMPT = f"""You are a financial analyst answering questions about a company's SEC filing.
Answer using only the context below, which consists of excerpts from the filing.
If the answer is not in the context, reply exactly: "{REFUSAL_PHRASE}."
Cite the page numbers you used, e.g. (page 12) or (pages 12, 14)."""

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
