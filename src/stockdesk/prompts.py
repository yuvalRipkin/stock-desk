"""Prompts for the RAG pipeline. No imports, so evals can use them without loading the LLM stack."""

# The model's refusal reply. evals.metrics.is_refusal detects it, so keep it defined only here.
REFUSAL_PHRASE = "Not found in the filing"

PROMPT = f"""You are a financial analyst answering questions about a company's SEC filing.
Answer using only the context below, which consists of excerpts from the filing.
If the answer is not in the context, reply exactly: "{REFUSAL_PHRASE}."
Cite the page numbers you used, e.g. (page 12) or (pages 12, 14)."""
