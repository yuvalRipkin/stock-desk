# Error analysis: naive RAG baseline (2026-11-23)

**Setup:** table `naive_1000` (chunk 1000 / overlap 200, `text-embedding-3-small`), `retrieve(question, doc_name, k=5)`,
`stockdesk.ask` prompt with strict "Not found in the filing." fallback, temperature 0.
**Sample:** 15 of the 30 FinanceBench open-source questions for the 5 manifest filings, mixed across filings and
question types. Correctness was judged by hand against the gold answer.

Gold `evidence_page_num` is 0-based and matches our `page` metadata (verified against the PDFs), so
"hit" means a gold page number appears in the retrieved pages.

## Results

| ID | Doc | Type | Correct | Hit (gold → retrieved) | Fault | Note |
|---|---|---|---|---|---|---|
| 00799 | AMCOR | ratio calc | ✗ | ✗ 51 → 31,29,2,12 | Retrieval | Quick ratio; balance sheet never retrieved. "Not found." |
| 01079 | AMCOR | extraction | ✗ | ✗ 63 → 29,108,31,49,34 | Retrieval | Acquisitions note not retrieved. "Not found." |
| 00684 | AMCOR | margin calc | ✓ | ✓ 49 | — | 19.4% → 18.6%, math correct. |
| 00222 | AMD | ratio calc | ✗ | ✗ 55 → 3,4,105,104 | Retrieval | Quick ratio; balance sheet never retrieved. "Not found." |
| 01198 | AMD | extraction | ✓ | ✓ 42 | — | EPYC, semi-custom, Xilinx. |
| 00917 | AMD | reasoning | ✗ | ~ 42 (page yes, chunk no) | Retrieval | Page 42 retrieved, but not the chunk with the Xilinx amortization sentence. Answer is generic. |
| 01279 | AMD | numerical | ✓ | ✓ 57 | — | Operations. Cites p51 instead of p57. |
| 00476 | AMEX | extraction | ✗ | ✗ 0 → 251,125,68,67,161 | Retrieval | Gold "none" is on the cover page; p0 is in the store but not ranked. "Not found." |
| 00723 | AMEX | judgment | ✗ | ✗ 95 → 48,44,45,63,6 | Retrieval (+prompt) | Gold: "not measured by operating margin". Strict "Not found" prompt also blocks this kind of answer. |
| 01351 | AMEX | numerical | ✓ | ✗ 43 → 48,98,162,156,45 | — | Correct from p48, which repeats the figures. Page-level hit misses alternative evidence. |
| 00517 | BOEING | numerical | ✓ | ✓ 61 | — | All three segments > 20%. |
| 01290 | BOEING | extraction | ✗ | ✗ 7,9,13 → 61,54,4,23,11 | Retrieval | Customers / US gov 40% not retrieved. "Not found." |
| 00678 | BOEING | margin calc | ✓ | ✓ 54 | — | Right numbers (3,502 vs 3,017); hedges that gross margin isn't useful, no %. |
| 03620 | PEPSICO | metric calc | ✗ | ½ 61,63 → 70,39,61,102 | Retrieval | Income statement (p61) found, cash flow (p63, D&A + capex) not. "Not found." |
| 01009 | PEPSICO | extraction | ✗ | ✗ 3,4 → 31,488,489,70 | Retrieval | Lists 4 international divisions, omits North America. p488–489 are exhibits. |

## Summary

- **Correct: 6/15 (40%).**
- **Gold page retrieved:** 5/15 fully, plus 2 partial (00917 page without the right chunk, 03620 one of two pages).
- **Failures by cause (9):** Retrieval 9, Generation 0, Arithmetic 0.
- "Not found in the filing." was the answer in 7 of the 9 failures. The model rarely hallucinates; it abstains when retrieval misses.
- Correct and hit mostly agree. The exception is 01351, which was correct without the gold page.

## Patterns

1. **Computed-metric questions miss the statements.** Quick ratio, EBITDA − capex: the question's terms
   ("quick ratio") don't appear in the balance sheet or cash-flow statement, so semantic search on the
   question doesn't find them. → BM25 won't fix this alone; consider query rewriting ("current assets,
   receivables, current liabilities") or always retrieving the financial statement pages.
2. **Multi-page evidence needs more than k=5.** Chunks cluster: k=5 often yields only 3–4 distinct pages
   (several chunks from one page), and questions needing 2+ statements lose one.
3. **Page hit ≠ evidence hit.** 00917 retrieved the right page but the wrong chunk. Measure at chunk/text level too.
4. **Exhibit noise.** PepsiCo has 503 pages; exhibit pages (488–489) take up retrieval slots.
5. **Prompt conflict.** "If the metric isn't useful, explain why" questions (00723) need judgment, but the
   strict "Not found" instruction pushes the model to abstain. Also, answers sometimes cite pages
   different from the evidence used (01279).

## What the eval runner must measure automatically

- **Answer correctness** vs gold: gold answers are free text, so this needs an LLM judge (Phase 5), with
  numeric tolerance for computed answers.
- **Retrieval recall@k at page level** (any / all gold pages retrieved), and **evidence-text recall**
  (does a retrieved chunk contain the gold `evidence_text`?), to catch cases like 00917.
- **Abstention rate** ("Not found in the filing.") and whether abstentions coincide with retrieval misses.
- **Distinct pages retrieved** per question (k vs unique pages).
- **Failure attribution:** retrieval miss → retrieval; hit but wrong → generation/arithmetic.
- Save `experiment_config()` and per-question records (retrieved pages, answer, judgment) in `results/*.json`.
