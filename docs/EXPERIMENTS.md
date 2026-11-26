# Experiments

30 FinanceBench questions over 5 filings, 3 runs each; full records in `results/<name>.json`.
Numbers are metrics v3. Accuracy moves up to ±7 points with no change (see 005b), so retrieval
changes are judged by evidence hit.

## 000: error analysis of the naive baseline (2026-11-23)

- **Hypothesis:** before automating evals, find out by hand where naive RAG fails.
- **Change:** none; 15 questions at k=5, graded by hand.
- **Result:** 6/15 correct; all 9 failures were retrieval misses, 7 of them "Not found".
- **Decision:** build the runner to measure page hit, evidence hit, refusals and distinct pages.
- **Example:** 00799 (Amcor quick ratio): "quick ratio" never appears on the balance sheet, so it never ranks.

## 001: naive baseline, k=5

- **Hypothesis:** baseline for everything after.
- **Change:** vector search, 1000-char chunks with 200 overlap, top 5 chunks.
- **Result:** accuracy 38.9%, evidence hit 40.0%, refusal 34.4%.
- **Decision:** baseline.
- **Example:** k=5 often yields only 3–4 distinct pages, so questions needing two statements lose one.

## 002: k=10

- **Hypothesis:** more chunks reach more of the evidence.
- **Change:** `RETRIEVAL_K=10`.
- **Result:** accuracy 50.0% (+11), evidence hit 50.0% (+3 questions), refusal 21.1%.
- **Decision:** adopt k=10; input tokens roughly double, latency barely moves.
- **Example:** with the gold passage retrieved, 64% of answers are correct; without it, 35%.

## 003 / 004: one chunk per page (dedup)

- **Hypothesis:** 002's gain came from more distinct pages, not more context.
- **Change:** keep the best chunk per page; 5 distinct pages (003) or 10 (004).
- **Result:** worse: 003 accuracy 33.3%, evidence hit 40.0%; 004 evidence hit unchanged at 50.0%.
- **Decision:** rejected; same-page chunks are one split passage, not duplicates. Retriever removed (003 is at `a20aa8d`).
- **Example:** 00917 (AMD operating margin): dedup dropped page 42's second chunk, which held the Xilinx answer; 3/3 → 0/3.

## 005: page expansion

- **Hypothesis:** send whole pages, so a passage split across chunks arrives intact.
- **Change:** search k=10 chunks, send the full text of the first 5 distinct pages.
- **Result:** evidence hit 56.7% (+2 over 002), accuracy 54.4%, ~15,200 chars of context.
- **Decision:** adopt; the 5-page cap cut 3 gold pages ranked 6th–7th, so test a higher cap.
- **Example:** 03620 (PepsiCo EBITDA less capex): full pages 61 and 63 hold all three inputs; 0/3 → 3/3.

## 005b: accidental repeat of 005 (the noise floor)

- **Hypothesis:** none; a run meant as 007 ran before the exhibit code existed, so it repeats 005.
- **Change:** none; 88 of 90 records retrieved identical pages.
- **Result:** accuracy 48.9% vs 54.4%, same evidence hit; 9 questions changed their correct count.
- **Decision:** accuracy differences under ±7 points are noise; judge retrieval by evidence hit.
- **Example:** 01328: 3/3 → 0/3 with the same pages.

## 006: 8 pages

- **Hypothesis:** a higher cap recovers the gold pages 005 cut.
- **Change:** `MAX_PAGES=8`.
- **Result:** evidence hit 63.3% (+2), page hit 63.3%, accuracy 56.7%, but context +47% (~22,500 chars).
- **Decision:** deferred; it costs too much context for the gain, so look for a cheaper way to fill 5 pages.
- **Example:** 00476 (AmEx cover page 0, ranked 6th): back in the context.

## 007: exclude exhibit pages

- **Hypothesis:** attached exhibits (plans, agreements) take page slots from real 10-K pages.
- **Change:** `EXCLUDE_EXHIBITS=true` filters out pages from the first exhibit after SIGNATURES.
- **Result:** evidence hit 63.3% (= 006) at ~15,500 chars (−31% vs 006), accuracy 55.6%.
- **Decision:** adopt; best config so far (expand, k=10, 5 pages, no exhibits).
- **Example:** 01009 (PepsiCo geographies): 3 exhibit pages replaced by gold pages 3 and 4; 1/3 → 3/3.

## Metrics v2: refusal rule

- **Hypothesis:** "Not found" anywhere in an answer marked real but hedged answers as refusals.
- **Change:** a refusal is the phrase alone; everything else is judged. Results rescored without rerunning RAG.
- **Result:** 9 of 720 records changed; 2 became correct; 005b and 007 accuracy +1.1.
- **Decision:** adopted (`metrics_version` in each file); since extended by v3, see 012.
- **Example:** 01964 run 2: "Customer deposits were the largest liability… Not found in the filing." is now judged correct.

## 008: multi-query rewriting

- **Hypothesis:** questions use analyst terms, 10-Ks use line items; rewrite queries into filing language.
- **Change:** an LLM writes 3 queries in 10-K terms; question + rewrites are merged with RRF, then expanded as in 007.
- **Result:** evidence hit 64.4% ± 6.9 (no gain); retrieval varies by run (5/30 stable vs 30/30); p50 latency 3x.
- **Decision:** rejected; `QUERY_REWRITE` stays `none`. If revisited, cache rewrites so runs are deterministic.
- **Example:** 01964 (AmEx largest liability): good rewrites, same pages as 007; gold page 97 still never ranks.

## 009: section-header chunking

- **Hypothesis:** chunks that carry their 10-K section ("Consolidated Balance Sheets") rank for questions phrased by statement.
- **Change:** `CHUNKING=section_header`: pages also split at section headings; `[<SEC Item> > <statement>]` prepended before embedding (table `section_hdr_1000`).
- **Result:** evidence hit 63.3% (identical per question), accuracy 48.9%, refusals 26.7% (+11); every extra refusal had different pages.
- **Decision:** rejected: headers dilute number-dense statement chunks (similarity drops 3–5x more than for MD&A). 009a (boundaries only) was skipped: it changes only 34 of 1,229 pages and 3 gold pages, none of them page 97.
- **Example:** 03620 (PepsiCo EBITDA less capex): cash-flow page 63 fell from 5th to 8th behind MD&A page 55; 3/3 → 0/3. 01964's page 97 rose to 8th, still cut by the 5-page cap.
- **Known quirk:** PepsiCo prints its statements inside Item 7, so they are labeled under Item 7, as filed.

## 010: cross-encoder re-ranking

- **Hypothesis:** a cross-encoder ranks the gold page into the 5-page budget (01964's page 97 was 8th in 009).
- **Change:** fetch 30 chunks (007's filters), re-rank (question, chunk) pairs with `BAAI/bge-reranker-base` on MPS, then pages as in 007.
- **Result:** evidence hit 43.3% vs 63.3% (−6 questions), accuracy 45.6%, refusals 34.4%; ~0.8 s per question to re-rank. Fixed 01964, 00585, 04481 (00585 and 04481 never answered before); broke 00563, 00917, 01091, 03620 (3/3 → 0/3).
- **Decision:** rejected. Not a bug: higher score = more relevant, inputs are chunks under the 512-token limit. The model is confident on statement tables (page 97 scores 0.70) but scores every chunk near zero on prose questions, where its order is noise (00917: all ≤ 0.01). Code is in `b8ecfa4`, removed afterwards. Follow-up, not done: fuse vector and re-ranker rankings with RRF.
- **Example:** 01091 (Boeing legal proceedings): gold page 112 fell from 2nd to 20th behind the 10-K index page (score 0.03).

## 011: text-embedding-3-large

- **Hypothesis:** a stronger embedding ranks the right pages higher with nothing else changed.
- **Change:** 007's config on `naive_1000_large`: same chunks (1000/200), embedded with `text-embedding-3-large` (3,072 dims).
- **Result:** accuracy 67.8% vs 55.6% (+12.2, beyond the ±7 noise floor), evidence hit 66.7% vs 63.3% (+5 questions, −4, with 3 of the 4 still correct), refusals 7.8% vs 15.6%, no latency cost.
- **Decision:** adopted; now the default config. Trade-off: 3,072 dims doubles vector storage and exceeds pgvector's 2,000-dimension index limit; fine for exact search at ~4,600 rows. At scale, use the API's `dimensions` parameter to shorten vectors.
- **Example:** 01964 (AmEx largest liability) 1/3 → 3/3 with gold page 97 retrieved at last; also 04481 and 00757 0/3 → 3/3, 00222 0/3 → 2/3, 00476 and 01290 0/3 → 1/3.

## 012: answer prompt v2

- **Hypothesis:** answers fail on generation too: conclusions before arithmetic (00684 said "Yes", then computed a decline) and refusals on judgment questions.
- **Change:** `ANSWER_PROMPT=v2`: list figures with pages, show arithmetic, reason from the filing's explanations on judgment questions, end with an "Answer:" line consistent with the steps; refuse only when facts are missing. Retrieval as in 011.
- **Result:** accuracy 73.3% vs 67.8% (+5.5, within the ±7 noise floor but consistent: 6 questions up, 2 down), identical retrieval, refusals 4.4% (metrics v3; 4 refusals hid behind an "Answer:" line), answers ~1,170 chars, p50 latency 1.2 → 1.9 s. A number-by-number check of wrong answers without gold evidence found no fabricated figures.
- **Decision:** adopted; now the default. Known cost: overconfidence when a key page is missing (00678, Boeing gross margin: income statement not retrieved, concluded from program margins; 1/3 → 0/3).
- **Example:** 00723 (AmEx operating margin, a judgment question): refused 3/3 in 011, correct 2/3 in 012 ("not the right metric for a spend-centric business").

## 013 / 014: held-out test

- **Split rule:** split by filing. Dev (5 filings, 30 questions) was used for every decision above; test (79 filings, 120 questions, `data_manifest_test.txt`) is for final measurement only, never for decisions.
- **Change:** none. Day-1 config (001: naive, k=5, 3-small, prompt v1) as 014 vs final config (012) as 013, both on `--split test`, 3 runs.
- **Result:** accuracy 38.1% ± 1.3 → 71.9% ± 1.7 (+33.8; dev gain was +34.4), evidence hit 47.5% → 79.2%, page hit 53.3% → 79.2%, refusals 40.0% → 5.0%, context 4.5k → 15.5k chars. Latency not compared: the two runs ran in parallel.
- **Decision:** the tuning generalizes. Retrieval is filtered to the question's filing (single-document setting), so results are not directly comparable to shared-store FinanceBench numbers.
- **Setup notes:** some test PDFs are AES-encrypted and need `pypdf[crypto]`; 14 PDFs have empty pages (63 in all, no gold pages); the exhibit cutoff was found in 29 of 79 filings (nothing is excluded in the rest). A btree index on `(doc_name, page)` cut per-query execution from ~50–62 ms to ~7–14 ms at 54,666 rows.

## 015: BM25 + vector hybrid

- **Hypothesis:** exact-term matching ranks pages that vector search misses (targets 00735, 01028: gold page never retrieved).
- **Change:** `HYBRID=bm25`: BM25 (`rank_bm25`, per-filing in-memory index, exhibits excluded) ranks 10 chunks, fused with vector's 10 by RRF, then pages as in 012. Dev only.
- **Result:** evidence hit 56.7% vs 66.7% (−3 questions: 00585, 01079, 01964; none gained), accuracy 61.1% vs 73.3% (−12.2, beyond noise), refusals 6.7%. 00735 and 01028 not fixed. 01964 and 00684 3/3 → 0/3, 00723 2/3 → 0/3. Cause: with equal RRF weights and a 5-page cap, BM25's rank-2/3 hits tie vector's and push out vector's 4th–5th pages; BM25 matches words ("liability", "margin"), not the statement.
- **Decision:** rejected; `HYBRID` stays `none`. Not tuned further (e.g. BM25 weighting), to avoid open-ended tuning on dev.
- **Example:** 01964 (AmEx largest liability): BM25-only pages 84 (Other matters) and 63 (Corporate & Other) took slots 3 and 5; balance sheet page 97 (vector rank 4) fell out.

## 016: LangGraph agent

- **Hypothesis:** an agent that can search again with new wording and calculate with a tool fixes questions where one search misses a page or the arithmetic goes wrong.
- **Change:** `ANSWER_MODE=agent`: a ReAct loop (gpt-5.4-mini) with `search_filings` (012's retrieval, bound to the question's filing) and `calculate` (safe `ast` arithmetic), at most 6 tool calls (`MAX_AGENT_STEPS`); page and evidence hit over all pages from all searches. Dev only.
- **Result:** accuracy 83.3% ± 3.3 vs 73.3% (+10.0, beyond the ±7 noise floor), evidence hit 87.8% ± 3.8 vs 66.7%, refusals 0%. 14.6k input / 384 output tokens per question: ~$0.0127 vs ~$0.0041 for 012 at list price (3.1x, before prompt caching; 012 tokens reconstructed with tiktoken). p50 latency 5.5 s vs 1.9 s. 1.78 searches and 0.78 calculate calls per question; step limit reached in 11/90 runs. Evidence hit now varies between runs because the model words the queries differently each run (01964, 00735, 01028, 01981); retrieval for a given query is unchanged.
- **Findings:** no invented filing figures in the 15 wrong answers. But 3 copying errors into `calculate` (03620: 11,212 for 11,512; 00723: 9,505 and 9,559), each reported as the result; and 11/90 answers cite `calculate` results without calling it (the arithmetic was right). Losses: 01964 3/3 → 1/3, the agent's keyword query ("current liabilities, long-term debt…") missed balance sheet page 97, which the verbatim question found in 012; 00723 2/3 → 1/3, the judgment question was treated as a calculation (6 calls on a pretax-margin proxy, business-model pages 6–7 never retrieved).
- **Decision:** stepping stone, not final; `ANSWER_MODE` stays `rag` by default. Next: search the original question first, quote `calculate` inputs and results verbatim, and measure cache reads (recorded per LLM call from now on: `cache_read_tokens`, `llm_usage`).
- **Example:** 00678 (Boeing gross margin) 0/3 → 3/3: the agent's own query ("gross margin net sales cost of sales … gross profit") retrieved the income statement (page 54) that 012 never found.

## 017: prefetch the verbatim question

- **Hypothesis:** searching the question verbatim before the loop (012's query, which found 01964's balance sheet) gives the agent the right pages up front and saves a model call.
- **Change:** `AGENT_PREFETCH=true`: `search_filings` on the question verbatim in code; its pages go in the first message; the prompt says to search again only if something is missing. Not counted in `MAX_AGENT_STEPS`. Otherwise 016.
- **Result:** accuracy 70.0% ± 3.3 vs 83.3% (−13.3, beyond noise), evidence hit 76.7% vs 87.8%: 00517, 00678, 00684, 00995 lost all evidence (3/3 → 0/3), though only 00678 lost accuracy with it (the other three were mostly answered from other pages); 01964 gained (1/3 → 3/3, accuracy 3/3). Largest accuracy losses: 00678 3/3 → 0/3, 00222 2/3 → 0/3, 04481 3/3 → 1/3. The model accepted the prefetched pages: no further tool call in 59/90 runs, a new search in 15/90. p50 latency 3.3 s vs 5.5 s.
- **Timing (first breakdown):** mean 4.0 s per question: model calls 3.3 s (81%; 1.66 calls at ~2.0 s each), searches 0.76 s (19%: embedding 0.37 s, DB 0.03 s, page loading ~0, PDF parse 0.35 s). Without the cold start (5 first-time PDF parses, ~6.4 s each, all in run 1) model calls are 89%. Latency rises with model calls: 3.2 s at 1 call, 6.9 s at 3, 9.9 s at 6. Embedding (~0.3 s) is most of a warm search.
- **Decision:** rejected; `AGENT_PREFETCH` stays false. The agent's own first query (016, worded like a 10-K) beats the verbatim question; prefetching anchors the model on worse pages. Latency is decided by model calls, not retrieval.
- **Example:** 00678 (Boeing gross margin) 3/3 → 0/3: the verbatim question never returned the income statement (page 54), and the model did not search for it.

## 018: prefetch + evidence checklist

- **Hypothesis:** forcing a checklist of needed figures with their pages makes the model search for what the prefetched pages lack.
- **Change:** `AGENT_PREFETCH_PROMPT=v2`: before answering, list each needed figure with its page or MISSING; search for every missing item (one query each, in one step); no general knowledge or substitute figures. Otherwise 017.
- **Result:** accuracy 70.0% ± 8.8, evidence hit 75.6% ± 5.1 (no change from 017). Checklist written in 87/90 runs, with missing items in 24, followed by a search in 16 of those; 17 runs still had missing items at the end. 00678 recovered (3/3); 00517, 00684, 00995 stayed at 0/3 evidence (the checklist marked nothing missing, citing the wrong pages). p50 latency 4.1 s.
- **Decision:** rejected. The checklist only catches gaps the model notices; it can't fix pages it wrongly accepts. Prefetch prompt v2 is kept for reference.

## 019: agent v2 (list tools, arithmetic discipline, short answer)

- **Hypothesis:** fewer model calls (several searches or expressions per call), stricter arithmetic and a shorter answer cut latency and cost, and remove 016's copying errors and uncalled `calculate` claims.
- **Change:** `AGENT_VERSION=v2` on 016: `search_filings(queries)` runs up to 3 queries per call (each page once), `calculate(expressions)` up to 10; one step per call. Prompt: put all searches and all writable expressions in one call each, copy inputs digit by digit, never write a self-computed number or claim an uncalled result; answer as Figures / Calculation / at most two sentences of Assessment / Answer.
- **Result:**

  | | 016 | 019 |
  |---|---|---|
  | Accuracy | 83.3% ± 3.3 | 75.6% ± 3.8 |
  | Evidence hit | 87.8% | 85.6% |
  | Latency p50 / mean | 5.5 s / 6.1 s | 4.6 s / 5.0 s |
  | Tool calls per question | 2.56 (limit hit 11/90) | 1.32 (limit hit 0/90) |
  | Tokens in / out | 14.6k / 384 | 9.4k / 252 |
  | Cost per question | $0.0127 (list) | $0.0082 (list), $0.0069 with measured cache reads (20%) |
  | Answer length | 1,179 chars | 703 chars |

  2.32 model calls per question at 1.55 s each (017: 1.98 s); latency still follows model calls (4.3 s at 2 calls, 6.3 s at 3). ~3 queries per search call, 2.8 expressions per calculate call.
- **Arithmetic discipline:** 0 copying errors into `calculate` (016: 3); arithmetic without a `calculate` call in 4/90 answers (016: 11/90 cited uncalled results). Not gone: 00517 run 1 shows `55893 / 66608 * 100 = 83.87247558544594` with no call.
- **Losses:** 00517 3/3 → 1/3, 00678 3/3 → 1/3, 01079 2/3 → 0/3, 00222 2/3 → 0/3, 00723 and 00799 1/3 → 0/3; gains 01964 1/3 → 3/3, 03620 2/3 → 3/3. Likely cause: the short format cuts reasoning on judgment questions. 00678 had the figures (revenues page 54; cost of sales 94.7% vs 95.2%, page 27) but skipped the calculation and concluded gross margin "isn't useful" for Boeing; 00517 split revenue by products/services instead of segments; 01079 stated FY2022/FY2021 had no acquisitions.
- **Decision:** rejected; `AGENT_VERSION` stays v1. Next (020): keep the list tools and discipline, restore the full answer format, to separate the two effects.
