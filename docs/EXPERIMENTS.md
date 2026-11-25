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
