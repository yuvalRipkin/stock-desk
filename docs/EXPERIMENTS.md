# Experiments

30 FinanceBench questions over 5 filings, 3 runs each; full records in `results/<name>.json`.
Numbers are metrics v2. Accuracy moves up to ±7 points with no change (see 005b), so retrieval
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
- **Decision:** all results are v2 (`metrics_version` in each file).
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
