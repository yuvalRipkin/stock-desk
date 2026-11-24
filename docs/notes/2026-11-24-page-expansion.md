# Experiment 005: page expansion (2026-11-24)

**Idea.** Experiments 003 and 004 showed that two chunks of the same page are usually one
passage split by the 1000-character splitter, so dropping either loses evidence
([dedup note](2026-11-24-dedup-negative.md)). Page expansion does the opposite: search for
k=10 chunks, keep the first `MAX_PAGES`=5 distinct pages in rank order, and send the model the
full text of each page.

**Result** ([results](../../results/005_expand_k10.json), 30 questions × 3 runs):

| | 002 naive k=10 | 005 expand k=10, 5 pages |
|---|---|---|
| Accuracy | 50.0% ± 3.3 | 54.4% ± 5.1 |
| Page hit | 63.3% | 53.3% |
| Evidence hit | 50.0% | 56.7% |
| Refusal | 21.1% ± 1.9 | 20.0% ± 0.0 |
| Distinct pages | 8.1 | 5.0 |
| Context per question | not measured (≤ ~10,000 chars) | ~15,200 chars |
| Latency p50 / p95 | 1.43 s / 2.39 s | 1.55 s / 4.41 s (cold start, see below) |

`context_chars` was added after 002, so 002's context is bounded (10 chunks of at most 1000
characters plus page labels) rather than measured.

## Gains

Evidence hit is deterministic across runs, so its change is real: 4 questions gained the gold
evidence (00494, 00917, 01328, 01981) and 2 lost it (00476, 01009, both from the page cap
below), +2 net. 00917 is one of the dedup casualties: the full page 42 now carries the Xilinx
amortization sentence that sat in its second chunk.

Two questions went from 0/3 to 3/3:

- **financebench_id_03620** (PepsiCo FY2022 EBITDA less capex; gold $9,068M, pages 61 and 63).
  A calculation over three numbers. With full pages 61 and 63 the model had operating profit,
  D&A and capex together and computed $11,512M + $2,763M − $5,207M = $9,068M.
- **financebench_id_01964** (American Express's largest liability; gold: customer deposits,
  page 97). Expansion never reached page 97, but a full page it did send states customer
  deposits of $189.5 billion, enough for a correct answer. Whole pages carry facts that chunks
  around them miss.

Accuracy rose 4.4 points over 002, but that is within noise (002 std 3.3, 005 std 5.1), so the
claim here is about retrieval: page expansion finds more gold evidence with fewer pages.

## The page cap

Page hit fell from 63.3% to 53.3% because `MAX_PAGES`=5 cuts pages that naive k=10 reached.
In all three questions that lost their page hit, the gold page was 6th or 7th among 002's
distinct pages:

| Question | Gold page | Rank among 002's distinct pages | Pages 005 sent |
|---|---|---|---|
| 00476 | 0 | 6th | 251, 125, 68, 67, 161 |
| 01009 | 3, 4 | 6th (page 3) | 31, 488, 489, 70, 477 |
| 01148 | 4 | 7th | 29, 54, 8, 5, 10 |

So the cap, not expansion, is what costs page hit. A higher cap (e.g. 8, matching 002's 8.1
distinct pages) should recover these, at the price of more context.

**financebench_id_00678 is not a page-cap loss.** It fell from 2/3 to 0/3, but its only gold
page (54) was the second page sent in every run, and evidence hit stayed true. The failure is in
the answer: given full pages, the model hedged ("only selectively"), argued gross margin is not
a useful metric for Boeing, and reported operating margin instead of gross profit ($3,017M →
$3,502M). It is a judgment question that 002 also got wrong once. More context did not help
here and may have pulled the answer toward the operating-margin discussion on the same pages.

## Cold-start p95

The p95 of 4.41 s ± 3.21 is driven by run 1 (8.1 s; runs 2 and 3: 2.9 s and 2.3 s). The five
slowest questions in run 1 (5.6–10.6 s) are each the first question about one of the five
filings: that is when `page_texts` parses the whole PDF with pypdf. After that the per-process
cache is warm. Warm latency is close to 002's (p95 2.39 s), slightly higher from the larger
context.

This is an artifact of reading pages from PDFs at query time. Production would store page text
in the database at ingestion; for evals, warming the cache before the timed loop would remove it.

## Cost

About 15,200 characters of context per question, versus at most about 10,000 for 002 and about
5,000 for the baseline, so the answering model's input is roughly 1.5× 002's and 3× the
baseline's. The judge does not see the context, so its cost is unchanged.

## Experiment 006: 8 pages

Same retriever with `MAX_PAGES`=8 ([results](../../results/006_expand_k10_p8.json)):

| | 005 (5 pages) | 006 (8 pages) |
|---|---|---|
| Accuracy | 54.4% ± 5.1 | 56.7% ± 3.3 |
| Page hit | 53.3% | 63.3% |
| Evidence hit | 56.7% | 63.3% |
| Refusal | 20.0% | 20.0% ± 3.3 |
| Distinct pages | 5.0 | 7.6 |
| Context per question | ~15,200 chars | ~22,500 chars (+47%) |
| Latency p50 / p95 | 1.55 s / 4.41 s | 1.57 s / 4.23 s (cold start again) |

- **The cap was the page-hit problem.** All three capped questions (00476, 01009, 01148) get
  their gold page back, and page hit returns to 002's 63.3%. Evidence hit gains 2 questions
  (00476, 01009) and loses none.
- **Accuracy +2.3 points, within noise.**
- **00678 recovered (0/3 to 3/3).** Not because of the cap: its gold page was present in 005 as
  well (see above). With more pages the model gave the gross-profit answer instead of hedging.
  One question, so this is weak evidence either way.
- **Evidence hit now exceeds accuracy** (63.3% vs 56.7%), the first time the gap opens. With
  more context, generation starts to fail on questions whose evidence is present. The clearest
  case is **01328**: evidence hit in both runs, but 3/3 in 005 and 1/3 in 006, likely
  distraction. **00723** also fell (2/3 to 0/3), but it has no evidence hit in either run, so it
  is not evidence of distraction.
- **Cost:** 47% more context for +2 evidence questions.

**Decision deferred.** 005 is the cost-effective setting, 006 finds more evidence. A re-ranker
should decide: if it can put the gold page in the top 5, the 5-page budget is enough.

## Next

- ~~Raise `MAX_PAGES` (e.g. 8)~~: done in 006, above.
- Add a re-ranker so the best pages fit in a small page budget.
- Warm the page cache before timing, so p95 reflects steady state.
