# Experiment 003: page dedup hurts (2026-11-24)

**Question.** Experiment 002 (k=10) beat the baseline (k=5) by 11 points of accuracy. It also
reached more distinct pages (8.1 vs 4.4). Was the gain from more pages, or from more context?
Page dedup at k=5 tests the first explanation at half the context of k=10: fetch 20 candidates,
keep the best-ranked chunk of each page, return 5 chunks from 5 distinct pages.

**Result.** It hurts ([results](../../results/003_dedup_k5.json), 30 questions × 3 runs):

| | 001 naive k=5 | 003 dedup k=5 | 002 naive k=10 |
|---|---|---|---|
| Accuracy | 38.9% ± 5.1 | 33.3% ± 3.3 | 50.0% ± 3.3 |
| Page hit | 50.0% | 53.3% | 63.3% |
| Evidence hit | 40.0% | 40.0% | 50.0% |
| Refusal | 34.4% ± 5.1 | 37.8% ± 3.8 | 21.1% ± 1.9 |
| Distinct pages | 4.4 | 5.0 | 8.1 |

Dedup reached one more gold page than the baseline but no more gold evidence text, and
accuracy fell. The aggregate drop is about one baseline std, so on its own it is weak. The
per-question evidence is clearer.

## Two examples

Both questions have their gold evidence on page 42 of AMD's 2022 10-K, in MD&A prose that the
1000-character splitter cut into more than one chunk.

**financebench_id_00917** (what drove AMD's FY22 operating margin change; gold: amortization of
Xilinx intangibles)

- Baseline pages: `[4, 42, 42, 53, 47]`, correct 3/3.
- Dedup pages: `[4, 42, 53, 47, 3]`, correct 0/3.

Dedup dropped the second chunk of page 42 and filled the slot with page 3. The evidence-text
metric did not match in either run, so here the change in correctness is the only signal that
the dropped chunk held the answer.

**financebench_id_01198** (what drove AMD's FY22 revenue change; gold: EPYC, semi-custom, Xilinx
embedded)

- Baseline pages: `[4, 47, 42, 47, 42]`, correct 3/3, evidence hit.
- Dedup pages: `[4, 47, 42, 53, 5]`, correct 0/3, no evidence hit.

Dedup dropped the second chunks of pages 47 and 42. The gold passage stopped matching any
retrieved chunk, and the answer went wrong.

## Conclusion

Chunk-level dedup treats two chunks of the same page as redundant, but on these filings they
are usually complementary: one passage split across chunks. Dropping the second chunk drops
evidence. So the k=10 gain is not explained by "more distinct pages" alone.

This motivates the opposite move, **page expansion**: when a chunk of a page is retrieved,
send the model the whole page, so a passage split across chunks arrives intact.

## Reproducing

The dedup retriever was removed in the commit after this note ("Remove dedup retriever
option"); its pure `dedup_by_page` helper was kept for page expansion. To reproduce experiment
003, check out the commit recorded in its results file (`a20aa8d`, `git.commit` in
`results/003_dedup_k5.json`) and run with `RETRIEVER=dedup`, `RETRIEVAL_K=5`,
`FETCH_MULTIPLIER=4`.
