# Experiment 008: multi-query rewriting (2026-11-24)

**Verdict: no measurable gain; adds retrieval variance and 3x latency.** `QUERY_REWRITE` stays
`none` by default, and 007 remains the best config.

**Idea.** Questions are phrased the way an analyst asks them ("quick ratio", "largest
liability"), while the evidence is phrased the way a 10-K writes it (line items on a named
statement). A cheap LLM rewrites each question into search queries in the filing's own terms,
so vector search reaches pages the question alone does not. The main target was 01964
(American Express's largest liability, gold page 97, never retrieved).

**Method.** On top of 007's config (expand, k=10, max 5 pages, exhibits excluded),
`QUERY_REWRITE=multi` asks `gpt-5.4-mini` (structured output, never an answer) for 3 queries
that name financial statements, the line items behind a metric, and 10-K sections, with one
query per statement when a question needs several. The original question and each rewrite
retrieve k=10 chunks with the same filters. The four ranked lists are merged with Reciprocal
Rank Fusion (k_rrf=60), then deduplicated by page and expanded to full pages as before. Each
record saves its `queries`.

**Result** ([results](../../results/008_rewrite_multi.json), 30 questions × 3 runs):

| | 007 | 008 (rewriting) |
|---|---|---|
| Evidence hit | 63.3% ± 0.0 | 64.4% ± 6.9 |
| Page hit | 60.0% ± 0.0 | 63.3% ± 5.8 |
| Accuracy | 55.6% ± 1.9 | 54.4% ± 13.5 |
| Refusal | 15.6% ± 1.9 | 11.1% ± 5.1 |
| Context per question | ~15,500 chars | ~13,800 chars |
| Latency p50 / p95 | 1.51 s / 4.33 s* | 4.48 s / 8.41 s* |

- **Evidence hit is unchanged within noise:** +1.1 points, or one question-run out of 90.
  Gains and losses nearly cancel (below).
- **Retrieval is no longer deterministic.** The rewriter writes different queries each run, even
  at temperature 0, so only 5 of 30 questions got the same page set in all 3 runs (007: 30 of
  30). Accuracy per run was 66.7%, 40.0% and 56.7%. The noise-floor rule ("judge retrieval by
  evidence hit, which is deterministic") does not hold for this retriever. A 3-run eval of it
  has noise on the retrieval side as well as on the answer side.
- **Latency triples** (p50 1.5 s → 4.5 s): one extra LLM call and three extra searches per
  question. \*Run 1's p95 (13.6 s) is a cold start; the warm runs' p95 is 5.4–6.3 s.
- **The rewriter behaved.** No rewrite answered the question; all were search queries in
  10-K vocabulary.

## Per question (evidence hit, runs out of 3)

| Question | 007 | 008 | What happened |
|---|---|---|---|
| 00222 AMD quick ratio | 0/3 | 3/3 | balance-sheet line-item queries reach pages 55–58 |
| 00799 Amcor quick ratio | 0/3 | 3/3 | same: the balance sheet now ranks |
| 00720 | 0/3 | 2/3 | |
| 00723, 01079 | 0/3 | 1/3 | |
| 00563 AMD segment sales | 3/3 | 0/3 | rewrites steer to segment notes (4, 53, 54) and push out page 66 |
| 00476, 01009, 01981 | 3/3 | 1/3 | lost in 2 of 3 runs |

The quick-ratio gains are what the prompt was designed for, and they hold in every run. But the
answers to 00222 and 00799 are still mostly wrong (1 of 6 correct): with the evidence now
retrieved, the answering step is where they fail.

## financebench_id_01964 (the target case): not fixed

The rewrites are what a human would write ("American Express 2022 Item 8 consolidated balance
sheet liabilities… customer deposits, long-term debt"), but retrieval returns the same five
pages as 007 (`125, 124, 98, 151, 138`), only reordered. Gold page 97 still never makes the top
5. Page 98 is next to it, so the balance sheet is close but loses on ranking. This points at
how the page is chunked or embedded, not at query wording.

## Decision

Keep `QUERY_REWRITE=none` as the default; 007 stays the best config. The quick-ratio gains are
real but offset by losses elsewhere, and they cost 3x latency and unstable retrieval.

## Follow-ups

- **If rewriting is revisited:** rewrite once per question and cache the queries, so every run
  (and every experiment) searches the same queries. That restores deterministic retrieval and
  removes the latency from repeat runs.
- **01964:** try section-aware chunking, so the balance sheet is one coherent chunk.
- **00222 / 00799:** the evidence now arrives, so the answer step is the bottleneck here
  (computing a ratio from line items, or judging whether the ratio applies).
