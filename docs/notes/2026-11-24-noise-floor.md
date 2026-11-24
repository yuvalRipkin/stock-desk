# The accuracy noise floor: an accidental repeat of 005 (2026-11-24)

**What happened.** A run intended as experiment 007 (exclude exhibit pages) was made before the
exhibit-exclusion code existed. It ran at commit `1655a7f` with exactly 005's config
(`RETRIEVER=expand`, k=10, `MAX_PAGES`=5), and it still retrieved the exhibit pages that
exclusion would remove (e.g. PepsiCo pages 477, 488, 489 for 01009). So it is not an exhibit
experiment; it is a repeat of 005. It is saved as
[results/005b_expand_k10_repeat.json](../../results/005b_expand_k10_repeat.json). The file's
internal `name` field still says `007_expand_p5_noexhibits`, because that is what the runner
recorded; the file is otherwise unchanged.

**Why it is useful.** Two runs of the same code and config measure the noise between
experiments directly.

| | 005 | 005b (repeat) |
|---|---|---|
| Accuracy | 54.4% ± 5.1 | 47.8% ± 1.9 |
| Accuracy per run | 60.0, 50.0, 53.3 | 50.0, 46.7, 46.7 |
| Page hit | 53.3% | 53.3% |
| Evidence hit | 56.7% | 56.7% |
| Refusal | 20.0% | 23.3% ± 3.3 |
| Context per question | ~15,240 chars | ~15,265 chars |

**Rescored with metrics v2** (see README): 005b accuracy is 48.9% ± 1.9, a 5.5-point move
from 005; the conclusion is unchanged. Numbers below are as first scored (v1).

Retrieval is essentially identical: 88 of 90 records retrieved the same pages; the other two
are one question (00720) where the vector search returned a different 4th/5th page. Yet
accuracy moved **6.6 points**, 6 fewer correct answers out of 90, about 2 questions per run.

Nine questions changed their correct count. The largest swings were **01328** (3/3 to 0/3) and
**00723** (2/3 to 0/3), with the same retrieved pages in both runs. The source is the
answering model and the judge, both of which vary between calls even at temperature 0.

## Consequences

- **Between experiments, accuracy differences under about ±7 points are noise.** That is wider
  than the per-experiment std (1.9 to 5.1), which only measures variation within one batch of
  3 runs.
- **Retrieval experiments are judged by evidence hit and page hit**, which are deterministic
  given the same retrieval. Accuracy confirms a change only when the gain is large (002's +11
  points qualifies; 005's +4.4 and 006's +2.3 do not).
- **Per-question accuracy changes are not evidence on their own.** The 006 note first cited
  01328 and 00723 as signs of distraction from more context; 005b shows the same drops with no
  change at all, so that claim was withdrawn.

## Exhibit exclusion

Still untested. Stage 1 analysis found exhibit pages in 4 of 30 questions' retrieved pages in
005 (6 of 150 pages) and 7 of 30 in 006 (12 of 227), concentrated in PepsiCo, whose filing is
74% exhibits. The real experiment 007 needs the `EXCLUDE_EXHIBITS` setting, which is
implemented in the commit after this note.
