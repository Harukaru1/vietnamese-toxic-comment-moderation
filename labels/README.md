# Labels for the 2026 YouTube pool (guideline v3.3)

Only ids and labels are stored here. `comment_id` and `video_id` are public YouTube ids, so the
text can be fetched again through the YouTube Data API; no comment text or author name is committed.

| File | Content |
|---|---|
| `testset2_v3.csv` | 2,000 sampled comments: `why` (sampling stratum), `label`, `uncertain`, `excluded` (English or shorter than 3 words; 72 rows) |
| `testset2_split.csv` | which half each kept row belongs to: `tune` (960 rows, used as training data for v6) or `test` (968 rows, held out, never trained on) |
| `pool_train_v33.csv` | 779 more comments from the same pool, labeled for training only |
| `pool_train_v33_hard.csv` | 229 hard examples: comments the v6 model flagged or was unsure about |
| `extra_batch1.csv` | 1,000 training comments from other sources (500 ViCTSD news comments, 300 VOZ forum posts, 200 more YouTube comments) and a random 200-comment ViCTSD news test set (`split`). Rows are identified by `source_split` + `source_row` in the original dataset (ViCTSD, VOZ-HSD) or by `comment_id` (YouTube) |

Labels were written by an LLM (Claude) without seeing model predictions, following
[guideline v3.3](../docs/annotation_guidelines.md). The author labeled 241 of the test-set comments blind
in two rounds: Cohen's kappa 0.68 on three classes and 0.74 on toxic vs clean
([`reports/annotation_agreement_testset2.json`](../reports/annotation_agreement_testset2.json)).

Columns: `label` is one of `clean`, `offensive`, `hate`; `uncertain` marks borderline rows.

The extra batch was cleaned first (URLs, @user tags and emoji removed; fewer than 4 words, mostly English
and duplicate comments dropped) and selected with `scripts/select_extra_batch.py`. `label_source` says how each
label was set: `manual_review` (PhoBERT and Gemini disagreed, or both called it toxic, so it was checked by
hand under the guideline) or `gemini_and_v7_agree` (both called it clean). The original datasets' own labels
were not used: their definitions of toxic differ from this guideline.
