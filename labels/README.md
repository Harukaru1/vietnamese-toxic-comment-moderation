# Labels for the 2026 YouTube pool (guideline v3.2)

Only ids and labels are stored here. `comment_id` and `video_id` are public YouTube ids, so the
text can be fetched again through the YouTube Data API; no comment text or author name is committed.

| File | Content |
|---|---|
| `testset2_v2.csv` | 2,000 sampled comments: `why` (sampling stratum), `label`, `uncertain`, `excluded` (English or shorter than 3 words; 72 rows) |
| `testset2_split.csv` | which half each kept row belongs to: `tune` (960 rows, used as training data for v6) or `test` (968 rows, held out, never trained on) |
| `pool_train_v32.csv` | 779 more comments from the same pool, labeled for training only |

Labels were written by an LLM (Claude) without seeing model predictions; the author labeled 100
comments blind (Cohen's kappa 0.70 on three classes, 0.84 on toxic vs clean after the v3.2 change).
