# Label patches (thesis data)

These patches fix labels in the older 52k-comment training data and were used for models v3–v5.
They follow earlier guideline versions (v3, v3.1); the current definitions are in
[guideline v3.3](../docs/annotation_guidelines.md) and the 2026 labels are in [`labels/`](../labels/).

`guidelines_v3.csv` re-labels training/validation/test comments to follow
[annotation guidelines v3](../docs/annotation_guidelines.md) (profanity used only as an intensifier is
*clean*; curse phrases without a target are *offensive*; anything aimed at a person/group is *hate*).

Columns: `split` (train/val/test of `data/clean_vihsd/`), `row` (0-based row index), `old`, `new`,
`uncertain`. Only row indices and labels are stored (no comment text), so the patch can be
published while the data itself stays private.

The 1,179 rows were selected with simple profanity patterns (intensifier-only comments labeled
offensive/hate, and comments with curse words labeled clean), then reviewed one by one
(LLM-assisted, following the guideline). Apply with `scripts/apply_relabels.py`.

`guidelines_v31.csv` is applied **on top of** `guidelines_v3.csv` (rows of `data/clean_vihsd_v3/`). It
reviews 790 short comments (≤200 characters) matching three error patterns found in the demo and the
v4 probe set: body shaming / appearance insults, intensifiers aimed at a person ("m … vl"), and "cút".
163 labels changed (129 train, 15 val, 19 test). The author blind-labeled 50 of them: κ 0.42 with
the first pass (0.21 with the original labels); after agreeing two rules (insulting one person = hate;
group generalizations are hate only with a demeaning word) κ = 0.59.
