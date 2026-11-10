# Label patches

`guidelines_v3.csv` re-labels training/validation/test comments to follow
[annotation guidelines v3](../docs/annotation_guidelines.md) (profanity used only as an intensifier is
*clean*; curse phrases without a target are *offensive*; anything aimed at a person/group is *hate*).

Columns: `split` (train/val/test of `data/clean_vihsd/`), `row` (0-based row index), `old`, `new`,
`uncertain`. Only row indices and labels are stored (no comment text), so the patch can be
published while the data itself stays private.

The 1,179 rows were selected with simple profanity patterns (intensifier-only comments labeled
offensive/hate, and comments with curse words labeled clean), then reviewed one by one
(LLM-assisted, following the guideline). Apply with `scripts/apply_relabels.py`.
