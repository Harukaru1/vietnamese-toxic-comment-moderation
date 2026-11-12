# Reports

Machine-written JSON results behind every number in the main README. Only aggregate numbers are
stored here (metrics, confusion matrices, counts); no comment text.

## Current results (test set 2, guideline v3.3)

| File | Content | Written by |
|---|---|---|
| `testset2_heldout.json` | **Main benchmark.** Held-out half of test set 2 (968 YouTube comments, 2026): weighted toxic precision/recall/F1 per model and paired-bootstrap intervals | `scripts/eval_heldout.py` |
| `llm_second_stage.json` | LLM re-check of the comments v7 flags, scored against the author's blind labels | `scripts/eval_llm_stage.py` |
| `annotation_agreement_testset2.json` | Cohen's κ between the author's blind labels (two rounds, 241 comments) and the reference labels, before and after the guideline revision | `scripts/agreement_testset2.py` |
| `phobert_results_v6.json`, `phobert_results_v7.json` | Kaggle training runs (v7: validation F1 of each seed and the selected one) | `notebooks/` |
| `build_testset2.json` | Sampling of test set 2 from the 12k-comment pool (strata and weights) | `scripts/build_testset_pool.py` |
| `testset2_eval.json`, `threshold_v3.json`, `threshold_v6.json` | Whole-set scores and decision-threshold study, computed before the final label revision | `scripts/eval_testset2.py`, `scripts/tune_threshold.py` |

## [`earlier/`](earlier/): thesis data and models v1–v5 (older label definitions)

| Prefix | Content |
|---|---|
| `label_audit`, `clean_splits*`, `add_vihsd`, `add_active_*`, `fix_bad_token` | data-pipeline statistics (leakage, cleanlab audit, merged datasets) |
| `baseline_results*` | TF-IDF baselines (LR / SVM / RF) |
| `phobert_results*` | PhoBERT v1–v5 training runs |
| `phobert*_eval_*` | a published model evaluated on another test set (dedup, UIT-ViHSD, 2026 set 1) |
| `realworld_2026_*_bootstrap` | paired bootstrap comparisons on 2026 set 1 (1,000 comments) |
| `annotation_agreement*` | κ for the earlier annotation rounds |
