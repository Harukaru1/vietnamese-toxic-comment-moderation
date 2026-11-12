# Reports

Machine-written JSON results of every experiment quoted in the main README (metrics, confusion
matrices, data-pipeline counts, annotation agreement). No comment text is stored here.

| Prefix | Content |
|---|---|
| `baseline_results*` | TF-IDF baselines (LR / SVM / RF) per data version |
| `phobert_results*` | PhoBERT training runs (validation + test) from the Kaggle notebooks |
| `phobert*_eval_*` | a published model evaluated on another test set (dedup, ViHSD, 2026 real-world) |
| `clean_splits*`, `label_audit`, `add_vihsd`, `add_active_v4` | data-pipeline statistics |
| `annotation_agreement*` | Cohen's κ between LLM pre-annotation and the author's blind labels |
| `realworld_2026_*` | analyses on the 2026 real-world test set (incl. paired bootstrap v2 vs v3) |
| `testset2_*`, `threshold_*`, `build_testset2` | test set 2 (2026 YouTube, guideline v3.2): sampling, v2-v6 scores, held-out comparison with paired bootstrap, decision thresholds |
