# Reports

Machine-written JSON results behind every number in the main README. Only aggregate numbers are stored
(metrics, confusion matrices, counts); no comment text.

| File | Content | Written by |
|---|---|---|
| `testset2_heldout.json` | **Main benchmark**: 968 held-out YouTube comments (2026), weighted toxic precision/recall/F1 per model, paired-bootstrap intervals | `scripts/eval_heldout.py` |
| `generative_vs_phobert.json` | Gemini and Qwen (zero-shot and QLoRA) on the same comments, alone and as a cascade, plus a check on the author's blind labels | `scripts/compare_generative.py` |
| `llm_second_stage.json` | LLM re-check of the comments PhoBERT flags, against the author's blind labels | `scripts/eval_llm_stage.py` |
| `annotation_agreement_testset2.json`, `annotation_agreement_extra.json` | Cohen's κ between the author's blind labels and the reference labels | `scripts/agreement_testset2.py` |
| `phobert_results_v7.json`, `phobert_results_v8.json` | Kaggle training runs (validation F1 per seed) | `notebooks/` |
| `build_testset2.json`, `extra_pool.json`, `extra_batch.json` | How the test set and the extra training batch were sampled and selected | `scripts/build_testset_pool.py`, `scripts/build_extra_pool.py`, `scripts/select_extra_batch.py` |

[`earlier/`](earlier/) holds the results of the thesis-era pipeline and models v1–v6 (older label
definitions): leakage and cleanlab statistics, TF-IDF baselines, cross-dataset tests and threshold studies.
