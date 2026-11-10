# Project notes for Claude

Portfolio project extending the author's graduation thesis. Working style: explain steps in
Vietnamese, keep tasks small, and let the author run Kaggle/GitHub/HF steps themselves.
Code, commits and README in English.

## What exists
- 3-class Vietnamese toxic comment classifier (clean / offensive / hate), extension of the thesis.
- PhoBERT v3 on HF Hub `HaruKaru/phobert-vi-toxic-comment` (`main`; branches `v2`, `v3`); demo on Streamlit Cloud
  (`streamlit_app/`, https://vi-toxic-moderation.streamlit.app/); FastAPI + Gradio + Docker in `app/`;
  GitHub Actions CI runs pytest (HF Space deploy is manual only: free Docker Spaces return HTTP 402).
- README has all results; read it first.

## Data pipeline (data/ is git-ignored and private)
`data/raw/{train,val,test}.csv` (thesis data) → `scripts/find_label_issues.py` (cleanlab) →
`scripts/make_clean_splits.py --prune` (`data/clean/`) → `scripts/add_vihsd.py`
(`data/clean_vihsd/`, ViHSD test kept as `data/vihsd_test/`) → `scripts/apply_relabels.py`
(`relabels/guidelines_v3.csv` → `data/clean_vihsd_v3/`). Deterministic; verified on a fresh clone.

## Annotation
- `docs/annotation_guidelines.md` **v3**: profanity used only as intensifier/exclamation = clean;
  curse/venting without target = offensive; aimed at person/group = hate.
- 2026 real-world test set: `data/realworld/realworld_2026_v3.csv` (700 YouTube via API + 300 VOZ,
  Claude-labeled); `data/realworld_eval/test.csv` = text,label for `scripts/evaluate_model.py`.
  Author's blind 100-comment labels are in `data/realworld/` (κ 0.72 independent, 0.91 after v3).
- YouTube API key is a network secret (header `X-Goog-Api-Key`) on the cloud environment
  `vietnamese-toxic-comment-moderation`; `src/toxic_moderation/youtube.py`.

## Key results (README has details)
In-domain test: TF-IDF ≈ PhoBERT v2 ≈ 0.80 F1-macro. 2026 set: PhoBERT v3 0.68 F1-macro (v2 0.59), toxic recall 0.59;
weak on implicit hate. Leakage (4.7%) and inconsistent profanity labels were the main data issues.

## Next steps
1. DONE: PhoBERT v3 is on HF `main` (v2 kept on branch `v2`); 2026 set F1 0.594 → 0.679, see README.
2. Active learning (IN PROGRESS): `scripts/build_active_pool.py` built `data/active/pool.csv` (9.5k: 39 new YouTube videos + VOZ)
   and `data/active/to_label.csv` (700 selected: uncertain / cue-but-predicted-clean / predicted toxic / random).
   Next: Claude labels them (guidelines v3), author spot-checks ~50, add to train, retrain v4 on Kaggle
   to fix implicit hate.
3. Short texts / context (author's request): (a) after v4, app mode for a short paragraph: split into
   sentences, classify each, text label = most severe sentence, highlight sentences by colour;
   (b) optional experiment: context-aware model (parent comment [SEP] reply, or whole paragraph,
   max_len 256) with YouTube reply threads; compare with (a). Guideline already has a
   "context-dependent comments" rule (appearance remarks → clean + uncertain unless demeaning on their own).
   Demo feedback from the author is in data/active/user_feedback.csv (add to v4 train).
4. YouTube moderation dashboard in the Streamlit app (needs the API key in Streamlit secrets).
5. At the end, write for the author (in Vietnamese, to read and learn from): (a) a project report (process, results,
   lessons); (b) a study guide explaining the theory, algorithms and code used (TF-IDF, LR/SVM, PhoBERT &
   fine-tuning, class weights, cleanlab/confident learning, bootstrap CI, Cohen's κ, active learning,
   deployment), pointing to the exact files in this repo.

## Conventions
- Local env: `source /home/user/.venv/bin/activate` (package installed with `pip install -e .`).
- Never commit data or comment texts; commit only code, aggregate reports and label patches.
- GPU training happens on Kaggle (user runs it); this container is CPU-only.
