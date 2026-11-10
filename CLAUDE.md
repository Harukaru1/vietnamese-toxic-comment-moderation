# Project notes for Claude

Portfolio project extending the author's graduation thesis. Working style: explain steps in
Vietnamese, keep tasks small, and let the author run Kaggle/GitHub/HF steps themselves.
Code, commits and README in English.

## What exists
- 3-class Vietnamese toxic comment classifier (clean / offensive / hate), extension of the thesis.
- PhoBERT v2 on HF Hub `HaruKaru/phobert-vi-toxic-comment` (`main`); demo on Streamlit Cloud
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
In-domain test: TF-IDF ≈ PhoBERT v2 ≈ 0.80 F1-macro. 2026 set: PhoBERT v2 0.59 F1-macro, toxic recall 0.50;
weak on implicit hate. Leakage (4.7%) and inconsistent profanity labels were the main data issues.

## Next steps
1. User runs `notebooks/train_phobert_v3_kaggle.ipynb` (pushes to HF branch `v3`), pastes results.
   Then evaluate: `python scripts/evaluate_model.py --revision v3 --data-dir data/realworld_eval --name phobert_v3 --tag _realworld`
   and compare with v2; if better, upload v3 to `main` and reboot the Streamlit app.
2. Active learning: pick uncertain YouTube/VOZ comments (not in the 2026 test set), label them, retrain
   to fix implicit hate.
3. YouTube moderation dashboard in the Streamlit app (needs the API key in Streamlit secrets).
4. Write up the project summary.

## Conventions
- Local env: `source /home/user/.venv/bin/activate` (package installed with `pip install -e .`).
- Never commit data or comment texts; commit only code, aggregate reports and label patches.
- GPU training happens on Kaggle (user runs it); this container is CPU-only.
