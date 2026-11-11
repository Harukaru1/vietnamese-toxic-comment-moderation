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

## Thesis context (author's report, read Nov 2026; do not copy names/text into the repo)
- Thesis data = VFYR-2025 (~39k FB/YouTube/Reddit comments, Jan-May 2025) + ~20k ViHSD; labels from
  ViHateT5 + PhoBERT-ViHSD auto-labels, human review ONLY where the two models disagreed (model kappa 0.65).
  Shared model errors were never reviewed -> explains the inconsistent profanity/intensifier labels we found.
- Thesis claimed macro F1 0.83-0.89 and offensive F1 0.52 -> 0.77 with mT5 paraphrase augmentation; our
  leakage-free re-run gives ~0.80 (README section 1). Do not quote thesis numbers as current results.
- Thesis pipeline removed stopwords before PhoBERT; ours does not (keep it that way: negations matter).
- Thesis future work that matches our roadmap: context-aware model, sarcasm, continued pretraining of
  PhoBERT on unlabeled social-media text (domain-adaptive pretraining; candidate for later).
- Final report (step 5) should include a "thesis -> this project: what changed and why" section.

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
2. Active learning v4 (WAITING FOR KAGGLE): 817 comments labeled (data/active/to_label_final.csv; author
   blind-checked 50: kappa 0.47 independent -> guideline v3.1 "behaviour criticism without insults = clean",
   0.835 after; 9 ambiguous dropped) + 15 author demo-feedback rows -> data/active/active_labeled_v4.csv
   (Kaggle copy: data/active/kaggle_upload/active_labeled_v4.csv, 821 rows, private dataset).
   scripts/add_active_labels.py (repeat x3, train only) -> data/clean_vihsd_v4; notebook
   notebooks/train_phobert_v4_kaggle.ipynb pushes HF branch `v4`. The 2026 test set was re-labeled under
   v3.1 (13 hate->clean): data/realworld/realworld_2026_v4.csv, eval dir data/realworld_eval_v4
   (baseline: v2 0.556, v3 0.647 F1-macro). When the author pastes v4 results: save
   reports/phobert_results_v4.json, run evaluate_model.py --revision v4 on data/realworld_eval_v4,
   data/clean_vihsd_v4 (test) and data/vihsd_test; paired bootstrap v3 vs v4 (see
   reports/realworld_2026_v2_vs_v3_bootstrap.json method); update README results table (v4 row) and
   MODEL_CARD in streamlit_app.py if v4 is promoted to `main`.
3. Short texts / context (author's request): (a) after v4, app mode for a short paragraph: split into
   sentences, classify each, text label = most severe sentence, highlight sentences by colour;
   (b) optional experiment: context-aware model (parent comment [SEP] reply, or whole paragraph,
   max_len 256) with YouTube reply threads; compare with (a). Guideline already has a
   "context-dependent comments" rule (appearance remarks → clean + uncertain unless demeaning on their own).
   Demo feedback from the author is in data/active/user_feedback.csv (add to v4 train).
3b. Shared demo feedback (LIVE, verified Nov 2026): streamlit app logs every vote anonymously via
   src/toxic_moderation/feedback_store.py (CommitScheduler, every 2 min) to private HF dataset
   `HaruKaru/vi_toxic_feedback` (data/*.jsonl, append-only, last vote per session+item wins). Needs
   Streamlit secrets HF_TOKEN (fine-grained, write to that dataset only) + optional FEEDBACK_REPO; without
   them it is disabled. Export for review: scripts/export_feedback.py -> data/active/demo_feedback.csv.
3c. LATER (author: after the model is more mature): spoonerism / "nói lái" decoder, e.g. "mẹo mày bé" ->
   "mẹ mày béo" (keep initial + tone, swap rhymes; or swap initials). Plan: syllable parser, candidates for
   syllable pairs within 2-3 positions, keep only candidates matching a toxic-phrase lexicon, re-score the
   decoded text, show "possible spoonerism" + needs-review (never auto-hate); small test set of harmful vs
   playful spoonerisms; guideline rule "label by decoded meaning".
4. YouTube moderation dashboard in the Streamlit app (needs the API key in Streamlit secrets).
5. At the end, write for the author (in Vietnamese, to read and learn from): (a) a project report (process, results,
   lessons); (b) a study guide explaining the theory, algorithms and code used (TF-IDF, LR/SVM, PhoBERT &
   fine-tuning, class weights, cleanlab/confident learning, bootstrap CI, Cohen's κ, active learning,
   deployment), pointing to the exact files in this repo.

## Conventions
- Local env: `source /home/user/.venv/bin/activate` (package installed with `pip install -e .`).
- Never commit data or comment texts; commit only code, aggregate reports and label patches.
- GPU training happens on Kaggle (user runs it); this container is CPU-only.
