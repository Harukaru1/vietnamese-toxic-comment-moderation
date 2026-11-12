# Project notes for Claude

Portfolio project extending the author's graduation thesis. Working style: explain steps in
Vietnamese, keep tasks small, and let the author run Kaggle/GitHub/HF steps themselves.
Code, commits and README in English. Read README.md first; docs/bao_cao_du_an.md has the full story.

## State (Nov 2026)
- 3-class Vietnamese toxic comment classifier (clean / offensive / hate), guideline **v3.2**
  (docs/annotation_guidelines.md): hate = identity-based attack, body shaming, threats; offensive = insult at
  one person without that, or crude venting; intensifier profanity = clean.
- HF Hub `HaruKaru/phobert-vi-toxic-comment`: `main` = v3, branches v2-v6. The Streamlit demo
  (https://vi-toxic-moderation.streamlit.app/, not reachable from this container) serves branch **v6** via
  MODEL_REVISION in streamlit_app/streamlit_app.py (set "v3" to roll back).
- v6 = v3 + 3 epochs (lr 1e-5) on 1,739 v3.2-labeled comments (data/adapt_v6; notebook
  notebooks/train_phobert_v6_kaggle.ipynb). Held-out half of test set 2, weighted: toxic P 0.45 / R 0.50 / F1 0.47
  vs v3 0.28 / 0.45 / 0.35 (delta +0.125, CI [0.045, 0.223]); threshold 0.8 -> P 0.62 / R 0.36. Hate F1 0.28.
  2026 set 1 (older labels): toxic F1 0.58 vs 0.63. Local CPU run of the same recipe gave P 0.53 (run variance).
- v4, v5: negative results, documented in README. Learning-curve check: more v3.2 data helps hate, not binary.
- App tabs: single comment, paragraph (src/toxic_moderation/segment.py), batch, YouTube video (queues on
  P(toxic): >=0.8 hide, 0.4-0.8 review), history & feedback. Emoji-only comments skip the model (preprocess.has_text).
- Streamlit secrets set by the author: HF_TOKEN (+ FEEDBACK_REPO), YOUTUBE_API_KEY. Feedback goes to private HF
  dataset `HaruKaru/vi_toxic_feedback` (feedback_store.py; export with scripts/export_feedback.py).

## Data (data/ is git-ignored and private; this container is the only copy of the texts)
- Thesis pipeline: data/raw -> find_label_issues.py -> make_clean_splits.py --prune -> add_vihsd.py ->
  apply_relabels.py (relabels/*.csv) -> data/clean_vihsd_v3 (v5 variant: fix_bad_token.py, add_active_labels.py).
- 2026 set 1: data/realworld_eval_v4/test.csv (1,000 rows, guideline v3.1).
- Test set 2: data/testset2/ (pool.csv 12k YouTube comments; testset2_labeled_v2.csv; train_to_label.csv +
  train_labels.txt). Public id-only labels: labels/testset2_v2.csv, labels/testset2_split.csv (tune half = training
  data, test half = 968 held-out rows, NEVER train on it), labels/pool_train_v32.csv.
- Evaluate a model: python scripts/eval_heldout.py --models v3 v6 [name=local_dir]; whole set: eval_testset2.py;
  set 1: bootstrap_compare.py --a v3 --b v6 --data-dir data/realworld_eval_v4.
- YouTube API key is a network secret (header X-Goog-Api-Key) on the cloud environment
  `vietnamese-toxic-comment-moderation`; src/toxic_moderation/youtube.py.

## Thesis context (do not copy names/text from the thesis into the repo)
Thesis labels came from two models with human review only on disagreements; thesis F1 0.83-0.89 is not
reproducible without leakage (~0.80). Use repo numbers only.

## Open ideas
- Label more hate examples under v3.2 (use v6 to pick candidates from the ~9k unlabeled pool rows).
- Average several fine-tuning runs; mix some old clean rows to limit forgetting.
- Context-aware model (parent + reply); spoonerism ("nói lái") decoder: author said later.
- Author still has to set the GitHub "About" section manually.

## Conventions
- Local env: `source /home/user/.venv/bin/activate` (package installed with `pip install -e .`).
- Never commit data or comment texts; commit only code, aggregate reports and label patches. No personal info.
- Never ask the author to paste secrets in chat. Do not store YouTube author names; no Facebook/TikTok scraping.
- GPU training happens on Kaggle (author runs it); this container is CPU-only (small fine-tunes take ~10 min).
- Vietnamese docs for the author: docs/bao_cao_du_an.md (report), docs/tai_lieu_hoc.md (study guide); keep them
  in sync with README when results change.
