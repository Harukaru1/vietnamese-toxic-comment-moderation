# Vietnamese Toxic Comment Moderation

Classify Vietnamese social-media comments as **clean / offensive / hate**, and compare
classic ML, a fine-tuned transformer (PhoBERT) and an LLM on the same test set.

> 🚧 Work in progress. Roadmap below.

**Live demo:** [vi-toxic-moderation.streamlit.app](https://vi-toxic-moderation.streamlit.app/) · **Model:** [HaruKaru/phobert-vi-toxic-comment](https://huggingface.co/HaruKaru/phobert-vi-toxic-comment)

<p align="center"><img src="docs/demo_single.png" width="600" alt="Demo: classifying a single comment"></p>

## Dataset

~37.6k Vietnamese comments (built for my graduation thesis at UIT), 3 classes:

| Split | clean | offensive | hate | Total |
|---|---:|---:|---:|---:|
| train | 17,822 | 2,857 | 9,389 | 30,068 |
| val | 2,228 | 358 | 1,173 | 3,759 |
| test | 2,228 | 357 | 1,174 | 3,759 |

The data is not published (see [`data/README.md`](data/README.md)).

## Results (de-duplicated test set)

Each model family is selected on the **validation** set; the test set is used once for reporting.
All numbers below use the **leakage-free** splits (see *Data quality* below).

| Model | Training data | Test F1-macro | Test accuracy | Test F1 offensive |
|---|---|---:|---:|---:|
| TF-IDF (word + char n-grams) + Logistic Regression | de-duplicated | 0.793 | 0.861 | 0.648 |
| TF-IDF (word + char n-grams) + Linear SVM | de-duplicated + label errors pruned | **0.799** | **0.865** | **0.662** |
| [PhoBERT-base-v2](https://huggingface.co/HaruKaru/phobert-vi-toxic-comment) v1 (class-weighted) | original | 0.780 | 0.853 | 0.608 |
| PhoBERT v2 (no class weights) | de-duplicated / + pruned | _running_ | | |
| LLM zero/few-shot | – | _todo_ | | |

The hardest class is **offensive**: it is the smallest class (8% of comments) and overlaps with *hate*.
With 281 offensive comments in the test set, differences below ~0.015 F1-macro are within
the bootstrap 95% confidence interval and should not be over-interpreted.

## Data quality: leakage and label noise

**1. Train → test leakage.** 178 test comments (4.7%) and 211 validation comments (5.6%) also
appear verbatim in the training set, mostly toxic comments that were posted repeatedly.
11 texts appear with *conflicting* labels. Removing them shows the original scores were inflated:

| TF-IDF + LR | Test F1-macro | Test F1 offensive |
|---|---:|---:|
| original splits (with leakage) | 0.820 | 0.710 |
| leakage-free splits | 0.793 | 0.647 |

**2. Label noise (confident learning).** Out-of-fold probabilities from 5-fold cross-validation
were passed to [cleanlab](https://github.com/cleanlab/cleanlab), which flagged **2,076 comments
(5.5%)** as likely mislabeled; the *offensive* class has the highest rate (11.6%).
A manual review of 120 flagged training comments (20 per confusion type) against the
annotation guidelines found:

| Review outcome | Share |
|---|---:|
| clearly mislabeled | 34% |
| ambiguous under the guidelines | 30% |
| label was correct | 36% |

The ambiguous cases come from an overlap in the guidelines: both *offensive* ("attacks an
issue, organization or person") and *hate* ("heavy language insulting or discriminating against
a group or someone") cover insults aimed at a person, and profanity without a target
(e.g. "đm hay vãi", which is actually praise) is labeled inconsistently.

**3. Effect.** Pruning the flagged comments from **train only** (val/test untouched) improves
TF-IDF on both validation (+0.008) and test (+0.007 F1-macro, +0.013 F1 offensive).

Reproduce: `scripts/find_label_issues.py` → `scripts/make_clean_splits.py [--prune]`.

## More data: adding UIT-ViHSD

[UIT-ViHSD](https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD) (Luu et al., 2021; CC BY-NC-SA 4.0)
uses the same clean / offensive / hate scheme. Its train + dev sets (22.5k comments after removing
anything that overlaps our splits or the ViHSD test set) were added to **our train only**; the
ViHSD test set (6.4k) is kept aside as an **out-of-domain benchmark**.

| Model | Training data | Our test F1-macro | ViHSD test F1-macro |
|---|---|---:|---:|
| TF-IDF + LR | ours (cleaned) | 0.801 | 0.603 |
| TF-IDF + LR | ours (cleaned) + ViHSD | 0.797 | **0.667** |
| PhoBERT v1 | ours (original) | 0.780 | 0.643 |

More data does not move the in-domain score (the difference is within noise), but it clearly
improves robustness to comments from another source, which matters for a public demo.


### Ablation: does teencode normalization help? (original splits)

| Model | Teencode dictionary | Val F1-macro | Test F1-macro | Test F1 offensive |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | yes | 0.829 | 0.812 | 0.695 |
| TF-IDF + Linear SVM | no | 0.823 | 0.822 | 0.718 |
| PhoBERT | yes | 0.828 | 0.798 | 0.659 |
| PhoBERT | no | 0.843 | 0.801 | 0.662 |

Result: **no measurable gain** for either model family. Character n-grams (TF-IDF) and
subword tokens (PhoBERT) already handle most spelling variants, and fine-tuning lets
PhoBERT learn slang directly from the data.

## Project structure

```
src/toxic_moderation/   preprocessing, teencode dictionary (650+ entries), evaluation, predictor
app/                    FastAPI service + Gradio UI (one app, Docker)
streamlit_app/          Streamlit demo (deployed on Streamlit Community Cloud)
scripts/                training / evaluation entry points
tests/                  unit tests (pytest)
notebooks/              Kaggle GPU notebooks (PhoBERT)
Dockerfile              CPU image for the API/demo (used by Hugging Face Spaces)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-app.txt && pip install -e .
pytest
python scripts/train_baseline.py      # trains LR / SVM / RF, picks best on val, reports on test
```

### Run the API + demo

```bash
uvicorn app.main:app --port 7860      # demo: http://localhost:7860  |  API docs: http://localhost:7860/docs
```

```bash
curl -X POST localhost:7860/predict -H "Content-Type: application/json" \
     -d '{"texts": ["Video rất hay và bổ ích", "Đá như này thì nghỉ mẹ đi cho rồi"]}'
```

The model (`HaruKaru/phobert-vi-toxic-comment`) is downloaded from the Hugging Face Hub on the
first request. On CPU, a single comment takes ~40 ms.

Docker: `docker build -t toxic-mod . && docker run -p 7860:7860 toxic-mod`

### Run the Streamlit demo

```bash
pip install streamlit
streamlit run streamlit_app/streamlit_app.py
```

Single-comment mode and batch mode (paste lines or upload a CSV, download the labeled results).

## Roadmap

- [x] Clean project structure, preprocessing with teencode normalization, tests
- [x] TF-IDF baselines (Logistic Regression, Linear SVM, Random Forest)
- [x] Fine-tune PhoBERT on Kaggle GPU, publish model to Hugging Face Hub
- [ ] LLM zero/few-shot classification and comparison (quality, latency, cost)
- [x] FastAPI inference service + Docker
- [x] Streamlit demo (single + batch/CSV moderation)
- [x] Deploy demo on Streamlit Community Cloud
- [x] CI with GitHub Actions (tests on every push; manual deploy to Hugging Face Spaces)
- [x] Data quality audit: leakage removal + confident learning label-noise detection
- [x] Add UIT-ViHSD as extra training data + out-of-domain benchmark
- [ ] Retrain PhoBERT on cleaned data (+ ViHSD)
