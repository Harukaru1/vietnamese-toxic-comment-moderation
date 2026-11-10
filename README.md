# Vietnamese Toxic Comment Moderation

Classify Vietnamese social-media comments as **clean / offensive / hate**, and compare
classic ML, a fine-tuned transformer (PhoBERT) and an LLM on the same test set.

> 🚧 Work in progress. Roadmap below.

## Dataset

~37.6k Vietnamese comments (built for my graduation thesis at UIT), 3 classes:

| Split | clean | offensive | hate | Total |
|---|---:|---:|---:|---:|
| train | 17,822 | 2,857 | 9,389 | 30,068 |
| val | 2,228 | 358 | 1,173 | 3,759 |
| test | 2,228 | 357 | 1,174 | 3,759 |

The data is not published (see [`data/README.md`](data/README.md)).

## Results (test set)

Each model family is selected on the **validation** set; the test set is used once for reporting.

| Model | Val F1-macro | Test F1-macro | Test accuracy | Test F1 offensive |
|---|---:|---:|---:|---:|
| TF-IDF (word + char n-grams) + Logistic Regression | 0.829 | **0.812** | **0.862** | **0.695** |
| [PhoBERT-base-v2 fine-tuned](https://huggingface.co/HaruKaru/phobert-vi-toxic-comment) | **0.843** | 0.801 | 0.856 | 0.662 |
| LLM zero/few-shot | | _todo_ | | |

The hardest class is **offensive**: it is the smallest class and is often confused with *hate*.

### Findings so far

- **PhoBERT is best on validation but does not beat TF-IDF on test.** Its val→test drop
  (0.843 → 0.801) is much larger than TF-IDF's (0.829 → 0.812).
- **PhoBERT over-predicts *offensive*:** recall 0.77 but precision only 0.58 (199 clean/hate
  comments predicted as offensive). The class-weighted loss pushes too hard toward the minority class.
- Next experiments: PhoBERT without class weights, and a TF-IDF + PhoBERT ensemble.

### Ablation: does teencode normalization help?

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
src/toxic_moderation/   preprocessing, data loading, shared evaluation
scripts/                training / evaluation entry points
resources/              Vietnamese teencode/slang dictionary (650+ entries)
tests/                  unit tests (pytest)
notebooks/              Kaggle GPU notebooks (PhoBERT)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/train_baseline.py      # trains LR / SVM / RF, picks best on val, reports on test
```

## Roadmap

- [x] Clean project structure, preprocessing with teencode normalization, tests
- [x] TF-IDF baselines (Logistic Regression, Linear SVM, Random Forest)
- [x] Fine-tune PhoBERT on Kaggle GPU, publish model to Hugging Face Hub
- [ ] LLM zero/few-shot classification and comparison (quality, latency, cost)
- [ ] FastAPI inference service + Docker
- [ ] Gradio demo deployed on Hugging Face Spaces
- [ ] CI with GitHub Actions
