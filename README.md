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

| Model | F1-macro | Accuracy | F1 offensive |
|---|---:|---:|---:|
| TF-IDF (word + char n-grams) + Logistic Regression | 0.812 | 0.862 | 0.695 |
| PhoBERT fine-tuned | _todo_ | | |
| LLM zero/few-shot | _todo_ | | |

The hardest class is **offensive**: it is the smallest class and is often confused with *hate*.

### Ablation: does teencode normalization help TF-IDF?

| Preprocessing | Best model (by val) | Val F1-macro | Test F1-macro | Test F1 offensive |
|---|---|---:|---:|---:|
| with teencode dictionary | Logistic Regression | 0.829 | 0.812 | 0.695 |
| without teencode dictionary | Linear SVM | 0.823 | 0.822 | 0.718 |

Result: no clear gain for TF-IDF. Character n-grams already capture most spelling
variants, and the gain seen on val did not carry over to test. Next step: check whether
normalization helps PhoBERT, whose subword vocabulary is trained on standard Vietnamese.

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
- [ ] Fine-tune PhoBERT on Kaggle GPU, publish model to Hugging Face Hub
- [ ] LLM zero/few-shot classification and comparison (quality, latency, cost)
- [ ] FastAPI inference service + Docker
- [ ] Gradio demo deployed on Hugging Face Spaces
- [ ] CI with GitHub Actions
