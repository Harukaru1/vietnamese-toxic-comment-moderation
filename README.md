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
- [x] CI with GitHub Actions (tests on every push, auto-deploy to Hugging Face Spaces)
