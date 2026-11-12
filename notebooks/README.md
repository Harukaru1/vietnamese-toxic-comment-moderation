# Kaggle notebooks

One notebook per model version. Each clones this repository, prepares the training data from
private Kaggle datasets with the scripts in `scripts/`, fine-tunes PhoBERT on a Kaggle GPU (T4) and
pushes the model to its own branch of
[`HaruKaru/phobert-vi-toxic-comment`](https://huggingface.co/HaruKaru/phobert-vi-toxic-comment).

| Notebook | Model | Training data |
|---|---|---|
| `train_phobert_kaggle.ipynb` | v1 | original thesis splits, class-weighted loss |
| `train_phobert_v2_clean_kaggle.ipynb` | v2 | leakage removed, cleanlab-pruned, + UIT-ViHSD |
| `train_phobert_v3_kaggle.ipynb` | v3 | v2 data re-labeled with guideline v3 |
| `train_phobert_v4_kaggle.ipynb` | v4 | v3 data + active-learning batch |
| `train_phobert_v5_kaggle.ipynb` | v5 | v4 + guideline v3.1 relabels (790 rows) + `bad`→`đéo` text fix |
| `train_phobert_v6_kaggle.ipynb` | v6 (deployed) | starts from v3; 3 epochs on 1,739 comments labeled under guideline v3.2 |

Kaggle setup: GPU T4 x2, Internet on, secret `HF_TOKEN`; inputs are private Kaggle datasets.
