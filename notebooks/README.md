# Kaggle notebooks

Each notebook clones this repository, copies its training data from a private Kaggle dataset, fine-tunes on a
Kaggle GPU (T4) and pushes the result to the Hugging Face Hub. Setup: GPU T4, Internet on, secret `HF_TOKEN`.

| Notebook | Produces | Training data |
|---|---|---|
| `train_phobert_v3_kaggle.ipynb` | PhoBERT v3 (branch `main`) | ~52k thesis + UIT-ViHSD comments after leakage removal, cleanlab pruning and the `relabels/` patch |
| `train_phobert_v8_kaggle.ipynb` | PhoBERT v8 (branch `v8`, **deployed**) | continues v3 on ~2.6k comments labeled under guideline v3.3 (`labels/`); best of 3 seeds |
| `train_qwen_qlora_kaggle.ipynb` | Qwen2.5-1.5B QLoRA adapter (`qwen-vi-toxic-lora`) | the same data as v8, as a generative classifier |

Notebooks for the intermediate versions (v1, v2, v4–v7) are in the git history; their results are in
[`reports/earlier/`](../reports/earlier/) and [`reports/testset2_heldout.json`](../reports/testset2_heldout.json).
