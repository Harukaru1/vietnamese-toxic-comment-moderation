"""Compare models on the held-out half of test set 2 (guideline v3.2 labels).

    python scripts/eval_heldout.py --models v3 v5 v6a=models/phobert_v6a/best --baseline v3

A model is a Hub revision (`v3`) or `name=local_dir`. The held-out half (labels/testset2_split.csv, half == test)
is never used for training or threshold tuning. Metrics are weighted back to the 12k-comment pool with the
inverse-probability weights described in scripts/tune_threshold.py, so they estimate performance on real
traffic. The paired bootstrap gives a 95% CI for the difference in weighted toxic F1 against the baseline.
Output: reports/testset2_heldout.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from toxic_moderation import LABELS  # noqa: E402
from toxic_moderation.predictor import Predictor  # noqa: E402
from tune_threshold import cell, prf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v3", "v5"])
    ap.add_argument("--baseline", default="v3")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    d = ROOT / "data" / "testset2"
    pool = pd.read_csv(d / "pool.csv", encoding="utf-8-sig")
    pool["cell"] = cell(pool)
    rows = (pd.read_csv(d / "testset2_labeled_v3.csv")
            .merge(pd.read_csv(ROOT / "labels" / "testset2_split.csv"), on="id")
            .merge(pool[["comment_id", "cell"]], on="comment_id"))
    weights = pool["cell"].value_counts() / rows["cell"].value_counts()
    te = rows[rows["half"] == "test"].reset_index(drop=True)
    y, w = te["label"].to_numpy(), te["cell"].map(weights).to_numpy()

    cache = d / "heldout_preds.csv"
    preds = pd.read_csv(cache) if cache.exists() else te[["id"]].copy()
    names = []
    for spec in args.models:
        name, _, path = spec.partition("=")
        names.append(name)
        if name not in preds:
            p = Predictor(model_id=path) if path else Predictor(revision=name)
            preds[name] = [o["label"] for o in p.predict(te["text"].astype(str).tolist(), batch_size=64)]
            preds.to_csv(cache, index=False)

    def summary(p: np.ndarray) -> dict:
        per = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
        return {"toxic_weighted": prf(y != "clean", p != "clean", w),
                "toxic_unweighted": prf(y != "clean", p != "clean", np.ones(len(y))),
                "f1_macro_3class": round(float(per.mean()), 3),
                **{f"f1_{lab}": round(float(v), 3) for lab, v in zip(LABELS, per)}}

    rep = {"n": int(len(te)), "class_counts": te["label"].value_counts().to_dict(),
           "models": {n: summary(preds[n].to_numpy()) for n in names}, "vs_baseline": {}}
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, len(y), len(y)) for _ in range(args.n)]
    base = preds[args.baseline].to_numpy()
    for n in names:
        if n == args.baseline:
            continue
        p = preds[n].to_numpy()
        diffs = np.array([prf(y[i] != "clean", p[i] != "clean", w[i])["f1"] - prf(y[i] != "clean", base[i] != "clean", w[i])["f1"]
                          for i in idxs])
        rep["vs_baseline"][n] = {"delta_weighted_toxic_f1": round(float(np.mean(diffs)), 3),
                                 "ci95": [round(float(np.percentile(diffs, q)), 3) for q in (2.5, 97.5)],
                                 "p_not_better": round(float((diffs <= 0).mean()), 4)}
    (ROOT / "reports" / "testset2_heldout.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
