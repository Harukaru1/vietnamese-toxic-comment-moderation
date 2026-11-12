"""Evaluate several model revisions on test set 2 (whole set and random stratum).

    python scripts/eval_testset2.py --revisions v2 v3 v4 v5

Reads data/testset2/eval_rows.csv (id, why, text, label; private). Reports 3-class F1-macro, per-class F1
and binary toxic-vs-clean precision / recall / F1, each with a 95% bootstrap CI, for the whole set (enriched
with likely-toxic comments) and for the random stratum (natural class mix). Predictions are cached in
data/testset2/preds.csv; the aggregate report goes to reports/testset2_eval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation import LABELS  # noqa: E402
from toxic_moderation.predictor import Predictor  # noqa: E402


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pr, rc, f1, _ = precision_recall_fscore_support(y != "clean", p != "clean", average="binary", zero_division=0)
    per = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
    return {"f1_macro": float(per.mean()), "toxic_precision": float(pr), "toxic_recall": float(rc),
            "toxic_f1": float(f1), **{f"f1_{lab}": float(v) for lab, v in zip(LABELS, per)}}


def with_ci(y: np.ndarray, p: np.ndarray, n: int, rng) -> dict:
    point = metrics(y, p)
    boots = [metrics(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return {k: {"value": round(v, 3), "ci95": [round(float(np.percentile([b[k] for b in boots], q)), 3) for q in (2.5, 97.5)]}
            for k, v in point.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revisions", nargs="+", default=["v2", "v3", "v4", "v5"])
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    d = ROOT / "data" / "testset2"
    df = pd.read_csv(d / "eval_rows.csv")
    cache = d / "preds.csv"
    preds = pd.read_csv(cache) if cache.exists() else df[["id"]].copy()
    preds = df[["id"]].merge(preds, on="id", how="left")
    for rev in args.revisions:
        if rev not in preds or preds[rev].isna().any():
            preds[rev] = [p["label"] for p in Predictor(revision=rev).predict(df["text"].astype(str).tolist(), batch_size=64)]
            preds.to_csv(cache, index=False)
    y = df["label"].to_numpy()
    rep = {"n": {"all": int(len(df)), "random": int((df.why == "random").sum())},
           "class_counts": {"all": df.label.value_counts().to_dict(),
                            "random": df[df.why == "random"].label.value_counts().to_dict()}}
    for name, mask in (("all", np.ones(len(df), bool)), ("random", (df.why == "random").to_numpy())):
        rng = np.random.default_rng(0)
        rep[name] = {rev: with_ci(y[mask], preds[rev].to_numpy()[mask], args.n, rng) for rev in args.revisions}
    out = ROOT / "reports" / "testset2_eval.json"
    out.write_text(json.dumps(rep, indent=2))
    for name in ("all", "random"):
        print(name, rep["class_counts"][name])
        for rev in args.revisions:
            r = rep[name][rev]
            print(f"  {rev}: F1-macro {r['f1_macro']['value']} | toxic P {r['toxic_precision']['value']} R {r['toxic_recall']['value']} "
                  f"F1 {r['toxic_f1']['value']} {r['toxic_f1']['ci95']} | off {r['f1_offensive']['value']} hate {r['f1_hate']['value']}")


if __name__ == "__main__":
    main()
