"""Paired bootstrap comparison of two model revisions on the same test set.

    python scripts/bootstrap_compare.py --a v3 --b v4 --data-dir data/realworld_eval_v4 \
        --out reports/realworld_2026_v3_vs_v4_bootstrap.json

Both models predict every comment once; then comments are resampled with replacement (same
indices for both models) and F1-macro is recomputed, giving a 95% CI for F1(b) - F1(a) and the
share of resamples where b is not better. Also reports toxic (offensive+hate vs clean) precision/recall.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation import LABELS  # noqa: E402
from toxic_moderation.predictor import Predictor  # noqa: E402


def toxic_pr(y, p) -> dict:
    yt, pt = np.asarray(y) != "clean", np.asarray(p) != "clean"
    return {"toxic_precision": round(precision_score(yt, pt), 3), "toxic_recall": round(recall_score(yt, pt), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="v3")
    ap.add_argument("--b", default="v4")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "realworld_eval_v4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data_dir / f"{args.split}.csv", encoding="utf-8-sig")
    y = df["label"].to_numpy()
    preds = {}
    for rev in (args.a, args.b):
        preds[rev] = np.array([p["label"] for p in Predictor(revision=rev).predict(df["text"].astype(str).tolist(), batch_size=64)])

    def f1(p, idx):
        return f1_score(y[idx], p[idx], labels=LABELS, average="macro", zero_division=0)

    rng = np.random.default_rng(0)
    full = np.arange(len(y))
    deltas = []
    for _ in range(args.n):
        idx = rng.integers(0, len(y), len(y))
        deltas.append(f1(preds[args.b], idx) - f1(preds[args.a], idx))
    deltas = np.array(deltas)
    rep = {f"delta_f1_macro_{args.b}_minus_{args.a}": round(f1(preds[args.b], full) - f1(preds[args.a], full), 4),
           "ci95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
           f"p_{args.b}_not_better": round(float((deltas <= 0).mean()), 4),
           "n_bootstrap": args.n, "data_dir": str(args.data_dir.relative_to(ROOT)),
           args.a: toxic_pr(y, preds[args.a]), args.b: toxic_pr(y, preds[args.b]),
           "n_changed_predictions": int((preds[args.a] != preds[args.b]).sum())}
    args.out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
