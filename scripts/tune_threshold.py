"""Choose decision thresholds for the deployed model on test set 2 (no retraining).

    python scripts/tune_threshold.py --revision v3

Test set 2 is a stratified sample of a 12k-comment pool (enriched with comments the model flagged or that
contain cue words), so raw precision/recall on it do not describe real traffic. Each labeled comment gets an
inverse-probability weight N_cell / n_cell, where the cell is (model flagged it | clean but has a cue word |
clean but matches a pattern | everything else) and N_cell is the cell size in the pool. Weighted metrics
estimate performance on the whole pool.

The labeled rows are split 50/50 (stratified by cell and label) into a tuning half and a held-out half.
On the tuning half we pick
  * t_flag : threshold on P(toxic) = 1 - P(clean) that maximizes weighted F1, and
  * a review band [t_review, t_auto): comments below t_review are shown, above t_auto hidden automatically
    (t_auto = lowest threshold with weighted precision >= --auto-precision), the rest go to a human.
All numbers in the report come from the held-out half. Output: reports/threshold_<revision>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.predictor import Predictor  # noqa: E402


def cell(df: pd.DataFrame) -> pd.Series:
    return np.select([df["pred"] != "clean", df["cue"].astype(bool), df["pattern"].astype(bool)],
                     ["flagged", "cue", "pattern"], "rest")


def prf(y: np.ndarray, flag: np.ndarray, w: np.ndarray) -> dict:
    tp, fp, fn = (w[y & flag]).sum(), (w[~y & flag]).sum(), (w[y & ~flag]).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(float(p), 3), "recall": round(float(r), 3),
            "f1": round(float(2 * p * r / (p + r)) if p + r else 0.0, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="v3")
    ap.add_argument("--auto-precision", type=float, default=0.8)
    ap.add_argument("--review-recall", type=float, default=0.8)
    args = ap.parse_args()
    d = ROOT / "data" / "testset2"
    pool = pd.read_csv(d / "pool.csv", encoding="utf-8-sig")
    pool["cell"] = cell(pool)
    rows = pd.read_csv(d / "testset2_labeled_v3.csv")
    rows = rows[rows["excluded"].isna() | (rows["excluded"] == "")].merge(pool[["comment_id", "cell"]], on="comment_id")
    rows["w"] = rows["cell"].map(pool["cell"].value_counts() / rows["cell"].value_counts())

    cache = d / f"probs_{args.revision}.csv"
    if cache.exists():
        probs = pd.read_csv(cache)
    else:
        out = Predictor(revision=args.revision).predict(rows["text"].astype(str).tolist(), batch_size=64)
        probs = pd.DataFrame({"id": rows["id"], "p_clean": [o["scores"]["clean"] for o in out],
                              "argmax": [o["label"] for o in out]})
        probs.to_csv(cache, index=False)
    rows = rows.merge(probs, on="id")
    rows["p_toxic"] = 1 - rows["p_clean"]

    rng = np.random.default_rng(0)
    rows["half"] = "test"
    for _, idx in rows.groupby(["cell", "label"]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        rows.loc[idx[: len(idx) // 2], "half"] = "tune"
    tune, test = rows[rows.half == "tune"], rows[rows.half == "test"]

    def arrays(df):
        return (df["label"] != "clean").to_numpy(), df["p_toxic"].to_numpy(), df["w"].to_numpy()

    yt, pt, wt = arrays(tune)
    grid = np.round(np.arange(0.30, 0.991, 0.01), 2)
    scores = {t: prf(yt, pt >= t, wt) for t in grid}
    t_flag = max(grid, key=lambda t: scores[t]["f1"])
    autos = [t for t in grid if scores[t]["precision"] >= args.auto_precision and (pt >= t).sum() >= 20]
    t_auto = float(min(autos)) if autos else None
    reviews = [t for t in grid if scores[t]["recall"] >= args.review_recall]
    t_review = float(max(reviews)) if reviews else float(grid[0])

    y, p, w = arrays(test)
    rep = {"revision": args.revision, "n_tune": int(len(tune)), "n_test": int(len(test)),
           "pool_cells": pool["cell"].value_counts().to_dict(), "labeled_cells": rows["cell"].value_counts().to_dict(),
           "estimated_toxic_rate_in_pool": round(float((w * y).sum() / w.sum()), 4),
           "held_out": {"argmax": prf(y, (test["argmax"] != "clean").to_numpy(), w),
                        f"t_flag={t_flag:.2f}": prf(y, p >= t_flag, w)},
           "t_flag": float(t_flag), "t_review": t_review, "t_auto": t_auto}
    if t_auto is not None:
        auto, band = p >= t_auto, (p >= t_review) & (p < t_auto)
        rep["held_out"]["review_band"] = {
            "auto_hide": prf(y, auto, w), "auto_hide_share_of_traffic": round(float(w[auto].sum() / w.sum()), 4),
            "review_share_of_traffic": round(float(w[band].sum() / w.sum()), 4),
            "toxic_caught_auto_or_review": round(float(w[y & (auto | band)].sum() / w[y].sum()), 3),
            "toxic_share_inside_review_band": round(float(w[y & band].sum() / w[band].sum()), 3) if w[band].sum() else None}
    (ROOT / "reports" / f"threshold_{args.revision}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
