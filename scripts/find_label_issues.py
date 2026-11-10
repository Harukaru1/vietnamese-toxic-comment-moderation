"""Find likely mislabeled comments with confident learning (cleanlab).

1. Get out-of-fold predicted probabilities for every comment (5-fold CV, TF-IDF + LR),
   so each comment is scored by a model that never saw it.
2. cleanlab flags comments whose given label is unlikely under those probabilities,
   using per-class confidence thresholds.

Outputs:
    data/label_issues.csv        flagged comments with texts (private: data/ is git-ignored)
    reports/label_audit.json     aggregate statistics only (safe to publish)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation import LABELS  # noqa: E402
from toxic_moderation.data import load_split  # noqa: E402
from toxic_moderation.features import build_tfidf  # noqa: E402
from toxic_moderation.preprocess import clean_text  # noqa: E402


def main() -> None:
    df = pd.concat([load_split(s).assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
    X = [clean_text(t) for t in df["text"]]
    y = df["label"].map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()

    pipe = Pipeline([("features", build_tfidf()),
                     ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Computing out-of-fold probabilities (5 folds)...")
    probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=5)

    issues = find_label_issues(labels=y, pred_probs=probs, return_indices_ranked_by="self_confidence")
    out = df.loc[issues, ["split", "text", "label"]].copy()
    out["suggested"] = [LABELS[i] for i in probs[issues].argmax(1)]
    out["p_given"] = probs[issues, y[issues]].round(3)
    out["p_suggested"] = probs[issues].max(1).round(3)
    out.insert(0, "row_id", issues)
    out.to_csv(ROOT / "data" / "label_issues.csv", index=False, encoding="utf-8-sig")

    stats = {
        "n_total": int(len(df)),
        "n_flagged": int(len(issues)),
        "flagged_rate": round(len(issues) / len(df), 4),
        "flagged_by_split": out["split"].value_counts().to_dict(),
        "flagged_by_given_label": {lab: int((out["label"] == lab).sum()) for lab in LABELS},
        "flagged_rate_by_given_label": {lab: round(float((out["label"] == lab).sum() / (df["label"] == lab).sum()), 4)
                                        for lab in LABELS},
        "given_to_suggested": {f"{a}->{b}": int(n) for (a, b), n in
                               out.groupby(["label", "suggested"]).size().items()},
        "oof_accuracy": round(float((probs.argmax(1) == y).mean()), 4),
    }
    (ROOT / "reports" / "label_audit.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
