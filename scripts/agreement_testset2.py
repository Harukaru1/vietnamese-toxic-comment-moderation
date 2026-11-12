"""Agreement between the author's blind labels and the reference labels of test set 2.

The author labeled two random batches of test set 2 without seeing the reference labels
(round 1: 100 comments, round 2: 160). Kappa is reported against the labels before and after the
guideline revision (v3.2 -> v3.3) that followed the disagreements.

    python scripts/agreement_testset2.py

Inputs (private, git-ignored): data/testset2/blindcheck1_author.csv, blindcheck2_author.csv
(columns id, author) and data/testset2/testset2_labeled_v3.csv (label, label_v32).
Output: reports/annotation_agreement_testset2.json (aggregate numbers only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "testset2"
LABELS = ["clean", "offensive", "hate"]


def scores(human: pd.Series, ref: pd.Series) -> dict:
    return {
        "raw_agreement": round(float((human == ref).mean()), 3),
        "cohen_kappa_3class": round(cohen_kappa_score(human, ref, labels=LABELS), 3),
        "cohen_kappa_binary_toxic": round(cohen_kappa_score(human != "clean", ref != "clean"), 3),
    }


def main() -> None:
    rounds = []
    for i in (1, 2):
        r = pd.read_csv(DATA / f"blindcheck{i}_author.csv")[["id", "author"]]
        rounds.append(r.assign(round=i, n_submitted=len(r)))
    human = pd.concat(rounds)
    submitted = {f"round_{i}": int(len(r)) for i, r in enumerate(rounds, 1)}
    human["author"] = human["author"].astype(str).str.strip().str.lower()
    human = human[human["author"].isin(LABELS)]
    ref = pd.read_csv(DATA / "testset2_labeled_v3.csv", usecols=["id", "label", "label_v32"])
    d = human.merge(ref, on="id")

    res = {
        "submitted": submitted,
        "n": int(len(d)),
        "guideline_v3_2": scores(d["author"], d["label_v32"]),
        "guideline_v3_3": scores(d["author"], d["label"]),
        "per_round_v3_3": {f"round_{i}": {"n": int((d["round"] == i).sum()),
                                          **scores(g["author"], g["label"])}
                           for i, g in d.groupby("round")},
        "confusion_v3_3_rows_author_cols_reference": confusion_matrix(
            d["author"], d["label"], labels=LABELS).tolist(),
    }
    print(json.dumps(res, indent=2))
    (ROOT / "reports" / "annotation_agreement_testset2.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
