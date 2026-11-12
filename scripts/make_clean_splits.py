"""Build de-duplicated, leakage-free splits and (optionally) prune likely label errors.

Steps
1. Drop train texts that appear with conflicting labels; keep one copy of exact duplicates.
2. Remove val/test comments whose normalized text also appears in train (leakage).
3. With --prune: drop train comments flagged by scripts/find_label_issues.py.
   Val/test labels are never changed, so evaluation stays comparable.

Output: data/clean/{train,val,test}.csv  (private, git-ignored)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.data import load_split  # noqa: E402
from toxic_moderation.preprocess import clean_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune", action="store_true", help="drop train rows flagged by find_label_issues.py")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "clean")
    args = ap.parse_args()

    splits = {s: load_split(s) for s in ("train", "val", "test")}
    for df in splits.values():
        df["norm"] = [clean_text(t) for t in df["text"]]
    stats = {"original": {s: len(d) for s, d in splits.items()}}

    train = splits["train"]
    n_labels = train.groupby("norm")["label"].nunique()
    conflicting = set(n_labels[n_labels > 1].index)
    train = train[~train["norm"].isin(conflicting)].drop_duplicates("norm")
    stats["train_conflicting_texts_dropped"] = len(conflicting)

    seen = set(train["norm"])
    for s in ("val", "test"):
        d = splits[s]
        stats[f"{s}_leaked_rows_dropped"] = int(d["norm"].isin(seen).sum())
        splits[s] = d[~d["norm"].isin(seen)].drop_duplicates("norm")

    if args.prune:
        issues = pd.read_csv(ROOT / "data" / "label_issues.csv", encoding="utf-8-sig")
        flagged = {clean_text(t) for t in issues.loc[issues["split"] == "train", "text"]}
        before = len(train)
        train = train[~train["norm"].isin(flagged)]
        stats["train_flagged_rows_pruned"] = before - len(train)
    splits["train"] = train

    args.out.mkdir(parents=True, exist_ok=True)
    for s, d in splits.items():
        d[["text", "label"]].to_csv(args.out / f"{s}.csv", index=False, encoding="utf-8-sig")
    stats["final"] = {s: len(d) for s, d in splits.items()}
    stats["final_label_counts"] = {s: d["label"].value_counts().to_dict() for s, d in splits.items()}
    (ROOT / "reports" / "earlier" / f"clean_splits{'_pruned' if args.prune else ''}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
