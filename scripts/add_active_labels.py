"""Add the active-learning batch (newly labeled real-world comments) to the training split.

    python scripts/add_active_labels.py --active data/active/active_labeled_v4.csv \
        --src data/clean_vihsd_v3 --dst data/clean_vihsd_v4

- Only the train split changes; val/test are copied unchanged, so v3 and v4 stay comparable.
- Comments that already appear in train/val/test or in the 2026 real-world test set (normalized
  text) are dropped, so the batch cannot leak into any evaluation set.
- The batch is small (~800 comments vs ~50k), so it is repeated `--repeat` times to give the
  model enough signal from these harder, more recent examples.

The labeled batch itself is private (YouTube/VOZ comments); on Kaggle it is uploaded as a
private dataset file named `active_labeled_v4.csv` with columns text,label.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation import LABELS  # noqa: E402
from toxic_moderation.preprocess import clean_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active", type=Path, default=ROOT / "data" / "active" / "active_labeled_v4.csv")
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "clean_vihsd_v3")
    ap.add_argument("--dst", type=Path, default=ROOT / "data" / "clean_vihsd_v4")
    ap.add_argument("--realworld-test", type=Path, default=ROOT / "data" / "realworld_eval" / "test.csv")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "add_active_v4.json")
    args = ap.parse_args()

    act = pd.read_csv(args.active, encoding="utf-8-sig")[["text", "label"]].dropna()
    bad = set(act["label"]) - set(LABELS)
    if bad:
        raise SystemExit(f"unknown labels in {args.active}: {bad}")

    splits = {s: pd.read_csv(args.src / f"{s}.csv", encoding="utf-8-sig") for s in ("train", "val", "test")}
    seen = set()
    for df in splits.values():
        seen |= set(df["text"].astype(str).map(clean_text))
    if args.realworld_test.exists():
        seen |= set(pd.read_csv(args.realworld_test, encoding="utf-8-sig")["text"].astype(str).map(clean_text))

    act = act.assign(norm=act["text"].astype(str).map(clean_text))
    n_in = len(act)
    act = act[~act["norm"].isin(seen)].drop_duplicates("norm")
    add = pd.concat([act[["text", "label"]]] * args.repeat, ignore_index=True)

    args.dst.mkdir(parents=True, exist_ok=True)
    train = pd.concat([splits["train"], add], ignore_index=True).sample(frac=1, random_state=42)
    train.to_csv(args.dst / "train.csv", index=False, encoding="utf-8-sig")
    for s in ("val", "test"):
        shutil.copy(args.src / f"{s}.csv", args.dst / f"{s}.csv")

    rep = {"active_rows": n_in, "dropped_overlap_or_duplicate": n_in - len(act), "added_unique": len(act),
           "repeat": args.repeat, "added_label_counts": act["label"].value_counts().to_dict(),
           "final_train": len(train)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
