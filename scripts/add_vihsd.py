"""Add the public UIT-ViHSD dataset (same clean/offensive/hate scheme) to the cleaned splits.

- ViHSD train + validation are added to OUR train only, after removing any comment that
  also appears in our train/val/test or in the ViHSD test set (no leakage).
- ViHSD test is kept aside as a second, out-of-domain benchmark (never trained on).

Source: https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD (CC BY-NC-SA 4.0, non-commercial).
Paper: Luu et al., "A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts", 2021.

Output: data/clean_vihsd/{train,val,test}.csv and data/vihsd_test/test.csv (private, git-ignored)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.preprocess import clean_text  # noqa: E402

URL = "https://huggingface.co/datasets/ura-hcmut/UIT-ViHSD/resolve/main/ViHSD_{split}.csv"
LABELS = {0: "clean", 1: "offensive", 2: "hate"}


def load_vihsd(split: str) -> pd.DataFrame:
    d = pd.read_csv(URL.format(split=split)).rename(columns={"free_text": "text"})
    d["text"] = d["text"].astype(str)
    d["label"] = d["label_id"].map(LABELS)
    d["norm"] = [clean_text(t) for t in d["text"]]
    return d[d["norm"].str.len() > 0]


def main() -> None:
    src, out, bench = ROOT / "data" / "clean", ROOT / "data" / "clean_vihsd", ROOT / "data" / "vihsd_test"
    ours = {s: pd.read_csv(src / f"{s}.csv", encoding="utf-8-sig") for s in ("train", "val", "test")}
    for d in ours.values():
        d["norm"] = [clean_text(t) for t in d["text"]]

    v_test = load_vihsd("test").drop_duplicates("norm")
    v_train = pd.concat([load_vihsd("train"), load_vihsd("validation")])
    blocked = set().union(*(set(d["norm"]) for d in ours.values()), set(v_test["norm"]))
    added = v_train[~v_train["norm"].isin(blocked)].drop_duplicates("norm")

    out.mkdir(parents=True, exist_ok=True)
    bench.mkdir(parents=True, exist_ok=True)
    pd.concat([ours["train"], added])[["text", "label"]].to_csv(out / "train.csv", index=False, encoding="utf-8-sig")
    for s in ("val", "test"):
        ours[s][["text", "label"]].to_csv(out / f"{s}.csv", index=False, encoding="utf-8-sig")
    v_test[["text", "label"]].to_csv(bench / "test.csv", index=False, encoding="utf-8-sig")

    stats = {"vihsd_rows_added_to_train": len(added), "added_label_counts": added["label"].value_counts().to_dict(),
             "vihsd_removed_as_duplicates_or_leakage": len(v_train) - len(added),
             "final_train": len(ours["train"]) + len(added), "vihsd_test_benchmark": len(v_test)}
    (ROOT / "reports" / "add_vihsd.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
