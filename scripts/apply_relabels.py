"""Apply a label patch (relabels/*.csv) to a split directory.

    python scripts/apply_relabels.py --src data/clean_vihsd --dst data/clean_vihsd_v3

Each patched row must still carry its `old` label; otherwise the splits were built differently
and the script stops instead of silently corrupting labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "clean_vihsd")
    ap.add_argument("--dst", type=Path, default=ROOT / "data" / "clean_vihsd_v3")
    ap.add_argument("--patch", type=Path, default=ROOT / "relabels" / "guidelines_v3.csv")
    args = ap.parse_args()

    patch = pd.read_csv(args.patch)
    args.dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.read_csv(args.src / f"{split}.csv", encoding="utf-8-sig")
        p = patch[patch["split"] == split]
        mismatch = (df.loc[p["row"].values, "label"].values != p["old"].values).sum()
        if mismatch:
            raise SystemExit(f"{split}: {mismatch} rows do not match the patch's old labels")
        df.loc[p["row"].values, "label"] = p["new"].values
        df.to_csv(args.dst / f"{split}.csv", index=False, encoding="utf-8-sig")
        print(f"{split}: {len(p)} rows patched, {(p['old'] != p['new']).sum()} labels changed")


if __name__ == "__main__":
    main()
