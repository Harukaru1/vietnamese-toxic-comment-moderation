"""Undo a text artifact in the thesis data: the Vietnamese swear word "đéo" (also written "đ", "dell")
was replaced by the English word "bad" in ~1,500 comments, e.g. "có sai bad đâu" (= "có sai đéo đâu").

Real comments never contain this "bad" (0 of 1,000 in the 2026 set), so the model learned a token it
will never see, and 69% of the affected rows are hate. This script maps it back to "đéo" in every split,
keeping genuine English uses ("too bad", "bad boy", "bad trip", ...).

    python scripts/fix_bad_token.py --dir data/clean_vihsd_v5      # edits the CSVs in place
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BAD = re.compile(r"\bbad\b", re.I)
ENGLISH = re.compile(
    r"\b(too|so|not|very|is|was|feel|feeling|the|a)\s+bad\b|"
    r"\bbad\s+(boys?|girls?|guys?|ending|news|day|luck|time|trip|blood|bunny|idea|game|words?|habits?|things?)\b",
    re.I)


def fix(text: str) -> str:
    if not isinstance(text, str) or not BAD.search(text) or ENGLISH.search(text):
        return text
    return BAD.sub("đéo", text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "fix_bad_token.json")
    args = ap.parse_args()
    rep = {}
    for split in ("train", "val", "test"):
        p = args.dir / f"{split}.csv"
        df = pd.read_csv(p, encoding="utf-8-sig")
        new = df["text"].map(fix)
        changed = new != df["text"]
        rep[split] = {"rows_changed": int(changed.sum()),
                      "label_counts_changed": df.loc[changed, "label"].value_counts().to_dict()}
        df["text"] = new
        df.to_csv(p, index=False, encoding="utf-8-sig")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
