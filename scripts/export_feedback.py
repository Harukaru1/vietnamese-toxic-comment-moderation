"""Download the demo's shared feedback (private HF dataset) for review / relabeling.

    HF_TOKEN=... python scripts/export_feedback.py            # -> data/active/demo_feedback.csv

Keeps each user's final vote per comment, then one row per distinct comment text with vote counts,
so disagreements between users are easy to spot. Review the rows (guideline v3.1) before adding
them to training: users can be wrong, joke, or try to poison the data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.feedback_store import load_feedback  # noqa: E402
from toxic_moderation.preprocess import clean_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="HaruKaru/vi_toxic_feedback")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "active" / "demo_feedback.csv")
    args = ap.parse_args()

    df = load_feedback(args.repo, token=os.environ.get("HF_TOKEN"))
    df = df[df["feedback"].notna()]
    if df.empty:
        print("no feedback yet")
        return
    # what the user says the label is: the model's label on 👍, their correction on 👎
    df["user_label"] = df["correct"].where(df["feedback"] == "sai", df["model_label"])
    df["norm"] = df["text"].astype(str).map(clean_text)
    agg = (df.groupby("norm")
             .agg(text=("text", "first"), model_label=("model_label", "last"), n_votes=("session", "size"),
                  n_wrong=("feedback", lambda s: int((s == "sai").sum())),
                  user_labels=("user_label", lambda s: json.dumps(s.dropna().value_counts().to_dict())),
                  last_ts=("ts", "max"))
             .reset_index(drop=True).sort_values(["n_wrong", "n_votes"], ascending=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(json.dumps({"votes": len(df), "sessions": df["session"].nunique(), "comments": len(agg),
                      "reported_wrong": int((agg["n_wrong"] > 0).sum()), "out": str(args.out)}, indent=1))


if __name__ == "__main__":
    main()
