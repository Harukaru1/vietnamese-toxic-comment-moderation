"""Measure the LLM second stage against the author's blind labels.

    python scripts/eval_llm_stage.py --model-rev v7

Uses the 155 held-out comments the author labeled blind (data/testset2/blindcheck2_author.csv, private). For the
comments the classifier flags as toxic it asks the LLM (src/toxic_moderation/llm_check.py) for a label and reports
how often the author also called them toxic: for the classifier alone, and where the LLM confirms. The sample
was drawn mostly from flagged comments, so these are precision-style figures, not full-traffic estimates.
Output: reports/llm_second_stage.json (aggregate numbers only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.llm_check import DEFAULT_MODEL, review  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-rev", default="v7", help="column of data/testset2/heldout_preds.csv")
    ap.add_argument("--llm", default=DEFAULT_MODEL)
    args = ap.parse_args()
    d = ROOT / "data" / "testset2"
    a = pd.read_csv(d / "blindcheck2_author.csv").dropna(subset=["author"])
    a["author"] = a["author"].str.strip().str.lower()
    a = a.merge(pd.read_csv(d / "heldout_preds.csv")[["id", args.model_rev]], on="id")
    a = a.merge(pd.read_csv(d / "testset2_labeled_v3.csv")[["id", "label"]], on="id")
    flagged = a[a[args.model_rev] != "clean"].copy()
    cache = d / f"llm_labels_{args.llm}.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["id", "llm"])
    todo = flagged[~flagged["id"].isin(done["id"])]
    for i in range(0, len(todo), 20):
        part = todo.iloc[i:i + 20]
        labels = review(part["text"].tolist(), api_key="set-by-proxy-or-env", model=args.llm, retries=6, backoff=5)
        done = pd.concat([done, pd.DataFrame({"id": part["id"], "llm": labels})])
        done.to_csv(cache, index=False)
    flagged = flagged.merge(done, on="id")
    tox = flagged["author"] != "clean"

    def stage(mask):
        return {"n": int(mask.sum()), "author_says_toxic": round(float(tox[mask].mean()), 3),
                "share_of_true_toxic_kept": round(float(tox[mask].sum() / tox.sum()), 3)}

    rep = {"classifier": args.model_rev, "llm": args.llm, "n_author_labeled": int(len(a)),
           "classifier_only": stage(pd.Series(True, index=flagged.index)),
           "llm_confirms": stage(flagged["llm"] != "clean"),
           "reference_labels_confirm": stage(flagged["label"] != "clean"),
           "llm_vs_author_3class_agreement": round(float((flagged["llm"] == flagged["author"]).mean()), 3)}
    (ROOT / "reports" / "llm_second_stage.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
