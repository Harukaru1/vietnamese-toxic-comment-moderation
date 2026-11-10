"""Inter-annotator agreement between the author's blind labels and the LLM pre-annotation.

    python scripts/agreement.py data/realworld/blind_100_for_tri.xlsx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["clean", "offensive", "hate"]


def main() -> None:
    human = pd.read_excel(sys.argv[1], sheet_name="annotate")
    human["label"] = human["label"].astype(str).str.strip().str.lower()
    human = human[human["label"].isin(LABELS)]
    llm = pd.read_csv(ROOT / "data" / "realworld" / "claude_labels.csv")
    d = human.merge(llm, on="id", suffixes=("_human", "_llm"))
    d_bin = d.assign(h=d["label_human"] != "clean", l=d["label_llm"] != "clean")

    res = {
        "n": len(d),
        "raw_agreement": round(float((d["label_human"] == d["label_llm"]).mean()), 3),
        "cohen_kappa_3class": round(cohen_kappa_score(d["label_human"], d["label_llm"], labels=LABELS), 3),
        "cohen_kappa_binary_toxic": round(cohen_kappa_score(d_bin["h"], d_bin["l"]), 3),
        "confusion_rows_human_cols_llm": confusion_matrix(d["label_human"], d["label_llm"], labels=LABELS).tolist(),
    }
    print(json.dumps(res, indent=2))
    (ROOT / "reports" / "annotation_agreement.json").write_text(json.dumps(res, indent=2))
    dis = d[d["label_human"] != d["label_llm"]]
    text = pd.read_csv(ROOT / "data" / "realworld" / "realworld_2026_unlabeled.csv", encoding="utf-8-sig")
    dis = dis.drop(columns=["text"], errors="ignore").merge(text[["id", "text"]], on="id")
    dis[["id", "label_human", "label_llm", "text"]].to_csv(ROOT / "data" / "realworld" / "disagreements.csv",
                                                           index=False, encoding="utf-8-sig")
    print(f"{len(dis)} disagreements -> data/realworld/disagreements.csv")


if __name__ == "__main__":
    main()
