"""Evaluate a published Hugging Face model on a local test split (CPU is fine).

    python scripts/evaluate_model.py --data-dir data/dedup --tag _dedup
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.data import load_split  # noqa: E402
from toxic_moderation.evaluate import evaluate  # noqa: E402
from toxic_moderation.predictor import DEFAULT_MODEL_ID, Predictor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--teencode", action="store_true")
    ap.add_argument("--name", default="phobert")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    df = load_split(args.split, args.data_dir)
    preds = Predictor(args.model, teencode=args.teencode).predict(df["text"].tolist(), batch_size=64)
    m = evaluate(df["label"], [p["label"] for p in preds])
    print(m["report"])
    out = ROOT / "reports" / f"{args.name}_eval{args.tag}.json"
    out.write_text(json.dumps({"model": args.model, "data_dir": str(args.data_dir), "split": args.split,
                               **{k: v for k, v in m.items() if k != "report"}}, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
