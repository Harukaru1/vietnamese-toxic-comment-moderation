"""Build a candidate pool of new comments from other sources, cleaned and scored by PhoBERT.

    python scripts/build_extra_pool.py --voz-sample 8000

Sources (all private, under data/external/, never committed):
  - ViCTSD: 10k comments on VnExpress news articles (UIT, research use)
  - VOZ-HSD: comments from the VOZ forum (the downloaded head of the dataset)
  - YouTube: the rows of the 2026 pool (data/testset2/pool.csv) that are not labeled yet

Every text is cleaned before anything else: URLs, @user tags and emoji are removed, and comments with fewer
than --min-words words, mostly-English comments and duplicates (also of every text already used for training
or testing) are dropped. The cleaned text is what will be labeled and trained on. Each row is scored with the
deployed PhoBERT so that labeling can focus on informative comments (scripts/select_extra_batch.py).
Output: data/extra/pool_scored.csv (private) and reports/extra_pool.json (counts only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import emoji
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXT = ROOT / "data" / "external"
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@[\w.\-À-ỹ]+")  # YouTube handles (@name) and forum tags
ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
WORD = re.compile(r"\w+")
VI_CHARS = set("ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ")
EN_WORDS = set("the and is are you to of it that this my in for what so was with have not but they be on i".split())


def clean(text: str) -> str:
    t = unicodedata.normalize("NFC", str(text))
    t = ZERO_WIDTH.sub("", t)
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = emoji.replace_emoji(t, replace=" ")
    t = " ".join(t.split())
    return t.strip(" ,.:;-–")


def keep(text: str, min_words: int) -> bool:
    words = WORD.findall(text.lower())
    if len(words) < min_words:
        return False
    english = sum(w in EN_WORDS for w in words) / len(words)
    return english < 0.2 or any(ch in VI_CHARS for ch in text.lower())


def key(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def used_texts() -> set[str]:
    """Every text already in training or test data, so the new pool holds only new comments."""
    files = [ROOT / "data" / d / f"{s}.csv" for d in ("clean_vihsd_v3", "adapt_v8") for s in ("train", "val", "test")]
    files += [ROOT / "data" / "realworld_eval_v4" / "test.csv", ROOT / "data" / "testset2" / "testset2_labeled_v3.csv"]
    seen = set()
    for f in files:
        if f.exists():
            seen |= set(pd.read_csv(f)["text"].astype(str).map(clean).map(key))
    return seen


def load_sources(voz_sample: int, seed: int) -> pd.DataFrame:
    vc = pd.concat([pd.read_csv(EXT / "victsd" / f"ViCTSD_{s}.csv") for s in ("train", "valid", "test")])
    victsd = pd.DataFrame({"source": "victsd", "text": vc["Comment"], "orig_label": vc["Toxicity"], "topic": vc["Topic"]})

    voz = pd.read_csv(EXT / "voz_head.csv", nrows=178_000)  # the downloaded file is cut off after this row
    voz = voz.sample(min(voz_sample, len(voz)), random_state=seed)
    voz = pd.DataFrame({"source": "voz", "text": voz["texts"], "orig_label": voz["labels"]})

    pool = pd.read_csv(ROOT / "data" / "testset2" / "pool.csv", encoding="utf-8-sig")
    done = set()
    for f in ("testset2_v3.csv", "pool_train_v33.csv", "pool_train_v33_hard.csv"):
        done |= set(pd.read_csv(ROOT / "labels" / f)["comment_id"])
    pool = pool[~pool["comment_id"].isin(done)]
    youtube = pd.DataFrame({"source": "youtube", "text": pool["text"], "comment_id": pool["comment_id"],
                            "video_id": pool["video_id"], "category": pool["category"]})
    return pd.concat([victsd, voz, youtube], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voz-sample", type=int, default=8000)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--revision", default="v7")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = load_sources(args.voz_sample, args.seed)
    counts = {"raw": raw["source"].value_counts().to_dict()}
    raw["text"] = raw["text"].astype(str).map(clean)
    df = raw[raw["text"].map(lambda t: keep(t, args.min_words))].copy()
    counts["after_clean_and_length"] = df["source"].value_counts().to_dict()
    df["key"] = df["text"].map(key)
    df = df[~df["key"].isin(used_texts())].drop_duplicates("key")
    counts["after_dedup"] = df["source"].value_counts().to_dict()

    from toxic_moderation.predictor import Predictor
    clf = Predictor(revision=args.revision)
    out = clf.predict(df["text"].tolist(), batch_size=64)
    df["pred"] = [o["label"] for o in out]
    df["p_toxic"] = [round(1 - o["scores"]["clean"], 4) for o in out]
    df = df.drop(columns="key").reset_index(drop=True)
    df.insert(0, "id", [f"e{i:05d}" for i in range(len(df))])

    (ROOT / "data" / "extra").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "extra" / "pool_scored.csv", index=False)
    counts["flagged_by_" + args.revision] = df[df["pred"] != "clean"]["source"].value_counts().to_dict()
    rep = {"min_words": args.min_words, "voz_sample": args.voz_sample, "counts": counts}
    (ROOT / "reports" / "extra_pool.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
