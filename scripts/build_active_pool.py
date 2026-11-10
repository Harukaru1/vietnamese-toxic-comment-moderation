"""Active learning, step 1: collect new unlabeled comments and pick the most useful ones to label.

Pool
- YouTube: comments from videos on topics where implicit hate is common (football rivalries,
  regional/gender debates, showbiz drama, international news), found with the YouTube search API.
  The 20 videos of the 2026 test set are excluded, so no test video leaks into training.
- VOZ: comments from the public VOZ-HSD corpus (its binary hate flag is used only for sampling).
Everything already in train/val/test or in the 2026 test set is removed (normalized text match).

Selection (default ~700 comments), scored with the deployed model (PhoBERT v3):
- uncertain: lowest max-probability (the model is unsure),
- predicted toxic (offensive/hate),
- implicit-hate cues: comments matching a small lexicon of group slurs / stereotypes that the
  model predicts as clean (the error type found on the 2026 set),
- a random slice of the rest, so the labeled batch is not only hard cases.

    python scripts/build_active_pool.py            # writes data/active/pool.csv and to_label.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_realworld_set import VI_CHARS, VIDEOS as TEST_VIDEOS  # noqa: E402
from toxic_moderation.predictor import Predictor  # noqa: E402
from toxic_moderation.preprocess import clean_text, load_teencode  # noqa: E402
from toxic_moderation.youtube import YouTubeClient, YouTubeError  # noqa: E402

QUERIES = {
    "football": ["Việt Nam vs Thái Lan", "Việt Nam vs Indonesia highlights", "bóng đá Việt Nam thua",
                 "V-League trọng tài"],
    "society": ["tranh cãi vùng miền", "nữ quyền tranh cãi", "phụ nữ nên", "người miền bắc miền nam",
                "giới trẻ bây giờ"],
    "showbiz": ["drama showbiz", "nghệ sĩ bị tẩy chay", "streamer drama"],
    "news": ["Trung Quốc biển Đông", "bình luận thời sự quốc tế", "lừa đảo bị bắt"],
}
CUES = re.compile(
    r"đông lào|thái lọ|bắc kỳ|bắc kì|nam kỳ|nam kì|parky|bake|3 que|ba que|bò đỏ|tàu khựa|khựa|"
    r"tàu cộng|chệt|mọi rợ|mọi|bê đê|bede|pede|đồng bóng|incel|simp|phò|đĩ|cave|gái gú|đàn bà|"
    r"con gái bây giờ|phụ nữ|nữ quyền|feminist|thiểu năng|não tôm|óc chó|óc lợn|ngu|súc vật|"
    r"thú vật|bọn|lũ|tụi|đám|chết đi|đáng chết|phơi xác|hoả thiêu|hỏa thiêu|xử bắn|giết",
    re.I)
# Error patterns reported through the demo's feedback button (Nov 2026): body shaming / animal
# comparisons, regional slurs on their own, and an intensifier aimed at a person ("m béo vl").
FEEDBACK_PATTERNS = re.compile(
    r"\b(béo|mập|lùn|xấu như)\b|như (con )?(lợn|heo|chó|khỉ)|parky|pake|\bcút\b|"
    r"\b(m|mày|bạn|thằng|con|mi)\b.{0,30}\b(vl|vcl|vãi|vkl)\b",
    re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "active")
    ap.add_argument("--videos-per-query", type=int, default=4)
    ap.add_argument("--min-comments", type=int, default=100)
    ap.add_argument("--per-video", type=int, default=200)
    ap.add_argument("--n-voz", type=int, default=4000)
    ap.add_argument("--n-select", type=int, default=700)
    ap.add_argument("--published-after", default="2025-01-01T00:00:00Z")
    ap.add_argument("--n-feedback", type=int, default=150, help="extra comments matching FEEDBACK_PATTERNS")
    ap.add_argument("--select-only", action="store_true",
                    help="reuse data/active/pool.csv (no API calls, no re-scoring) and only redo the selection")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for d in ["clean_vihsd_v3", "vihsd_test"]:
        for split in ["train", "val", "test"]:
            p = ROOT / "data" / d / f"{split}.csv"
            if p.exists():
                seen |= {clean_text(t) for t in pd.read_csv(p, encoding="utf-8-sig")["text"].astype(str)}
    seen |= {clean_text(t) for t in pd.read_csv(ROOT / "data" / "realworld" / "realworld_2026_v3.csv")["text"].astype(str)}
    teencode = set(load_teencode())

    def keep(text: str) -> bool:
        norm = clean_text(text)
        words = norm.split()
        if not 2 <= len(words) <= 80 or norm in seen:
            return False
        return bool(VI_CHARS.search(text)) or any(w in teencode for w in words)

    if args.select_only:
        pool = pd.read_csv(args.out_dir / "pool.csv", encoding="utf-8-sig")
        pool["norm"] = pool["text"].astype(str).map(clean_text)
        pool = pool.drop(columns=["why"], errors="ignore")
        n_videos = pool.loc[pool["source"] == "youtube", "video_id"].nunique()
    else:
        pool, n_videos = collect_and_score(args, keep)
    select(pool, args, n_videos)


def collect_and_score(args, keep) -> tuple[pd.DataFrame, int]:
    # --- YouTube ---------------------------------------------------------------------------
    test_ids = {v for vs in TEST_VIDEOS.values() for v in vs}
    yt = YouTubeClient()
    found: dict[str, tuple[str, str]] = {}
    for cat, qs in QUERIES.items():
        for q in qs:
            for vid in yt.search_videos(q, max_results=10, published_after=args.published_after):
                if vid not in test_ids and vid not in found:
                    found[vid] = (cat, q)
    infos = [dict(i, category=found[i["video_id"]][0], query=found[i["video_id"]][1])
             for i in yt.video_info(list(found))]
    vids = (pd.DataFrame(infos).query("comment_count >= @args.min_comments")
            .sort_values("comment_count", ascending=False).groupby("query").head(args.videos_per_query))
    (args.out_dir / "videos.json").write_text(json.dumps(vids.to_dict("records"), ensure_ascii=False, indent=1))
    rows = []
    for v in vids.itertuples():
        try:
            rows += [dict(c, category=v.category) for c in yt.iter_comments(v.video_id, max_comments=args.per_video)]
        except YouTubeError as e:  # comments disabled etc.
            print("skip", v.video_id, e)
    ys = pd.DataFrame(rows).assign(source="youtube")

    # --- VOZ -------------------------------------------------------------------------------
    voz = pd.read_csv(ROOT / "data" / "external" / "voz_head.csv", on_bad_lines="skip", engine="python")
    voz = voz.iloc[:-1].rename(columns={"texts": "text", "labels": "voz_hate"})
    voz = voz[voz["text"].astype(str).map(keep)]
    voz_hate = voz[voz["voz_hate"] == 1]
    voz_s = pd.concat([voz_hate.sample(min(args.n_voz // 2, len(voz_hate)), random_state=1),
                       voz[voz["voz_hate"] != 1].sample(args.n_voz // 2, random_state=1)])
    voz_s = voz_s.assign(source="voz", category="forum")

    pool = pd.concat([ys, voz_s], ignore_index=True)
    pool = pool[pool["text"].astype(str).map(keep)]
    pool = pool.assign(norm=pool["text"].map(clean_text)).drop_duplicates("norm").reset_index(drop=True)

    # --- score with the deployed model -----------------------------------------------------
    preds = Predictor().predict(pool["text"].tolist(), batch_size=64)
    pool["pred"] = [p["label"] for p in preds]
    for lab in ["clean", "offensive", "hate"]:
        pool[f"p_{lab}"] = [p["scores"][lab] for p in preds]
    pool["p_max"] = pool[["p_clean", "p_offensive", "p_hate"]].max(axis=1)
    pool["cue"] = pool["text"].astype(str).str.contains(CUES)
    return pool, len(vids)


def select(pool: pd.DataFrame, args, n_videos: int) -> None:
    # --- select ----------------------------------------------------------------------------
    n = args.n_select
    rng = np.random.default_rng(0)
    picked: list[int] = []

    def take(idx, k, why):
        done = set(picked)
        idx = [i for i in idx if i not in done][:k]
        picked.extend(idx)
        pool.loc[idx, "why"] = why

    take(pool.sort_values("p_max").index, int(n * 0.35), "uncertain")
    take(pool[pool["cue"] & (pool["pred"] == "clean")].sample(frac=1, random_state=2).index, int(n * 0.25), "cue_pred_clean")
    take(pool[pool["pred"] != "clean"].sample(frac=1, random_state=3).index, int(n * 0.25), "pred_toxic")
    take(list(rng.permutation(pool.index)), n - len(picked), "random")
    fb = pool[pool["text"].astype(str).str.contains(FEEDBACK_PATTERNS)].sample(frac=1, random_state=4)
    take(fb.index, args.n_feedback, "feedback_pattern")

    pool.drop(columns="norm").to_csv(args.out_dir / "pool.csv", index=False, encoding="utf-8-sig")
    sel = pool.loc[picked].sample(frac=1, random_state=7).reset_index(drop=True)
    sel.insert(0, "id", [f"al{i:04d}" for i in range(len(sel))])
    cols = ["id", "source", "category", "video_id", "comment_id", "why", "pred", "p_max", "text"]
    sel.reindex(columns=cols).to_csv(args.out_dir / "to_label.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"videos": n_videos, "pool": len(pool), "pool_by_source": pool["source"].value_counts().to_dict(),
                      "pool_pred": pool["pred"].value_counts().to_dict(), "selected": len(sel),
                      "selected_by_why": sel["why"].value_counts().to_dict(),
                      "selected_by_source": sel["source"].value_counts().to_dict()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
