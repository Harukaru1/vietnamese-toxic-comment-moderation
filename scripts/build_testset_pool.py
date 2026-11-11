"""Collect a second, larger real-world test set (YouTube, new videos) and pick comments to label.

The 2026 test set has only 78 toxic comments, so its 95% CI (about +-0.06 F1-macro) cannot resolve
small differences between model versions. This script collects comments from videos that were used
in neither the first test set nor the active-learning batch, removes everything already in
train/val/test, and selects a stratified sample:

- random: a plain random slice (keeps an unbiased estimate of the real class mix),
- pred_toxic: comments the deployed model calls offensive/hate,
- cue_pred_clean: comments with group/insult cues that the model calls clean (possible misses),
- pattern: error patterns reported through the demo (animal idioms, body remarks, literal slur words).

    python scripts/build_testset_pool.py          # writes data/testset2/{pool,to_label}.csv, videos.json
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

from build_active_pool import CUES, FEEDBACK_PATTERNS  # noqa: E402
from build_realworld_set import VI_CHARS, VIDEOS as TEST_VIDEOS  # noqa: E402
from toxic_moderation.predictor import Predictor  # noqa: E402
from toxic_moderation.preprocess import clean_text, load_teencode  # noqa: E402
from toxic_moderation.youtube import YouTubeClient, YouTubeError  # noqa: E402

# Topics deliberately different from build_active_pool.QUERIES, so the test set is not a re-sample
# of the training batch's videos or themes.
QUERIES = {
    "esports": ["liên quân chung kết", "tốc chiến highlight", "streamer rank thách đấu"],
    "music": ["MV mới ra mắt", "rap việt battle", "reaction MV"],
    "review": ["review quán ăn", "đánh giá điện thoại", "review phim chiếu rạp"],
    "life": ["va chạm giao thông", "chủ trọ người thuê", "học sinh giáo viên tranh cãi"],
    "talk": ["talkshow hôn nhân", "phỏng vấn đường phố", "tranh luận gen z"],
    # heated topics (different videos from the training batch), to get enough toxic comments
    "drama": ["bóc phốt tiktoker", "fan war kpop", "livestream bán hàng bị phốt", "drama streamer mới nhất"],
    "sport": ["VAR trọng tài tranh cãi", "U23 Việt Nam thua", "fan bóng đá cãi nhau"],
    "news": ["vụ án gây phẫn nộ", "xung đột hàng xóm", "tài xế đánh nhau"],
}
PATTERNS = re.compile(
    FEEDBACK_PATTERNS.pattern + r"|trò con bò|như (trâu|bò|chó|lợn|heo|khỉ)|\bbò đỏ\b|\b(3|ba) que\b|"
    r"\b(mẹ|bố|má|ba) (mày|m|bạn)\b|\bsủa\b|thái độ", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "testset2")
    ap.add_argument("--videos-per-query", type=int, default=4)
    ap.add_argument("--min-comments", type=int, default=150)
    ap.add_argument("--per-video", type=int, default=300)
    ap.add_argument("--n-select", type=int, default=2000)
    ap.add_argument("--published-after", default="2026-01-01T00:00:00Z")
    ap.add_argument("--select-only", action="store_true", help="reuse pool.csv and only redo the selection")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    sources = [ROOT / "data" / d / f"{s}.csv" for d in ("clean_vihsd_v5", "vihsd_test") for s in ("train", "val", "test")]
    sources += [ROOT / "data" / "realworld" / "realworld_2026_v4.csv", ROOT / "data" / "active" / "pool.csv"]
    for p in sources:
        if p.exists():
            seen |= {clean_text(t) for t in pd.read_csv(p, encoding="utf-8-sig")["text"].astype(str)}
    teencode = set(load_teencode())

    def keep(text: str) -> bool:
        norm = clean_text(text)
        words = norm.split()
        if not 2 <= len(words) <= 80 or norm in seen:
            return False
        return bool(VI_CHARS.search(text)) or any(w in teencode for w in words)

    if args.select_only:
        pool = pd.read_csv(args.out_dir / "pool.csv", encoding="utf-8-sig").drop(columns=["why"], errors="ignore")
    else:
        pool = collect_and_score(args, keep)
    select(pool, args)


def collect_and_score(args, keep) -> pd.DataFrame:
    used = {v for vs in TEST_VIDEOS.values() for v in vs}
    active_videos = ROOT / "data" / "active" / "videos.json"
    if active_videos.exists():
        used |= {v["video_id"] for v in json.loads(active_videos.read_text())}
    yt = YouTubeClient()
    found: dict[str, tuple[str, str]] = {}
    for cat, qs in QUERIES.items():
        for q in qs:
            for vid in yt.search_videos(q, max_results=10, published_after=args.published_after):
                if vid not in used and vid not in found:
                    found[vid] = (cat, q)
    infos = [dict(i, category=found[i["video_id"]][0], query=found[i["video_id"]][1]) for i in yt.video_info(list(found))]
    vids = (pd.DataFrame(infos).query("comment_count >= @args.min_comments")
            .sort_values("comment_count", ascending=False).groupby("query").head(args.videos_per_query))
    (args.out_dir / "videos.json").write_text(json.dumps(vids.to_dict("records"), ensure_ascii=False, indent=1))
    rows = []
    for v in vids.itertuples():
        try:
            rows += [dict(c, category=v.category) for c in yt.iter_comments(v.video_id, max_comments=args.per_video)]
        except YouTubeError as e:  # comments disabled etc.
            print("skip", v.video_id, e)
    pool = pd.DataFrame(rows).assign(source="youtube")
    pool = pool[pool["text"].astype(str).map(keep)]
    pool = pool.assign(norm=pool["text"].map(clean_text)).drop_duplicates("norm").drop(columns="norm").reset_index(drop=True)

    preds = Predictor().predict(pool["text"].tolist(), batch_size=64)
    pool["pred"] = [p["label"] for p in preds]
    pool["p_max"] = [max(p["scores"].values()) for p in preds]
    pool["cue"] = pool["text"].astype(str).str.contains(CUES)
    pool["pattern"] = pool["text"].astype(str).str.contains(PATTERNS)
    pool.to_csv(args.out_dir / "pool.csv", index=False, encoding="utf-8-sig")
    return pool


def select(pool: pd.DataFrame, args) -> None:
    n = args.n_select
    rng = np.random.default_rng(0)
    picked: list[int] = []

    def take(idx, k, why):
        done = set(picked)
        idx = [i for i in idx if i not in done][:k]
        picked.extend(idx)
        pool.loc[idx, "why"] = why

    take(list(rng.permutation(pool.index)), int(n * 0.35), "random")
    take(pool[pool["pred"] != "clean"].sample(frac=1, random_state=1).index, int(n * 0.30), "pred_toxic")
    take(pool[pool["cue"] & (pool["pred"] == "clean")].sample(frac=1, random_state=2).index, int(n * 0.20), "cue_pred_clean")
    take(pool[pool["pattern"]].sample(frac=1, random_state=3).index, n - len(picked), "pattern")
    take(list(rng.permutation(pool.index)), n - len(picked), "random")   # top up if a stratum ran short

    sel = pool.loc[picked].sample(frac=1, random_state=7).reset_index(drop=True)
    sel.insert(0, "id", [f"t{i:04d}" for i in range(len(sel))])
    cols = ["id", "category", "video_id", "comment_id", "why", "pred", "p_max", "text"]
    sel.reindex(columns=cols).to_csv(args.out_dir / "to_label.csv", index=False, encoding="utf-8-sig")
    rep = {"videos": int(pool["video_id"].nunique()), "pool": len(pool), "pool_pred": pool["pred"].value_counts().to_dict(),
           "selected": len(sel), "selected_by_why": sel["why"].value_counts().to_dict(),
           "selected_pred": sel["pred"].value_counts().to_dict(),
           "selected_by_category": sel["category"].value_counts().to_dict()}
    (ROOT / "reports" / "build_testset2.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
