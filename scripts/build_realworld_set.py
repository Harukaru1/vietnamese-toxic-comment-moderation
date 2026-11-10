"""Build the 2026 real-world test set: YouTube comments (via the official API) + VOZ forum comments.

- YouTube: up to 300 comments from each of 20 videos (8 categories), filtered and de-duplicated
  against the training data, then 35 random comments per video (700 total).
- VOZ: 300 random comments from the public VOZ-HSD corpus (first ~30 MB of data.csv).
- Comments are short-ish (2-80 words). Author names are never stored.

Comments on YouTube change over time, so re-running gives a different sample; the
snapshot used in the README is kept privately (data/ is git-ignored; YouTube ToS and
VOZ-HSD's research-only license do not allow redistribution).

    YOUTUBE_API_KEY=... python scripts/build_realworld_set.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.preprocess import clean_text, load_teencode  # noqa: E402
from toxic_moderation.youtube import YouTubeClient  # noqa: E402

VIDEOS = {
    "news": ["sdjMqIbT55A", "L7QxM1xHF5Q"],
    "society": ["MXh7uECsUCs", "u1flC_kUZj0", "nUgaNDHxYPM"],
    "football": ["_wTmGkgUH_4", "8IeW8ZUwwD8", "H2yGv6l1agI"],
    "showbiz": ["6WHnEiwokd0", "OPPKIjccXJo"],
    "gaming": ["rBDinocbnbc", "JyPHIlw-tvY", "xr7-fRhG1Z4", "hof1COwKJE0"],
    "music": ["HEblLwDSBSE", "gJAbDSse5WM", "tMOQ_lII7Ao"],
    "tech": ["Qf4PZlxMerE"],
    "film": ["EH8_7gP_7js", "Cd2BYYCJLpc"],
}
VOZ_URL = "https://huggingface.co/datasets/tarudesu/VOZ-HSD/resolve/main/data.csv"
VI_CHARS = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "clean_vihsd" / "train.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "realworld" / "realworld_2026_unlabeled.csv")
    ap.add_argument("--per-video", type=int, default=35)
    ap.add_argument("--n-voz", type=int, default=300)
    ap.add_argument("--max-videos", type=int, default=None, help="for quick tests")
    args = ap.parse_args()

    teencode = set(load_teencode())
    seen = {clean_text(t) for t in pd.read_csv(args.train, encoding="utf-8-sig")["text"]}

    def keep(text: str) -> bool:
        norm = clean_text(text)
        words = norm.split()
        if not 2 <= len(words) <= 80 or norm in seen:
            return False
        return bool(VI_CHARS.search(text)) or any(w in teencode for w in words)

    yt = YouTubeClient()
    rows = []
    videos = [(c, v) for c, vs in VIDEOS.items() for v in vs][: args.max_videos]
    for cat, vid in videos:
        rows += [dict(c, category=cat) for c in yt.iter_comments(vid, max_comments=300)]
    pool = pd.DataFrame(rows)
    pool = pool[pool["text"].astype(str).map(keep)]
    pool = pool.assign(norm=pool["text"].map(clean_text)).drop_duplicates("norm")
    ys = pd.concat([g.sample(min(args.per_video, len(g)), random_state=42) for _, g in pool.groupby("video_id")])
    ys = ys.assign(source="youtube")

    voz_path = ROOT / "data" / "external" / "voz_head.csv"
    if not voz_path.exists():
        voz_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-r", "0-31457279", "-o", str(voz_path), VOZ_URL], check=True)
    voz = pd.read_csv(voz_path, on_bad_lines="skip", engine="python").iloc[:-1].rename(columns={"texts": "text"})
    voz = voz[voz["text"].astype(str).map(keep)]
    voz = voz.assign(norm=voz["text"].map(clean_text)).drop_duplicates("norm")
    vs = voz.sample(min(args.n_voz, len(voz)), random_state=42).assign(source="voz", category="forum")

    cols = ["source", "category", "video_id", "comment_id", "published_at", "text"]
    out = pd.concat([ys, vs]).reindex(columns=cols).sample(frac=1, random_state=7).reset_index(drop=True)
    out.insert(0, "id", [f"rw{i:04d}" for i in range(len(out))])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(len(out), out["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
