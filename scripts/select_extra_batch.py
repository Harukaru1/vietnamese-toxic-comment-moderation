"""Pick the most informative new comments to label, using PhoBERT scores and a Gemini pre-label.

    python scripts/select_extra_batch.py --quota victsd=500 voz=300 youtube=200 --test-victsd 200

1. A random --test-victsd rows of ViCTSD are set aside first as a small news-comment test set (random, so the
   scores on it are unbiased; never used for training).
2. From the rest, Gemini (src/toxic_moderation/llm_check.py, guideline v3.3 in the prompt) labels every comment
   PhoBERT finds at least slightly suspicious (P(toxic) >= --min-p) or the source dataset flagged as toxic,
   plus a random sample of the others.
3. Per source, the batch takes first the comments where PhoBERT and Gemini disagree on toxic vs clean (what the
   model can learn most from), then comments both call toxic, then comments both call clean.
Disagreements and the test rows are marked needs_review for a manual check under the guideline.
Inputs/outputs are private (data/extra/); reports/extra_batch.json holds counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.llm_check import review  # noqa: E402

EXTRA = ROOT / "data" / "extra"


def gemini_labels(df: pd.DataFrame, cache: Path) -> pd.DataFrame:
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["id", "gemini"])
    todo = df[~df["id"].isin(done["id"])]
    for i in range(0, len(todo), 20):
        part = todo.iloc[i:i + 20]
        labels = review(part["text"].tolist(), api_key="set-by-proxy-or-env", retries=6, backoff=5)
        done = pd.concat([done, pd.DataFrame({"id": part["id"], "gemini": labels})])
        done.to_csv(cache, index=False)
        if (i // 20) % 10 == 0:
            print(f"gemini: {len(done)} labeled", flush=True)
    return df.merge(done, on="id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota", nargs="+", default=["victsd=500", "voz=300", "youtube=200"])
    ap.add_argument("--test-victsd", type=int, default=200)
    ap.add_argument("--min-p", type=float, default=0.2)
    ap.add_argument("--random-low", type=int, default=400, help="random low-score rows per source sent to Gemini")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    quota = {k: int(v) for k, v in (q.split("=") for q in args.quota)}

    pool = pd.read_csv(EXTRA / "pool_scored.csv")
    test = pool[pool["source"] == "victsd"].sample(args.test_victsd, random_state=args.seed)
    rest = pool.drop(test.index)

    cand = []
    for src in quota:
        s = rest[rest["source"] == src]
        # ViCTSD's own toxicity flag is only a hint for selection (its definition differs from ours)
        hint = (s["p_toxic"] >= args.min_p) | (s["orig_label"].fillna(0) == 1)
        high, low = s[hint], s[~hint]
        cand += [high, low.sample(min(args.random_low, len(low)), random_state=args.seed)]
    cand = pd.concat(cand + [test])
    cand = gemini_labels(cand, EXTRA / "gemini_labels.csv")

    test = cand[cand["id"].isin(test["id"])].assign(split="test_victsd", needs_review=1)
    cand = cand[~cand["id"].isin(test["id"])]
    batch, rep = [], {"sent_to_gemini": int(len(cand) + len(test)), "per_source": {}}
    for src, n in quota.items():
        s = cand[cand["source"] == src].sample(frac=1, random_state=args.seed)
        v7_tox, gem_tox = s["pred"] != "clean", s["gemini"] != "clean"
        groups = [s[v7_tox != gem_tox], s[v7_tox & gem_tox], s[~v7_tox & ~gem_tox]]
        take, left = [], n
        for g, cap in zip(groups, (int(0.45 * n), int(0.30 * n), n)):
            k = min(len(g), cap, left)
            take.append(g.head(k))
            left -= k
        if left:  # not enough in the capped groups: fill with whatever is left
            used = pd.concat(take).index
            take.append(s.drop(used).head(left))
        b = pd.concat(take).assign(split="train")
        b["needs_review"] = ((b["pred"] != "clean") != (b["gemini"] != "clean")).astype(int)
        batch.append(b)
        rep["per_source"][src] = {"selected": int(len(b)), "disagreements": int(b["needs_review"].sum()),
                                  "gemini_toxic": int((b["gemini"] != "clean").sum())}
    out = pd.concat(batch + [test], ignore_index=True)
    out.to_csv(EXTRA / "batch1.csv", index=False)
    rep["test_victsd"] = {"n": int(len(test)), "gemini_toxic": int((test["gemini"] != "clean").sum()),
                          "v7_toxic": int((test["pred"] != "clean").sum())}
    rep["needs_review"] = int(out["needs_review"].sum())
    (ROOT / "reports" / "extra_batch.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
