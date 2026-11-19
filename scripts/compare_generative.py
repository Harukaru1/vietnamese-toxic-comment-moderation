"""Compare generative models (LLMs) with PhoBERT on the held-out half of test set 2.

    python scripts/compare_generative.py --gemini gemini-flash-lite-latest
    python scripts/compare_generative.py --local qwen_qlora=HaruKaru/qwen-vi-toxic-lora --local-base Qwen/Qwen2.5-1.5B-Instruct

Each generative model labels the same 968 held-out comments (never used for training) and is scored with the
same pool weights and paired bootstrap as scripts/eval_heldout.py. Also scored: the cascade the demo uses,
where a comment counts as toxic only if PhoBERT flags it and the LLM agrees. Predictions are cached as extra
columns of data/testset2/heldout_preds.csv (private); seconds per comment are recorded.
Output: reports/generative_vs_phobert.json (aggregate numbers only).

Note: the reference labels were written by an LLM (Claude) following the guideline and checked blind by the
author (kappa 0.74 toxic vs clean), so they may favour LLMs that are given the same guideline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from toxic_moderation.llm_check import review  # noqa: E402
from tune_threshold import cell, prf  # noqa: E402

DATA = ROOT / "data" / "testset2"
OUT = ROOT / "reports" / "generative_vs_phobert.json"


def heldout() -> tuple[pd.DataFrame, np.ndarray]:
    pool = pd.read_csv(DATA / "pool.csv", encoding="utf-8-sig")
    pool["cell"] = cell(pool)
    rows = (pd.read_csv(DATA / "testset2_labeled_v3.csv")
            .merge(pd.read_csv(ROOT / "labels" / "testset2_split.csv"), on="id")
            .merge(pool[["comment_id", "cell"]], on="comment_id"))
    weights = pool["cell"].value_counts() / rows["cell"].value_counts()
    te = rows[rows["half"] == "test"].reset_index(drop=True)
    return te, te["cell"].map(weights).to_numpy()


def run_gemini(texts: list[str], model: str) -> tuple[list[str], float]:
    out, start = [], time.time()
    for i in range(0, len(texts), 20):
        out += review(texts[i:i + 20], api_key="set-by-proxy-or-env", model=model, retries=6, backoff=5)
        print(f"{model}: {len(out)}/{len(texts)}", flush=True)
    return out, (time.time() - start) / len(texts)


def run_local(texts: list[str], adapter: str | None, base: str) -> tuple[list[str], float]:
    from toxic_moderation.llm_local import LocalLLMClassifier
    clf = LocalLLMClassifier(base, adapter=adapter)
    start = time.time()
    out = clf.predict(texts, progress=True)
    return out, (time.time() - start) / len(texts)


def author_check(preds: pd.DataFrame, cols: list[str], n_boot: int) -> dict:
    """Same comparison against the author's own blind labels (no LLM in the reference). Unweighted: the blind
    rounds were drawn mostly from flagged comments, so recall is high for every model."""
    a = pd.concat([pd.read_csv(DATA / f"blindcheck{i}_author.csv") for i in (1, 2)])[["id", "author"]].dropna()
    a["author"] = a["author"].str.strip().str.lower()
    m = a[a["author"].isin(["clean", "offensive", "hate"])].merge(preds, on="id")
    y = (m["author"] != "clean").to_numpy()
    ones = np.ones(len(y))
    res = {"n": int(len(m)), "toxic_unweighted": {c: prf(y, m[c].to_numpy() != "clean", ones) for c in cols}}
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    base = m[cols[0]].to_numpy() != "clean"
    res[f"vs_{cols[0]}"] = {}
    for c in cols[1:]:
        p = m[c].to_numpy() != "clean"
        d = np.array([prf(y[i], p[i], ones[i])["f1"] - prf(y[i], base[i], ones[i])["f1"] for i in idxs])
        res[f"vs_{cols[0]}"][c] = {"delta_f1": round(float(d.mean()), 3),
                                    "ci95": [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini", nargs="*", default=[], help="Gemini model names")
    ap.add_argument("--local", nargs="*", default=[], help="name=adapter_repo, or name= for the base model alone")
    ap.add_argument("--local-base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--classifier", default="v7", help="PhoBERT column of heldout_preds.csv")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    te, w = heldout()
    y = te["label"].to_numpy()
    cache = DATA / "heldout_preds.csv"
    preds = pd.read_csv(cache)
    assert (preds["id"] == te["id"]).all(), "heldout_preds.csv is not aligned with the held-out rows"
    rep = json.loads(OUT.read_text()) if OUT.exists() else {"seconds_per_comment": {}}
    texts = te["text"].astype(str).tolist()

    jobs = [(m, lambda m=m: run_gemini(texts, m)) for m in args.gemini]
    for spec in args.local:
        name, _, adapter = spec.partition("=")
        jobs.append((name, lambda a=adapter: run_local(texts, a or None, args.local_base)))
    for name, job in jobs:
        if name in preds:
            continue
        labels, sec = job()
        preds[name] = labels
        preds.to_csv(cache, index=False)
        rep["seconds_per_comment"][name] = round(sec, 3)

    base = preds[args.classifier].to_numpy() != "clean"
    systems = {args.classifier: base}
    for col in rep["seconds_per_comment"]:  # every generative model run so far
        llm = preds[col].to_numpy() != "clean"
        systems[col] = llm
        systems[f"{args.classifier}+{col} cascade"] = base & llm

    yt = y != "clean"
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, len(y), len(y)) for _ in range(args.n)]
    rep["n"] = int(len(te))
    rep["share_flagged_weighted"] = {}
    rep["toxic_weighted"] = {}
    rep[f"vs_{args.classifier}"] = {}
    for name, p in systems.items():
        rep["toxic_weighted"][name] = prf(yt, p, w)
        rep["share_flagged_weighted"][name] = round(float((p * w).sum() / w.sum()), 3)
        if name == args.classifier:
            continue
        diffs = np.array([prf(yt[i], p[i], w[i])["f1"] - prf(yt[i], base[i], w[i])["f1"] for i in idxs])
        rep[f"vs_{args.classifier}"][name] = {
            "delta_f1": round(float(diffs.mean()), 3),
            "ci95": [round(float(np.percentile(diffs, q)), 3) for q in (2.5, 97.5)]}
    rep["author_blind_check"] = author_check(preds, [args.classifier, *rep["seconds_per_comment"]], args.n)
    OUT.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
