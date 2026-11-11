"""Short-paragraph mode: split a text into sentences, classify each, and summarize.

The classifier was trained on single comments (mostly one or two sentences), so a paragraph is
judged sentence by sentence. The paragraph label is the most severe label among sentences the
model is reasonably sure about; unsure toxic sentences only raise a "needs review" flag, so one
low-confidence false alarm does not condemn a whole paragraph.
"""

from __future__ import annotations

import re

SEVERITY = {"clean": 0, "offensive": 1, "hate": 2}
# split after sentence punctuation followed by whitespace (keeps "5.5", "TP.HCM", "v.v." intact), or at line breaks
_SPLIT = re.compile(r"(?<=[.!?…])\s+|\s*\n+\s*")


def split_sentences(text: str, min_words: int = 3, max_sentences: int = 40) -> list[str]:
    """Split on sentence punctuation / line breaks; glue very short fragments to a neighbour."""
    parts = [p.strip() for p in _SPLIT.split(text or "") if p and p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(out[-1].split()) < min_words:   # previous fragment too short: extend it
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1].split()) < min_words:  # short tail ("Thế thôi.") joins the previous one
        out[-2:] = [f"{out[-2]} {out[-1]}"]
    if len(out) > max_sentences:                        # keep the tail as one block rather than dropping text
        out = out[:max_sentences - 1] + [" ".join(out[max_sentences - 1:])]
    return out


def summarize(preds: list[dict], min_conf: float = 0.6) -> dict:
    """Paragraph label from per-sentence predictions ({"label", "scores"} as returned by Predictor)."""
    label, worst, needs_review = "clean", None, False
    for i, p in enumerate(preds):
        conf = p["scores"][p["label"]]
        if p["label"] != "clean" and conf < min_conf:
            needs_review = True
        elif SEVERITY[p["label"]] > SEVERITY[label]:
            label, worst = p["label"], i
    counts = {lab: sum(p["label"] == lab for p in preds) for lab in SEVERITY}
    return {"label": label, "worst_sentence": worst, "needs_review": needs_review, "counts": counts}
