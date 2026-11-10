"""Inference wrapper around the fine-tuned PhoBERT model (shared by the API and the demo)."""

from __future__ import annotations

import os

import torch
from pyvi import ViTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocess import clean_text, load_teencode

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "HaruKaru/phobert-vi-toxic-comment")
DEFAULT_REVISION = os.getenv("MODEL_REVISION") or None  # e.g. a branch such as "v3"


class Predictor:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, teencode: bool = False,
                 max_len: int = 160, device: str | None = None, revision: str | None = DEFAULT_REVISION):
        # Must match the preprocessing used at training time (the published model: no teencode).
        self.model_id, self.revision = model_id, revision
        self.teencode = load_teencode() if teencode else None
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        self.model = self.model.to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def preprocess(self, text: str) -> str:
        return ViTokenizer.tokenize(clean_text(text, self.teencode))

    @torch.inference_mode()
    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = [self.preprocess(t) for t in texts[i:i + batch_size]]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            for p in probs:
                scores = {lab: round(s, 4) for lab, s in zip(self.labels, p)}
                results.append({"label": max(scores, key=scores.get), "scores": scores})
        return results
