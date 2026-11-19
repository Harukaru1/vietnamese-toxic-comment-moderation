"""A small open LLM (e.g. Qwen2.5-1.5B-Instruct) used as a 3-class classifier, with or without a LoRA adapter.

The comment goes into a chat prompt that contains the annotation guideline; the model's answer is read by
comparing the probabilities of the three label words as the first answer token, so every comment gets
exactly one valid label in a single forward pass. The guideline part of the prompt is the same for every
comment, so its key/value cache is computed once and reused. The same prompt is used to fine-tune the adapter
(notebooks/train_qwen_qlora_kaggle.ipynb), so zero-shot and fine-tuned runs differ only in the weights.
"""

from __future__ import annotations

import copy

import torch

from .llm_check import GUIDELINE, LABELS


def build_messages(text: str) -> list[dict]:
    comment = " ".join(str(text).split())[:500]
    return [{"role": "system", "content": GUIDELINE},
            {"role": "user", "content": f"Bình luận: {comment}\nNhãn (clean, offensive hoặc hate):"}]


def _read_threshold(repo: str, revision: str | None) -> float | None:
    import json
    from pathlib import Path
    path = Path(repo) / "classifier_config.json"
    if not path.exists():
        try:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo, "classifier_config.json", revision=revision))
        except Exception:  # older adapters have no config: fall back to argmax
            return None
    return float(json.loads(path.read_text())["clean_threshold"])


class LocalLLMClassifier:
    def __init__(self, base: str = "Qwen/Qwen2.5-1.5B-Instruct", adapter: str | None = None,
                 device: str | None = None):
        """adapter: a local folder or a Hub repo, optionally with a branch ("user/repo@v2"). If the adapter
        folder has classifier_config.json (written by scripts/train_qwen_qlora.py), its P(clean) threshold
        is used; otherwise the most likely label is taken."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(base)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        self.clean_threshold = None
        if adapter:
            from peft import PeftModel
            repo, _, rev = adapter.partition("@")
            model = PeftModel.from_pretrained(model, repo, revision=rev or None)
            self.clean_threshold = _read_threshold(repo, rev or None)
        self.model = model.to(self.device).eval()
        first = [self.tok.encode(lab, add_special_tokens=False)[0] for lab in LABELS]
        assert len(set(first)) == len(LABELS), "label words must start with different tokens"
        self.label_ids = torch.tensor(first)
        # Everything before the comment is identical for all comments: cache it once.
        full = self.prompt("x")
        self.prefix = full[:full.index("Bình luận:")]  # cut at a special-token boundary (after "user\n")
        self._prefix_ids = self.tok(self.prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        with torch.inference_mode():
            self._prefix_cache = self.model(self._prefix_ids, use_cache=True).past_key_values

    def prompt(self, text: str) -> str:
        return self.tok.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def predict(self, texts: list[str], progress: bool = False) -> list[str]:
        out = []
        for k, t in enumerate(texts):
            full = self.prompt(t)
            if full.startswith(self.prefix):
                rest = self.tok(full[len(self.prefix):], return_tensors="pt",
                                add_special_tokens=False).input_ids.to(self.device)
                ids = torch.cat([self._prefix_ids, rest], dim=1)
                logits = self.model(rest, past_key_values=copy.deepcopy(self._prefix_cache),
                                    attention_mask=torch.ones_like(ids)).logits[0, -1]
            else:  # should not happen; fall back to the full prompt
                ids = self.tok(full, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                logits = self.model(ids).logits[0, -1]
            scores = logits[self.label_ids.to(logits.device)].float()
            if self.clean_threshold is None:
                out.append(LABELS[int(scores.argmax())])
            else:
                p = torch.softmax(scores, -1).tolist()
                out.append("clean" if p[0] >= self.clean_threshold else ("offensive" if p[1] >= p[2] else "hate"))
            if progress and (k + 1) % 50 == 0:
                print(f"local LLM: {k + 1}/{len(texts)}", flush=True)
        return out
