"""Fine-tune a small instruction LLM (default Qwen2.5-1.5B-Instruct) with QLoRA as a 3-class classifier.

    python scripts/train_qwen_qlora.py --data-dir data/adapt_v6 --output-dir models/qwen_qlora --push HaruKaru/qwen-vi-toxic-lora

The base model is loaded in 4-bit (NF4) and only LoRA adapters on the attention and MLP projections are trained.
Each example is the chat prompt from toxic_moderation.llm_local.build_messages (guideline + comment) followed by
the label word; the loss is computed on the answer only. After every epoch the adapter is scored on the
validation split by reading the label from the first answer token (as LocalLLMClassifier does) and the best
epoch is kept. Toxic rows are rare, so they are repeated (--oversample) and the P(clean) threshold for
calling a comment toxic is tuned on validation and saved next to the adapter (classifier_config.json); the
epoch is chosen by toxic-vs-clean F1 at that threshold. Needs a CUDA GPU (Kaggle T4 works). Writes reports/qwen_qlora_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.llm_check import LABELS  # noqa: E402
from toxic_moderation.llm_local import build_messages  # noqa: E402


def encode(tok, text: str, label: str | None, max_len: int) -> dict:
    prompt = tok.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(label + "<|im_end|>", add_special_tokens=False).input_ids if label else []
    p_ids = p_ids[-(max_len - len(a_ids)):]  # comments are short; truncation only guards the length
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):  # left padding, so the last position is always the prompt end
        k = len(b["input_ids"])
        ids[i, n - k:] = torch.tensor(b["input_ids"])
        labels[i, n - k:] = torch.tensor(b["labels"])
        mask[i, n - k:] = 1
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)  # positions must ignore the left padding
    return {"input_ids": ids, "labels": labels, "attention_mask": mask, "position_ids": position_ids}


@torch.inference_mode()
def predict(model, tok, texts: list[str], label_ids: torch.Tensor, max_len: int, bs: int = 16) -> list[list[float]]:
    """Probabilities of clean / offensive / hate (softmax over the three label words)."""
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        batch = collate([encode(tok, t, None, max_len) for t in texts[i:i + bs]], tok.pad_token_id)
        logits = model(**{k: v.to(model.device) for k, v in batch.items() if k != "labels"}).logits[:, -1]
        out += torch.softmax(logits[:, label_ids.to(logits.device)].float(), -1).tolist()
    return out


def decide(probs: list[list[float]], clean_threshold: float) -> list[str]:
    """Toxic when P(clean) < threshold; then the more likely of offensive / hate."""
    return ["clean" if p[0] >= clean_threshold else ("offensive" if p[1] >= p[2] else "hate") for p in probs]


def tune_threshold(probs: list[list[float]], gold: list[str]) -> tuple[float, float]:
    """The P(clean) threshold that maximizes toxic-vs-clean F1 on validation."""
    y = [g != "clean" for g in gold]
    best = (0.5, -1.0)
    for t in [i / 100 for i in range(5, 100)]:
        p = [lab != "clean" for lab in decide(probs, t)]
        tp = sum(a and b for a, b in zip(p, y))
        f1 = 2 * tp / (sum(p) + sum(y)) if sum(p) + sum(y) else 0.0
        if f1 > best[1]:
            best = (t, f1)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "adapt_v6")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "models" / "qwen_qlora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push", default=None, help="Hub repo for the best adapter")
    ap.add_argument("--revision", default="main", help="Hub branch to push to")
    ap.add_argument("--oversample", nargs="*", default=["offensive=3", "hate=5"],
                    help="repeat the training rows of a label, e.g. offensive=3 hate=5 (toxic rows are rare)")
    ap.add_argument("--max-train", type=int, default=None, help="use only the first N rows (smoke test)")
    ap.add_argument("--cpu-smoke-test", action="store_true", help="no 4-bit, no GPU: only checks the code runs")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = pd.read_csv(args.data_dir / "train.csv").dropna(subset=["text", "label"])
    val = pd.read_csv(args.data_dir / "val.csv").dropna(subset=["text", "label"])
    if args.max_train:
        train, val = train.head(args.max_train), val.head(args.max_train)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    label_ids = torch.tensor([tok.encode(lab, add_special_tokens=False)[0] for lab in LABELS])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    if args.cpu_smoke_test:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quant, device_map={"": 0})
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    repeat = {k: int(v) for k, v in (o.split("=") for o in args.oversample)}
    data = [encode(tok, t, lab, args.max_len) for t, lab in zip(train["text"].astype(str), train["label"])
            for _ in range(repeat.get(lab, 1))]
    steps = math.ceil(len(data) / (args.batch_size * args.grad_accum)) * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    best, history, start = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(data)
        for k in range(0, len(data), args.batch_size):
            batch = {n: v.to(model.device) for n, v in collate(data[k:k + args.batch_size], tok.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=not args.cpu_smoke_test):
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if (k // args.batch_size + 1) % args.grad_accum == 0 or k + args.batch_size >= len(data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if (k // args.batch_size) % 50 == 0:
                print(f"epoch {epoch} step {k // args.batch_size} loss {loss.item() * args.grad_accum:.4f}", flush=True)
        probs = predict(model, tok, val["text"].astype(str).tolist(), label_ids, args.max_len)
        gold = val["label"].tolist()
        macro = float(f1_score(gold, decide(probs, 0.5), labels=list(LABELS), average="macro", zero_division=0))
        thr, f1 = tune_threshold(probs, gold)
        history.append({"epoch": epoch, "val_f1_macro_argmax": round(macro, 4),
                        "val_toxic_f1_tuned": round(f1, 4), "clean_threshold": thr})
        print(history[-1], flush=True)
        if f1 > best:  # choose the epoch by the decision we actually use: toxic vs clean
            best = f1
            model.save_pretrained(args.output_dir / "best")
            tok.save_pretrained(args.output_dir / "best")
            (args.output_dir / "best" / "classifier_config.json").write_text(
                json.dumps({"labels": list(LABELS), "clean_threshold": thr}))

    rep = {"base": args.base, "config": {k: str(v) for k, v in vars(args).items()}, "epochs": history,
           "best_val_toxic_f1": round(best, 4), "train_rows": len(data), "minutes": round((time.time() - start) / 60, 1)}
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, exist_ok=True)
        if args.revision != "main":
            api.create_branch(args.push, branch=args.revision, exist_ok=True)
        api.upload_folder(folder_path=str(args.output_dir / "best"), repo_id=args.push, revision=args.revision,
                          commit_message=f"QLoRA adapter for {args.base} (best validation epoch)")
        rep["pushed_to"] = f"{args.push}@{args.revision}"
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "qwen_qlora_results.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
