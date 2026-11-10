"""Fine-tune PhoBERT for 3-class Vietnamese toxic comment classification.

Designed to run on a Kaggle GPU (see notebooks/train_phobert_kaggle.ipynb), but also
works on CPU for a quick smoke test:

    python scripts/train_phobert.py --max-train-samples 64 --epochs 1 --no-push

Main options:
    --teencode            apply the teencode dictionary before tokenization
    --push-to-hub REPO    upload the best model to the Hugging Face Hub (needs HF_TOKEN)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from pyvi import ViTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation import LABELS  # noqa: E402
from toxic_moderation.data import DATA_DIR, load_split  # noqa: E402
from toxic_moderation.evaluate import evaluate  # noqa: E402
from toxic_moderation.preprocess import clean_text, load_teencode  # noqa: E402

LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))


def prepare_text(text: str, teencode: dict | None) -> str:
    # PhoBERT was pre-trained on word-segmented text ("học_sinh"), so we segment with pyvi.
    return ViTokenizer.tokenize(clean_text(text, teencode))


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (offensive is the minority class)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/phobert-base-v2")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "models" / "phobert")
    ap.add_argument("--teencode", action="store_true")
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--max-train-samples", type=int, default=None, help="for smoke tests")
    ap.add_argument("--push-to-hub", default=None, help="e.g. HaruKaru/phobert-vi-toxic-comment")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. _clean")
    args = ap.parse_args()
    set_seed(args.seed)

    teencode = load_teencode() if args.teencode else None
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def to_dataset(split: str) -> Dataset:
        df = load_split(split, args.data_dir)
        if args.max_train_samples:
            df = df.sample(n=min(args.max_train_samples, len(df)), random_state=args.seed)
        ds = Dataset.from_dict({
            "text": [prepare_text(t, teencode) for t in df["text"]],
            "label": [LABEL2ID[lab] for lab in df["label"]],
        })
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])

    print("Preparing data (cleaning + word segmentation)...")
    train_ds, val_ds, test_ds = (to_dataset(s) for s in ("train", "val", "test"))

    class_weights = None
    if not args.no_class_weights:
        counts = np.bincount(train_ds["label"], minlength=len(LABELS)).astype(float)
        counts[counts == 0] = 1.0
        w = counts.sum() / (len(LABELS) * counts)
        class_weights = torch.tensor(np.sqrt(w), dtype=torch.float)  # sqrt: softer than full balancing
        print("Class weights:", dict(zip(LABELS, class_weights.round(decimals=3).tolist())))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        m = evaluate([ID2LABEL[i] for i in labels], [ID2LABEL[i] for i in preds])
        return {"f1_macro": m["f1_macro"], "accuracy": m["accuracy"]}

    n_devices = max(1, torch.cuda.device_count())
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * n_devices))
    warmup_steps = int(0.1 * steps_per_epoch * args.epochs)  # 10% warmup

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )
    trainer.train()

    val_metrics = trainer.evaluate(val_ds)
    preds = trainer.predict(test_ds)
    y_true = [ID2LABEL[i] for i in preds.label_ids]
    y_pred = [ID2LABEL[i] for i in preds.predictions.argmax(-1)]
    test = evaluate(y_true, y_pred)
    print(f"\nVal F1-macro: {val_metrics['eval_f1_macro']:.4f}\nTest results:\n{test['report']}")
    print("Confusion matrix (rows=true, cols=pred; clean/offensive/hate):", test["confusion_matrix"])

    best_dir = args.output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    results = {
        "model": args.model,
        "teencode": args.teencode,
        "config": {k: str(v) for k, v in vars(args).items()},
        "val_f1_macro": round(val_metrics["eval_f1_macro"], 4),
        "test": {k: v for k, v in test.items() if k != "report"},
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    out = ROOT / "reports" / f"phobert_results{'_teencode' if args.teencode else ''}{args.tag}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("Saved", out)

    if args.push_to_hub and not args.no_push:
        trainer.model.push_to_hub(args.push_to_hub, private=False)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
