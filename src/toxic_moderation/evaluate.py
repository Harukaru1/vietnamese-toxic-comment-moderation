"""Shared evaluation so every model (TF-IDF, PhoBERT, LLM) is scored the same way."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import LABELS


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", labels=LABELS), 4),
        "f1_per_class": {
            lab: round(s, 4)
            for lab, s in zip(LABELS, f1_score(y_true, y_pred, average=None, labels=LABELS))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "report": classification_report(y_true, y_pred, labels=LABELS, digits=4),
    }
