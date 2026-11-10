"""Train TF-IDF + linear/tree baselines, pick the best on val, report on test.

Usage: python scripts/train_baseline.py [--no-teencode]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moderation.data import load_split  # noqa: E402
from toxic_moderation.evaluate import evaluate  # noqa: E402
from toxic_moderation.preprocess import clean_text, load_teencode  # noqa: E402


def features() -> FeatureUnion:
    # Word n-grams capture phrases; char n-grams are robust to misspellings/teencode.
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 max_features=200_000, sublinear_tf=True)),
    ])


MODELS = {
    "logreg": LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"),
    "linear_svm": LinearSVC(C=0.5, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced_subsample",
                                            random_state=42),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-teencode", action="store_true")
    args = ap.parse_args()

    teencode = None if args.no_teencode else load_teencode()
    splits = {s: load_split(s) for s in ("train", "val", "test")}
    X = {s: [clean_text(t, teencode) for t in df["text"]] for s, df in splits.items()}
    y = {s: df["label"].tolist() for s, df in splits.items()}

    results, best_name, best_pipe, best_f1 = {}, None, None, -1.0
    for name, clf in MODELS.items():
        pipe = Pipeline([("features", features()), ("clf", clf)])
        pipe.fit(X["train"], y["train"])
        val = evaluate(y["val"], pipe.predict(X["val"]))
        results[name] = {"val_f1_macro": val["f1_macro"], "val_accuracy": val["accuracy"]}
        print(f"{name:14s} val F1-macro={val['f1_macro']:.4f} acc={val['accuracy']:.4f}")
        if val["f1_macro"] > best_f1:
            best_name, best_pipe, best_f1 = name, pipe, val["f1_macro"]

    test = evaluate(y["test"], best_pipe.predict(X["test"]))
    print(f"\nBest on val: {best_name}. Test results:\n{test['report']}")
    print("Confusion matrix (rows=true, cols=pred; clean/offensive/hate):", test["confusion_matrix"])

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(best_pipe, ROOT / "models" / "tfidf_best.joblib")
    out = {"teencode": teencode is not None, "val": results, "best": best_name,
           "test": {k: v for k, v in test.items() if k != "report"}}
    suffix = "" if teencode else "_no_teencode"
    (ROOT / "reports" / f"baseline_results{suffix}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
