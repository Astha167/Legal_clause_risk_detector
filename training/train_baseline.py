"""
STEP 5 - Classical baseline: TF-IDF + Logistic Regression.

* input text   : clause_text (clean original legal text; no stop-word removal,
                 so negations such as "not", "no", "without" are preserved)
* vectorizer   : fitted on the TRAIN split only
* C selection  : small grid, chosen on the VALIDATION split (macro F1)
* final report : TEST split, untouched until the end

Usage (from the project root):
    python training/train_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from training.common import (  # noqa: E402
    SplitError,
    compute_metrics,
    load_split_frames,
    print_metrics,
    save_model_results,
)

C_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]


def build_vectorizer():
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=config.TFIDF_WORD_NGRAM_RANGE,
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=config.TFIDF_MIN_DF,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w'-]+\b",  # keeps "not", "no", "non-exclusive"
        stop_words=None,
    )
    if not config.TFIDF_USE_CHAR_NGRAMS:
        return word
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=config.TFIDF_MIN_DF,
                           max_features=config.TFIDF_MAX_FEATURES, sublinear_tf=True)
    return FeatureUnion([("word", word), ("char", char)])


def main() -> int:
    config.set_seed(config.RANDOM_SEED)
    try:
        frames, info = load_split_frames()
    except SplitError as exc:
        print(f"ERROR: {exc}")
        return 1
    train, val, test = frames["train"], frames["val"], frames["test"]
    print(f"Train / Val / Test clauses: {len(train)} / {len(val)} / {len(test)}")

    start = time.time()
    vec = build_vectorizer()
    X_train = vec.fit_transform(train["clause_text"])     # fit on train only
    X_val = vec.transform(val["clause_text"])
    X_test = vec.transform(test["clause_text"])
    print(f"Vocabulary size: {X_train.shape[1]}")

    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=config.LOGREG_MAX_ITER, class_weight="balanced",
                                 random_state=config.RANDOM_SEED)
        clf.fit(X_train, train["label"])
        m = compute_metrics(val["label"], clf.predict(X_val))
        print(f"  C={c:<4} validation macro F1 = {m['f1_macro']:.4f}")
        if best is None or m["f1_macro"] > best[1]["f1_macro"]:
            best = (c, m, clf)
    best_c, val_metrics, clf = best
    print(f"Selected C = {best_c} (by validation macro F1)")

    test_metrics = compute_metrics(test["label"], clf.predict(X_test))
    elapsed = round(time.time() - start, 1)
    print_metrics("TEST RESULTS - TF-IDF + Logistic Regression", test_metrics)
    print(test_metrics["classification_report_text"])

    config.BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, config.BASELINE_MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(clf, config.BASELINE_MODEL_DIR / "logistic_regression.joblib")
    meta = {
        "model": "TF-IDF + Logistic Regression",
        "labels": config.LABELS,
        "C": best_c,
        "class_weight": "balanced",
        "ngram_range": list(config.TFIDF_WORD_NGRAM_RANGE),
        "max_features": config.TFIDF_MAX_FEATURES,
        "min_df": config.TFIDF_MIN_DF,
        "char_ngrams": config.TFIDF_USE_CHAR_NGRAMS,
        "vocabulary_size": int(X_train.shape[1]),
        "train_size": len(train), "val_size": len(val), "test_size": len(test),
        "random_seed": config.RANDOM_SEED,
        "training_time_seconds": elapsed,
        "test_macro_f1": test_metrics["f1_macro"],
    }
    (config.BASELINE_MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out = save_model_results("tfidf_logreg", "TF-IDF + Logistic Regression", "TF-IDF", test_metrics, val_metrics,
                             extra={"hyperparameters": meta, "split_info": info})
    print(f"Saved model -> {config.BASELINE_MODEL_DIR.relative_to(config.PROJECT_ROOT)}")
    print(f"Saved metrics -> {out.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
