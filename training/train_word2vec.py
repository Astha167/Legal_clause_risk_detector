"""
STEP 6 (optional) - Word2Vec + Logistic Regression experiment.

1. Analyse the TRAIN corpus (size, vocabulary, frequency distribution) to
   decide whether a Word2Vec experiment is meaningful.
2. If it is, train gensim Word2Vec on TRAIN clauses only (never on val/test).
3. Represent each clause as the mean of its word vectors.
4. Train a class-weighted Logistic Regression (C chosen on validation).
5. Report test metrics next to the TF-IDF baseline.

Word2Vec is an experiment only. The application's main risk prediction uses
Contracts-BERT.

Usage (from the project root):
    python training/train_word2vec.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from training.common import (  # noqa: E402
    SplitError,
    compute_metrics,
    load_split_frames,
    print_metrics,
    save_model_results,
)

TOKEN_RE = re.compile(r"[a-z][a-z'\-]*|\d+")
MIN_TOTAL_TOKENS = 100_000
MIN_VOCAB = 3_000


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def corpus_stats(token_lists):
    counts = Counter(t for toks in token_lists for t in toks)
    total = sum(counts.values())
    low = sum(1 for c in counts.values() if c < config.W2V_MIN_COUNT)
    return {
        "documents": len(token_lists),
        "total_tokens": int(total),
        "unique_tokens": len(counts),
        "vocab_at_min_count": int(sum(1 for c in counts.values() if c >= config.W2V_MIN_COUNT)),
        "low_frequency_tokens": int(low),
        "low_frequency_ratio": round(low / max(1, len(counts)), 3),
        "avg_tokens_per_clause": round(total / max(1, len(token_lists)), 1),
        "top_20": counts.most_common(20),
    }


def doc_vectors(model, token_lists):
    dim = model.vector_size
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        vecs = [model.wv[t] for t in toks if t in model.wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def main() -> int:
    config.set_seed(config.RANDOM_SEED)
    try:
        frames, info = load_split_frames()
    except SplitError as exc:
        print(f"ERROR: {exc}")
        return 1
    tok = {k: [tokenize(t) for t in v["clause_text"]] for k, v in frames.items()}

    stats = corpus_stats(tok["train"])
    print("TRAIN corpus characteristics")
    for k, v in stats.items():
        if k != "top_20":
            print(f"  {k:<24}: {v}")
    suitable = stats["total_tokens"] >= MIN_TOTAL_TOKENS and stats["vocab_at_min_count"] >= MIN_VOCAB
    reason = (
        f"total_tokens={stats['total_tokens']} (>= {MIN_TOTAL_TOKENS}) and "
        f"vocab_at_min_count={stats['vocab_at_min_count']} (>= {MIN_VOCAB})"
    )
    print(f"Suitable for a Word2Vec experiment: {suitable} ({reason})")
    if not suitable:
        out = config.EVALUATION_DIR / "word2vec_metrics.json"
        config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": "Word2Vec + Logistic Regression", "status": "Not used",
                                   "reason": "insufficient benefit / unsuitable corpus characteristics",
                                   "corpus": stats}, indent=2), encoding="utf-8")
        print("Word2Vec skipped - result documented in reports/model_evaluation/word2vec_metrics.json")
        return 0

    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("gensim is not installed, so the optional Word2Vec experiment was skipped.\n"
              "This does not affect the app, the TF-IDF baseline or Contracts-BERT.\n"
              "gensim has no pre-built wheel for very new Python versions on Windows; to run this experiment use\n"
              "Python 3.10-3.12 (pip install gensim) or install Microsoft C++ Build Tools.\n"
              "The previously measured Word2Vec results in reports/model_evaluation/ are kept unchanged.")
        return 0
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    start = time.time()
    try:
        w2v = Word2Vec(
            sentences=tok["train"],             # TRAIN ONLY
            vector_size=config.W2V_VECTOR_SIZE,
            window=config.W2V_WINDOW,
            min_count=config.W2V_MIN_COUNT,
            workers=config.W2V_WORKERS,
            epochs=config.W2V_EPOCHS,
            sg=1,
            seed=config.RANDOM_SEED,
        )
    except Exception as exc:
        print(f"ERROR: Word2Vec training failed: {exc}")
        return 1

    X = {k: doc_vectors(w2v, v) for k, v in tok.items()}
    scaler = StandardScaler().fit(X["train"])
    X = {k: scaler.transform(v) for k, v in X.items()}
    y = {k: frames[k]["label"].to_numpy() for k in frames}

    best = None
    for c in [0.1, 0.5, 1.0, 4.0]:
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=config.RANDOM_SEED)
        clf.fit(X["train"], y["train"])
        m = compute_metrics(y["val"], clf.predict(X["val"]))
        print(f"  C={c:<4} validation macro F1 = {m['f1_macro']:.4f}")
        if best is None or m["f1_macro"] > best[1]["f1_macro"]:
            best = (c, m, clf)
    best_c, val_metrics, clf = best
    test_metrics = compute_metrics(y["test"], clf.predict(X["test"]))
    elapsed = round(time.time() - start, 1)
    print_metrics("TEST RESULTS - Word2Vec + Logistic Regression", test_metrics)
    print(test_metrics["classification_report_text"])

    # Probe the embedding space for a few legal terms (qualitative sanity check).
    probes = {}
    for term in ["terminate", "liability", "exclusive", "compete", "royalty"]:
        if term in w2v.wv:
            probes[term] = [w for w, _ in w2v.wv.most_similar(term, topn=5)]
    print("Nearest neighbours:", probes)

    baseline_path = config.EVALUATION_DIR / "tfidf_logreg_metrics.json"
    comparison_note = ""
    if baseline_path.exists():
        base_f1 = json.loads(baseline_path.read_text())["test"]["f1_macro"]
        comparison_note = (
            f"Word2Vec macro F1 {test_metrics['f1_macro']:.4f} vs TF-IDF macro F1 {base_f1:.4f} on the same test split."
        )
        print(comparison_note)

    config.WORD2VEC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    w2v.save(str(config.WORD2VEC_MODEL_DIR / "word2vec.model"))
    import joblib

    joblib.dump({"scaler": scaler, "classifier": clf}, config.WORD2VEC_MODEL_DIR / "w2v_logreg.joblib")
    decision = (
        "Word2Vec was evaluated as an optional classical embedding approach but was not selected as the "
        "production representation because averaged static word vectors discard word order and negation "
        "scope, and Contracts-BERT is the designated primary model. " + comparison_note
    )
    save_model_results(
        "word2vec_logreg", "Word2Vec + Logistic Regression", "Word2Vec (mean-pooled)", test_metrics, val_metrics,
        extra={"status": "Implemented (experimental)", "corpus": stats, "C": best_c,
               "hyperparameters": {"vector_size": config.W2V_VECTOR_SIZE, "window": config.W2V_WINDOW,
                                   "min_count": config.W2V_MIN_COUNT, "epochs": config.W2V_EPOCHS, "sg": 1,
                                   "seed": config.RANDOM_SEED},
               "training_time_seconds": elapsed, "nearest_neighbours": probes, "decision": decision},
    )
    print("\n" + decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
