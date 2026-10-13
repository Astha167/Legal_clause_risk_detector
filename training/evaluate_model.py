"""
STEP 8 - Evaluate Contracts-BERT on the untouched TEST split and build the
model comparison table (TF-IDF, Word2Vec, Contracts-BERT).

Only measured results are reported. Models that were not trained are shown
as such - no numbers are invented.

Usage (from the project root):
    python training/evaluate_model.py                     # evaluate BERT + comparison
    python training/evaluate_model.py --compare-only      # comparison table only
    python training/evaluate_model.py --model-dir model/_smoke_test   # pipeline check only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from training.common import (  # noqa: E402
    SplitError,
    compute_metrics,
    load_split_frames,
    print_metrics,
    save_confusion_matrix_png,
    save_model_results,
)

COMPARISON_ROWS = [
    ("tfidf_logreg", "TF-IDF + Logistic Regression", "TF-IDF"),
    ("word2vec_logreg", "Word2Vec + Logistic Regression", "Word2Vec"),
    ("contracts_bert", "Contracts-BERT", "contextual embeddings"),
]


def evaluate_bert(model_dir: Path) -> int:
    from services.predictor import BertRiskPredictor, ModelLoadError, ModelNotFoundError

    try:
        predictor = BertRiskPredictor(model_dir)
    except (ModelNotFoundError, ModelLoadError) as exc:
        print(f"Contracts-BERT not evaluated: {exc}")
        return 1
    try:
        frames, info = load_split_frames()
    except SplitError as exc:
        print(f"ERROR: {exc}")
        return 1
    test = frames["test"]
    config.get_device()
    print(f"Evaluating {model_dir} on {len(test)} test clauses ...")
    probs = predictor.predict_proba(test["clause_text"].tolist())
    preds = probs.argmax(axis=1)
    metrics = compute_metrics(test["label"].to_numpy(), preds)
    print_metrics("TEST RESULTS - Contracts-BERT", metrics)
    print(metrics["classification_report_text"])

    meta_path = model_dir / "training_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if predictor.smoke_test:
        out_dir = config.EVALUATION_DIR / "smoke_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "smoke_test_metrics.json").write_text(json.dumps(
            {"WARNING": "Smoke-test model trained on a tiny subset - NOT a real result.",
             "test": {k: v for k, v in metrics.items() if k != "classification_report_text"}}, indent=2))
        save_confusion_matrix_png(metrics["confusion_matrix"], "Smoke test (not a result)",
                                  out_dir / "smoke_test_confusion_matrix.png")
        print(f"Smoke-test evaluation written to {out_dir.relative_to(config.PROJECT_ROOT)} "
              "(excluded from the model comparison).")
        return 0

    save_model_results("contracts_bert", "Contracts-BERT", "contextual embeddings", metrics,
                       meta.get("best_val_metrics"),
                       extra={"model_dir": str(model_dir.relative_to(config.PROJECT_ROOT)),
                              "training_metadata": {k: v for k, v in meta.items() if k != "split_info"},
                              "split_info": info})
    pred_df = test[["contract_id", "clause_type", "clause_text", "risk_label"]].copy()
    pred_df["predicted"] = [config.ID2LABEL[int(p)] for p in preds]
    for i, lab in enumerate(config.LABELS):
        pred_df[f"p_{lab.lower()}"] = np.round(probs[:, i], 4)
    pred_df.to_csv(config.EVALUATION_DIR / "contracts_bert_test_predictions.csv", index=False)
    return 0


def build_comparison() -> pd.DataFrame:
    rows = []
    for prefix, name, rep in COMPARISON_ROWS:
        path = config.EVALUATION_DIR / f"{prefix}_metrics.json"
        if not path.exists():
            status = ("Not used - insufficient benefit / unsuitable corpus characteristics"
                      if prefix == "word2vec_logreg" else "Not trained yet - run the training script")
            rows.append({"Model": name, "Representation": rep, "Accuracy": status, "Precision": "", "Recall": "",
                         "Macro F1": "", "Weighted F1": ""})
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        if "test" not in d:
            rows.append({"Model": name, "Representation": rep,
                         "Accuracy": d.get("status", "Not used") + " - " + d.get("reason", ""),
                         "Precision": "", "Recall": "", "Macro F1": "", "Weighted F1": ""})
            continue
        t = d["test"]
        rows.append({"Model": name, "Representation": rep, "Accuracy": t["accuracy"],
                     "Precision": t["precision_macro"], "Recall": t["recall_macro"],
                     "Macro F1": t["f1_macro"], "Weighted F1": t["f1_weighted"]})
    df = pd.DataFrame(rows)
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.EVALUATION_DIR / "model_comparison.csv", index=False)
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(v) for v in r.values) + " |")
    (config.EVALUATION_DIR / "model_comparison.md").write_text(
        "Test-split results (precision/recall are macro-averaged).\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(config.BERT_MODEL_DIR))
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    code = 0
    if not args.compare_only:
        code = evaluate_bert(Path(args.model_dir).resolve())
    df = build_comparison()
    print("\nMODEL COMPARISON (test split)")
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(df.to_string(index=False))
    print(f"\nSaved -> {(config.EVALUATION_DIR / 'model_comparison.csv').relative_to(config.PROJECT_ROOT)}")
    return 0 if args.compare_only else code


if __name__ == "__main__":
    raise SystemExit(main())
