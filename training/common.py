"""
Shared training utilities: loading the fixed contract-level split, metrics,
and non-destructive saving of evaluation results.

Every model (TF-IDF, Word2Vec, Contracts-BERT) loads its data through
``load_split_frames`` so all models see exactly the same partitions.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


class SplitError(Exception):
    pass


def load_split_frames() -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """Return {'train','val','test'} DataFrames plus leakage-control info.

    * Contracts are assigned to exactly one split (split_assignments.csv).
    * Within each split, exact duplicate clause texts are collapsed (the same
      span is often annotated under several CUAD categories).
    * Validation/test rows whose exact text also occurs in train (cross-contract
      boilerplate) are removed so evaluation measures generalisation.
    """
    if not config.RISK_ANNOTATED_PATH.exists():
        raise SplitError("data/risk_annotated_clauses.csv not found. Run training/annotate_risk.py first.")
    if not config.SPLIT_ASSIGNMENTS_PATH.exists():
        raise SplitError("data/split_assignments.csv not found. Run training/split_dataset.py first.")
    df = pd.read_csv(config.RISK_ANNOTATED_PATH, keep_default_na=False)
    splits = pd.read_csv(config.SPLIT_ASSIGNMENTS_PATH)
    df = df.merge(splits[["contract_id", "split"]], on="contract_id", how="left")
    if df["split"].isna().any():
        raise SplitError("Some contracts have no split assignment. Re-run training/split_dataset.py.")
    df = df[df["risk_label"].isin(config.LABELS)].copy()
    df["label"] = df["risk_label"].map(config.LABEL2ID).astype(int)

    frames = {}
    info = {"within_split_duplicates_removed": {}, "cross_split_duplicates_removed": {}}
    for name in ["train", "val", "test"]:
        part = df[df["split"] == name]
        before = len(part)
        part = part.drop_duplicates("clause_text").reset_index(drop=True)
        info["within_split_duplicates_removed"][name] = before - len(part)
        frames[name] = part
    train_texts = set(frames["train"]["clause_text"])
    for name in ["val", "test"]:
        before = len(frames[name])
        frames[name] = frames[name][~frames[name]["clause_text"].isin(train_texts)].reset_index(drop=True)
        info["cross_split_duplicates_removed"][name] = before - len(frames[name])

    overlap = {
        "train_val": len(set(frames["train"].contract_id) & set(frames["val"].contract_id)),
        "train_test": len(set(frames["train"].contract_id) & set(frames["test"].contract_id)),
        "val_test": len(set(frames["val"].contract_id) & set(frames["test"].contract_id)),
    }
    if any(overlap.values()):
        raise SplitError(f"Contract leakage detected between splits: {overlap}")
    info["contract_overlap"] = overlap
    info["sizes"] = {k: int(len(v)) for k, v in frames.items()}
    for name, part in frames.items():
        if part["label"].nunique() < 2:
            raise SplitError(f"Insufficient data: split '{name}' contains fewer than two risk classes.")
    return frames, info


def compute_metrics(y_true, y_pred) -> Dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = list(range(len(config.LABELS)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=config.LABELS, zero_division=0, output_dict=True
        ),
        "classification_report_text": classification_report(
            y_true, y_pred, labels=labels, target_names=config.LABELS, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def archive_existing(paths) -> None:
    """Never silently overwrite previous results: move them into an archive folder."""
    existing = [Path(p) for p in paths if Path(p).exists()]
    if not existing:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = config.EVALUATION_DIR / "archive" / stamp
    archive.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.move(str(p), archive / p.name)
    print(f"WARNING: previous results were moved to {archive.relative_to(config.PROJECT_ROOT)}")


def save_confusion_matrix_png(cm, title: str, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.asarray(cm)
        fig, ax = plt.subplots(figsize=(4.6, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(config.LABELS)), config.LABELS)
        ax.set_yticks(range(len(config.LABELS)), config.LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title, fontsize=10)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    except Exception as exc:  # plotting is optional
        print(f"(confusion-matrix image skipped: {exc})")


def save_model_results(prefix: str, model_name: str, representation: str, test_metrics: Dict,
                       val_metrics: Dict | None = None, extra: Dict | None = None) -> Path:
    """Write JSON metrics, text report, confusion matrix CSV/PNG for one model."""
    config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.EVALUATION_DIR / f"{prefix}_metrics.json"
    report_path = config.EVALUATION_DIR / f"{prefix}_classification_report.txt"
    cm_csv = config.EVALUATION_DIR / f"{prefix}_confusion_matrix.csv"
    cm_png = config.EVALUATION_DIR / f"{prefix}_confusion_matrix.png"
    archive_existing([json_path, report_path, cm_csv, cm_png])

    payload = {
        "model": model_name,
        "representation": representation,
        "evaluated_on": "test split (contract-level, untouched during training)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "test": {k: v for k, v in test_metrics.items() if k != "classification_report_text"},
    }
    if val_metrics:
        payload["validation"] = {k: v for k, v in val_metrics.items() if k != "classification_report_text"}
    if extra:
        payload.update(extra)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(f"{model_name} - test split\n\n{test_metrics['classification_report_text']}", encoding="utf-8")
    pd.DataFrame(test_metrics["confusion_matrix"], index=[f"actual_{l}" for l in config.LABELS],
                 columns=[f"pred_{l}" for l in config.LABELS]).to_csv(cm_csv)
    save_confusion_matrix_png(test_metrics["confusion_matrix"], f"{model_name} (test)", cm_png)
    return json_path


def print_metrics(title: str, m: Dict) -> None:
    print(f"\n{title}")
    print(f"  Accuracy     : {m['accuracy']:.4f}")
    print(f"  Precision(M) : {m['precision_macro']:.4f}")
    print(f"  Recall(M)    : {m['recall_macro']:.4f}")
    print(f"  Macro F1     : {m['f1_macro']:.4f}")
    print(f"  Weighted F1  : {m['f1_weighted']:.4f}")
