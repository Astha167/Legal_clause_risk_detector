"""
STEP 3 - Project-defined risk annotation layer.

CUAD does NOT contain Low / Medium / High risk labels. This script creates
*screening* labels with the deterministic, text-based rule engine in
``services/risk_rules.py``. The clause category is never an input to the
score, so the same category can receive different labels depending on its
actual wording.

Usage (from the project root):
    python training/annotate_risk.py

Output:
    data/risk_annotated_clauses.csv
    data/analysis/risk_distribution.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from services.risk_rules import annotate  # noqa: E402

OUTPUT_COLUMNS = [
    "contract_id", "document_name", "clause_type", "clause_text", "answer_value",
    "risk_label", "annotation_score", "risk_factors",
]


def main() -> int:
    if not config.EXTRACTED_CLAUSES_PATH.exists():
        print("ERROR: data/extracted_clauses.csv not found. Run: python training/prepare_dataset.py")
        return 1
    df = pd.read_csv(config.EXTRACTED_CLAUSES_PATH, keep_default_na=False)
    required = {"contract_id", "clause_type", "clause_text"}
    if not required.issubset(df.columns):
        print(f"ERROR: extracted_clauses.csv is missing columns {required - set(df.columns)}")
        return 1

    results = df["clause_text"].map(annotate)
    df["risk_label"] = results.map(lambda r: r["risk_label"])
    df["annotation_score"] = results.map(lambda r: r["annotation_score"])
    df["risk_factors"] = results.map(lambda r: "; ".join(r["risk_factor_names"]))
    df[OUTPUT_COLUMNS].to_csv(config.RISK_ANNOTATED_PATH, index=False, encoding="utf-8")

    dist = df["risk_label"].value_counts().reindex(config.LABELS, fill_value=0)
    print("Risk distribution (project-defined screening labels):")
    for label in config.LABELS:
        print(f"  {label:<6}: {dist[label]:>5}  ({dist[label] / len(df) * 100:.1f}%)")
    ratio = dist.max() / max(1, dist.min())
    imbalance_note = (
        f"Largest/smallest class ratio = {ratio:.2f}. "
        + ("Classes are imbalanced; class-weighted loss is used during training." if ratio > 1.5
           else "Classes are reasonably balanced.")
    )
    print(imbalance_note)

    per_cat = pd.crosstab(df["clause_type"], df["risk_label"]).reindex(columns=config.LABELS, fill_value=0)
    n_mixed = int((per_cat.gt(0).sum(axis=1) > 1).sum())
    print(f"\nCategories whose instances receive more than one risk label: {n_mixed}/{len(per_cat)}")
    print("(Confirms labels depend on clause wording, not on a fixed category mapping.)")
    with pd.option_context("display.max_rows", 100, "display.width", 120):
        print(per_cat.to_string())

    factor_counts = (
        df["risk_factors"].str.split("; ").explode().replace("", pd.NA).dropna().value_counts()
    )
    summary = {
        "n_clauses": int(len(df)),
        "distribution": {k: int(v) for k, v in dist.items()},
        "imbalance_note": imbalance_note,
        "thresholds": {"low_max": config.ANNOTATION_LOW_MAX, "high_min": config.ANNOTATION_HIGH_MIN},
        "categories_with_mixed_labels": n_mixed,
        "per_category": per_cat.to_dict(orient="index"),
        "factor_frequency": {k: int(v) for k, v in factor_counts.items()},
        "score_stats": df["annotation_score"].describe().round(2).to_dict(),
    }
    config.DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_REPORTS_DIR / "risk_distribution.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {config.RISK_ANNOTATED_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
