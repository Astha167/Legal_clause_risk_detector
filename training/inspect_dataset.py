"""
STEP 1 - Inspect the actual master_clauses.csv before any modelling.

Usage (from the project root):
    python training/inspect_dataset.py

Writes a machine-readable summary to data/analysis/dataset_inspection.json.
Nothing about the structure is hard-coded: column pairs, metadata, answer
types and categories are detected from the file itself.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from training.dataset_utils import (  # noqa: E402
    DatasetError,
    clean_answer,
    clean_clause_text,
    detect_structure,
    load_raw_dataset,
    parse_context,
)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    try:
        df = load_raw_dataset()
        structure = detect_structure(df)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    hr("BASIC SHAPE")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    hr("EXACT COLUMN NAMES, DATA TYPES AND MISSING VALUES")
    missing = df.isna().sum()
    info = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": missing,
            "missing_pct": (missing / len(df) * 100).round(2),
        }
    )
    with pd.option_context("display.max_rows", 200, "display.width", 140):
        print(info.to_string())

    hr("SAMPLE ROWS (first 2 rows, first 8 columns, truncated)")
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(df.iloc[:2, :8].to_string())

    hr("DETECTED STRUCTURE")
    print(f"Identifier columns : {structure.identifier_columns}")
    print(f"Paired categories  : {len(structure.pairs)}")
    print(f"Metadata categories: {[p.category for p in structure.metadata_pairs]}")
    print(f"Clause categories  : {len(structure.clause_pairs)}")
    print(f"Unpaired columns   : {structure.unpaired_columns}")
    irregular = [
        p.answer_column
        for p in structure.pairs
        if p.answer_column != f"{p.context_column}-Answer"
    ]
    print(f"Irregular answer-column names detected: {irregular}")

    hr("PER-CATEGORY ANNOTATION STATISTICS (clause categories only)")
    rows = []
    all_spans = []
    for p in structure.clause_pairs:
        ctx = df[p.context_column].map(parse_context)
        ans = df[p.answer_column].map(clean_answer)
        has_ctx = ctx.map(len) > 0
        spans = [clean_clause_text(s) for lst in ctx for s in lst]
        spans = [s for s in spans if s]
        all_spans.extend(spans)
        rows.append(
            {
                "category": p.category,
                "answer_type": p.answer_type,
                "contracts_with_clause": int(has_ctx.sum()),
                "extracted_spans": len(spans),
                "yes_answers": int((ans.str.lower() == "yes").sum()),
                "no_answers": int((ans.str.lower() == "no").sum()),
                "value_answers": int(((ans != "") & ~ans.str.lower().isin(["yes", "no"])).sum()),
                "empty_answers": int((ans == "").sum()),
                "empty_context_lists": int((~has_ctx).sum()),
            }
        )
    stats = pd.DataFrame(rows).sort_values("extracted_spans", ascending=False)
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(stats.to_string(index=False))

    hr("ANSWER TOTALS")
    print(f"Yes annotations   : {int(stats['yes_answers'].sum())}")
    print(f"No annotations    : {int(stats['no_answers'].sum())}")
    print(f"Value annotations : {int(stats['value_answers'].sum())} (e.g. jurisdictions, durations)")
    print(f"Empty annotations : {int(stats['empty_answers'].sum())}")

    hr("CLAUSE SPAN LENGTHS AND DUPLICATES")
    lengths = pd.Series([len(s) for s in all_spans])
    words = pd.Series([len(s.split()) for s in all_spans])
    span_series = pd.Series(all_spans)
    dup_count = int(span_series.duplicated().sum())
    print(f"Total non-empty spans : {len(all_spans)}")
    print(f"Unique spans          : {span_series.nunique()}")
    print(f"Exact duplicate spans : {dup_count}")
    print(f"Avg length (chars)    : {lengths.mean():.1f}")
    print(f"Min length (chars)    : {lengths.min()}")
    print(f"Max length (chars)    : {lengths.max()}")
    print(f"Avg words             : {words.mean():.1f}   median: {words.median():.0f}")

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "identifier_columns": structure.identifier_columns,
        "metadata_categories": [p.category for p in structure.metadata_pairs],
        "clause_categories": [p.category for p in structure.clause_pairs],
        "irregular_answer_columns": irregular,
        "answer_types": {p.category: p.answer_type for p in structure.clause_pairs},
        "category_stats": stats.to_dict(orient="records"),
        "totals": {
            "yes": int(stats["yes_answers"].sum()),
            "no": int(stats["no_answers"].sum()),
            "value": int(stats["value_answers"].sum()),
            "empty": int(stats["empty_answers"].sum()),
            "spans": len(all_spans),
            "unique_spans": int(span_series.nunique()),
            "duplicate_spans": dup_count,
            "avg_chars": round(float(lengths.mean()), 1),
            "min_chars": int(lengths.min()),
            "max_chars": int(lengths.max()),
            "avg_words": round(float(words.mean()), 1),
        },
    }
    config.DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.DATA_REPORTS_DIR / "dataset_inspection.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved inspection summary -> {out.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
