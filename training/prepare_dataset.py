"""
STEP 2 - Convert contract-level CUAD rows into clause-level instances.

Usage (from the project root):
    python training/prepare_dataset.py

Rules implemented (see README "Dataset Preparation"):
* metadata categories (Filename, Document Name, Parties, dates) are excluded;
* one instance is produced per extracted CUAD span of a clause category;
* ``clause_text`` is always the legal span, ``answer_value`` the CUAD answer
  (Yes / Nevada / 30 days / ...), stored separately;
* no rows are created for empty lists, empty text, or ``No`` answers;
* exact duplicates of (contract, category, text) are removed.

Outputs:
    data/extracted_clauses.csv
    data/contract_index.csv
    data/analysis/data_quality.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from config import get_logger  # noqa: E402
from training.dataset_utils import (  # noqa: E402
    DatasetError,
    clean_answer,
    clean_clause_text,
    detect_structure,
    load_raw_dataset,
    parse_context,
)

log = get_logger("prepare_dataset")
OUTPUT_COLUMNS = ["contract_id", "document_name", "clause_type", "clause_text", "answer_value"]


def extract_clauses(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    structure = detect_structure(df)
    id_col = structure.identifier_columns[0] if structure.identifier_columns else None
    doc_col = structure.document_name_column

    contracts = []
    records = []
    skipped = {"empty_list": 0, "no_answer_with_context": 0, "empty_text_after_cleaning": 0}
    for idx, row in df.reset_index(drop=True).iterrows():
        contract_id = f"C{idx + 1:04d}"
        filename = str(row[id_col]) if id_col else ""
        doc_name = clean_answer(row[doc_col]) if doc_col else ""
        if not doc_name:
            doc_name = Path(filename).stem if filename else contract_id
        contracts.append({"contract_id": contract_id, "document_name": doc_name, "source_filename": filename})

        for pair in structure.clause_pairs:
            spans = parse_context(row[pair.context_column])
            answer = clean_answer(row[pair.answer_column])
            if not spans:
                skipped["empty_list"] += 1
                continue
            if answer.lower() == "no":
                # An explicit "No" means the category is absent; never create a clause from it.
                skipped["no_answer_with_context"] += 1
                continue
            for span in spans:
                text = clean_clause_text(span)
                if not text:
                    skipped["empty_text_after_cleaning"] += 1
                    continue
                records.append(
                    {
                        "contract_id": contract_id,
                        "document_name": doc_name,
                        "clause_type": pair.category,
                        "clause_text": text,
                        "answer_value": answer,
                    }
                )

    clauses = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    before = len(clauses)
    clauses = clauses.drop_duplicates(subset=["contract_id", "clause_type", "clause_text"]).reset_index(drop=True)
    skipped["exact_duplicates_removed"] = before - len(clauses)
    return clauses, pd.DataFrame(contracts), skipped


def near_duplicate_pairs(texts: list[str], contract_ids: list[str], threshold: float = 0.95) -> dict:
    """Count near-duplicate clause pairs across *different* contracts (TF-IDF cosine)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    unique = pd.DataFrame({"t": texts, "c": contract_ids}).drop_duplicates("t").reset_index(drop=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(unique["t"])
    cids = unique["c"].to_numpy()
    n_pairs = 0
    chunk = 500
    for start in range(0, X.shape[0], chunk):
        sims = (X[start:start + chunk] @ X.T).tocoo()
        for i, j, v in zip(sims.row, sims.col, sims.data):
            gi = start + i
            if j > gi and v >= threshold and cids[gi] != cids[j]:
                n_pairs += 1
    return {"threshold": threshold, "near_duplicate_pairs_across_contracts": int(n_pairs),
            "unique_texts_compared": int(X.shape[0])}


def quality_report(clauses: pd.DataFrame) -> dict:
    chars = clauses["clause_text"].str.len()
    words = clauses["clause_text"].str.split().str.len()
    try:
        from services.nlp_processor import get_nlp

        nlp = get_nlp()
        tokens = pd.Series([len(d) for d in nlp.tokenizer.pipe(clauses["clause_text"].tolist())])
        token_source = "spaCy tokenizer"
    except Exception as exc:  # spaCy optional here
        log.warning("spaCy unavailable for token counts: %s", exc)
        tokens = words
        token_source = "whitespace (spaCy unavailable)"

    q1, q3 = words.quantile(0.25), words.quantile(0.75)
    iqr = q3 - q1
    short_thr = max(3, int(q1 - 1.5 * iqr))
    long_thr = int(q3 + 1.5 * iqr)

    dup_text = clauses.duplicated("clause_text", keep=False)
    by_text = clauses.groupby("clause_text")["contract_id"].nunique()
    cross_contract = by_text[by_text > 1]

    report = {
        "n_clause_instances": int(len(clauses)),
        "n_contracts_with_clauses": int(clauses["contract_id"].nunique()),
        "n_categories": int(clauses["clause_type"].nunique()),
        "length": {
            "chars": {"mean": round(float(chars.mean()), 1), "min": int(chars.min()), "max": int(chars.max()),
                      "median": float(chars.median())},
            "words": {"mean": round(float(words.mean()), 1), "min": int(words.min()), "max": int(words.max()),
                      "median": float(words.median())},
            "tokens": {"mean": round(float(tokens.mean()), 1), "max": int(tokens.max()), "source": token_source},
            "short_threshold_words": short_thr,
            "long_threshold_words": long_thr,
            "n_unusually_short": int((words < short_thr).sum()),
            "n_unusually_long": int((words > long_thr).sum()),
            "note": "Long clauses are kept; the transformer truncates at MAX_LENGTH tokens.",
        },
        "duplicates": {
            "rows_sharing_text_with_another_row": int(dup_text.sum()),
            "texts_appearing_in_multiple_contracts": int(len(cross_contract)),
            "leakage_note": (
                "Identical boilerplate can appear in several contracts. The contract-level split "
                "prevents same-contract leakage; cross-contract exact duplicates that land in "
                "validation/test are flagged in split_assignments and excluded from evaluation."
            ),
        },
        "category_distribution": clauses["clause_type"].value_counts().to_dict(),
    }
    try:
        report["duplicates"]["near_duplicates"] = near_duplicate_pairs(
            clauses["clause_text"].tolist(), clauses["contract_id"].tolist()
        )
    except Exception as exc:
        log.warning("Near-duplicate analysis skipped: %s", exc)
        report["duplicates"]["near_duplicates"] = f"skipped: {exc}"
    return report


def main() -> int:
    try:
        df = load_raw_dataset()
        clauses, contracts, skipped = extract_clauses(df)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    if len(clauses) < 50:
        print("ERROR: insufficient data - fewer than 50 clause instances were extracted.")
        return 1

    # Sanity checks required by the specification.
    assert not clauses["clause_text"].str.fullmatch(r"(?i)\s*(yes|no)\s*").any(), "Yes/No leaked into clause_text"
    assert (clauses["clause_text"].str.len() > 0).all()

    clauses.to_csv(config.EXTRACTED_CLAUSES_PATH, index=False, encoding="utf-8")
    contracts.to_csv(config.CONTRACT_INDEX_PATH, index=False, encoding="utf-8")

    report = quality_report(clauses)
    report["skipped"] = skipped
    config.DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Contracts read                 : {len(contracts)}")
    print(f"Clause instances extracted     : {len(clauses)}")
    print(f"Clause categories              : {clauses['clause_type'].nunique()}")
    print(f"Skipped (empty lists / No)     : {skipped['empty_list']} / {skipped['no_answer_with_context']}")
    print(f"Exact duplicates removed       : {skipped['exact_duplicates_removed']}")
    print(f"Avg words / clause             : {report['length']['words']['mean']}")
    print(f"Avg tokens / clause            : {report['length']['tokens']['mean']} ({report['length']['tokens']['source']})")
    print(f"Unusually short / long clauses : {report['length']['n_unusually_short']} / {report['length']['n_unusually_long']}")
    print(f"Texts in multiple contracts    : {report['duplicates']['texts_appearing_in_multiple_contracts']}")
    nd = report["duplicates"]["near_duplicates"]
    if isinstance(nd, dict):
        print(f"Near-duplicate pairs (>=0.95)  : {nd['near_duplicate_pairs_across_contracts']}")
    print("\nAnswer value examples (kept separate from clause_text):")
    vals = clauses[~clauses["answer_value"].isin(["Yes", ""])]
    print(vals[["clause_type", "answer_value"]].drop_duplicates().head(8).to_string(index=False))
    print(f"\nSaved -> {config.EXTRACTED_CLAUSES_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
