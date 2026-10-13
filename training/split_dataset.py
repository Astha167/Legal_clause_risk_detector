"""
STEP 4 - Reproducible contract-level train / validation / test split.

All clauses of a contract go to exactly one split, so correlated clauses from
the same agreement can never appear in both training and evaluation data.

Method: StratifiedGroupKFold (groups = contract_id, y = clause risk label)
with 20 folds and a fixed seed. 3 folds (~15%) form the test set, 3 folds
(~15%) the validation set and the remaining 14 folds (~70%) the training set.
Stratification is approximate because whole contracts are moved together.

Usage (from the project root):
    python training/split_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from training.common import SplitError, load_split_frames  # noqa: E402

N_FOLDS = 20


def main() -> int:
    if not config.RISK_ANNOTATED_PATH.exists():
        print("ERROR: run training/annotate_risk.py first.")
        return 1
    df = pd.read_csv(config.RISK_ANNOTATED_PATH, keep_default_na=False)
    if df["contract_id"].nunique() < N_FOLDS:
        print("ERROR: insufficient contracts for a contract-level split.")
        return 1
    config.set_seed(config.RANDOM_SEED)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    fold_of_contract = {}
    for fold, (_, idx) in enumerate(sgkf.split(df, df["risk_label"], groups=df["contract_id"])):
        for cid in df.iloc[idx]["contract_id"].unique():
            fold_of_contract[cid] = fold
    n_test = round(N_FOLDS * config.TEST_RATIO)
    n_val = round(N_FOLDS * config.VAL_RATIO)

    def split_for(fold: int) -> str:
        if fold < n_test:
            return "test"
        if fold < n_test + n_val:
            return "val"
        return "train"

    # Contracts without any clause instance still get a (train) assignment for completeness.
    all_contracts = pd.read_csv(config.CONTRACT_INDEX_PATH)["contract_id"] if config.CONTRACT_INDEX_PATH.exists() \
        else df["contract_id"].drop_duplicates()
    rows = [{"contract_id": c, "fold": fold_of_contract.get(c, -1),
             "split": split_for(fold_of_contract[c]) if c in fold_of_contract else "train"} for c in all_contracts]
    assign = pd.DataFrame(rows)
    assign.to_csv(config.SPLIT_ASSIGNMENTS_PATH, index=False)

    try:
        frames, info = load_split_frames()
    except SplitError as exc:
        print(f"ERROR: {exc}")
        return 1

    summary = {"seed": config.RANDOM_SEED, "n_folds": N_FOLDS, "method": "StratifiedGroupKFold by contract_id",
               "contracts_per_split": assign["split"].value_counts().to_dict(), **info, "label_distribution": {}}
    print("Contract-level split (seed = %d)" % config.RANDOM_SEED)
    print(f"{'split':<6} {'contracts':>9} {'clauses':>8}   " + "  ".join(f"{l:>7}" for l in config.LABELS))
    for name, part in frames.items():
        dist = part["risk_label"].value_counts().reindex(config.LABELS, fill_value=0)
        summary["label_distribution"][name] = {k: int(v) for k, v in dist.items()}
        pct = "  ".join(f"{dist[l] / len(part) * 100:6.1f}%" for l in config.LABELS)
        print(f"{name:<6} {part['contract_id'].nunique():>9} {len(part):>8}   {pct}")
    print(f"Within-split duplicate texts collapsed : {info['within_split_duplicates_removed']}")
    print(f"Cross-split duplicates removed (val/test texts also in train): {info['cross_split_duplicates_removed']}")
    print(f"Contract overlap between splits        : {info['contract_overlap']}  (must be all zero)")
    config.DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_REPORTS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {config.SPLIT_ASSIGNMENTS_PATH.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
