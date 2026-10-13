# Data

| File | Created by | Description |
|---|---|---|
| `master_clauses.csv` | provided | CUAD master clauses file (510 contracts x 83 columns). The only dataset used. |
| `extracted_clauses.csv` | `training/prepare_dataset.py` | 8,683 clause instances: `contract_id, document_name, clause_type, clause_text, answer_value` |
| `contract_index.csv` | `training/prepare_dataset.py` | `contract_id` -> document name and original CUAD filename |
| `risk_annotated_clauses.csv` | `training/annotate_risk.py` | adds `risk_label, annotation_score, risk_factors` (project-defined screening labels) |
| `split_assignments.csv` | `training/split_dataset.py` | contract-level train / val / test assignment (seed 42) shared by all models |
| `analysis/*.json` | the scripts above | inspection, data-quality, risk-distribution and split summaries |

Important:

* `Yes` / `No` in CUAD answer columns are annotation answers, never clause text. No rows are created for `No` or for empty lists `[]`.
* `answer_value` keeps CUAD answers such as `Nevada`, `30 days` or `successive 1 year`, separately from `clause_text`.
* CUAD has no Low/Medium/High labels; `risk_label` comes from the deterministic rule engine in `services/risk_rules.py` and is a screening label, not legal ground truth.
* Metadata categories (Filename, Document Name, Parties, Agreement/Effective/Expiration Date) are excluded from training text.
