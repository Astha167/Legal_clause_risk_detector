# LegalRisk AI - AI-Powered Legal Clause Risk Detector with Explainable Risk Analysis

> Understand contractual risks before you sign.

LegalRisk AI is a local Streamlit application that reads a contract PDF, splits it into clauses, classifies each clause as **Low / Medium / High** screening risk with a fine-tuned **Contracts-BERT** model, and explains *why* in plain language using a rule engine and **SHAP**.

> **Just unzipped?** See [SETUP_GUIDE.md](SETUP_GUIDE.md). The TF-IDF baseline and Demo Mode work immediately; Contracts-BERT must be trained on your computer (`python training/train_model.py`, or `--cpu-fast` on a CPU). Run `python check_setup.py` to see your status.

**LegalRisk AI provides automated contract screening and educational information. It does not provide legal advice or replace a qualified legal professional.**

---

## 1. Project Overview

| Layer | What it does |
|---|---|
| Data pipeline | Turns the CUAD `master_clauses.csv` into 8,683 clause-level instances and adds a project-defined risk annotation layer |
| Models | TF-IDF + Logistic Regression baseline, optional Word2Vec experiment, Contracts-BERT (primary) |
| Explainability | SHAP term importance + rule-based plain-English explanations + document-level observations |
| Application | Streamlit UI: upload, processing screen, dashboard, clause analysis, highlighted document viewer, PDF/CSV reports, Demo Mode |

## 2. Problem Statement

Rental agreements, employment and internship contracts, privacy policies and service contracts are written in complex legal language. Non-experts often sign without noticing clauses that impose unfair obligations, financial liabilities, restrictions or penalties. Most tools only say *risky / not risky*. This project also explains **why**, for example:

* "Makes one party responsible for all costs or obligations of a certain kind."
* "No clearly defined termination notice period was detected."
* "Most detected obligations (5 of 6) fall on 'tenant', which may indicate a one-sided allocation of responsibilities."

(These three lines are real outputs for `assets/sample_rental_agreement.pdf`.)

## 3. Objectives

Extract text from legal PDFs; segment clauses; preprocess with spaCy and NLTK; classify clauses with a BERT-based model; assign Low/Medium/High; highlight risky clauses; generate concise explanations; present everything in a simple interface.

## 4. Features

* PDF upload (text-based PDFs) with clear errors for empty, corrupted, password-protected and image-only files
* Clause segmentation from numbered sections, ARTICLE headings, ALL-CAPS headings and paragraphs, with clause-type identification (unknown types are labelled `Other / Unclassified`)
* spaCy tokenisation, POS tagging, NER, lemmatisation, dependency-based obligation detection ("who must do what"); NLTK tokenisation and bigrams
* Contracts-BERT risk classification with probabilities, confidence and a 0-100 screening score
* SHAP explanations (on demand for BERT, automatic for the baseline) with a safe fallback
* Rule-based explanations and contract-level observations (missing liability cap, missing notice period, one-sided rights, automatic renewal, obligation imbalance)
* Risk dashboard, filterable/searchable clause analysis, colour-highlighted document viewer with navigation
* Professional PDF report and CSV export
* Demo Mode that works without any trained model (clearly labelled)

## 5. System Architecture

```
                    master_clauses.csv
                           |
                  Dataset Preparation  (training/prepare_dataset.py)
                           |
                  Legal Text Cleaning  (training/dataset_utils.py)
                           |
              Project risk annotation  (training/annotate_risk.py -> services/risk_rules.py)
                           |
            Contract-level split 70/15/15  (training/split_dataset.py)
                 |                  |                    |
           TF-IDF + LR        Word2Vec + LR        Contracts-BERT (primary)
                 \__________________|____________________/
                           Model comparison (training/evaluate_model.py)

Streamlit (app.py, pages/, components/)
    -> services/api.py
        -> pdf_extractor (PyMuPDF) -> clause_segmenter -> nlp_processor (spaCy/NLTK)
        -> predictor (Contracts-BERT / baseline) -> explainer (SHAP) -> risk_rules
        -> results -> dashboard / clause analysis / document viewer / reports/report_generator.py
```

`app.py` only configures the page and routes. All logic lives in `services/`.

## 6. Technology Stack

Python, Streamlit, PyTorch, Hugging Face Transformers (`nlpaueb/bert-base-uncased-contracts`), scikit-learn, spaCy (`en_core_web_sm`), NLTK, gensim, SHAP, PyMuPDF, Plotly, ReportLab, pandas, NumPy, matplotlib.

## 7. Dataset

**The project uses the CUAD `master_clauses.csv` file provided as the dataset input** (`data/master_clauses.csv`). It is loaded directly with `pandas.read_csv`. No other dataset is downloaded and the Hugging Face `datasets` package is not used.

* CUAD (Contract Understanding Atticus Dataset) provides expert annotations of clause categories in commercial contracts.
* The CSV contains **paired clause/context and answer fields**, e.g. `Governing Law` / `Governing Law-Answer`.
* **`Yes` and `No` are annotation answers, not clause text.**
* Extracted answer values such as jurisdictions (`Nevada`) and durations (`30 days`, `successive 1 year`) are **stored separately** in `answer_value`.
* **CUAD does not provide Low/Medium/High risk labels.** This project creates a custom risk annotation layer, and those project-defined labels are used for model training.

## 8. Dataset Structure (measured by `training/inspect_dataset.py`)

| Property | Value |
|---|---|
| Rows (contracts) | 510 |
| Columns | 83 |
| Identifier column | `Filename` |
| Detected category pairs | 41 |
| Metadata categories (excluded) | Document Name, Parties, Agreement Date, Effective Date, Expiration Date (+ Filename) |
| Legal clause categories | 36 |
| Irregular answer-column name handled | `Notice Period To Terminate Renewal- Answer` |
| Yes / No / value / empty answers | 3,688 / 13,142 / 698 / 832 |
| Non-empty spans | 8,683 (7,667 unique) |
| Span length (chars) | mean 384.1, min 12, max 4,218 |

Structure is detected dynamically: any column ending in "Answer" (with flexible spacing/hyphenation) is paired with the context column of the same normalised name, and a context column must contain Python-list literals such as `[]` or `['clause text']`. Binary (Yes/No) and value answer types are detected from the data.

## 9. Dataset Preparation (`training/prepare_dataset.py`)

For every clause category and contract:

* empty list `[]` -> no row (13,950 skipped);
* answer `No` -> no row (0 cases had a `No` answer with non-empty context);
* every non-empty span -> one row with `clause_text` = the legal span and `answer_value` = the CUAD answer (`Yes`, `Nevada`, `30 days`, or empty when CUAD left the answer blank);
* exact duplicates of (contract, category, text) removed (0 found).

Result: **8,683 clause instances**, **36 categories**, **475 contracts** containing at least one clause. Outputs: `data/extracted_clauses.csv`, `data/contract_index.csv`.

## 10. Data Cleaning

Unicode normalisation, newline/tab normalisation, collapsing of excessive whitespace, replacement of CUAD `<omitted>` markers with `[...]`, removal of empty clauses and exact duplicates. **Stop-words are not removed** from `clause_text`; legally important words (`not`, `no`, `shall`, `unless`, `except`, `without`, `must`, `may`) are preserved because they change legal meaning.

Data-quality analysis (`data/analysis/data_quality.json`):

* Words per clause: mean 60.9, median 48, max 706; spaCy tokens mean 72.2, max 812.
* IQR outliers: 1 unusually short, 468 unusually long clauses. Long clauses are kept (documented reason: they are genuine CUAD spans); the transformer truncates at `MAX_LENGTH` tokens.
* 74 clause texts appear in more than one contract (boilerplate); 24 near-duplicate pairs across contracts (TF-IDF cosine >= 0.95). Leakage handling is described in section 13.

## 11. NLP Preprocessing (`services/nlp_processor.py`)

spaCy performs tokenisation, POS tagging, NER, lemmatisation, noun-chunk extraction and dependency parsing (subjects of `shall/must/will/may` give the obligated party and whether it is negated). NLTK provides offline Treebank tokenisation and bigram frequency analysis. Two views are kept:

* `clean_text` - original cleaned legal text, **the only text given to Contracts-BERT** and shown to users;
* `processed_text` - lemmatised, non-legal stop-words removed, negations and modal verbs kept; used for linguistic analysis, keywords, explanations and classical experiments.

If `en_core_web_sm` is missing, a blank English pipeline is used and the UI states that POS/NER/lemmas are unavailable.

## 12. Risk Annotation (`training/annotate_risk.py`, `services/risk_rules.py`)

A deterministic, reproducible, **text-only** rule engine (the clause category is never an input). 45 indicators in groups: liability, financial, restriction, termination, duration, IP, obligation, protective (negative weights) and informational (weight 0), plus duration and amount adjustments.

```
annotation_score = max(0, sum(indicator weights) + duration/amount adjustments)
Low    : score <= 1
Medium : 2 <= score <= 3
High   : score >= 4
```

* Weight 3: uncapped liability, exception to liability limits, liquidated damages, non-compete
* Weight 2: penalty, cancellation fee/forfeiture, indemnification, minimum commitment, responsibility for all costs, business-activity restriction, non-solicitation, exclusivity, non-disparagement, assignment restriction, change of control, ROFR, most-favoured terms, termination without cause, sole discretion, automatic renewal, survival, IP assignment, covenant not to sue
* Weight 1: broad "any and all" scope, revenue share/royalty, monetary amount, late charges, costs at one party's expense, non-transferable rights, price/volume limits, broad territory, immediate termination, perpetual/irrevocable, post-termination obligation, joint IP, audit rights, insurance, third-party rights
* Duration: +2 if a restriction lasts >= 2 years, +1 if >= 1 year; +1 for other periods >= 5 years. Amount: +1 for amounts >= 1,000,000
* Protective, weight -1: liability cap, mutual wording, consent not unreasonably withheld, defined notice period
* Negation/scope handling: `non-exclusive` is not exclusivity; `exclusive remedy/jurisdiction/property` are excluded; "World Wide Web" is not worldwide scope; "no third-party beneficiaries" is not a third-party right; carve-outs only count when liability language is present.
* Conflicts are resolved additively (protective indicators offset risk indicators).

**Measured distribution:** Low 5,684 (65.5%), Medium 2,350 (27.1%), High 649 (7.5%). Classes are imbalanced (ratio 8.76), so class-weighted loss/`class_weight="balanced"` is used. All 36 categories contain more than one risk label, confirming labels are not a fixed category mapping.

These are **project-generated screening labels, not ground-truth legal risk** and not expert annotations.

## 13. Train / Validation / Test Split (`training/split_dataset.py`)

StratifiedGroupKFold (groups = `contract_id`, 20 folds, seed 42): 3 folds test, 3 folds validation, 14 folds train. All clauses from one contract are in exactly one split, and the assignment is saved in `data/split_assignments.csv` so **every model uses identical partitions**. Stratification is approximate because whole contracts move together.

Leakage controls applied when loading (`training/common.py`): duplicate texts inside a split are collapsed; validation/test texts that also occur in train are removed (11 val, 6 test); contract overlap is asserted to be zero.

| Split | Contracts | Clauses | Low | Medium | High |
|---|---|---|---|---|---|
| Train | 328* | 5,346 | 3,690 | 1,322 | 334 |
| Validation | 76 | 1,170 | 802 | 299 | 69 |
| Test | 70 | 1,158 | 797 | 289 | 72 |

\*Contracts with at least one clause. The 35 contracts without any extractable clause are also assigned to train.

## 14. TF-IDF Baseline (`training/train_baseline.py`)

Word 1-2-grams, max 30,000 features, min_df 2, sublinear TF, no stop-word removal (negations kept), vectorizer fitted on train only, class-weighted Logistic Regression; C selected on validation macro F1 from {0.5, 1, 2, 4, 8} (selected C = 8). Character n-grams are configurable (`TFIDF_USE_CHAR_NGRAMS`) but off by default.

## 15. Word2Vec Experiment (`training/train_word2vec.py`)

Corpus analysis on the training split: 316,452 tokens, 5,492 vocabulary items at min_count 2, 30.9% of word types occur once. This met the project threshold (>= 100k tokens, >= 3k vocabulary), so a skip-gram Word2Vec (100 dims, window 5, 20 epochs) was trained **on train only**; clause vectors are mean-pooled and classified with class-weighted Logistic Regression.

**Word2Vec was evaluated as an optional classical embedding approach but was not selected as the production representation because** averaged static vectors discard word order and negation scope, and it scored well below TF-IDF on the same test split.

## 16. Contracts-BERT (`training/train_model.py`)

`nlpaueb/bert-base-uncased-contracts` with `AutoTokenizer` / `AutoModelForSequenceClassification`, 3 labels (`0 = Low, 1 = Medium, 2 = High`). Input is the clean original `clause_text` only - not the clause type, not the spaCy-processed text.

## 17. Model Training

Configurable in `config.py` (or environment variables `LRAI_*`, or CLI flags): `MODEL_NAME`, `MAX_LENGTH` (256), `BATCH_SIZE` (16 on CUDA) / `CPU_BATCH_SIZE` (8), `LEARNING_RATE` (2e-5), `EPOCHS` (4), `WEIGHT_DECAY` (0.01), `WARMUP_RATIO` (0.1), `RANDOM_SEED` (42), `EARLY_STOPPING_PATIENCE` (2).

The script auto-detects the device and prints `Selected device: CUDA` or `Selected device: CPU`. Batches are length-grouped by default (similar-length clauses together, less padding). `--freeze-layers N` freezes the embeddings and lowest N encoder layers, and `--cpu-fast` is a CPU preset (`--epochs 2 --max-length 128 --freeze-layers 6`). The options used are recorded in `training_metadata.json`; measured speed-ups are in SETUP_GUIDE.md. It uses class-weighted cross-entropy, AdamW with linear warm-up, gradient clipping, mixed precision on CUDA, validation macro F1 each epoch, best-model checkpointing and early stopping. The test split is not touched. Saved to `model/contracts_bert_risk/`: weights, config, tokenizer, `label_mapping.json`, `training_metadata.json` (sizes, hyper-parameters, seed, device, time, history).

`--smoke-test` trains on 48 clauses for 1 epoch and saves to `model/_smoke_test/`, only to verify the pipeline. Its results are never reported as model results.

The Streamlit app never trains; it only loads an existing model.

## 18. Model Evaluation (`training/evaluate_model.py`)

Evaluates Contracts-BERT on the untouched test split and writes JSON metrics, classification report, confusion matrix (CSV + PNG), per-clause test predictions and the comparison table to `reports/model_evaluation/`. Existing result files are moved to `reports/model_evaluation/archive/<timestamp>/` with a warning instead of being overwritten.

## 19. Model Comparison (measured, test split)

| Model | Representation | Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | TF-IDF | 0.7927 | 0.6573 | 0.6191 | 0.6351 | 0.7875 |
| Word2Vec + Logistic Regression | Word2Vec | 0.5449 | 0.4461 | 0.4834 | 0.4262 | 0.5954 |
| Contracts-BERT | contextual embeddings | *run training/train_model.py, then training/evaluate_model.py* | | | | |

Per-class F1 (TF-IDF): Low 0.887, Medium 0.605, High 0.413.

**Contracts-BERT results are not included** because the model was not fully trained in the environment where this package was built (a single-CPU sandbox, about 6 s per training step). The training and evaluation code was executed end to end in smoke-test mode to verify that it works. After you train locally, `evaluate_model.py` fills this row with your measured numbers in `reports/model_evaluation/model_comparison.csv/.md`.

## 20. SHAP Explainability (`services/explainer.py`)

* Contracts-BERT: `shap.Explainer` with a `Text` masker (partition explainer) on the predicted-class probability; sub-word tokens are merged into words. Computed automatically for the 3 highest-risk clauses and on demand for others (`SHAP_MAX_EVALS`, default 200).
* Baseline: `shap.LinearExplainer` on TF-IDF features (log-odds, empty-clause reference).
* Wording used: "These terms contributed to the model's prediction." SHAP explains model behaviour; it does not establish legal validity.
* On failure, a fallback shows rule-matched phrases explicitly labelled "not SHAP". No SHAP values are fabricated. Demo Mode shows no SHAP values.

## 21. Rule-Based Explanations (`services/risk_rules.py`)

Each matched indicator has a cautious plain-English sentence, e.g. non-compete: "May restrict future employment or business activities during or after the agreement." Uncapped liability: "May expose a party to financial liability without a stated upper limit." Automatic renewal: "May cause the agreement to continue unless termination notice is provided." These are simplified screening explanations, not legal advice.

**Risk score.** Clause screening score = `100 x (0.0 x P(Low) + 0.5 x P(Medium) + 1.0 x P(High))`; confidence = probability of the predicted class. Overall score = `0.5 x mean(clause scores) + 0.5 x mean(top-quartile clause scores)`; Low < 35 <= Medium < 65 <= High. It is always shown as an **AI-generated screening score**, never as a legal measurement. In Demo Mode the clause score is `min(100, 100 x annotation_score / 8)` and confidence is not applicable.

## 22. Streamlit Application

Pages: Home, Analyze Contract (upload + processing steps + "View Risk Dashboard"), Risk Dashboard (metrics, overall score, donut, risk by clause type, clause score chart, observations, High/Medium/Low sections), Clause Analysis (search, All/High/Medium/Low filter, sort by Highest Risk / Lowest Risk / Clause Type, full clause detail with SHAP), Document Viewer (highlighted text, legend, previous/next, clause index), Report (PDF + CSV). Custom CSS in `styles/style.css`.

## 23. Installation (Windows, PowerShell)

```powershell
cd legal_clause_risk_detector
python -m venv .venv
.venv\Scripts\activate
# optional, NVIDIA GPU only - install CUDA PyTorch first:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python check_setup.py
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. If `streamlit` is not recognised, use `python -m streamlit`. On Python 3.13+ gensim is skipped automatically (it is only needed for the optional Word2Vec experiment); see SETUP_GUIDE.md.

## 24. Usage

The derived CSVs and the trained baseline/Word2Vec artefacts are already included. To regenerate everything from `master_clauses.csv`:

```powershell
python training/inspect_dataset.py
python training/prepare_dataset.py
python training/annotate_risk.py
python training/split_dataset.py
python training/train_baseline.py
python training/train_word2vec.py
python training/train_model.py              # Contracts-BERT on GPU  (CPU: add --cpu-fast)
python training/evaluate_model.py           # test metrics + comparison table
python -m streamlit run app.py
```

Quick pipeline check without full training: `python training/train_model.py --smoke-test` then `python training/evaluate_model.py --model-dir model/_smoke_test`. Run the tests with `python -m pytest -q`. A sample contract for uploading is in `assets/sample_rental_agreement.pdf`.

**Hardware.** CPU works for everything, but full BERT fine-tuning on a CPU takes several hours; `--cpu-fast` measured about 55% faster per step than the original settings on the build machine. With a CUDA GPU (about 6 GB VRAM or more) training should be much faster. 8 GB RAM is recommended. The first training run downloads the pre-trained Contracts-BERT weights (about 440 MB).

## 25. Demo Mode

Click **Try Demo** (Home) or **Try Demo Contract** (Analyze). A fictional consulting agreement containing non-compete, uncapped liability, renewal, termination, insurance, governing law, audit, assignment and liquidated-damages clauses is analysed by the rule engine only. Every page shows a "Demo Mode" badge; confidence is shown as not applicable and no SHAP values are shown, so demo output is never presented as a model result.

## 26. Folder Structure

```
legal_clause_risk_detector/
  app.py, config.py, check_setup.py, requirements.txt, README.md, SETUP_GUIDE.md, .gitignore, .streamlit/config.toml
  data/        master_clauses.csv, extracted_clauses.csv, risk_annotated_clauses.csv,
               contract_index.csv, split_assignments.csv, analysis/*.json, README.md
  model/       README.md, contracts_bert_risk/ (created by training), baseline/, word2vec/
  training/    inspect_dataset.py, prepare_dataset.py, annotate_risk.py, split_dataset.py,
               train_baseline.py, train_word2vec.py, train_model.py, evaluate_model.py,
               dataset_utils.py, common.py
  services/    pdf_extractor.py, clause_segmenter.py, nlp_processor.py, predictor.py,
               explainer.py, risk_rules.py, api.py, mock_data.py
  components/  navbar.py, sidebar.py, cards.py, charts.py, clause_card.py, explanation.py
  pages/       home.py, analyze.py, dashboard.py, clauses.py, document.py, report.py
  reports/     report_generator.py, model_evaluation/
  styles/      style.css
  assets/      README.md, make_sample_pdf.py, sample_rental_agreement.pdf
  tests/       test_pipeline.py
```

## Academic Contribution

1. CUAD is used as the source of annotated contract clauses. 2. The dataset is transformed into clause-level training instances. 3. A project-defined risk annotation layer is created. 4. A contract-level split reduces data leakage. 5. TF-IDF + Logistic Regression provides a classical baseline. 6. Word2Vec is evaluated as an optional embedding experiment. 7. Contracts-BERT is fine-tuned for three-class risk classification. 8. SHAP provides model interpretability. 9. Rule-based explanations translate risk indicators into simpler language. 10. Streamlit provides an interactive screening interface. This project did not invent BERT, SHAP, CUAD, TF-IDF or Word2Vec.

## 27. Limitations

* Risk labels are project-defined heuristics, not expert legal annotations; the models learn to reproduce these screening labels.
* Predictions are not legal advice; this is a screening tool and predictions can be wrong.
* CUAD covers commercial contracts; rental, employment and privacy documents are outside its distribution, so results for them are less reliable.
* The dataset (510 contracts) is small compared with large language-model corpora.
* Scanned/image-only PDFs are not supported (no OCR).
* Clause segmentation and clause-type rules can mis-split or mis-type unusual layouts.
* SHAP explains model behaviour, not legal validity; BERT SHAP is slow on CPU.
* Rule-based explanations are simplified interpretations.
* Word2Vec has limited value on this small legal corpus.
* Contracts-BERT metrics must be produced locally (section 19).

## 28. Legal Disclaimer

LegalRisk AI provides automated contract screening and educational information. It does not provide legal advice or replace a qualified legal professional.

**Privacy & Security.** Contracts may contain confidential information. This prototype is intended for research and demonstration purposes. Avoid uploading sensitive real-world documents unless appropriate security controls are in place. Uploaded files are processed in memory by the local app and are not written to disk by the application.
