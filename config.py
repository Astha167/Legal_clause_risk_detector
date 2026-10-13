"""
Central configuration for LegalRisk AI.

All paths are resolved relative to this file so the project works from any
working directory (VS Code, PowerShell, Streamlit) on Windows, Linux or macOS.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATASET_PATH = DATA_DIR / "master_clauses.csv"
EXTRACTED_CLAUSES_PATH = DATA_DIR / "extracted_clauses.csv"
RISK_ANNOTATED_PATH = DATA_DIR / "risk_annotated_clauses.csv"
CONTRACT_INDEX_PATH = DATA_DIR / "contract_index.csv"
SPLIT_ASSIGNMENTS_PATH = DATA_DIR / "split_assignments.csv"
DATA_REPORTS_DIR = DATA_DIR / "analysis"

MODEL_DIR = PROJECT_ROOT / "model"
BERT_MODEL_DIR = MODEL_DIR / "contracts_bert_risk"
BASELINE_MODEL_DIR = MODEL_DIR / "baseline"
WORD2VEC_MODEL_DIR = MODEL_DIR / "word2vec"

REPORTS_DIR = PROJECT_ROOT / "reports"
EVALUATION_DIR = REPORTS_DIR / "model_evaluation"

STYLES_PATH = PROJECT_ROOT / "styles" / "style.css"
ASSETS_DIR = PROJECT_ROOT / "assets"
LOG_DIR = PROJECT_ROOT / "logs"

# --------------------------------------------------------------------------
# Dataset handling
# --------------------------------------------------------------------------
# CUAD metadata categories. These describe the document itself (title, parties,
# dates) and are NOT legal clauses, so they are excluded from ML training text.
# Matching is case-insensitive and whitespace-insensitive.
METADATA_CATEGORIES = [
    "Filename",
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
]

# Marker CUAD uses where the annotator skipped text inside a span.
CUAD_OMITTED_MARKER = "<omitted>"
OMITTED_REPLACEMENT = " [...] "

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
LABELS = ["Low", "Medium", "High"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Project-defined annotation thresholds on the rule-based annotation score.
# score <= LOW_MAX            -> Low
# LOW_MAX < score < HIGH_MIN  -> Medium
# score >= HIGH_MIN           -> High
ANNOTATION_LOW_MAX = 1
ANNOTATION_HIGH_MIN = 4

# --------------------------------------------------------------------------
# Reproducibility and split
# --------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --------------------------------------------------------------------------
# Contracts-BERT hyper-parameters (override with environment variables or CLI)
# --------------------------------------------------------------------------
MODEL_NAME = os.getenv("LRAI_MODEL_NAME", "nlpaueb/bert-base-uncased-contracts")
MAX_LENGTH = int(os.getenv("LRAI_MAX_LENGTH", "256"))
BATCH_SIZE = int(os.getenv("LRAI_BATCH_SIZE", "16"))          # used on CUDA
CPU_BATCH_SIZE = int(os.getenv("LRAI_CPU_BATCH_SIZE", "8"))   # conservative CPU value
LEARNING_RATE = float(os.getenv("LRAI_LEARNING_RATE", "2e-5"))
EPOCHS = int(os.getenv("LRAI_EPOCHS", "4"))
WEIGHT_DECAY = float(os.getenv("LRAI_WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.getenv("LRAI_WARMUP_RATIO", "0.1"))
EARLY_STOPPING_PATIENCE = int(os.getenv("LRAI_EARLY_STOPPING_PATIENCE", "2"))
GRAD_CLIP_NORM = 1.0
INFERENCE_BATCH_SIZE = int(os.getenv("LRAI_INFERENCE_BATCH_SIZE", "8"))

# --------------------------------------------------------------------------
# TF-IDF baseline
# --------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 30000
TFIDF_MIN_DF = 2
TFIDF_WORD_NGRAM_RANGE = (1, 2)
TFIDF_USE_CHAR_NGRAMS = False   # word n-grams are sufficient for this corpus
LOGREG_C = 4.0
LOGREG_MAX_ITER = 3000

# --------------------------------------------------------------------------
# Word2Vec experiment
# --------------------------------------------------------------------------
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = 1   # >1 is faster but makes gensim results non-deterministic
W2V_EPOCHS = 20

# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
SHAP_MAX_EVALS = int(os.getenv("LRAI_SHAP_MAX_EVALS", "200"))
SHAP_TOP_K = 10

# --------------------------------------------------------------------------
# Risk score (see README "Risk score")
# screening_score = 100 * (0.0 * P(Low) + 0.5 * P(Medium) + 1.0 * P(High))
# --------------------------------------------------------------------------
RISK_SCORE_WEIGHTS = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
OVERALL_LOW_MAX = 35      # overall score < 35  -> Low
OVERALL_HIGH_MIN = 65     # overall score >= 65 -> High

# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------
MAX_UPLOAD_MB = 25
MIN_CLAUSE_CHARS = 25

APP_NAME = "LegalRisk AI"
APP_VERSION = "1.0"
APP_TITLE = "AI-Powered Legal Clause Risk Detector"
APP_TAGLINE = "Understand contractual risks before you sign."

DISCLAIMER = (
    "LegalRisk AI provides automated contract screening and educational information. "
    "It does not provide legal advice or replace a qualified legal professional."
)
PRIVACY_NOTICE = (
    "Contracts may contain confidential information. This prototype is intended for "
    "research and demonstration purposes. Avoid uploading sensitive real-world documents "
    "unless appropriate security controls are in place."
)
MODEL_MISSING_MESSAGE = (
    "Trained model not found. Please train the model first or use Demo Mode."
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_device(verbose: bool = True):
    """Return a torch.device, preferring CUDA when an NVIDIA GPU is available."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("Selected device: CUDA")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Selected device: CPU")
    return device


def effective_batch_size(device) -> int:
    return BATCH_SIZE if getattr(device, "type", "cpu") == "cuda" else CPU_BATCH_SIZE


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    """File + console logger. Technical details go to logs/app.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    logger.propagate = False
    return logger
