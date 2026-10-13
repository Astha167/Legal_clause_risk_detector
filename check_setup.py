"""
Environment and model status check for LegalRisk AI.

Usage (from the project root):
    python check_setup.py

Prints what is installed, which models are ready, and the next command to run.
It does not change anything.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"

REQUIRED = [("streamlit", "streamlit"), ("pandas", "pandas"), ("numpy", "numpy"), ("scipy", "scipy"),
            ("torch", "torch"), ("transformers", "transformers"), ("sklearn", "scikit-learn"),
            ("joblib", "joblib"), ("spacy", "spacy"), ("nltk", "nltk"), ("shap", "shap"),
            ("pymupdf", "pymupdf"), ("plotly", "plotly"), ("reportlab", "reportlab"),
            ("matplotlib", "matplotlib")]
OPTIONAL = [("gensim", "gensim", "only for the optional Word2Vec experiment"),
            ("pytest", "pytest", "only for running the tests")]


def _version(mod) -> str:
    return getattr(mod, "__version__", "") or "installed"


def main() -> int:
    missing_required = []
    print("LegalRisk AI - setup check\n")
    v = sys.version_info
    py = f"{v.major}.{v.minor}.{v.micro}"
    if (3, 10) <= (v.major, v.minor) <= (3, 12):
        print(f"{OK} Python {py}")
    elif (v.major, v.minor) >= (3, 13):
        print(f"{WARN} Python {py}: the app works, but gensim (optional) has no Windows wheel for this "
              "version. 3.10-3.12 is recommended.")
    else:
        print(f"{BAD} Python {py}: please use Python 3.10 or newer.")

    print("\nRequired packages")
    for mod_name, pip_name in REQUIRED:
        try:
            mod = importlib.import_module(mod_name)
            print(f"{OK} {pip_name} {_version(mod)}")
        except Exception as exc:  # ImportError or a broken install
            print(f"{BAD} {pip_name} missing ({type(exc).__name__})")
            missing_required.append(pip_name)
    print("\nOptional packages")
    for mod_name, pip_name, why in OPTIONAL:
        try:
            mod = importlib.import_module(mod_name)
            print(f"{OK} {pip_name} {_version(mod)}")
        except Exception:
            print(f"{WARN} {pip_name} not installed - {why}")

    print("\nLanguage model")
    try:
        import spacy

        spacy.load("en_core_web_sm")
        print(f"{OK} spaCy en_core_web_sm")
    except Exception:
        print(f"{BAD} spaCy en_core_web_sm missing - run: python -m spacy download en_core_web_sm")
        missing_required.append("en_core_web_sm")

    print("\nHardware")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"{OK} CUDA GPU: {torch.cuda.get_device_name(0)} - Contracts-BERT training will use it")
        else:
            print(f"{WARN} No CUDA GPU detected - training runs on CPU (use --cpu-fast for a quicker run)")
    except Exception:
        print(f"{WARN} PyTorch unavailable, hardware not checked")

    import config

    print("\nData")
    for p in [config.RAW_DATASET_PATH, config.RISK_ANNOTATED_PATH, config.SPLIT_ASSIGNMENTS_PATH]:
        print(f"{OK if p.exists() else BAD} {p.relative_to(ROOT)}")

    print("\nModels")
    base_ok = (config.BASELINE_MODEL_DIR / "logistic_regression.joblib").exists()
    print(f"{OK if base_ok else WARN} TF-IDF baseline: {'ready' if base_ok else 'missing - python training/train_baseline.py'}")
    bert_dir = config.BERT_MODEL_DIR
    has_weights = any((bert_dir / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))
    smoke = False
    meta = bert_dir / "training_metadata.json"
    if meta.exists():
        try:
            smoke = bool(json.loads(meta.read_text(encoding="utf-8")).get("smoke_test"))
        except ValueError:
            pass
    if has_weights and (bert_dir / "config.json").exists():
        state = "smoke-test weights only (not meaningful)" if smoke else "ready"
        print(f"{WARN if smoke else OK} Contracts-BERT: {state}")
        bert_ready = not smoke
    else:
        print(f"{WARN} Contracts-BERT: not trained yet (this is expected right after unzipping)")
        bert_ready = False
    eval_ok = (config.EVALUATION_DIR / "contracts_bert_metrics.json").exists()

    print("\nNext step")
    if missing_required:
        print("  pip install -r requirements.txt")
        if "en_core_web_sm" in missing_required:
            print("  python -m spacy download en_core_web_sm")
    elif not bert_ready:
        print("  The app already works with the TF-IDF baseline and Demo Mode:  python -m streamlit run app.py")
        print("  To add Contracts-BERT, train it:")
        print("    GPU : python training/train_model.py")
        print("    CPU : python training/train_model.py --cpu-fast")
    elif not eval_ok:
        print("  python training/evaluate_model.py      (measure Contracts-BERT on the test split)")
    else:
        print("  All set:  python -m streamlit run app.py")
    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
