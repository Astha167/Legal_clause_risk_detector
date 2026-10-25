"""
Model inference.

Engines
-------
* ``contracts_bert`` - fine-tuned Contracts-BERT in model/contracts_bert_risk/
                       (primary model).
* ``baseline``       - TF-IDF + Logistic Regression in model/baseline/
                       (clearly labelled fallback when BERT is not trained yet).

Screening risk score (documented in README):
    risk_score = 100 * (0.0 * P(Low) + 0.5 * P(Medium) + 1.0 * P(High))
Confidence is the probability of the predicted class. Both are model outputs,
not legal measurements.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config
from config import get_logger

log = get_logger("predictor")

ENGINE_BERT = "contracts_bert"
ENGINE_BASELINE = "baseline"
ENGINE_DEMO = "demo"
ENGINE_LABELS = {
    ENGINE_BERT: "Contracts-BERT",
    ENGINE_BASELINE: "TF-IDF + Logistic Regression (baseline)",
    ENGINE_DEMO: "Demo Mode (rule-based, no ML model)",
}


class ModelNotFoundError(Exception):
    pass


class ModelLoadError(Exception):
    pass


def risk_score_from_probs(probs: Dict[str, float]) -> float:
    return round(100.0 * sum(config.RISK_SCORE_WEIGHTS[l] * float(probs.get(l, 0.0)) for l in config.LABELS), 1)


BERT_WEIGHTS_RELEASE_URL = "https://github.com/Astha167/Legal_clause_risk_detector/releases/download/v1.0.0/model.safetensors"


def ensure_bert_weights(model_dir: Path) -> None:
    """Download model.safetensors from GitHub Release if not present locally."""
    model_dir = Path(model_dir)
    weights_path = model_dir / "model.safetensors"
    if weights_path.exists() and weights_path.stat().st_size > 1_000_000:
        return

    import urllib.request
    log.info("Downloading Contracts-BERT model weights from %s ...", BERT_WEIGHTS_RELEASE_URL)
    model_dir.mkdir(parents=True, exist_ok=True)
    temp_path = model_dir / "model.safetensors.download"
    try:
        urllib.request.urlretrieve(BERT_WEIGHTS_RELEASE_URL, temp_path)
        temp_path.replace(weights_path)
        log.info("Contracts-BERT weights downloaded successfully.")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        log.exception("Failed to download Contracts-BERT weights: %s", e)
        raise


def _bert_files_present(model_dir: Path) -> bool:
    has_weights = any((model_dir / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))
    has_config = (model_dir / "config.json").exists()
    return has_weights or has_config


def _is_smoke(model_dir: Path) -> bool:
    meta = model_dir / "training_metadata.json"
    if not meta.exists():
        return False
    try:
        return bool(json.loads(meta.read_text(encoding="utf-8")).get("smoke_test"))
    except (OSError, ValueError):
        return False


class BertRiskPredictor:
    engine = ENGINE_BERT

    def __init__(self, model_dir: Path = config.BERT_MODEL_DIR):
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise ModelNotFoundError(config.MODEL_MISSING_MESSAGE)
        has_weights = any((model_dir / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))
        if not has_weights:
            ensure_bert_weights(model_dir)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = config.get_device(verbose=False)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device).eval()
            if getattr(self.device, "type", "cpu") == "cpu":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            self._torch = torch
        except Exception as exc:
            log.exception("Failed to load Contracts-BERT from %s", model_dir)
            raise ModelLoadError(
                "The trained Contracts-BERT model could not be loaded. The files may be incomplete or "
                "corrupted - retrain with: python training/train_model.py"
            ) from exc
        id2label = self.model.config.id2label or config.ID2LABEL
        self.labels = [id2label.get(i, id2label.get(str(i))) for i in range(len(id2label))]
        if sorted(self.labels) != sorted(config.LABELS):
            raise ModelLoadError(f"Unexpected label set in model: {self.labels}")
        self.model_dir = model_dir
        self.smoke_test = _is_smoke(model_dir)
        self.max_length = config.MAX_LENGTH
        meta = model_dir / "training_metadata.json"
        if meta.exists():
            try:
                self.max_length = json.loads(meta.read_text())["hyperparameters"]["max_length"]
            except (KeyError, ValueError, OSError):
                pass
        self._lock = threading.Lock()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        out = []
        with self._lock, torch.no_grad():
            for i in range(0, len(texts), config.INFERENCE_BATCH_SIZE):
                chunk = texts[i:i + config.INFERENCE_BATCH_SIZE]
                enc = self.tokenizer(chunk, truncation=True, max_length=self.max_length, padding=True,
                                     return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probs = np.vstack(out) if out else np.zeros((0, len(self.labels)))
        # reorder columns to config.LABELS order
        order = [self.labels.index(l) for l in config.LABELS]
        return probs[:, order]


class BaselineRiskPredictor:
    engine = ENGINE_BASELINE

    def __init__(self, model_dir: Path = config.BASELINE_MODEL_DIR):
        model_dir = Path(model_dir)
        vec_p, clf_p = model_dir / "tfidf_vectorizer.joblib", model_dir / "logistic_regression.joblib"
        if not (vec_p.exists() and clf_p.exists()):
            raise ModelNotFoundError("Baseline model not found. Run: python training/train_baseline.py")
        try:
            import joblib

            self.vectorizer = joblib.load(vec_p)
            self.model = joblib.load(clf_p)
        except Exception as exc:
            log.exception("Failed to load baseline model")
            raise ModelLoadError("The baseline model files could not be loaded (they may be corrupted).") from exc
        self.labels = [config.ID2LABEL[int(c)] for c in self.model.classes_]
        self.smoke_test = False

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        probs = self.model.predict_proba(self.vectorizer.transform(texts))
        full = np.zeros((len(texts), len(config.LABELS)))
        for j, lab in enumerate(self.labels):
            full[:, config.LABEL2ID[lab]] = probs[:, j]
        return full


_CACHE: Dict[str, object] = {}
_CACHE_LOCK = threading.Lock()


def load_predictor(engine: str):
    """Load (and cache) a predictor. Raises ModelNotFoundError / ModelLoadError."""
    with _CACHE_LOCK:
        if engine in _CACHE:
            return _CACHE[engine]
        if engine == ENGINE_BERT:
            pred = BertRiskPredictor()
        elif engine == ENGINE_BASELINE:
            pred = BaselineRiskPredictor()
        else:
            raise ValueError(f"Unknown engine '{engine}'")
        _CACHE[engine] = pred
        return pred


def engine_status() -> Dict[str, Dict]:
    """Availability of each engine without loading heavy weights."""
    bert_ok = _bert_files_present(config.BERT_MODEL_DIR)
    base_ok = (config.BASELINE_MODEL_DIR / "logistic_regression.joblib").exists() and \
        (config.BASELINE_MODEL_DIR / "tfidf_vectorizer.joblib").exists()
    return {
        ENGINE_BERT: {"available": bert_ok, "label": ENGINE_LABELS[ENGINE_BERT],
                      "smoke_test": bert_ok and _is_smoke(config.BERT_MODEL_DIR),
                      "message": "" if bert_ok else config.MODEL_MISSING_MESSAGE},
        ENGINE_BASELINE: {"available": base_ok, "label": ENGINE_LABELS[ENGINE_BASELINE], "smoke_test": False,
                          "message": "" if base_ok else "Baseline not trained (python training/train_baseline.py)."},
    }


def default_engine() -> Optional[str]:
    st = engine_status()
    if st[ENGINE_BERT]["available"]:
        return ENGINE_BERT
    if st[ENGINE_BASELINE]["available"]:
        return ENGINE_BASELINE
    return None


def predict_clauses(clauses: List[Dict], engine: str) -> List[Dict]:
    """Add risk_label, risk_score, confidence and probabilities to each clause dict."""
    predictor = load_predictor(engine)
    texts = [c["clause_text"] for c in clauses]
    if not texts:
        return []
    probs = predictor.predict_proba(texts)
    results = []
    for c, p in zip(clauses, probs):
        pd_ = {l: round(float(v), 4) for l, v in zip(config.LABELS, p)}
        label = config.LABELS[int(np.argmax(p))]
        results.append({
            **c,
            "risk_label": label,
            "risk_score": risk_score_from_probs(pd_),
            "confidence": round(float(np.max(p)), 4),
            "probabilities": pd_,
        })
    return results
