"""
Model explainability with SHAP.

* Contracts-BERT : shap.Explainer with a Text masker (partition explainer).
                   Values explain the predicted-class probability.
* Baseline       : shap.LinearExplainer on the TF-IDF features (log-odds),
                   relative to an empty-clause reference.

Interpretation: "These terms contributed to the model's prediction."
SHAP describes model behaviour; it does not establish legal validity.

If SHAP is unavailable, slow, or fails, a clearly-labelled fallback based on
the rule engine's matched phrases is returned instead - no SHAP values are
invented.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List

import numpy as np

import config
from config import get_logger
from services import risk_rules

log = get_logger("explainer")

SHAP_WORDING = "These terms contributed to the model's prediction."
_PUNCT = re.compile(r"^[\W_]+$")


def _aggregate_words(tokens: List[str], values: np.ndarray) -> List[Dict]:
    """Merge sub-word tokens into words (a token ending in whitespace closes a word)."""
    words, cur_txt, cur_val = [], "", 0.0
    for tok, val in zip(tokens, values):
        cur_txt += tok
        cur_val += float(val)
        if tok.endswith((" ", "\n")) or tok == tokens[-1]:
            w = cur_txt.strip()
            if w and not _PUNCT.match(w):
                words.append({"term": w.strip(".,;:()\"'"), "value": cur_val})
            cur_txt, cur_val = "", 0.0
    # merge repeated words
    merged: Dict[str, float] = {}
    for w in words:
        key = w["term"].lower()
        if key:
            merged[key] = merged.get(key, 0.0) + w["value"]
    return [{"term": k, "value": v} for k, v in merged.items()]


def _top(terms: List[Dict], k: int) -> List[Dict]:
    terms = sorted(terms, key=lambda t: -abs(t["value"]))[:k]
    return [{"term": t["term"], "value": round(float(t["value"]), 5)} for t in terms]


def explain_bert(predictor, text: str, target_label: str, max_evals: int = config.SHAP_MAX_EVALS) -> Dict:
    import shap

    target = config.LABEL2ID[target_label]

    def f(batch):
        return predictor.predict_proba([str(b) for b in batch])

    start = time.time()
    masker = shap.maskers.Text(predictor.tokenizer)
    explainer = shap.Explainer(f, masker, output_names=config.LABELS, silent=True)
    sv = explainer([text], max_evals=max_evals, batch_size=config.INFERENCE_BATCH_SIZE, silent=True)
    tokens = list(sv.data[0])
    values = np.asarray(sv.values[0])[:, target]
    terms = _aggregate_words(tokens, values)
    return {
        "available": True,
        "method": "SHAP partition explainer (text masker) on Contracts-BERT",
        "target_label": target_label,
        "value_meaning": f"contribution to P({target_label})",
        "terms": _top(terms, config.SHAP_TOP_K),
        "base_value": round(float(np.asarray(sv.base_values[0])[target]), 4),
        "seconds": round(time.time() - start, 1),
        "message": SHAP_WORDING,
    }


def explain_baseline(predictor, text: str, target_label: str) -> Dict:
    target = config.LABEL2ID[target_label]
    X = predictor.vectorizer.transform([text])
    names = predictor.vectorizer.get_feature_names_out()
    cls_index = list(predictor.model.classes_).index(target)
    method = "SHAP linear explainer on TF-IDF + Logistic Regression"
    try:
        import scipy.sparse as sp
        import shap

        background = sp.csr_matrix((1, X.shape[1]))
        explainer = shap.LinearExplainer(predictor.model, shap.maskers.Independent(background))
        vals = explainer.shap_values(X)
        vals = np.asarray(vals)
        # shap returns (n, features, classes) or list per class depending on version
        if vals.ndim == 3:
            row = vals[0, :, cls_index] if vals.shape[-1] == len(predictor.model.classes_) else vals[cls_index, 0]
        else:
            row = vals[0]
        row = np.asarray(row).ravel()
    except Exception as exc:
        # Exact linear SHAP values with an all-zero reference equal coef * x.
        log.info("shap.LinearExplainer unavailable (%s); computing linear SHAP values directly.", exc)
        row = np.asarray(X.multiply(predictor.model.coef_[cls_index]).todense()).ravel()
        method = "Linear SHAP values (coefficient x feature) on TF-IDF + Logistic Regression"
    nz = X.nonzero()[1]
    terms = [{"term": names[j], "value": float(row[j])} for j in nz if row[j] != 0]
    return {
        "available": True,
        "method": method,
        "target_label": target_label,
        "value_meaning": f"contribution to the log-odds of {target_label} (reference: empty clause)",
        "terms": _top(terms, config.SHAP_TOP_K),
        "message": SHAP_WORDING,
    }


def fallback_explanation(text: str, reason: str) -> Dict:
    factors = risk_rules.find_indicators(text)
    terms = [{"term": f["matched"].strip()[:48], "value": float(f["weight"])} for f in factors if f["matched"]]
    return {
        "available": False,
        "method": "Rule-based key phrases (fallback - not SHAP)",
        "target_label": None,
        "value_meaning": "rule weight (not a SHAP value)",
        "terms": _top(terms, config.SHAP_TOP_K),
        "message": f"SHAP explanation unavailable: {reason} Showing rule-based key phrases instead.",
    }


def explain(engine: str, text: str, target_label: str, max_evals: int = config.SHAP_MAX_EVALS) -> Dict:
    """Never raises: returns a SHAP explanation or a labelled fallback."""
    from services.predictor import ENGINE_BASELINE, ENGINE_BERT, load_predictor

    try:
        if engine not in (ENGINE_BERT, ENGINE_BASELINE):
            return fallback_explanation(text, "no trained model is in use (Demo Mode).")
        predictor = load_predictor(engine)
        if engine == ENGINE_BERT:
            return explain_bert(predictor, text, target_label, max_evals=max_evals)
        return explain_baseline(predictor, text, target_label)
    except ImportError:
        return fallback_explanation(text, "the 'shap' package is not installed.")
    except Exception as exc:
        log.exception("SHAP explanation failed")
        return fallback_explanation(text, f"the computation failed ({type(exc).__name__}).")
