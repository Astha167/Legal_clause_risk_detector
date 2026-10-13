"""
Service layer between the Streamlit UI and the NLP / ML components.

    Streamlit -> api.py -> PDF extraction -> clause segmentation -> spaCy NLP
              -> Contracts-BERT (or baseline) -> SHAP -> rule engine -> results

The UI never imports models directly; it only calls the functions below and
renders the returned plain-Python dictionaries.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

import config
from config import get_logger
from services import clause_segmenter, explainer, mock_data, risk_rules
from services.pdf_extractor import PDFExtractionError, extract_text_from_pdf
from services.predictor import (
    ENGINE_BASELINE,
    ENGINE_BERT,
    ENGINE_DEMO,
    ENGINE_LABELS,
    ModelLoadError,
    ModelNotFoundError,
    engine_status,
    load_predictor,
    predict_clauses,
)

log = get_logger("api")

STEPS = [
    "Document uploaded",
    "Text extraction",
    "Clause detection",
    "NLP processing",
    "Risk classification",
    "Explainability analysis",
]
SHAP_PRECOMPUTE_BERT = 0     # BERT SHAP is slow on CPU: disable precompute
DEMO_SCORE_SCALE = 8.0       # demo_score = min(100, 100 * annotation_score / 8)

ProgressFn = Optional[Callable[[int, str], None]]


class AnalysisError(Exception):
    """Error with a message that is safe to show to end users."""


def _progress(cb: ProgressFn, i: int, detail: str = "") -> None:
    if cb:
        try:
            cb(i, detail)
        except Exception:  # UI callbacks must never break the pipeline
            pass


def overall_score(scores: List[float]) -> float:
    """0.5 * mean(all clause scores) + 0.5 * mean(top-quartile clause scores).

    The top-quartile term stops a few serious clauses from being diluted by
    many routine ones.
    """
    if not scores:
        return 0.0
    s = sorted(scores, reverse=True)
    top = s[: max(1, int(np.ceil(len(s) * 0.25)))]
    return round(0.5 * float(np.mean(s)) + 0.5 * float(np.mean(top)), 1)


def overall_label(score: float) -> str:
    if score < config.OVERALL_LOW_MAX:
        return "Low"
    if score >= config.OVERALL_HIGH_MIN:
        return "High"
    return "Medium"


def _nlp_view(a: Dict) -> Dict:
    keep = ["processed_text", "n_tokens", "n_sentences", "pos_counts", "pos_tags", "entities", "obligations",
            "permissions", "negations", "modal_verbs", "keywords", "top_bigrams"]
    return {k: a.get(k) for k in keep}


def _obligation_bullet(nlp: Dict) -> Optional[str]:
    obs = [o for o in (nlp.get("obligations") or []) if not o.get("negated")]
    neg = [o for o in (nlp.get("obligations") or []) if o.get("negated")]
    parts = []
    if obs:
        parties = list(dict.fromkeys(o["party"] for o in obs))[:3]
        parts.append("Obligations in this clause are placed on: " + ", ".join(parties) + ".")
    if neg:
        parties = list(dict.fromkeys(o["party"] for o in neg))[:3]
        parts.append("Prohibitions apply to: " + ", ".join(parties) + ".")
    return " ".join(parts) or None


def document_observations(clauses: List[Dict]) -> List[str]:
    """Contract-level screening observations, e.g. missing protections."""
    obs: List[str] = []
    ids = Counter(f["id"] for c in clauses for f in c["factors"])
    types = {c["clause_type"] for c in clauses}
    if (ids["uncapped_liability"] or ids["cap_carveout"] or ids["indemnification"]) and not ids["liability_cap"]:
        obs.append("Liability or indemnity terms were found, but no liability cap was detected in the document.")
    has_termination = any(t in types for t in ("Termination", "Termination For Convenience")) or \
        ids["unilateral_termination"] or ids["immediate_termination"]
    if has_termination and not ids["notice_period"]:
        obs.append("No clearly defined termination notice period was detected.")
    if ids["unilateral_termination"] or ids["sole_discretion"]:
        obs.append("At least one clause gives a party a one-sided right (termination at will or sole discretion).")
    if "Governing Law" not in types and not ids["governing_law"]:
        obs.append("No governing-law clause was detected.")
    if ids["auto_renewal"]:
        obs.append("The agreement appears to renew automatically; note the deadline for sending a non-renewal notice.")
    parties = Counter()
    for c in clauses:
        for o in (c.get("nlp") or {}).get("obligations") or []:
            parties[re.sub(r"^the\s+", "", o["party"].lower().strip())] += 1
    total = sum(parties.values())
    if total >= 5:
        party, n = parties.most_common(1)[0]
        if n / total >= 0.6:
            obs.append(f"Most detected obligations ({n} of {total}) fall on '{party}', which may indicate a one-sided "
                       "allocation of responsibilities.")
    return obs


def _apply_rules_and_nlp(clauses: List[Dict]) -> List[Dict]:
    from services.nlp_processor import analyze_many

    try:
        analyses = analyze_many([c["clause_text"] for c in clauses])
    except Exception:
        log.exception("NLP processing failed; continuing without linguistic features")
        analyses = [{} for _ in clauses]
    for c, a in zip(clauses, analyses):
        ann = risk_rules.annotate(c["clause_text"])
        c["factors"] = ann["factors"]
        c["rule_label"] = ann["risk_label"]
        c["annotation_score"] = ann["annotation_score"]
        c["risk_factors"] = ann["risk_factor_names"]
        c["nlp"] = _nlp_view(a) if a else {}
    return clauses


def _finalise_explanations(clauses: List[Dict]) -> None:
    for c in clauses:
        bullets = risk_rules.plain_language_explanation(c["factors"], c["risk_label"], c["clause_type"])
        ob = _obligation_bullet(c.get("nlp") or {})
        if ob and c["risk_label"] != "Low":
            bullets.insert(1 if len(bullets) > 1 else len(bullets), ob)
        if c["risk_label"] != c["rule_label"] and c.get("engine") != ENGINE_DEMO:
            bullets.append(f"Note: the model's label ({c['risk_label']}) differs from the rule-based screening "
                           f"indicators ({c['rule_label']}); consider reading this clause yourself.")
        c["explanation_bullets"] = bullets
        c["short_explanation"] = risk_rules.short_explanation(c["factors"], c["risk_label"])


def analyze_text(text: str, contract_name: str, engine: str, file_size: int = 0, page_count: int = 0,
                 progress: ProgressFn = None) -> Dict:
    if not text or not text.strip():
        raise AnalysisError("The document contains no text to analyse.")
    _progress(progress, 2, "Detecting clauses")
    clauses = clause_segmenter.segment_clauses(text)
    if not clauses:
        raise AnalysisError("No clauses could be detected in this document.")

    _progress(progress, 3, f"Processing {len(clauses)} clauses with spaCy")
    clauses = _apply_rules_and_nlp(clauses)

    _progress(progress, 4, f"Classifying with {ENGINE_LABELS[engine]}")
    smoke = False
    if engine == ENGINE_DEMO:
        for c in clauses:
            c["risk_label"] = c["rule_label"]
            c["risk_score"] = round(min(100.0, 100.0 * c["annotation_score"] / DEMO_SCORE_SCALE), 1)
            c["confidence"] = None
            c["probabilities"] = None
    else:
        try:
            predicted = predict_clauses(clauses, engine)
        except ModelNotFoundError as exc:
            raise AnalysisError(str(exc)) from exc
        except ModelLoadError as exc:
            raise AnalysisError(str(exc)) from exc
        except Exception as exc:
            log.exception("Inference failed")
            raise AnalysisError("Risk classification failed. See logs/app.log for technical details.") from exc
        clauses = predicted
        smoke = bool(getattr(load_predictor(engine), "smoke_test", False))
    for c in clauses:
        c["engine"] = engine
        c["shap"] = None

    _progress(progress, 5, "Generating explanations")
    _finalise_explanations(clauses)
    if engine in (ENGINE_BERT, ENGINE_BASELINE):
        ranked = sorted(clauses, key=lambda c: -c["risk_score"])
        n = len(ranked) if engine == ENGINE_BASELINE else SHAP_PRECOMPUTE_BERT
        for c in ranked[:n]:
            c["shap"] = explainer.explain(engine, c["clause_text"], c["risk_label"])

    counts = Counter(c["risk_label"] for c in clauses)
    score = overall_score([c["risk_score"] for c in clauses])
    result = {
        "contract_name": contract_name,
        "file_size": file_size,
        "page_count": page_count,
        "engine": engine,
        "engine_label": ENGINE_LABELS[engine],
        "is_demo": engine == ENGINE_DEMO,
        "smoke_test": smoke,
        "text": text,
        "clauses": clauses,
        "summary": {
            "total": len(clauses),
            "High": counts.get("High", 0),
            "Medium": counts.get("Medium", 0),
            "Low": counts.get("Low", 0),
            "overall_score": score,
            "overall_label": overall_label(score),
        },
        "observations": document_observations(clauses),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    try:
        from services.nlp_processor import nlp_status

        result["nlp_status"] = nlp_status()
    except Exception:
        result["nlp_status"] = {"message": "spaCy unavailable"}
    return result


def analyze_pdf(pdf_bytes: bytes, filename: str, engine: str, progress: ProgressFn = None) -> Dict:
    _progress(progress, 0, filename)
    if len(pdf_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise AnalysisError(f"The file is larger than {config.MAX_UPLOAD_MB} MB.")
    _progress(progress, 1, "Extracting text with PyMuPDF")
    try:
        doc = extract_text_from_pdf(pdf_bytes)
    except PDFExtractionError as exc:
        raise AnalysisError(str(exc)) from exc
    return analyze_text(doc.text, filename, engine, file_size=len(pdf_bytes), page_count=doc.page_count,
                        progress=progress)


def analyze_demo(progress: ProgressFn = None) -> Dict:
    _progress(progress, 0, mock_data.DEMO_CONTRACT_NAME)
    _progress(progress, 1, "Loading demo contract text")
    return analyze_text(mock_data.DEMO_CONTRACT_TEXT, mock_data.DEMO_CONTRACT_NAME, ENGINE_DEMO,
                        file_size=len(mock_data.DEMO_CONTRACT_TEXT.encode()), page_count=1, progress=progress)


def explain_clause(result: Dict, clause_id: int, max_evals: int = config.SHAP_MAX_EVALS) -> Dict:
    """On-demand SHAP for one clause (stored in the result)."""
    for c in result["clauses"]:
        if c["clause_id"] == clause_id:
            c["shap"] = explainer.explain(result["engine"], c["clause_text"], c["risk_label"], max_evals=max_evals)
            return c["shap"]
    raise AnalysisError(f"Clause {clause_id} not found.")


def available_engines() -> Dict[str, Dict]:
    return engine_status()


CSV_FIELDS = ["contract_id", "clause_id", "clause_type", "clause_text", "risk_label", "risk_score", "confidence",
              "explanation"]


def results_to_csv(result: Dict) -> bytes:
    import csv

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    for c in result["clauses"]:
        w.writerow({
            "contract_id": result["contract_name"],
            "clause_id": c["clause_id"],
            "clause_type": c["clause_type"],
            "clause_text": c["clause_text"],
            "risk_label": c["risk_label"],
            "risk_score": c["risk_score"],
            "confidence": "" if c["confidence"] is None else c["confidence"],
            "explanation": " ".join(c["explanation_bullets"]),
        })
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel on Windows opens UTF-8 correctly
