"""Model explanation panel: prediction, confidence, probabilities and SHAP terms."""
from __future__ import annotations

import streamlit as st

from components import charts
from components.cards import chips, esc
from services import api


def render_model_explanation(c: dict, result: dict, key_prefix: str = "") -> None:
    engine_label = result.get("engine_label", "")
    is_demo = result.get("is_demo")
    conf = c.get("confidence")
    rows = [("Model", "Rule-based screening (Demo Mode)" if is_demo else engine_label),
            ("Prediction", f"{c['risk_label'].upper()} RISK"),
            ("Confidence", "Not applicable in Demo Mode" if conf is None else f"{conf * 100:.0f}%"),
            ("Screening score", f"{c['risk_score']:.0f} / 100 (AI-generated screening score)")]
    st.markdown('<div class="lr-explain"><div class="lr-kv">' +
                "".join(f"<div>{esc(k)}</div><div><b>{esc(v)}</b></div>" for k, v in rows) +
                "</div></div>", unsafe_allow_html=True)
    if result.get("smoke_test"):
        st.warning("This model was produced by the smoke test (tiny subset, 1 epoch). Its outputs are not "
                   "meaningful - train the full model with: python training/train_model.py")

    if c.get("probabilities"):
        st.caption("Class probabilities from the model")
        st.plotly_chart(charts.probability_bars(c["probabilities"]), width="stretch",
                        config={"displayModeBar": False}, key=f"{key_prefix}prob_{c['clause_id']}")

    st.markdown("**Important terms influencing the prediction**")
    shap = c.get("shap")
    if is_demo:
        terms = [f["matched"].strip()[:48] for f in c.get("factors", []) if f.get("matched") and f["weight"] > 0]
        st.caption("Demo Mode does not run a machine-learning model, so there are no SHAP values. "
                   "The phrases below are the wording matched by the rule-based screening indicators.")
        st.markdown(chips(terms) if terms else "<i>No risk phrases matched.</i>", unsafe_allow_html=True)
        return
    if shap is None:
        st.caption("SHAP is computed on demand for this clause (it can take a little while on a CPU).")
        if st.button("Compute SHAP explanation", key=f"{key_prefix}shap_{c['clause_id']}"):
            with st.spinner("Computing SHAP values ..."):
                api.explain_clause(result, c["clause_id"])
            st.rerun()
        return
    if shap.get("terms"):
        st.plotly_chart(charts.term_importance(shap["terms"], shap.get("target_label") or c["risk_label"]),
                        width="stretch", config={"displayModeBar": False},
                        key=f"{key_prefix}terms_{c['clause_id']}")
    if shap.get("available"):
        st.caption(f"{shap['message']} Method: {shap['method']}. Values show the {shap['value_meaning']}; "
                   "red bars push towards the predicted level, blue bars push away from it. "
                   "SHAP explains the model's behaviour; it does not establish legal validity.")
    else:
        st.info(shap.get("message", "SHAP explanation unavailable."))
        st.caption(f"Values shown: {shap.get('value_meaning', '')}.")
