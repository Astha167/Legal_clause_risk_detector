"""Home page."""
from __future__ import annotations

import streamlit as st

import config
from components.cards import disclaimer, feature_card
from components.sidebar import go


def _start_demo():
    st.session_state["run_demo"] = True
    go("Analyze Contract")


def render() -> None:
    st.markdown(
        """
<div class="lr-hero">
  <div>
    <h1>Understand Your Contract's Risk</h1>
    <p class="lr-lede">AI-powered clause analysis that identifies potentially risky contractual terms and explains them in simple language.</p>
  </div>
  <div class="lr-specimen" aria-label="Example of an analysed clause">
    <div class="lr-spec-label">Clause 4: Non-Compete</div>
    <div class="lr-spec-text">During the term of this Agreement and <span class="lr-mark">for a period of two (2) years following its termination</span>, the Consultant <span class="lr-mark">shall not, directly or indirectly, engage in any business that competes</span> with the Company anywhere in the world.</div>
    <div class="lr-note"><div class="lr-note-bar"></div><div><b>High risk</b> - may restrict future business activities for two years after the agreement ends, across a very wide geographic area.</div></div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("Analyze Contract", type="primary", on_click=go, args=("Analyze Contract",), width="stretch")
    c2.button("Try Demo", on_click=_start_demo, width="stretch")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    features = [
        ("&#9888;", "Clause Risk Detection",
         "Each clause is segmented from your PDF and classified as low, medium or high screening risk."),
        ("&#9673;", "Explainable AI",
         "SHAP shows which words pushed the model towards its prediction, so the result is not a black box."),
        ("&#9998;", "Simple-Language Explanations",
         "Risk indicators such as uncapped liability or automatic renewal are explained in plain English."),
        ("&#9636;", "Contract Risk Overview",
         "A dashboard, highlighted document view and downloadable report summarise the whole agreement."),
    ]
    for col, (icon, title, body) in zip(cols, features):
        with col:
            feature_card(icon, title, body)

    st.markdown("### How an analysis works")
    steps = ["Upload a text-based PDF contract", "Text is extracted and split into clauses",
             "spaCy analyses each clause; Contracts-BERT predicts its risk level",
             "Rules and SHAP explain the result; you review the dashboard and report"]
    st.markdown('<ol class="lr-bullets" style="max-width:70ch">' + "".join(f"<li>{s}</li>" for s in steps) + "</ol>",
                unsafe_allow_html=True)
    disclaimer(config.DISCLAIMER)
    with st.expander("Privacy & Security"):
        st.write(config.PRIVACY_NOTICE)
