"""Risk Dashboard page."""
from __future__ import annotations

import streamlit as st

from components import charts
from components.cards import disclaimer, empty_state, esc, metric_card, score_card
from components.clause_card import render_clause_card
from components.navbar import page_header
from components.sidebar import go
import config


def no_result_state() -> None:
    empty_state("No contract analysed yet", "Upload a PDF or run the demo contract to see results here.")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("Analyze Contract", type="primary", on_click=go, args=("Analyze Contract",), key="empty_an")

    def demo():
        st.session_state["run_demo"] = True
        go("Analyze Contract")

    c2.button("Try Demo", on_click=demo, key="empty_demo")


def render() -> None:
    result = st.session_state.get("result")
    page_header("Risk Dashboard", result["contract_name"] if result else "", result)
    if not result:
        no_result_state()
        return
    s = result["summary"]
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        m = st.columns(4)
        with m[0]:
            metric_card("Total clauses", s["total"])
        with m[1]:
            metric_card("High risk", s["High"], "high")
        with m[2]:
            metric_card("Medium risk", s["Medium"], "medium")
        with m[3]:
            metric_card("Low risk", s["Low"], "low")
        st.markdown("#### Screening score by clause")
        st.plotly_chart(charts.clause_score_strip(result["clauses"]), width="stretch",
                        config={"displayModeBar": False})
    with right:
        score_card(s["overall_score"], s["overall_label"],
                   "AI-generated screening score" + (" (Demo Mode)" if result.get("is_demo") else ""))

    a, b = st.columns([1, 1.4], gap="large")
    with a:
        st.markdown("#### Risk distribution")
        st.plotly_chart(charts.risk_donut(s), width="stretch", config={"displayModeBar": False})
    with b:
        st.markdown("#### Risk by clause type")
        st.plotly_chart(charts.risk_by_type(result["clauses"]), width="stretch", config={"displayModeBar": False})

    st.markdown("#### Risk summary")
    if result.get("observations"):
        st.markdown('<ul class="lr-bullets">' + "".join(f"<li>{esc(o)}</li>" for o in result["observations"]) + "</ul>",
                    unsafe_allow_html=True)
    else:
        st.caption("No document-level observations.")

    for label in ["High", "Medium", "Low"]:
        group = sorted([c for c in result["clauses"] if c["risk_label"] == label], key=lambda c: -c["risk_score"])
        with st.expander(f"{label}-Risk Clauses ({len(group)})", expanded=(label == "High")):
            if not group:
                st.caption("None.")
            for c in group:
                render_clause_card(c)
    st.button("Open Clause Analysis", on_click=go, args=("Clause Analysis",))
    disclaimer(config.DISCLAIMER)
