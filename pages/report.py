"""Report page: PDF report and CSV export."""
from __future__ import annotations

import re

import streamlit as st

import config
from components.cards import disclaimer, esc
from components.navbar import page_header
from config import get_logger
from pages.dashboard import no_result_state
from reports.report_generator import ReportError, generate_pdf_report
from services import api

log = get_logger("ui.report")


def _base_name(name: str) -> str:
    stem = re.sub(r"\.pdf$", "", name, flags=re.I)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:60] or "contract"


def render() -> None:
    result = st.session_state.get("result")
    page_header("Report", "Download a professional PDF report or the clause-level results as CSV.", result)
    if not result:
        no_result_state()
        return
    s = result["summary"]
    st.markdown(
        f'<div class="lr-card compact"><h4>{esc(result["contract_name"])}</h4>'
        f'<p>{s["total"]} clauses: {s["High"]} high, {s["Medium"]} medium, {s["Low"]} low risk. '
        f'Overall screening score {s["overall_score"]:.0f} / 100 ({s["overall_label"]}). '
        f'Engine: {esc(result["engine_label"])}.</p></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    st.markdown("The PDF report contains: contract information, analysis summary, risk distribution, overall "
                "screening score, high-, medium- and low-risk clauses, risk factors, model explanations and the "
                "disclaimer.")

    base = _base_name(result["contract_name"])
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("Generate PDF report", type="primary", width="stretch"):
            try:
                with st.spinner("Building report ..."):
                    st.session_state["report_pdf"] = (id(result), generate_pdf_report(result))
            except ReportError as exc:
                st.error(str(exc))
        cached = st.session_state.get("report_pdf")
        if cached and cached[0] == id(result):
            st.download_button("Download PDF", cached[1], file_name=f"{base}_risk_report.pdf",
                               mime="application/pdf", width="stretch")
    with c2:
        try:
            csv_bytes = api.results_to_csv(result)
            st.download_button("Download CSV", csv_bytes, file_name=f"{base}_clauses.csv", mime="text/csv",
                               width="stretch")
        except Exception:
            log.exception("CSV export failed")
            st.error("The CSV export could not be created.")
    unshap = sum(1 for c in result["clauses"] if c.get("shap") is None)
    if not result.get("is_demo") and unshap:
        st.caption(f"{unshap} clause(s) have no SHAP explanation yet; compute them on the Clause Analysis page to "
                   "include them in the PDF.")
    disclaimer(config.DISCLAIMER)
