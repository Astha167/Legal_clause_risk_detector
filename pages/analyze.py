"""Analyze Contract page: upload, processing screen, completion."""
from __future__ import annotations

import streamlit as st

import config
from components.cards import esc
from components.navbar import page_header
from components.sidebar import go
from config import get_logger
from services import api
from services.predictor import ENGINE_BASELINE, ENGINE_BERT

log = get_logger("ui.analyze")


def _steps_html(done: int, active: int | None, detail: str = "") -> str:
    rows = []
    for i, name in enumerate(api.STEPS):
        if i < done:
            cls, mark = "done", "&#10003;"
        elif i == active:
            cls, mark = "active", ""
        else:
            cls, mark = "", ""
        d = f'<span class="lr-step-detail">{esc(detail)}</span>' if i == active and detail else ""
        rows.append(f'<div class="lr-step {cls}"><span class="lr-step-mark">{mark}</span>{esc(name)}{d}</div>')
    return '<div class="lr-steps">' + "".join(rows) + "</div>"


def _run(kind: str, engine: str | None = None, data: bytes | None = None, name: str = "") -> None:
    box = st.empty()
    box.markdown(_steps_html(0, 0), unsafe_allow_html=True)

    def progress(i: int, detail: str) -> None:
        box.markdown(_steps_html(i, i, detail), unsafe_allow_html=True)

    try:
        with st.spinner("Analysing contract ..."):
            if kind == "demo":
                result = api.analyze_demo(progress)
            else:
                result = api.analyze_pdf(data, name, engine, progress)
    except api.AnalysisError as exc:
        box.empty()
        st.error(str(exc))
        return
    except Exception:
        log.exception("Unexpected analysis failure")
        box.empty()
        st.error("Something went wrong during the analysis. Technical details were written to logs/app.log.")
        return
    box.markdown(_steps_html(len(api.STEPS), None), unsafe_allow_html=True)
    st.session_state["result"] = result
    st.session_state["selected_clause"] = None
    st.session_state["doc_focus"] = 1
    st.session_state["analysis_done"] = True


def render() -> None:
    page_header("Analyze Your Contract", "Upload a text-based PDF. Clauses are extracted, classified and explained.")
    engines = api.available_engines()
    options = [e for e in (ENGINE_BERT, ENGINE_BASELINE) if engines[e]["available"]]

    if not engines[ENGINE_BERT]["available"]:
        if options:
            # Expected right after unzipping: the baseline is ready, Contracts-BERT still has to be trained.
            st.info(
                "Contracts-BERT has not been trained on this computer yet, so the included "
                "TF-IDF + Logistic Regression baseline is selected and ready to use. "
                "To add Contracts-BERT, run `python training/train_model.py` (GPU) or "
                "`python training/train_model.py --cpu-fast` (CPU), then restart the app. "
                "Run `python check_setup.py` to see your setup status."
            )
        else:
            st.warning(config.MODEL_MISSING_MESSAGE)
    elif engines[ENGINE_BERT].get("smoke_test"):
        st.warning("The installed Contracts-BERT model is a smoke-test model and its predictions are not meaningful.")

    engine = None
    if options:
        engine = st.radio("Analysis model", options, format_func=lambda e: engines[e]["label"], horizontal=True,
                          help="Contracts-BERT is the primary model. The baseline is a lighter classical model.")

    upload = st.file_uploader("Drop a contract PDF here", type=["pdf"], accept_multiple_files=False,
                              help=f"Text-based PDFs up to {config.MAX_UPLOAD_MB} MB. Scanned images are not supported.")
    if upload is not None:
        size_kb = upload.size / 1024
        st.markdown(f'<div class="lr-card compact" style="padding:.7rem 1rem"><b>{esc(upload.name)}</b>'
                    f'<p>{size_kb:,.1f} KB &nbsp;|&nbsp; Uploaded and ready for analysis</p></div>',
                    unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 2.2])
    analyze_clicked = c1.button("Analyze Contract", type="primary", width="stretch",
                                disabled=upload is None or engine is None)
    demo_clicked = c2.button("Try Demo Contract", width="stretch")
    if upload is not None and engine is None:
        st.info("No trained model is available yet, so only Demo Mode can run. See the README for training steps.")

    if st.session_state.pop("run_demo", False):
        demo_clicked = True
    if analyze_clicked and upload is not None and engine:
        st.session_state["analysis_done"] = False
        _run("pdf", engine, upload.getvalue(), upload.name)
    elif demo_clicked:
        st.session_state["analysis_done"] = False
        _run("demo")

    result = st.session_state.get("result")
    if st.session_state.get("analysis_done") and result:
        s = result["summary"]
        st.success(f"Analysis Complete - {s['total']} clauses analysed in '{result['contract_name']}'"
                   + (" (Demo Mode)" if result.get("is_demo") else ""))
        st.button("View Risk Dashboard", type="primary", on_click=go, args=("Risk Dashboard",))

    with st.expander("Privacy & Security"):
        st.write(config.PRIVACY_NOTICE)
