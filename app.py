"""
LegalRisk AI - AI-Powered Legal Clause Risk Detector.

Run from the project root:
    streamlit run app.py

This file only sets up the page, styling and routing. All NLP / ML logic lives in
services/ (see services/api.py) and every page lives in pages/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

import config  # noqa: E402

st.set_page_config(page_title=f"{config.APP_NAME} | {config.APP_TITLE}", page_icon="⚖️", layout="wide",
                   initial_sidebar_state="expanded")

from components.sidebar import render_sidebar  # noqa: E402
from config import get_logger  # noqa: E402
from pages import analyze, clauses, dashboard, document, home, report  # noqa: E402
from services.api import available_engines  # noqa: E402

log = get_logger("app")
ROUTES = {
    "Home": home.render,
    "Analyze Contract": analyze.render,
    "Risk Dashboard": dashboard.render,
    "Clause Analysis": clauses.render,
    "Document Viewer": document.render,
    "Report": report.render,
}


@st.cache_resource(show_spinner=False)
def _announce_device() -> str:
    try:
        return config.get_device().type   # prints "Selected device: CUDA/CPU" once per server start
    except Exception:
        return "cpu"


def _load_css() -> None:
    try:
        css = config.STYLES_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except OSError:
        log.warning("style.css not found; using default styling")


def router() -> None:
    _load_css()
    _announce_device()
    st.session_state.setdefault("page", "Home")
    render_sidebar(available_engines())
    page = st.session_state["page"]
    try:
        ROUTES.get(page, home.render)()
    except Exception:
        log.exception("Unhandled error while rendering page %s", page)
        st.error("This page could not be displayed because of an unexpected error. "
                 "Technical details were written to logs/app.log.")


# st.navigation replaces Streamlit's automatic pages/ sidebar with our own navigation.
st.navigation([st.Page(router, title=config.APP_NAME, default=True)], position="hidden").run()
