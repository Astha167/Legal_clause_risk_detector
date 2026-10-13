"""Sidebar navigation."""
from __future__ import annotations

import streamlit as st

import config
from components.cards import esc

PAGES = ["Home", "Analyze Contract", "Risk Dashboard", "Clause Analysis", "Document Viewer", "Report"]


def go(page: str) -> None:
    st.session_state["page"] = page


def render_sidebar(engines: dict) -> None:
    with st.sidebar:
        st.markdown(f'<div class="lr-brand">{esc(config.APP_NAME)}</div>'
                    f'<div class="lr-brand-sub">{esc(config.APP_TAGLINE)}</div>', unsafe_allow_html=True)
        current = st.session_state.get("page", "Home")
        for p in PAGES:
            st.button(p, key=f"nav_{p}", type="primary" if p == current else "secondary",
                      on_click=go, args=(p,), width="stretch")

        result = st.session_state.get("result")
        lines = []
        for key, info in engines.items():
            color = "#4CAF7D" if info["available"] else "#7C8699"
            state = "ready" if info["available"] else "not trained yet"
            if info.get("smoke_test"):
                state = "smoke test only"
            lines.append(f'<span class="lr-dot" style="background:{color}"></span>{esc(info["label"])}: {state}')
        if result:
            lines.append(f"<br>Current: {esc(result['contract_name'])[:40]}")
        st.markdown('<div class="lr-side-status">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="lr-side-foot">AI-assisted contract screening<br>'
                    f'{esc(config.APP_NAME)} v{esc(config.APP_VERSION)}</div>', unsafe_allow_html=True)
