"""Page header with an analysis-mode badge."""
from __future__ import annotations

import streamlit as st

from components.cards import esc


def mode_badge(result) -> str:
    if not result:
        return ""
    if result.get("is_demo"):
        return '<span class="lr-mode demo">Demo Mode - rule-based, no ML model</span>'
    if result.get("smoke_test"):
        return '<span class="lr-mode warn">Smoke-test model - results not meaningful</span>'
    return f'<span class="lr-mode model">{esc(result.get("engine_label", ""))}</span>'


def page_header(title: str, subtitle: str = "", result=None) -> None:
    st.markdown(
        f'<div class="lr-page-head"><h1>{esc(title)}{mode_badge(result)}</h1>'
        + (f"<p>{esc(subtitle)}</p>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )
