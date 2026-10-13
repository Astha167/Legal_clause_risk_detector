"""Small HTML building blocks (badges, metric cards, score card, feature cards)."""
from __future__ import annotations

import html

import streamlit as st

RISK_COLORS = {"High": "#B83A33", "Medium": "#B7790B", "Low": "#2E7D55"}


def esc(text) -> str:
    return html.escape("" if text is None else str(text))


def risk_badge(label: str) -> str:
    label = label if label in RISK_COLORS else "Low"
    return f'<span class="lr-badge {label}">{esc(label.upper())} RISK</span>'


def metric_card(label: str, value, tone: str = "neutral") -> None:
    st.markdown(
        f'<div class="lr-metric {tone}"><div class="lr-metric-label">{esc(label)}</div>'
        f'<div class="lr-metric-value">{esc(value)}</div></div>',
        unsafe_allow_html=True,
    )


def score_card(score: float, level: str, foot: str = "AI-generated screening score") -> None:
    color = RISK_COLORS.get(level, "#2F4B7C")
    st.markdown(
        f'<div class="lr-score"><div class="lr-metric-label" style="color:#5A6477;font-size:.82rem">'
        f'Overall risk score</div><div class="lr-score-num">{score:.0f}<span> / 100</span></div>'
        f'<div class="lr-score-level" style="color:{color}">{esc(level.upper())} RISK</div>'
        f'<div class="lr-meter"><div style="width:{max(2, min(100, score))}%;background:{color}"></div></div>'
        f'<div class="lr-score-foot">{esc(foot)}</div></div>',
        unsafe_allow_html=True,
    )


def feature_card(icon: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="lr-card"><div class="lr-feature-icon">{icon}</div><h4>{esc(title)}</h4>'
        f'<p>{esc(body)}</p></div>',
        unsafe_allow_html=True,
    )


def chips(items, protect: bool = False) -> str:
    cls = "lr-chip protect" if protect else "lr-chip"
    return "".join(f'<span class="{cls}">{esc(i)}</span>' for i in items)


def disclaimer(text: str) -> None:
    st.markdown(f'<div class="lr-disclaimer">{esc(text)}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="lr-empty"><h3>{esc(title)}</h3><p>{esc(body)}</p></div>', unsafe_allow_html=True)
