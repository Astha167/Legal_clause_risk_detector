"""Document Viewer page: full extracted text with risk highlighting."""
from __future__ import annotations

import streamlit as st

from components.cards import esc, risk_badge
from components.clause_card import clause_title
from components.navbar import page_header
from pages.dashboard import no_result_state


def _move(delta: int, n: int) -> None:
    st.session_state["doc_focus"] = min(n, max(1, st.session_state.get("doc_focus", 1) + delta))


def render() -> None:
    result = st.session_state.get("result")
    page_header("Document Viewer", "Extracted contract text with each clause highlighted by screening risk.", result)
    if not result:
        no_result_state()
        return
    clauses = result["clauses"]
    n = len(clauses)
    st.markdown('<div class="lr-legend"><span class="h">High risk</span><span class="m">Medium risk</span>'
                '<span class="l">Low risk</span></div>', unsafe_allow_html=True)

    focus = st.session_state.get("doc_focus", 1)
    focus = focus if 1 <= focus <= n else 1
    nav1, nav2, nav3, _ = st.columns([0.8, 3, 0.8, 1.2])
    nav1.button("Previous", on_click=_move, args=(-1, n), disabled=focus <= 1, width="stretch")
    sel = nav2.selectbox("Go to clause", list(range(1, n + 1)), index=focus - 1, label_visibility="collapsed",
                         format_func=lambda i: f"{clause_title(clauses[i - 1])} - {clauses[i - 1]['risk_label']}")
    nav3.button("Next", on_click=_move, args=(1, n), disabled=focus >= n, width="stretch")
    if sel != focus:
        st.session_state["doc_focus"] = focus = sel

    fc = clauses[focus - 1]
    st.markdown(f'<div class="lr-card compact" style="margin-bottom:1rem"><h4>{esc(clause_title(fc))} &nbsp;'
                f'{risk_badge(fc["risk_label"])}</h4><p>{esc(fc["short_explanation"])} '
                f'<a href="#clause-{focus}">Jump to it in the document</a></p></div>', unsafe_allow_html=True)

    doc_col, toc_col = st.columns([4, 1], gap="medium")
    with toc_col:
        st.markdown("**Clauses**")
        st.markdown('<div class="lr-toc">' + "".join(
            f'<a href="#clause-{c["clause_id"]}">{c["clause_id"]}. {esc(c["clause_type"])[:28]}</a>'
            for c in clauses) + "</div>", unsafe_allow_html=True)
    with doc_col:
        parts = []
        for c in clauses:
            focus_cls = " focus" if c["clause_id"] == focus else ""
            head = f"{c['number']} {c['heading']}".strip() if (c.get("heading") or c.get("number")) else ""
            factors = ", ".join(f["name"] for f in c["factors"] if f["weight"] > 0)[:140]
            margin = (f'<b>{esc(c["risk_label"])}</b> - score {c["risk_score"]:.0f}<br>{esc(c["clause_type"])}'
                      + (f"<br><span>{esc(factors)}</span>" if factors else ""))
            parts.append(
                f'<div class="lr-doc-clause {c["risk_label"]}{focus_cls}" id="clause-{c["clause_id"]}">'
                f'<div>{f"<div class=lr-doc-head>{esc(head)}</div>" if head else ""}'
                f'<div class="lr-doc-text">{esc(c["clause_text"])}</div></div>'
                f'<div class="lr-margin">{margin}</div></div>')
        st.markdown('<div class="lr-doc">' + "".join(parts) + "</div>", unsafe_allow_html=True)
    with st.expander("Raw extracted text"):
        st.text(result["text"][:200000])
