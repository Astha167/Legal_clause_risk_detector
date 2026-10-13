"""Clause list card and full clause detail view."""
from __future__ import annotations

import streamlit as st

from components.cards import chips, esc, risk_badge
from components.explanation import render_model_explanation


def clause_title(c: dict) -> str:
    head = c.get("heading") or ""
    num = f"Clause {c['clause_id']}"
    return f"{num}: {c['clause_type']}" + (f" ({head})" if head and head.lower() != c["clause_type"].lower() else "")


def clause_card_html(c: dict, excerpt_chars: int = 230) -> str:
    text = c["clause_text"]
    excerpt = text if len(text) <= excerpt_chars else text[:excerpt_chars].rsplit(" ", 1)[0] + " ..."
    conf = "n/a (demo)" if c.get("confidence") is None else f"{c['confidence'] * 100:.0f}%"
    return (
        f'<div class="lr-clause {c["risk_label"]}"><div class="lr-clause-top">'
        f'<div class="lr-clause-title">{esc(clause_title(c))}</div>{risk_badge(c["risk_label"])}</div>'
        f'<div class="lr-clause-meta">Screening score {c["risk_score"]:.0f} / 100 &nbsp;|&nbsp; Confidence {conf}</div>'
        f'<div class="lr-clause-excerpt">{esc(excerpt)}</div>'
        f'<div class="lr-clause-why">{esc(c.get("short_explanation", ""))}</div></div>'
    )


def render_clause_card(c: dict) -> None:
    st.markdown(clause_card_html(c), unsafe_allow_html=True)


def render_clause_detail(c: dict, result: dict, key_prefix: str = "detail_") -> None:
    st.markdown(f"### {esc(clause_title(c))}", unsafe_allow_html=True)
    st.markdown(risk_badge(c["risk_label"]), unsafe_allow_html=True)
    if True:  # single-column layout keeps the model panel readable inside the detail pane
        st.markdown("**Original clause text**")
        st.markdown(f'<div class="lr-clause-full">{esc(c["clause_text"])}</div>', unsafe_allow_html=True)

        risky = [f for f in c.get("factors", []) if f["weight"] > 0]
        protective = [f for f in c.get("factors", []) if f["weight"] < 0]
        st.markdown("**Risk factors**")
        if risky:
            st.markdown(chips(f["name"] for f in risky), unsafe_allow_html=True)
        else:
            st.caption("No rule-based risk indicators matched this wording.")
        if protective:
            st.markdown("**Balancing factors**")
            st.markdown(chips((f["name"] for f in protective), protect=True), unsafe_allow_html=True)

        st.markdown("**Plain-English explanation**")
        st.markdown('<ul class="lr-bullets">' + "".join(f"<li>{esc(b)}</li>" for b in c["explanation_bullets"]) +
                    "</ul>", unsafe_allow_html=True)
        st.caption("Screening explanation based on project-defined indicators - not legal advice.")
        st.markdown("#### Model explanation")
        render_model_explanation(c, result, key_prefix=key_prefix)

    nlp = c.get("nlp") or {}
    if nlp:
        with st.expander("Linguistic analysis (spaCy + NLTK)"):
            a, b = st.columns(2)
            with a:
                st.markdown("**Named entities**")
                ents = [f"{e['text']} ({e['label']})" for e in nlp.get("entities") or []]
                st.markdown(chips(ents) if ents else "<i>None detected</i>", unsafe_allow_html=True)
                st.markdown("**Obligations / prohibitions**")
                obs = [f"{o['party']}: {'not ' if o['negated'] else ''}{o['modal']} {o['verb']}"
                       for o in nlp.get("obligations") or []]
                st.markdown(chips(obs) if obs else "<i>None detected</i>", unsafe_allow_html=True)
                st.markdown("**Negations kept**")
                st.markdown(chips(nlp.get("negations") or []) or "<i>None</i>", unsafe_allow_html=True)
            with b:
                st.markdown("**Key phrases (noun chunks)**")
                st.markdown(chips(nlp.get("keywords") or []) or "<i>None</i>", unsafe_allow_html=True)
                st.markdown("**Frequent bigrams (NLTK)**")
                st.markdown(chips(nlp.get("top_bigrams") or []) or "<i>None</i>", unsafe_allow_html=True)
                st.markdown("**Part-of-speech counts**")
                pos = [f"{k}: {v}" for k, v in (nlp.get("pos_counts") or {}).items()]
                st.markdown(chips(pos) or "<i>None</i>", unsafe_allow_html=True)
            st.markdown("**Processed text** (lemmatised, legal negations and modal verbs preserved)")
            st.code(nlp.get("processed_text", ""), language=None, wrap_lines=True)
