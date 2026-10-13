"""Clause Analysis page: search, filter, sort and inspect clauses."""
from __future__ import annotations

import streamlit as st

from components.clause_card import clause_title, render_clause_card, render_clause_detail
from components.navbar import page_header
from pages.dashboard import no_result_state

ORDER = {"High": 2, "Medium": 1, "Low": 0}


def render() -> None:
    result = st.session_state.get("result")
    page_header("Clause Analysis", "Search, filter and open any clause to see why it was flagged.", result)
    if not result:
        no_result_state()
        return
    clauses = result["clauses"]
    c1, c2, c3 = st.columns([2, 1.6, 1.2])
    query = c1.text_input("Search clauses", placeholder="e.g. terminate, liability, renewal")
    level = c2.radio("Risk level", ["All", "High", "Medium", "Low"], horizontal=True)
    sort = c3.selectbox("Sort by", ["Highest Risk", "Lowest Risk", "Clause Type"])

    items = clauses
    if level != "All":
        items = [c for c in items if c["risk_label"] == level]
    if query.strip():
        q = query.strip().lower()
        items = [c for c in items if q in c["clause_text"].lower() or q in c["clause_type"].lower()
                 or q in (c.get("heading") or "").lower()]
    if sort == "Highest Risk":
        items = sorted(items, key=lambda c: (-ORDER[c["risk_label"]], -c["risk_score"]))
    elif sort == "Lowest Risk":
        items = sorted(items, key=lambda c: (ORDER[c["risk_label"]], c["risk_score"]))
    else:
        items = sorted(items, key=lambda c: (c["clause_type"], c["clause_id"]))

    st.caption(f"{len(items)} of {len(clauses)} clauses")
    if not items:
        st.info("No clauses match the current search and filter.")
        return

    ids = [c["clause_id"] for c in items]
    current = st.session_state.get("selected_clause")
    idx = ids.index(current) if current in ids else 0
    list_col, detail_col = st.columns([1, 1.9], gap="large")
    with list_col:
        choice = st.selectbox("Open clause", ids, index=idx,
                              format_func=lambda i: clause_title(next(c for c in clauses if c["clause_id"] == i)))
        st.session_state["selected_clause"] = choice
        for c in items:
            render_clause_card(c)
    with detail_col:
        clause = next(c for c in clauses if c["clause_id"] == choice)
        render_clause_detail(clause, result)
