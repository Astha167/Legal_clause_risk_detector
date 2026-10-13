"""Plotly charts."""
from __future__ import annotations

from collections import Counter, defaultdict

import plotly.graph_objects as go

import config
from components.cards import RISK_COLORS

FONT = dict(family="IBM Plex Sans, Segoe UI, Arial, sans-serif", color="#1C2536", size=13)


def _layout(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=FONT, showlegend=True,
                      legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
    return fig


def risk_donut(summary: dict):
    labels = config.LABELS[::-1]
    values = [summary.get(l, 0) for l in labels]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, sort=False, direction="clockwise",
                           marker=dict(colors=[RISK_COLORS[l] for l in labels], line=dict(color="#fff", width=2)),
                           textinfo="value", hovertemplate="%{label}: %{value} clauses<extra></extra>"))
    fig.add_annotation(text=f"<b>{summary.get('total', 0)}</b><br>clauses", showarrow=False,
                       font=dict(size=16, family="Source Serif 4, Georgia, serif"))
    return _layout(fig)


def risk_by_type(clauses: list, top_n: int = 12):
    counts = defaultdict(Counter)
    for c in clauses:
        counts[c["clause_type"]][c["risk_label"]] += 1
    order = sorted(counts, key=lambda t: (counts[t]["High"], counts[t]["Medium"], sum(counts[t].values())))[-top_n:]
    fig = go.Figure()
    for label in config.LABELS:
        fig.add_bar(y=order, x=[counts[t][label] for t in order], name=label, orientation="h",
                    marker_color=RISK_COLORS[label], hovertemplate="%{y}: %{x}<extra>" + label + "</extra>")
    fig.update_layout(barmode="stack", xaxis=dict(title="Clauses", gridcolor="#E2E5EB", dtick=1),
                      yaxis=dict(automargin=True))
    return _layout(fig, height=max(260, 28 * len(order) + 80))


def clause_score_strip(clauses: list):
    fig = go.Figure(go.Bar(
        x=[f"C{c['clause_id']}" for c in clauses], y=[c["risk_score"] for c in clauses],
        marker_color=[RISK_COLORS[c["risk_label"]] for c in clauses],
        hovertext=[c["clause_type"] for c in clauses],
        hovertemplate="Clause %{x}<br>%{hovertext}<br>score %{y}<extra></extra>"))
    fig.update_layout(yaxis=dict(range=[0, 100], title="Screening score", gridcolor="#E2E5EB"),
                      xaxis=dict(type="category", title="Clause"), showlegend=False)
    fig = _layout(fig, height=240)
    fig.update_layout(showlegend=False)
    return fig


def term_importance(terms: list, label: str):
    terms = list(reversed(terms))
    fig = go.Figure(go.Bar(
        x=[t["value"] for t in terms], y=[t["term"] for t in terms], orientation="h",
        marker_color=["#B83A33" if t["value"] > 0 else "#7C8FB0" for t in terms],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(xaxis=dict(title=f"Contribution towards {label}", gridcolor="#E2E5EB", zeroline=True,
                                 zerolinecolor="#9AA4B8"), yaxis=dict(automargin=True))
    fig = _layout(fig, height=max(220, 30 * len(terms) + 70))
    fig.update_layout(showlegend=False)
    return fig


def probability_bars(probs: dict):
    labels = config.LABELS
    fig = go.Figure(go.Bar(x=[probs[l] * 100 for l in labels], y=labels, orientation="h",
                           marker_color=[RISK_COLORS[l] for l in labels],
                           text=[f"{probs[l] * 100:.1f}%" for l in labels], textposition="outside",
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, 110], visible=False))
    fig = _layout(fig, height=160)
    fig.update_layout(showlegend=False)
    return fig
