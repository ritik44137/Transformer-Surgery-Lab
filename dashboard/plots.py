"""Plotly chart builders tuned for the dark NeuralAI-style dashboard."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from dashboard.loaders import loss_series
from dashboard.metrics import short_run_name

# Palette matching references/Dashboard.jpg
PURPLE = "#8b5cf6"
PURPLE_SOFT = "#a78bfa"
CYAN = "#22d3ee"
GREEN = "#34d399"
AMBER = "#fbbf24"
PINK = "#f472b6"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "rgba(148, 163, 184, 0.12)"
CARD_BG = "rgba(15, 23, 42, 0)"

PALETTE = [PURPLE, CYAN, GREEN, AMBER, PINK, "#60a5fa", "#c084fc"]


def _base_layout(title: str | None = None, height: int = 320) -> dict[str, Any]:
    # Never pass title=None — Plotly/JS can render the literal string "undefined".
    layout: dict[str, Any] = dict(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=MUTED, family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=40, r=20, t=48 if title else 40, b=48),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            x=0,
            xanchor="left",
            font=dict(color=MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(gridcolor=GRID, zeroline=False, color=MUTED, autorange=True),
        yaxis=dict(gridcolor=GRID, zeroline=False, color=MUTED, autorange=True),
        dragmode="zoom",
    )
    if title:
        layout["title"] = dict(text=title, font=dict(color=TEXT, size=14), x=0, xanchor="left")
    else:
        layout["title"] = dict(text="")
    return layout


def loss_curves_figure(bundles: list[dict[str, Any]], *, mode: str = "train") -> go.Figure:
    """Overlay train or eval loss curves for selected runs."""
    fig = go.Figure()
    for i, b in enumerate(bundles):
        series = loss_series(b)
        color = PALETTE[i % len(PALETTE)]
        name = short_run_name(b.get("run_name"))
        if mode == "eval":
            xs, ys = series["eval_steps"], series["eval_loss"]
            label = f"{name} · val"
        else:
            xs, ys = series["train_steps"], series["train_loss"]
            label = f"{name} · train"
        if not xs or not ys:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
                hovertemplate="%{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    layout = _base_layout(height=340)
    layout["uirevision"] = f"loss:{mode}:" + "|".join(
        short_run_name(b.get("run_name")) for b in bundles
    )
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Loss")
    return fig


def bar_metric_figure(
    bundles: list[dict[str, Any]],
    key: str,
    *,
    title: str,
    y_title: str,
) -> go.Figure:
    names: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for i, b in enumerate(bundles):
        row = b.get("comparison") or {}
        bench = b.get("benchmark") or {}
        summary = b.get("summary") or {}
        val = row.get(key)
        if val is None:
            val = bench.get(key, summary.get(key))
        if val is None:
            continue
        names.append(short_run_name(b.get("run_name")))
        values.append(float(val))
        colors.append(PALETTE[i % len(PALETTE)])

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y:.3g}<extra>%{x}</extra>",
        )
    )
    layout = _base_layout(title=title, height=300)
    layout["uirevision"] = f"bar:{key}:" + "|".join(names)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text=y_title)
    return fig


def _padded_axis(values: list[float], pad_frac: float = 0.18) -> dict[str, Any]:
    """Explicit axis range with breathing room; double-click resets back to it."""
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = span * pad_frac if span else (abs(hi) * 0.05 or 1.0)
    return dict(autorange=False, range=[lo - pad, hi + pad])


def throughput_vs_params_figure(bundles: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    points: list[tuple[float, float, str, str]] = []
    for i, b in enumerate(bundles):
        row = b.get("comparison") or {}
        params = row.get("param_count")
        tok = row.get("tokens_per_sec")
        if params is None or tok is None:
            continue
        points.append(
            (
                float(params),
                float(tok),
                short_run_name(b.get("run_name")),
                PALETTE[i % len(PALETTE)],
            )
        )

    # Place labels away from the top edge / legend: highest points get labels below.
    if points:
        max_tok = max(p[1] for p in points)
        for x, y, name, color in points:
            near_top = y >= max_tok * 0.97 if max_tok else False
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    name=name,
                    text=[name],
                    textposition="bottom center" if near_top else "top center",
                    textfont=dict(size=11, color=TEXT),
                    marker=dict(size=16, color=color, line=dict(width=0)),
                    hovertemplate="params=%{x:,.0f}<br>tok/s=%{y:.0f}<extra></extra>",
                )
            )

    layout = _base_layout(title="Throughput vs parameters", height=320)
    layout["uirevision"] = "scatter:" + "|".join(p[2] for p in points)
    if points:
        layout["xaxis"] = {**layout["xaxis"], **_padded_axis([p[0] for p in points])}
        layout["yaxis"] = {**layout["yaxis"], **_padded_axis([p[1] for p in points])}
    # Legend under the plot so it never collides with point labels.
    layout["legend"] = dict(
        orientation="h",
        yanchor="top",
        y=-0.22,
        x=0,
        xanchor="left",
        font=dict(color=MUTED),
        bgcolor="rgba(0,0,0,0)",
    )
    layout["margin"] = dict(l=40, r=20, t=48, b=72)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Parameters")
    fig.update_yaxes(title_text="Tokens / sec")
    return fig


def run_color(index: int) -> str:
    """Palette entry a run keeps across every chart and the model picker."""
    return PALETTE[index % len(PALETTE)]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def variant_radar_figure(bundle: dict[str, Any], *, color: str = PURPLE) -> go.Figure:
    """Simple normalized radar for one run (relative visual, not absolute scores)."""
    row = bundle.get("comparison") or {}
    # Normalize rough desirability proxies for display only.
    loss = row.get("best_val_loss")
    tok = row.get("tokens_per_sec")
    lat = row.get("forward_latency_ms")
    params = row.get("param_count")

    def _score_lower(v: float | None, scale: float) -> float:
        if v is None:
            return 0.3
        return max(0.05, min(1.0, 1.0 - (float(v) / scale)))

    def _score_higher(v: float | None, scale: float) -> float:
        if v is None:
            return 0.3
        return max(0.05, min(1.0, float(v) / scale))

    categories = ["Quality", "Speed", "Latency", "Compact"]
    values = [
        _score_lower(loss, 80.0),
        _score_higher(tok, 250_000.0),
        _score_lower(lat, 20.0),
        _score_lower(params, 200_000.0),
    ]
    values = values + values[:1]
    categories = categories + categories[:1]

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor=_rgba(color, 0.25),
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
            name=short_run_name(bundle.get("run_name")),
        )
    )
    layout = _base_layout(height=260)
    layout.update(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID, showticklabels=False),
            angularaxis=dict(gridcolor=GRID, color=MUTED),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    fig.update_layout(**layout)
    return fig
