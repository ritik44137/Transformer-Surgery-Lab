#!/usr/bin/env python3
"""Transformer Surgery Lab — Streamlit experiment dashboard.

Visual language matches references/Dashboard.jpg: dark navy canvas,
purple accents, rounded KPI cards, loss panel, and run tiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    import streamlit as st

    from dashboard.components import (
        inject_css,
        render_brand,
        render_kpi_row,
        render_mini_stats,
        render_model_picker,
        render_note,
        render_page_header,
        render_panel_head,
        render_run_cards,
    )
    from dashboard.loaders import discover_runs, load_dashboard_runs, overview_stats
    from dashboard.metrics import (
        fmt_compact,
        fmt_float,
        fmt_ms,
        fmt_params,
        interpret_selection,
        short_run_name,
    )
    from dashboard.plots import (
        bar_metric_figure,
        loss_curves_figure,
        run_color,
        throughput_vs_params_figure,
        variant_radar_figure,
    )

    st.set_page_config(
        page_title="TSL · Experiment Dashboard",
        page_icon="λ",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    available = discover_runs()
    available_names = [p.name for p in available]

    with st.sidebar:
        render_brand(len(available))
        st.markdown("**Compare runs**")
        # Pills toggle on each click, so no dropdown to open and close.
        selected = st.pills(
            "Runs",
            options=available_names,
            default=available_names[:2],
            selection_mode="multi",
            format_func=short_run_name,
            label_visibility="collapsed",
        )
        selected = selected or []
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    bundles = (
        load_dashboard_runs([_ROOT / "runs" / n for n in selected]) if selected else []
    )
    stats = overview_stats(bundles)

    render_page_header()
    render_kpi_row(stats)
    render_note(interpret_selection(bundles))

    if not available:
        st.warning(
            "No runs found under `runs/`. Train a mini experiment first: "
            "`make train-mini`, then `make benchmark RUN_DIR=runs/baseline_mini_gate`."
        )
        return

    if not bundles:
        st.info("Select at least one run in the sidebar (top-left › if closed).")
        return

    # Mode bar shows on hover only; keeps a visible escape hatch next to double-click reset.
    chart_kw = {
        "width": "stretch",
        "config": {
            "displaylogo": False,
            "doubleClick": "reset+autosize",
            "scrollZoom": False,
            "showTips": False,
            "modeBarButtonsToRemove": [
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "toImage",
            ],
        },
    }

    left, right = st.columns([1.55, 1.0], gap="large")

    with left:
        with st.container(border=True):
            render_panel_head("Training Progress", "Loss")
            mode = st.radio(
                "Curve",
                ["Train loss", "Val loss"],
                horizontal=True,
                label_visibility="collapsed",
            )
            curve_mode = "train" if mode.startswith("Train") else "eval"
            st.plotly_chart(loss_curves_figure(bundles, mode=curve_mode), **chart_kw)
            render_mini_stats(bundles)

    with right:
        with st.container(border=True):
            render_panel_head("Model Metrics", "Profile")
            colors = [run_color(i) for i in range(len(bundles))]
            focus_idx = render_model_picker(bundles, colors)
            focus = bundles[focus_idx]
            st.plotly_chart(
                variant_radar_figure(focus, color=colors[focus_idx]), **chart_kw
            )
            row = focus.get("comparison") or {}
            st.markdown(
                f"""
| Metric | Value |
|---|---|
| Val loss | `{fmt_float(row.get('best_val_loss'), 3)}` |
| Params | `{fmt_params(row.get('param_count'))}` |
| Tok/s | `{fmt_compact(row.get('tokens_per_sec'))}` |
| Fwd ms | `{fmt_ms(row.get('forward_latency_ms'))}` |
                """
            )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.plotly_chart(
                bar_metric_figure(
                    bundles, "tokens_per_sec", title="Throughput", y_title="tok/s"
                ),
                **chart_kw,
            )
    with c2:
        with st.container(border=True):
            st.plotly_chart(
                bar_metric_figure(
                    bundles,
                    "forward_latency_ms",
                    title="Forward latency",
                    y_title="ms",
                ),
                **chart_kw,
            )

    with st.container(border=True):
        st.plotly_chart(throughput_vs_params_figure(bundles), **chart_kw)

    render_run_cards(bundles)


if __name__ == "__main__":
    main()
