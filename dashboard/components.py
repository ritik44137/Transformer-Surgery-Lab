"""Reusable Streamlit UI fragments matching the dark dashboard reference."""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.metrics import (
    fmt_compact,
    fmt_float,
    fmt_params,
    short_run_name,
    variant_label,
)


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #070b16;
  --bg-elevated: #0d1424;
  --card: #121a2e;
  --card-2: #162036;
  --border: rgba(148, 163, 184, 0.14);
  --text: #f1f5f9;
  --muted: #94a3b8;
  --purple: #8b5cf6;
  --purple-2: #a78bfa;
  --cyan: #22d3ee;
  --green: #34d399;
  --amber: #fbbf24;
  --pink: #f472b6;
}

html, body, [class*="css"] {
  font-family: Inter, Segoe UI, sans-serif;
}

/* Top-align the whole app; do not vertically center content */
.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, rgba(139,92,246,0.18), transparent 55%),
              radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.08), transparent 45%),
              var(--bg);
  color: var(--text);
  align-items: flex-start !important;
  justify-content: flex-start !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main,
.main,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
div[data-testid="stVerticalBlock"],
div[data-testid="column"],
div[data-testid="stHorizontalBlock"] {
  align-items: flex-start !important;
  justify-content: flex-start !important;
}

/* Main content: flush to the top, uniform padding on every edge */
.main .block-container,
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
  padding-top: 0.9rem !important;
  padding-bottom: 1.5rem !important;
  padding-left: 1.25rem !important;
  padding-right: 1.25rem !important;
  /* No max-width cap: it left-aligns the block and dumps the remainder as an
     uneven gutter on the right. Padding alone defines both edges. */
  max-width: none !important;
  margin-top: 0 !important;
}

/* Collapse Streamlit chrome so content sits at the top; reopen control is fixed below */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stToolbar"] {
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
}
header[data-testid="stHeader"],
.stAppHeader {
  background: transparent !important;
  height: 0 !important;
  min-height: 0 !important;
  max-height: 0 !important;
  padding: 0 !important;
  border: none !important;
  visibility: hidden !important;
  overflow: visible !important;
}

/* A style-only markdown block still claims a slot in the vertical flow, adding
   one flex gap above the first visible element. Drop it from layout entirely;
   the <style> rules keep applying from a display:none subtree. */
[data-testid="stElementContainer"]:has(style) {
  display: none !important;
}

/* Anchor links Streamlit injects next to headings */
[data-testid="stHeaderActionElements"],
.stMarkdown h1 > a,
.stMarkdown h2 > a,
.stMarkdown h3 > a,
a.headerlink, a.anchor-link {
  display: none !important;
}

/* The expand button only exists while the sidebar is collapsed; indent the
   content by its width so it never sits on top of the page title. */
.stApp:has([data-testid="stExpandSidebarButton"]) [data-testid="stMainBlockContainer"],
.stApp:has([data-testid="stExpandSidebarButton"]) .block-container {
  padding-left: 4.25rem !important;
  padding-right: 4.25rem !important;
}

/* Explicitly show collapsed-sidebar reopen control (Streamlit versions differ) */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] {
  visibility: visible !important;
  display: flex !important;
  opacity: 1 !important;
  color: var(--text) !important;
  z-index: 999999 !important;
  background: rgba(18, 26, 46, 0.92) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  position: fixed !important;
  top: 0.9rem !important;
  left: 1.25rem !important;
}

/* Sidebar fills the viewport so its background never stops mid-screen */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1020 0%, #0b1222 100%);
  border-right: 1px solid var(--border);
  height: 100vh !important;
  min-height: 100vh !important;
}
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
  background: transparent !important;
  height: 100% !important;
  padding-top: 0 !important;
}
[data-testid="stMain"] {
  padding-top: 0 !important;
}

/* Same top padding as the main column (see .block-container above) */
[data-testid="stSidebarUserContent"] {
  padding-top: 0.9rem !important;
  padding-bottom: 1.5rem !important;
}
/* Float the collapse control instead of letting it reserve vertical space.
   Padding matches the content columns so it lines up with them. */
[data-testid="stSidebarHeader"] {
  position: absolute !important;
  top: 0;
  right: 0;
  left: auto;
  width: auto !important;
  padding: 0.9rem 1rem !important;
  height: auto !important;
  min-height: 0 !important;
  z-index: 5;
}
/* Keep the collapse arrow on screen instead of revealing it on hover */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {
  visibility: visible !important;
  display: flex !important;
  opacity: 1 !important;
  transform: none !important;
  margin: 0 !important;
}
/* Zero the button's own padding so its right edge lands on the header's
   padding edge, level with the widgets below it. */
[data-testid="stSidebarCollapseButton"] button {
  padding: 0 !important;
  width: 2rem !important;
  height: 2rem !important;
  min-width: 0 !important;
  align-items: center !important;
  justify-content: center !important;
}

/* Drag-to-resize handle on the sidebar edge */
[data-testid="stSidebar"] div[style*="col-resize"] {
  display: none !important;
  pointer-events: none !important;
}

/* Run selector pills */
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button {
  border-radius: 999px !important;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.03) !important;
  color: var(--muted) !important;
  font-size: 0.8rem !important;
}
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[kind="pillsActive"],
[data-testid="stSidebar"] [data-testid="stButtonGroup"] button[aria-pressed="true"] {
  background: rgba(139,92,246,0.20) !important;
  border-color: rgba(139,92,246,0.45) !important;
  color: var(--text) !important;
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
  color: var(--text);
}

.brand-row {
  display: flex;
  gap: 0.85rem;
  align-items: center;
  margin: 0 0 1rem 0;
}
.brand-mark {
  width: 44px;
  height: 44px;
  border-radius: 14px;
  background: linear-gradient(135deg, #7c4dff, #c084fc 60%, #22d3ee);
  display: grid;
  place-items: center;
  font-weight: 800;
  font-size: 0.95rem;
  box-shadow: 0 8px 24px rgba(124, 77, 255, 0.35);
  flex-shrink: 0;
}
.brand-title { font-weight: 800; font-size: 1.05rem; letter-spacing: 0.01em; }
.brand-sub { color: var(--muted); font-size: 0.78rem; margin-top: 0.1rem; }

.status-card {
  background: rgba(139, 92, 246, 0.08);
  border: 1px solid rgba(139, 92, 246, 0.25);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  margin-bottom: 1rem;
}
.status-card .label { color: var(--muted); font-size: 0.75rem; }
.status-card .value { font-weight: 700; margin-top: 0.15rem; }
.progress {
  margin-top: 0.65rem;
  height: 8px;
  border-radius: 999px;
  background: rgba(148,163,184,0.15);
  overflow: hidden;
}
.progress > span {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #7c4dff, #f472b6);
}

.page-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0 0 0.15rem 0;
}
.page-title .page-h1 {
  font-size: 1.55rem;
  font-weight: 800;
  margin: 0;
  color: var(--text);
  line-height: 1.2;
}
.accent-bar {
  width: 4px;
  height: 1.45rem;
  border-radius: 999px;
  background: linear-gradient(180deg, #8b5cf6, #22d3ee);
  flex-shrink: 0;
}
.page-sub {
  color: var(--muted);
  margin: 0 0 1rem 0.9rem;
  font-size: 0.92rem;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1rem;
  width: 100%;
}
@media (max-width: 1100px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
.kpi-card {
  background: linear-gradient(180deg, rgba(22,32,54,0.95), rgba(15,23,42,0.9));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1rem 1.05rem;
  position: relative;
  min-height: 108px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  overflow: hidden;
}
.kpi-icon {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  font-size: 0.95rem;
  margin-bottom: 0.7rem;
}
.kpi-label { color: var(--muted); font-size: 0.78rem; font-weight: 500; }
.kpi-value { font-size: 1.45rem; font-weight: 800; color: var(--text); margin-top: 0.15rem; word-break: break-word; }
.kpi-badge {
  position: absolute;
  top: 0.9rem;
  right: 0.9rem;
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.22rem 0.55rem;
  border-radius: 999px;
  max-width: 55%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Inline status pills — never position:absolute (avoids viewport anchoring) */
.status-pill {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.22rem 0.55rem;
  border-radius: 999px;
  flex-shrink: 0;
}
.badge-green { background: rgba(52,211,153,0.15); color: #34d399; }
.badge-purple { background: rgba(139,92,246,0.18); color: #c4b5fd; }
.badge-amber { background: rgba(251,191,36,0.15); color: #fbbf24; }
.badge-cyan { background: rgba(34,211,238,0.15); color: #22d3ee; }

.panel-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 0.15rem 0 0.55rem 0;
}
.panel-title { font-weight: 700; font-size: 1.02rem; color: var(--text); }
.pill {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  background: rgba(52,211,153,0.15);
  color: #34d399;
}

.mini-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.7rem;
  margin: 0.4rem 0 0.2rem 0;
  width: 100%;
}
@media (max-width: 900px) {
  .mini-grid { grid-template-columns: repeat(2, 1fr); }
}
.mini-card {
  border-radius: 14px;
  padding: 0.7rem 0.8rem;
  border: 1px solid transparent;
}
.mini-card .m-label { font-size: 0.72rem; color: var(--muted); }
.mini-card .m-value { font-weight: 700; margin-top: 0.15rem; color: var(--text); }
.tint-purple { background: rgba(139,92,246,0.12); border-color: rgba(139,92,246,0.2); }
.tint-blue { background: rgba(96,165,250,0.12); border-color: rgba(96,165,250,0.2); }
.tint-cyan { background: rgba(34,211,238,0.10); border-color: rgba(34,211,238,0.2); }
.tint-teal { background: rgba(45,212,191,0.10); border-color: rgba(45,212,191,0.2); }

.run-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  width: 100%;
  align-items: start;
}
@media (max-width: 1100px) {
  .run-grid { grid-template-columns: 1fr; }
}
.run-card {
  position: relative;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(0,0,0,0.22);
}
.run-card-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.7rem;
  gap: 0.5rem;
}
.run-icon {
  width: 36px; height: 36px; border-radius: 12px;
  background: rgba(139,92,246,0.18);
  display: grid; place-items: center;
  flex-shrink: 0;
}
.run-name { font-weight: 700; margin-top: 0.55rem; color: var(--text); }
.run-variants { color: var(--muted); font-size: 0.75rem; margin-top: 0.2rem; }
.run-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.55rem;
  margin-top: 0.85rem;
}
.run-stat {
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  padding: 0.55rem 0.65rem;
}
.run-stat .s-label { color: var(--muted); font-size: 0.7rem; }
.run-stat .s-value { font-weight: 700; font-size: 0.92rem; margin-top: 0.1rem; color: var(--text); }

.note {
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.55;
  padding: 0.15rem 0.1rem 0.75rem 0.1rem;
}

div[data-testid="stPlotlyChart"] {
  background: transparent !important;
}

.modebar-container .modebar {
  background: transparent !important;
}
.modebar-btn path { fill: var(--muted) !important; }
.modebar-btn:hover path { fill: var(--text) !important; }

/* Soft card look around Streamlit chart blocks without broken HTML wrappers */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(180deg, rgba(18,26,46,0.98), rgba(13,20,36,0.96));
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_brand(n_runs: int) -> None:
    st.markdown(
        f"""
<div class="brand-row">
  <div class="brand-mark">TSL</div>
  <div>
    <div class="brand-title">Surgery Lab</div>
    <div class="brand-sub">Transformer experiments</div>
  </div>
</div>
<div class="status-card">
  <div class="label">Tracked runs</div>
  <div class="value">{n_runs} available</div>
  <div class="progress"><span style="width:{min(100, max(12, n_runs * 18))}%"></span></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_head(title: str, pill: str) -> None:
    st.markdown(
        f"""
<div class="panel-head">
  <div class="panel-title">{title}</div>
  <div class="pill">{pill}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_model_picker(
    bundles: list[dict[str, Any]],
    colors: list[str],
    *,
    state_key: str = "focus_run",
) -> int:
    """Legend-style run picker: colored dot per run, dimmed unless selected.

    Returns the index of the selected bundle.
    """
    n = len(bundles)
    current = min(int(st.session_state.get(state_key, 0)), n - 1)

    def _entry(i: int) -> None:
        nonlocal current
        if st.button(
            short_run_name(bundles[i].get("run_name")),
            key=f"{state_key}_{i}",
            type="tertiary",
        ):
            current = i
            st.session_state[state_key] = i

    try:
        # Packs the entries side by side instead of spreading them over columns.
        row = st.container(horizontal=True, gap="medium")
    except TypeError:
        row = None

    if row is not None:
        with row:
            for i in range(n):
                _entry(i)
    else:
        for i, col in enumerate(st.columns(n)):
            with col:
                _entry(i)

    # Emitted after the buttons so the click above is reflected in this same run.
    rules = []
    for i in range(n):
        selected = i == current
        rules.append(
            f"""
.st-key-{state_key}_{i} button {{
  display: inline-flex !important;
  width: auto !important;
  align-items: center;
  gap: 0.4rem;
  justify-content: flex-start !important;
  padding: 0.1rem 0 !important;
  min-height: 0 !important;
  border: none !important;
  background: transparent !important;
}}
.st-key-{state_key}_{i} button > div {{
  flex: 0 0 auto !important;
  width: auto !important;
  text-align: left !important;
}}
.st-key-{state_key}_{i} button::before {{
  content: "";
  width: 11px;
  height: 11px;
  border-radius: 999px;
  flex-shrink: 0;
  background: {colors[i]};
  opacity: {1 if selected else 0.4};
}}
.st-key-{state_key}_{i} button p {{
  color: {"var(--text)" if selected else "var(--muted)"} !important;
  font-weight: {700 if selected else 500} !important;
  font-size: 0.85rem !important;
  margin: 0 !important;
}}
.st-key-{state_key}_{i} button:hover p {{ color: var(--text) !important; }}
"""
        )
    st.markdown("<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)
    return current


def render_page_header() -> None:
    st.markdown(
        """
<div class="page-title">
  <div class="accent-bar"></div>
  <div class="page-h1">AI Model Training Dashboard</div>
</div>
<p class="page-sub">Architecture swap tracking — loss, throughput, latency, and parameter tradeoffs</p>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(stats: dict[str, Any]) -> None:
    best_loss = fmt_float(stats.get("best_val_loss"), 3)
    tok = fmt_compact(stats.get("best_tok_s"))
    params = fmt_params(stats.get("total_params"))
    n = stats.get("n_runs", 0)
    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-icon" style="background:rgba(139,92,246,0.18);color:#c4b5fd;">◎</div>
    <div class="kpi-badge badge-green">{stats.get('best_run') or '—'}</div>
    <div class="kpi-label">Best Val Loss</div>
    <div class="kpi-value">{best_loss}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon" style="background:rgba(34,211,238,0.15);color:#22d3ee;">⚡</div>
    <div class="kpi-badge badge-purple">Live</div>
    <div class="kpi-label">Peak Throughput</div>
    <div class="kpi-value">{tok}<span style="font-size:0.85rem;color:#94a3b8"> tok/s</span></div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon" style="background:rgba(251,191,36,0.15);color:#fbbf24;">▣</div>
    <div class="kpi-badge badge-amber">Params</div>
    <div class="kpi-label">Model Size</div>
    <div class="kpi-value" style="font-size:1.25rem">{params}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon" style="background:rgba(52,211,153,0.15);color:#34d399;">⇆</div>
    <div class="kpi-badge badge-green">Selected</div>
    <div class="kpi-label">Compared Runs</div>
    <div class="kpi-value">{n}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_stats(bundles: list[dict[str, Any]]) -> None:
    if not bundles:
        return
    # Aggregate simple training context from first / best run
    steps = []
    losses = []
    finals = []
    for b in bundles:
        s = b.get("summary") or {}
        if s.get("total_steps") is not None:
            steps.append(int(s["total_steps"]))
        if s.get("best_val_loss") is not None:
            losses.append(float(s["best_val_loss"]))
        if s.get("final_train_loss") is not None:
            finals.append(float(s["final_train_loss"]))
    st.markdown(
        f"""
<div class="mini-grid">
  <div class="mini-card tint-purple">
    <div class="m-label">Max Steps</div>
    <div class="m-value">{max(steps) if steps else '—'}</div>
  </div>
  <div class="mini-card tint-blue">
    <div class="m-label">Best Val Loss</div>
    <div class="m-value">{fmt_float(min(losses) if losses else None, 3)}</div>
  </div>
  <div class="mini-card tint-cyan">
    <div class="m-label">Final Train Loss</div>
    <div class="m-value">{fmt_float(min(finals) if finals else None, 3)}</div>
  </div>
  <div class="mini-card tint-teal">
    <div class="m-label">Runs</div>
    <div class="m-value">{len(bundles)}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_run_cards(bundles: list[dict[str, Any]]) -> None:
    if not bundles:
        st.info("No runs selected.")
        return
    cards = []
    for b in bundles:
        row = b.get("comparison") or {}
        variants = b.get("model_variants") or {}
        status = "Ready" if b.get("benchmark") else "No bench"
        badge_class = "badge-green" if b.get("benchmark") else "badge-amber"
        cards.append(
            f"""
<div class="run-card">
  <div class="run-card-top">
    <div class="run-icon">λ</div>
    <div class="status-pill {badge_class}">{status}</div>
  </div>
  <div class="run-name">{short_run_name(b.get('run_name'))}</div>
  <div class="run-variants">{variant_label(variants)}</div>
  <div class="run-stats">
    <div class="run-stat"><div class="s-label">Val loss</div><div class="s-value">{fmt_float(row.get('best_val_loss'), 3)}</div></div>
    <div class="run-stat"><div class="s-label">Params</div><div class="s-value">{fmt_compact(row.get('param_count'))}</div></div>
    <div class="run-stat"><div class="s-label">Tok/s</div><div class="s-value">{fmt_compact(row.get('tokens_per_sec'))}</div></div>
    <div class="run-stat"><div class="s-label">Fwd latency</div><div class="s-value">{fmt_float(row.get('forward_latency_ms'), 2)} ms</div></div>
  </div>
</div>
            """
        )
    st.markdown(
        f"""
<div class="panel-head" style="margin:0.4rem 0 0.8rem 0">
  <div class="panel-title">Active Models</div>
  <div class="pill">{len(bundles)} Selected</div>
</div>
<div class="run-grid">{''.join(cards)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)
