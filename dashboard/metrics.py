"""Display formatting and derived metric helpers for the dashboard."""

from __future__ import annotations

from typing import Any, Mapping


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def fmt_compact(value: float | int | None) -> str:
    """Compact number formatting (1.2K, 3.4M)."""
    if value is None:
        return "—"
    n = float(value)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{sign}{n / 1_000:.1f}K"
    if n >= 100:
        return f"{sign}{n:.0f}"
    if n >= 10:
        return f"{sign}{n:.1f}"
    return f"{sign}{n:.2f}"


def fmt_params(value: int | float | None) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f} ms"


def fmt_pct_delta(current: float | None, baseline: float | None, *, lower_is_better: bool = True) -> str:
    if current is None or baseline is None or baseline == 0:
        return ""
    delta = (current - baseline) / abs(baseline) * 100.0
    if lower_is_better:
        improved = delta < 0
    else:
        improved = delta > 0
    arrow = "↓" if delta < 0 else "↑"
    sign = "+" if delta > 0 else ""
    label = "better" if improved else "worse"
    return f"{arrow}{sign}{delta:.1f}% {label}"


def variant_label(variants: Mapping[str, Any] | None) -> str:
    if not variants:
        return "unknown"
    parts = [
        str(variants.get("norm", "?")),
        str(variants.get("positional", "?")),
        str(variants.get("feedforward", "?")),
        str(variants.get("attention", "?")),
    ]
    return " · ".join(parts)


def short_run_name(name: str | None) -> str:
    if not name:
        return "run"
    return name.replace("_", " ")


def interpret_selection(bundles: list[dict[str, Any]]) -> str:
    """One-paragraph interpretation for the selected runs (HTML-safe)."""
    if not bundles:
        return "Select one or more runs in the sidebar to compare architecture tradeoffs."
    if len(bundles) == 1:
        b = bundles[0]
        v = b.get("model_variants") or {}
        return (
            f"<strong>{b.get('run_name')}</strong> uses {variant_label(v)}. "
            "Add another run to see single-axis or modern-stack tradeoffs."
        )

    rows = [b.get("comparison") or {} for b in bundles]
    with_loss = [r for r in rows if r.get("best_val_loss") is not None]
    with_tok = [r for r in rows if r.get("tokens_per_sec") is not None]

    bits: list[str] = [f"Comparing <strong>{len(bundles)}</strong> runs."]
    if with_loss:
        best = min(with_loss, key=lambda r: float(r["best_val_loss"]))
        bits.append(
            f"Lowest val loss: <strong>{best.get('run_name')}</strong> "
            f"({fmt_float(best.get('best_val_loss'), 3)})."
        )
    if with_tok:
        fastest = max(with_tok, key=lambda r: float(r["tokens_per_sec"]))
        bits.append(
            f"Highest throughput: <strong>{fastest.get('run_name')}</strong> "
            f"({fmt_compact(fastest.get('tokens_per_sec'))} tok/s)."
        )
    bits.append(
        "Keep dataset, tokenizer, and train budget fixed so differences reflect architecture only."
    )
    return " ".join(bits)
