"""Phase 4: daily-signal dashboard for the validated core (QQQ/GLD rotation).

Computes the core signal from the latest daily bars, paper-tracks a model
portfolio that assumes every signal was followed, and renders a single
self-contained HTML dashboard (hand-rolled SVG charts, zero new
dependencies) that Vin opens locally.

The model portfolio is *not* Vin's brokerage account: he places the trades
himself. The dashboard tells him what to do; execution stays manual.
"""
from __future__ import annotations

import html
from datetime import date
from pathlib import Path

import pandas as pd

from .data import get_daily_bars
from .engine import Costs, backtest
from .indicators import sma
from .strategies.core import core_rotation_signal


def summarize_core_signal(
    prices: dict[str, pd.DataFrame],
    risky: str = "QQQ",
    safe: str = "GLD",
    ma: int = 200,
    watch_pct: float = 0.02,
) -> dict:
    """Current core signal from the latest available bar.

    Returns the model position decided at the last close, the action for the
    next open (matching the backtest's no-lookahead convention: signal at
    close t, trade at open t+1), QQQ's distance from its MA, and a
    rotation-watch flag when a flip is near.
    """
    signals = core_rotation_signal(prices, risky=risky, safe=safe, ma=ma)
    pos = signals.idxmax(axis=1)
    asof = signals.index[-1]
    current = str(pos.iloc[-1])
    previous = str(pos.iloc[-2])

    changes = pos[pos != pos.shift(1)]
    last_rotation_date = changes.index[-1]
    last_rotation_from = str(pos.shift(1).loc[last_rotation_date])

    close = prices[risky]["Close"]
    ma_line = sma(close, ma)
    distance = float((close.iloc[-1] - ma_line.iloc[-1]) / ma_line.iloc[-1])

    return {
        "asof": asof,
        "position": current,
        "previous_position": previous,
        "action": "HOLD" if current == previous else "ROTATE",
        "rotate_from": previous if current != previous else None,
        "rotate_to": current if current != previous else None,
        "ma_distance_pct": distance * 100.0,
        "qqq_close": float(close.iloc[-1]),
        "ma_value": float(ma_line.iloc[-1]),
        "rotation_watch": abs(distance) < watch_pct,
        "last_rotation_date": last_rotation_date,
        "last_rotation_from": last_rotation_from,
        "last_rotation_to": str(pos.loc[last_rotation_date]),
    }


def rotation_history(signals: pd.DataFrame, equity: pd.Series) -> pd.DataFrame:
    """Every rotation the model portfolio made: date, from -> to, days held,
    and the holding-period return from the engine's own equity series."""
    pos = signals.idxmax(axis=1)
    change_idx = pos[pos != pos.shift(1)].index
    rows = []
    for i in range(1, len(change_idx)):
        start, end = change_idx[i - 1], change_idx[i]
        days_held = int((pos.index.get_loc(end) - pos.index.get_loc(start)))
        ret = float(equity.loc[end] / equity.loc[start] - 1.0)
        rows.append(
            {
                "date": end,
                "from": str(pos.loc[start]),
                "to": str(pos.loc[end]),
                "days_held": days_held,
                "holding_return": ret,
            }
        )
    return pd.DataFrame(rows, columns=["date", "from", "to", "days_held", "holding_return"])


def _describe_rotation_row(r) -> str:
    return (
        f"on {r['date'].date()} it sold {r['from']} and bought {r['to']}, "
        f"after holding {r['from']} for {int(r['days_held'])} trading days "
        f"({r['holding_return']:+.1%})"
    )


def rotation_reading_note(rotations: pd.DataFrame) -> str:
    """Plain-language note explaining how to read the rotation table.

    The table is newest-first, so the note walks through the current top
    rows. "Days held" / "Holding return" always describe the leg that just
    ended (the FROM side), counted in trading days.
    """
    newest = rotations.iloc[::-1].reset_index(drop=True)
    if len(newest) == 0:
        return "No rotations yet — the model has held its initial position since inception."
    note = (
        "How to read this table (newest first): the top row says "
        + _describe_rotation_row(newest.iloc[0])
        + '. "Days held" and "Holding return" always describe the leg that just '
        "ended — the FROM side — counted in trading days, not calendar days."
    )
    if len(newest) > 1:
        note += (
            " The next row reads the same way: "
            + _describe_rotation_row(newest.iloc[1])
            + "."
        )
    return note


def current_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float(1.0 - equity.iloc[-1] / running_max.iloc[-1])


# ---------------------------------------------------------------------------
# Hand-rolled SVG charts (no new dependencies).
# ---------------------------------------------------------------------------

_PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed"]


def _svg_line_chart(
    series: list[tuple[str, pd.Series]],
    *,
    title: str,
    width: int = 780,
    height: int = 300,
    markers: list[pd.Timestamp] | None = None,
    hlines: list[tuple[float, str, str]] | None = None,
    fill_negative: bool = False,
    fmt_y=lambda v: f"{v:,.0f}",
) -> str:
    """Minimal multi-series time-series SVG. All series share the date index
    space; each point maps by position in its own index."""
    pad_l, pad_r, pad_t, pad_b = 64, 16, 28, 30
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b

    all_vals = [v for _, s in series for v in s.values]
    for _, _, hv in [(None, None, h[0]) for h in (hlines or [])]:
        all_vals.append(hv)
    lo, hi = min(all_vals), max(all_vals)
    span = hi - lo or 1.0
    lo -= span * 0.06
    hi += span * 0.06

    # Shared x domain: union of all dates.
    x0 = min(s.index.min() for _, s in series)
    x1 = max(s.index.max() for _, s in series)
    total_days = max((x1 - x0).days, 1)

    def x(d: pd.Timestamp) -> float:
        return pad_l + (d - x0).days / total_days * iw

    def y(v: float) -> float:
        return pad_t + (1.0 - (v - lo) / (hi - lo)) * ih

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img">']
    parts.append(f'<text x="{pad_l}" y="18" font-size="14" font-weight="600" fill="#111">{html.escape(title)}</text>')

    # Gridlines + y labels.
    for i in range(6):
        v = lo + (hi - lo) * i / 5
        yy = y(v)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" font-size="10" text-anchor="end" fill="#6b7280">{fmt_y(v)}</text>')

    # X labels: ~6 evenly spaced dates.
    for i in range(7):
        d = x0 + (x1 - x0) * i / 6
        xx = x(d)
        label = d.strftime("%Y-%m")
        parts.append(f'<text x="{xx:.1f}" y="{height - 8}" font-size="10" text-anchor="middle" fill="#6b7280">{label}</text>')

    for (label, s), color in zip(series, _PALETTE):
        pts = " ".join(f"{x(d):.1f},{y(float(v)):.1f}" for d, v in s.items())
        if fill_negative and (s < 0).any():
            zero_y = y(0.0)
            neg = s[s < 0]
            if len(neg):
                area = f"{x(neg.index[0]):.1f},{zero_y:.1f} " + " ".join(
                    f"{x(d):.1f},{y(float(v)):.1f}" for d, v in neg.items()
                ) + f" {x(neg.index[-1]):.1f},{zero_y:.1f}"
                parts.append(f'<polygon points="{area}" fill="#fecaca" opacity="0.7"/>')
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        )

    for hv, hcolor, hlabel in hlines or []:
        yy = y(hv)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" '
            f'stroke="{hcolor}" stroke-dasharray="6,4" stroke-width="1.5"/>'
        )
        parts.append(f'<text x="{width - pad_r}" y="{yy - 6:.1f}" font-size="10" text-anchor="end" fill="{hcolor}">{html.escape(hlabel)}</text>')

    for m in markers or []:
        if x0 <= m <= x1:
            xx = x(m)
            parts.append(f'<line x1="{xx:.1f}" y1="{pad_t}" x2="{xx:.1f}" y2="{pad_t + ih}" stroke="#9ca3af" stroke-dasharray="2,3"/>')
            parts.append(f'<circle cx="{xx:.1f}" cy="{pad_t}" r="4" fill="#f59e0b"/>')

    # Legend.
    lx = pad_l
    for (label, _), color in zip(series, _PALETTE):
        parts.append(f'<rect x="{lx}" y="{height - 24}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx + 14}" y="{height - 15}" font-size="11" fill="#374151">{html.escape(label)}</text>')
        lx += 14 + len(label) * 6.5 + 18

    parts.append("</svg>")
    return "".join(parts)


def render_dashboard_html(
    *,
    summary: dict,
    equity: pd.Series,
    rotations: pd.DataFrame,
    metrics: dict,
    drawdown_cap: float,
    asof_label: str,
) -> str:
    """Assemble the full self-contained dashboard page."""
    cur_dd = current_drawdown(equity)
    running_max = equity.cummax()
    dd_series = (equity / running_max - 1.0) * 100.0

    action = summary["action"]
    action_class = "hold" if action == "HOLD" else "rotate"
    if action == "HOLD":
        action_text = f"HOLD {summary['position']}"
        action_detail = (
            f"No trade at tomorrow's open. Model position: 100% {summary['position']} "
            f"since {summary['last_rotation_date'].date()}."
        )
    else:
        action_text = f"ROTATE {summary['rotate_from']} → {summary['rotate_to']}"
        action_detail = (
            f"At tomorrow's open: sell {summary['rotate_from']}, buy {summary['rotate_to']}. "
            f"Model cost for the rotation ≈ 6 bps of traded notional."
        )

    dist = summary["ma_distance_pct"]
    above = dist >= 0
    ma_line_text = (
        f"QQQ closed at ${summary['qqq_close']:,.2f}, "
        f"{abs(dist):.2f}% {'above' if above else 'below'} its 200-day MA "
        f"(${summary['ma_value']:,.2f})."
    )
    watch_text = (
        " <strong>Rotation watch:</strong> within 2% of the MA — a flip may be near."
        if summary["rotation_watch"]
        else ""
    )

    equity_svg = _svg_line_chart(
        [("Model equity ($)", equity)],
        title="Model portfolio equity (assumes every signal followed)",
        markers=list(rotations["date"]),
        fmt_y=lambda v: f"${v:,.0f}",
    )
    dd_svg = _svg_line_chart(
        [("Drawdown %", dd_series)],
        title="Drawdown vs your cap",
        hlines=[(-drawdown_cap * 100.0, "#dc2626", f"{drawdown_cap:.0%} cap")],
        fill_negative=True,
        fmt_y=lambda v: f"{v:.0f}%",
    )

    rot_rows = "\n".join(
        "<tr>"
        f"<td>{r['date'].date()}</td><td>{r['from']} → {r['to']}</td>"
        f"<td>{r['days_held']}</td><td>{r['holding_return']:+.1%}</td>"
        "</tr>"
        for _, r in rotations.iloc[::-1].iterrows()
    )
    reading_note = rotation_reading_note(rotations)

    css = """
    body{font-family:-apple-system,system-ui,'Segoe UI',sans-serif;margin:0 auto;max-width:900px;padding:24px;color:#111827;background:#f9fafb}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px;margin-bottom:20px}
    .signal{border-left:6px solid}.signal.hold{border-color:#16a34a}.signal.rotate{border-color:#d97706;background:#fffbeb}
    .action{font-size:30px;font-weight:800;margin:4px 0}.hold .action{color:#16a34a}.rotate .action{color:#b45309}
    .muted{color:#6b7280;font-size:13px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    .stat .v{font-size:22px;font-weight:700}.stat .k{font-size:12px;color:#6b7280}
    table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:left}
    th{color:#6b7280;font-weight:600}svg{width:100%;height:auto;display:block}
    h1{font-size:22px}h2{font-size:16px;margin:0 0 8px}
    """
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Core Signal Dashboard</title><style>{css}</style></head>
<body>
<h1>Core signal dashboard <span class="muted">· as of {html.escape(asof_label)}</span></h1>
<div class="card signal {action_class}">
  <h2>Tomorrow's signal</h2>
  <div class="action">{html.escape(action_text)}</div>
  <div>{html.escape(action_detail)}</div>
  <div style="margin-top:8px">{ma_line_text}{watch_text}</div>
</div>
<div class="card"><h2>Model portfolio</h2>
  <div class="stats">
    <div class="stat"><div class="v">{metrics['cagr']:.1%}</div><div class="k">CAGR (since {equity.index[0].date()})</div></div>
    <div class="stat"><div class="v">{metrics['sharpe']:.2f}</div><div class="k">Sharpe</div></div>
    <div class="stat"><div class="v">{metrics['max_drawdown']:.1%}</div><div class="k">Max drawdown</div></div>
    <div class="stat"><div class="v">{cur_dd:.1%}</div><div class="k">Current drawdown</div></div>
    <div class="stat"><div class="v">${equity.iloc[-1]:,.0f}</div><div class="k">Model equity ($100k start)</div></div>
    <div class="stat"><div class="v">{int(metrics['num_trades'])}</div><div class="k">Trades modeled</div></div>
    <div class="stat"><div class="v">{len(rotations)}</div><div class="k">Rotations</div></div>
    <div class="stat"><div class="v">{drawdown_cap:.0%}</div><div class="k">Your drawdown cap</div></div>
  </div>
</div>
<div class="card">{equity_svg}</div>
<div class="card">{dd_svg}<div class="muted">Amber dots on the equity chart mark rotations.</div></div>
<div class="card"><h2>Rotation history</h2>
<table><tr><th>Date</th><th>Rotation</th><th>Days held</th><th>Holding return</th></tr>
{rot_rows}</table>
<div class="muted" style="margin-top:10px">{reading_note}</div></div>
<div class="card muted"><h2>Methodology</h2>
Rule: hold 100% QQQ when QQQ's close is above its 200-day moving average, otherwise 100% GLD.
Signals use each day's close and trade at the next open (no lookahead). Costs: 1 bp commission + 5 bps
slippage per rotation. Data: yfinance daily bars. This tracks a <em>model</em> portfolio that assumes every
signal was followed — it is not your brokerage account. You place the trades; reconcile the model position
with your actual holdings before acting. Not financial advice.
</div>
</body></html>
"""


def generate_dashboard_html(
    *,
    cache_dir: str | Path = "data/cache",
    costs: Costs | None = None,
    drawdown_cap: float = 0.30,
    start: str = "2010-01-01",
    end: str | None = None,
    prices: dict[str, pd.DataFrame] | None = None,
    risky: str = "QQQ",
    safe: str = "GLD",
    ma: int = 200,
) -> tuple[str, dict]:
    """Refresh bars, recompute the core signal, render the dashboard.

    The single code path used by both the local script and the Modal
    deployment: fetch daily bars (or use ``prices`` when given, which skips
    the download and keeps tests offline), run the core rotation signal and
    model backtest, and render the self-contained HTML page.

    Returns ``(html_page, summary)`` where ``summary`` is the dict from
    :func:`summarize_core_signal`.
    """
    if prices is None:
        prices = get_daily_bars(
            [risky, safe],
            start=start,
            end=end or date.today().isoformat(),
            cache_dir=cache_dir,
        )
    signals = core_rotation_signal(prices, risky=risky, safe=safe, ma=ma)
    result = backtest(prices, signals, costs or Costs())

    summary = summarize_core_signal(prices, risky=risky, safe=safe, ma=ma)
    rotations = rotation_history(signals, result.equity)
    asof = summary["asof"]
    asof_label = f"{pd.Timestamp(asof).date()} close (data through latest bar)"

    html_page = render_dashboard_html(
        summary=summary,
        equity=result.equity,
        rotations=rotations,
        metrics=result.metrics,
        drawdown_cap=drawdown_cap,
        asof_label=asof_label,
    )
    return html_page, summary
