#!/usr/bin/env python3
"""Phase 4: generate the daily-signal dashboard for the validated core.

Refreshes QQQ/GLD daily bars, recomputes the core rotation signal through the
latest close, paper-tracks the model portfolio, and writes a single
self-contained HTML dashboard. Open it in any browser; no server needed.

Usage:
    python scripts/make_dashboard.py --config configs/growth_daily.yaml
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

import pandas as pd

from trading_system import dashboard as dash
from trading_system.data import get_daily_bars
from trading_system.engine import Costs, backtest
from trading_system.strategies.core import core_rotation_signal


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/growth_daily.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(cfg["cache_dir"])
    out_path = Path(cfg["results_dir"]) / "dashboard.html"
    costs = Costs(**cfg.get("costs", {}))
    drawdown_cap = float(cfg["criteria"]["max_drawdown"])

    tickers = ["QQQ", "GLD"]
    print("Refreshing daily bars ...", flush=True)
    prices = get_daily_bars(
        tickers,
        start="2010-01-01",
        end=date.today().isoformat(),
        cache_dir=cache_dir,
    )

    signals = core_rotation_signal(prices, risky="QQQ", safe="GLD", ma=200)
    result = backtest(prices, signals, costs)

    summary = dash.summarize_core_signal(prices)
    rotations = dash.rotation_history(signals, result.equity)
    asof = summary["asof"]
    asof_label = f"{pd.Timestamp(asof).date()} close (data through latest bar)"

    html_page = dash.render_dashboard_html(
        summary=summary,
        equity=result.equity,
        rotations=rotations,
        metrics=result.metrics,
        drawdown_cap=drawdown_cap,
        asof_label=asof_label,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_page, encoding="utf-8")

    if summary["action"] == "HOLD":
        sig = f"HOLD {summary['position']}"
    else:
        sig = f"ROTATE {summary['rotate_from']} -> {summary['rotate_to']}"
    print(f"Signal for next open: {sig}")
    print(f"QQQ {summary['ma_distance_pct']:+.2f}% vs 200-day MA")
    print(f"Dashboard written to {out_path}")


if __name__ == "__main__":
    main()
