#!/usr/bin/env python3
"""Walk-forward sleeve repick simulation.

At each quarterly rebalance date from walkforward.start, picks the sleeve using
only data before that date, trades the picks for one quarter, steps forward,
and stitches everything into one track record. Writes
results/walkforward_equity.csv, results/walkforward_signals.parquet and
results/walkforward_picks.jsonl.

Usage:
    python scripts/walkforward.py --config configs/growth_daily.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from trading_system.criteria import PassCriteria
from trading_system.data import get_daily_bars
from trading_system.engine import Costs
from trading_system.research.walkforward import (
    WalkForwardConfig,
    print_summary,
    run_walkforward,
    save_outputs,
)
from trading_system.strategies.sleeve import SLEEVE_STRATEGIES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wf_raw = cfg.get("walkforward", {})
    wf_cfg = WalkForwardConfig.from_dict(wf_raw)
    if wf_cfg.rebalance != "quarterly":
        raise ValueError(f"Unsupported rebalance {wf_cfg.rebalance!r}: only 'quarterly' is implemented")

    # Data must cover [start - lookback - warmup, today] for indicator warmup
    # and the full 5-year selection window at the first rebalance.
    dl_start = (
        pd.Timestamp(wf_cfg.start)
        - pd.DateOffset(years=wf_cfg.selection_lookback_years)
        - pd.Timedelta(days=wf_cfg.warmup_days + 30)
    ).strftime("%Y-%m-%d")
    end = date.today().isoformat()
    data = get_daily_bars(
        cfg["profile"]["universe"], dl_start, end, cache_dir=cfg.get("cache_dir", "data/cache")
    )
    print(f"Data: {len(data)} tickers, {dl_start} -> {end}")

    grid = wf_raw.get("walkforward_grid") or cfg["param_grids"]
    n_combos = 0
    for s in grid:
        n = 1
        for v in grid[s].values():
            n *= len(v)
        n_combos += n
    print(f"Selection grid: {n_combos} (strategy, params) combos per rebalance step")

    costs = Costs(**cfg.get("costs", {}))
    criteria = PassCriteria(**cfg.get("criteria", {}))

    t0 = time.time()
    result = run_walkforward(
        data, SLEEVE_STRATEGIES, grid, costs, criteria, wf_cfg, progress=True
    )
    results_dir = Path(cfg.get("results_dir", "results"))
    save_outputs(result, results_dir)
    print(f"\nWrote {results_dir}/walkforward_equity.csv, "
          f"walkforward_signals.parquet, walkforward_picks.jsonl")
    print_summary(result)
    print(f"Total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
