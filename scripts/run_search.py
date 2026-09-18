#!/usr/bin/env python3
"""Run the strategy search on TRAIN data.

For each sleeve strategy: exhaustive grid search, then mutation search seeded
from the top grid results. Also scores the core rotation strategy. Every test
is appended to the ledger; passing candidates are ranked into candidates.csv.

Usage:
    python scripts/run_search.py --config configs/growth_daily.yaml
    python scripts/run_search.py --config configs/smoke.yaml --smoke   # synthetic data, fast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from trading_system import synthetic
from trading_system.criteria import PassCriteria, evaluate
from trading_system.data import get_daily_bars
from trading_system.engine import Costs, backtest, buy_and_hold
from trading_system.research import ledger as ledger_mod
from trading_system.research.search import (
    grid_search,
    mutate_search,
    rank_passers,
    save_candidates,
)
from trading_system.strategies.core import core_rotation_signal
from trading_system.strategies.sleeve import SLEEVE_STRATEGIES


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Use deterministic synthetic data instead of downloading",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    universe = cfg["profile"]["universe"]
    train_start, train_end = cfg["train_start"], cfg["train_end"]
    results_dir = Path(cfg.get("results_dir", "results"))
    ledger_path = results_dir / "ledger.jsonl"

    if args.smoke:
        print("SMOKE MODE: synthetic data (no downloads)")
        data = synthetic.random_walk_prices(universe, train_start, train_end, seed=7)
    else:
        data = get_daily_bars(
            universe, train_start, train_end, cache_dir=cfg.get("cache_dir", "data/cache")
        )
    print(f"Data: {len(data)} tickers, "
          f"{min(d.index[0] for d in data.values()).date()} -> "
          f"{max(d.index[-1] for d in data.values()).date()}")

    costs = Costs(**cfg.get("costs", {}))
    criteria = PassCriteria(**cfg.get("criteria", {}))
    mut_cfg = cfg.get("mutate", {})

    if "QQQ" not in data:
        raise KeyError("Config universe must include QQQ (benchmark)")
    benchmark = buy_and_hold(data, "QQQ", costs=costs)
    bm = benchmark.metrics
    print(f"Benchmark QQQ buy&hold: total_return={bm['total_return']:.3f} "
          f"sharpe={bm['sharpe']:.2f} max_dd={bm['max_drawdown']:.3f}")

    all_results: list[pd.DataFrame] = []
    for name, fn in SLEEVE_STRATEGIES.items():
        grid = cfg["param_grids"][name]
        print(f"\n=== {name}: grid search ({name}) ===")
        df_grid = grid_search(
            fn, name, grid, data, costs, criteria, benchmark, ledger_path=ledger_path
        )
        print(f"  grid: {len(df_grid)} tested, {int(df_grid['passed'].sum())} passed")
        top_params = (
            df_grid.sort_values("sharpe", ascending=False).head(mut_cfg.get("top_k", 3))["params"].tolist()
        )
        print(f"=== {name}: mutate search ===")
        df_mut = mutate_search(
            fn, name, top_params, data, costs, criteria, benchmark,
            n_rounds=mut_cfg.get("n_rounds", 2),
            n_children=mut_cfg.get("n_children", 6),
            perturb=mut_cfg.get("perturb", 0.25),
            top_k=mut_cfg.get("top_k", 3),
            seed=mut_cfg.get("seed", 42),
            ledger_path=ledger_path,
        )
        print(f"  mutate: {len(df_mut)} tested, "
              f"{int(df_mut['passed'].sum()) if len(df_mut) else 0} passed")
        all_results.extend([df_grid, df_mut])

    # Core strategy: scored, not searched (no parameter grid by design).
    print("\n=== core_rotation: scoring ===")
    core_signals = core_rotation_signal(data)
    core_result = backtest(data, core_signals, costs=costs)
    core_passed, core_reasons = evaluate(core_result, benchmark, criteria=criteria)
    cm = core_result.metrics
    core_record = {
        "strategy": "core_rotation",
        "params": {"risky": "QQQ", "safe": "GLD", "ma": 200},
        "n_trades": int(cm["num_trades"]),
        "cagr": float(cm["cagr"]),
        "sharpe": float(cm["sharpe"]),
        "max_dd": float(cm["max_drawdown"]),
        "passed": bool(core_passed),
        "fail_reasons": list(core_reasons),
    }
    ledger_mod.append_test(core_record, path=ledger_path)
    print(f"  core: passed={core_passed} sharpe={cm['sharpe']:.2f} "
          f"max_dd={cm['max_drawdown']:.3f} reasons={core_reasons}")
    all_results.append(pd.DataFrame([core_record]))

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    passers = rank_passers(combined)
    out_path = results_dir / "candidates.csv"
    save_candidates(passers, out_path)
    print(f"\nWrote {len(passers)} passing candidates -> {out_path}")
    print(f"Ledger: {ledger_path} ({len(ledger_mod.read_ledger(ledger_path))} rows)")
    if len(passers):
        print(passers[["strategy", "sharpe", "cagr", "max_dd", "n_trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
