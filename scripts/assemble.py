#!/usr/bin/env python3
"""Assemble the final portfolio: core + sleeve with correlation filter and risk overlay.

Sleeve source (config portfolio.sleeve_source):
  * "walkforward" (default): stitched signals from scripts/walkforward.py.
  * "validated": equal-weighted survivors from results/validation.csv.

Writes results/portfolio_equity.csv and results/portfolio_summary.txt, and
prints a core-only vs sleeve-only vs combined vs QQQ buy-and-hold comparison
with the configured drawdown filter verdict.

Usage:
    python scripts/assemble.py --config configs/growth_daily.yaml
    python scripts/assemble.py --config configs/growth_daily.yaml --sleeve-source validated
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from trading_system.criteria import PassCriteria
from trading_system.data import get_daily_bars
from trading_system.engine import Costs, backtest, buy_and_hold
from trading_system.portfolio import assemble
from trading_system.research.validate import STRATEGY_REGISTRY
from trading_system.risk import RiskConfig, backtest_with_risk
from trading_system.strategies.core import core_rotation_signal


def _validated_sleeve_signals(data, validation_path: Path) -> pd.DataFrame:
    df = pd.read_csv(validation_path)
    surv = df[df["survived"]]
    if surv.empty:
        raise ValueError(f"No surviving candidates in {validation_path}; cannot build validated sleeve")
    acc = None
    for _, row in surv.iterrows():
        params = json.loads(row["params"])
        sig = STRATEGY_REGISTRY[row["strategy"]](data, **params)
        acc = sig if acc is None else acc.add(sig, fill_value=0.0)
    return (acc / len(surv)).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--sleeve-source", choices=["walkforward", "validated"], default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg.get("results_dir", "results"))
    end = date.today().isoformat()
    data = get_daily_bars(
        cfg["profile"]["universe"], cfg["train_start"], end,
        cache_dir=cfg.get("cache_dir", "data/cache"),
    )
    print(f"Data: {len(data)} tickers -> {end}")

    port_cfg = cfg.get("portfolio", {})
    sleeve_source = args.sleeve_source or port_cfg.get("sleeve_source", "walkforward")

    core_params = cfg.get("core", {})
    core_sig = core_rotation_signal(data, **core_params) if core_params else core_rotation_signal(data)

    if sleeve_source == "walkforward":
        sig_path = results_dir / "walkforward_signals.parquet"
        if not sig_path.exists():
            raise FileNotFoundError(f"{sig_path} not found -- run scripts/walkforward.py first.")
        sleeve_sig = pd.read_parquet(sig_path)
        print(f"Sleeve: walk-forward stitched signals ({len(sleeve_sig)} days)")
    else:
        val_path = results_dir / "validation.csv"
        if not val_path.exists():
            raise FileNotFoundError(f"{val_path} not found -- run scripts/validate.py first.")
        sleeve_sig = _validated_sleeve_signals(data, val_path)
        print("Sleeve: equal-weighted validation survivors")

    weights = {
        "core": float(port_cfg.get("core_weight", 0.7)),
        "sleeve": float(port_cfg.get("sleeve_weight", 0.3)),
    }
    max_corr = float(port_cfg.get("max_pairwise_corr", cfg.get("criteria", {}).get("max_pairwise_corr", 0.7)))
    costs = Costs(**cfg.get("costs", {}))
    criteria = PassCriteria(**cfg.get("criteria", {}))

    asm = assemble(
        data, {"core": core_sig, "sleeve": sleeve_sig}, weights,
        costs=costs, max_corr=max_corr,
    )
    print(f"Window: {asm['window'][0]} -> {asm['window'][1]}")
    print(f"Weights used: {asm['weights_used']}")
    for d in asm["dropped"]:
        print(f"  correlation filter: {d['reason']}")

    # Risk overlay on the combined portfolio.
    risk_cfg = RiskConfig(**cfg.get("risk", {}))
    common_idx = asm["equity"].index
    risk_data = {t: df.loc[df.index.isin(common_idx)] for t, df in data.items()}
    risk_res, risk_stats = backtest_with_risk(risk_data, asm["signals"], risk_cfg, costs=costs)

    m, b = asm["metrics"], asm["benchmark_metrics"]
    rm = risk_res.metrics
    dd_ok = m["max_drawdown"] <= criteria.max_drawdown
    rows = [
        ("core-only", asm["component_metrics"]["core"]),
        ("sleeve-only", asm["component_metrics"]["sleeve"]),
        ("combined", m),
        ("combined+risk", rm),
        ("QQQ buy&hold", b),
    ]
    lines = []
    lines.append("=== Portfolio comparison ===")
    lines.append(f"{'component':14s} {'CAGR':>7s} {'Sharpe':>7s} {'maxDD':>7s} {'trades':>7s} {'exposure':>8s}")
    for name, mm in rows:
        lines.append(
            f"{name:14s} {mm['cagr']:7.3f} {mm['sharpe']:7.2f} "
            f"{mm['max_drawdown']:7.3f} {mm['num_trades']:7d} {mm['exposure_pct']:7.0f}%"
        )
    lines.append("")
    lines.append(
        f"Drawdown filter: combined maxDD {m['max_drawdown']:.3f} -> "
        f"{'PASS' if dd_ok else 'FAIL'} (limit {criteria.max_drawdown})"
    )
    lines.append(
        f"Risk overlay: capped_days={risk_stats['capped_days']}, "
        f"total_borrow_cost=${risk_stats['total_borrow_cost']:,.0f}, "
        f"max_single_name=${risk_stats['max_single_name_dollars']:,.0f}"
    )
    # Illustrative live-trade sizing (matches the unit test's example).
    from trading_system.risk import size_position

    ex_shares = size_position(100.0, risk_cfg.default_stop_pct, risk_cfg.max_loss_per_trade)
    lines.append(
        f"Illustrative sizing: a ${risk_cfg.max_loss_per_trade:,.0f} max-loss trade with "
        f"{risk_cfg.default_stop_pct:.0%} stop at $100 entry -> {ex_shares:,.1f} shares"
    )
    summary = "\n".join(lines)
    print("\n" + summary)

    eq = pd.DataFrame({"date": common_idx})
    eq["equity_plain"] = asm["equity"].reindex(common_idx).values
    eq["equity_risk"] = risk_res.equity.reindex(common_idx).values
    eq["equity_core"] = asm["component_equity"]["core"].reindex(common_idx).values
    eq["equity_sleeve"] = asm["component_equity"]["sleeve"].reindex(common_idx).values
    eq["equity_qqq"] = asm["benchmark_equity"].reindex(common_idx).values
    eq.to_csv(results_dir / "portfolio_equity.csv", index=False)
    (results_dir / "portfolio_summary.txt").write_text(summary + "\n")
    print(f"\nWrote {results_dir}/portfolio_equity.csv and portfolio_summary.txt")


if __name__ == "__main__":
    main()
