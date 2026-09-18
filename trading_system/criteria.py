"""Pass/fail criteria for backtest results.

The criteria must be written down BEFORE any testing (see README): the search
loop scores candidates against these rules so results can't bend the rules.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import BacktestResult


@dataclass
class PassCriteria:
    min_trades: int = 30
    min_sharpe: float = 0.5
    max_drawdown: float = 0.15  # as a fraction, e.g. 0.15 = 15%
    must_beat_benchmark: bool = True
    max_pairwise_corr: float = 0.7  # vs reference return series, when provided


def evaluate(
    result: BacktestResult,
    benchmark_result: BacktestResult,
    criteria: PassCriteria | None = None,
    reference_returns: list[pd.Series] | None = None,
) -> tuple[bool, list[str]]:
    """Score a backtest result against the criteria.

    Returns ``(passed, reasons)``; ``reasons`` lists every failed check and is
    empty when the result passes.
    """
    criteria = criteria or PassCriteria()
    reasons: list[str] = []
    m = result.metrics

    n_trades = int(m.get("num_trades", 0))
    if n_trades < criteria.min_trades:
        reasons.append(f"num_trades {n_trades} < min_trades {criteria.min_trades}")

    sharpe = float(m.get("sharpe", 0.0))
    if sharpe < criteria.min_sharpe:
        reasons.append(f"sharpe {sharpe:.3f} < min_sharpe {criteria.min_sharpe}")

    max_dd = float(m.get("max_drawdown", 1.0))
    if max_dd > criteria.max_drawdown:
        reasons.append(f"max_drawdown {max_dd:.3f} > {criteria.max_drawdown}")

    if criteria.must_beat_benchmark:
        r_ret = float(m.get("total_return", 0.0))
        b_ret = float(benchmark_result.metrics.get("total_return", 0.0))
        if not r_ret > b_ret:
            reasons.append(f"total_return {r_ret:.4f} did not beat benchmark {b_ret:.4f}")

    if reference_returns:
        rets = result.equity.pct_change().dropna()
        worst = 0.0
        for ref in reference_returns:
            aligned = pd.concat([rets, ref.pct_change().dropna()], axis=1, join="inner").dropna()
            if len(aligned) < 2:
                continue
            c = aligned.corr().iloc[0, 1]
            if pd.notna(c):
                worst = max(worst, abs(float(c)))
        if worst > criteria.max_pairwise_corr:
            reasons.append(
                f"max correlation to reference {worst:.3f} > {criteria.max_pairwise_corr}"
            )

    return (len(reasons) == 0), reasons
