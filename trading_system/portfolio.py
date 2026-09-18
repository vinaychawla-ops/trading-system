"""Portfolio assembly: combine core + sleeve into one portfolio.

Steps:
  1. Backtest each component standalone on the common date window.
  2. Correlation filter: while any pair of components has |correlation| of daily
     returns above ``max_corr``, drop the lower-Sharpe member of the worst pair
     and log the drop. (Prevents e.g. two near-identical pullback variants
     doubling down on the same bets.)
  3. Renormalize the surviving components' weights and backtest the weighted
     combination in one engine run.
"""
from __future__ import annotations

import pandas as pd

from .engine import BacktestResult, Costs, backtest, buy_and_hold


def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def correlation_filter(
    component_results: dict[str, BacktestResult], max_corr: float
) -> tuple[dict[str, BacktestResult], list[dict]]:
    """Iteratively drop the weaker member of over-correlated pairs.

    Returns (kept, dropped_log).
    """
    kept = dict(component_results)
    dropped: list[dict] = []
    while len(kept) >= 2:
        names = list(kept.keys())
        rets = pd.DataFrame({n: _daily_returns(kept[n].equity) for n in names}).dropna()
        if rets.empty or len(rets) < 2:
            break
        corr = rets.corr().abs()
        worst = 0.0
        worst_pair: tuple[str, str] | None = None
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                c = corr.iloc[i, j]
                if pd.notna(c) and c > worst:
                    worst, worst_pair = float(c), (names[i], names[j])
        if worst_pair is None or worst <= max_corr:
            break
        a, b = worst_pair
        drop = a if kept[a].metrics["sharpe"] < kept[b].metrics["sharpe"] else b
        keep = b if drop == a else a
        dropped.append(
            {
                "dropped": drop,
                "kept": keep,
                "correlation": round(worst, 4),
                "dropped_sharpe": round(kept[drop].metrics["sharpe"], 4),
                "kept_sharpe": round(kept[keep].metrics["sharpe"], 4),
                "reason": (
                    f"|corr| {worst:.3f} > max {max_corr}: dropped lower-Sharpe "
                    f"component {drop} in favor of {keep}"
                ),
            }
        )
        del kept[drop]
    return kept, dropped


def assemble(
    prices: dict[str, pd.DataFrame],
    components: dict[str, pd.DataFrame],
    weights: dict[str, float],
    costs: Costs = Costs(),
    max_corr: float = 0.7,
    initial_capital: float = 100_000.0,
    benchmark_ticker: str = "QQQ",
) -> dict:
    """Combine component signal series into one portfolio.

    Args:
        prices: ticker -> OHLC DataFrame.
        components: name -> target-weight signals DataFrame.
        weights: name -> portfolio weight (renormalized after correlation filter).
    Returns dict with ``equity``, ``metrics``, ``component_metrics``,
    ``dropped``, ``weights_used``, ``benchmark_metrics`` and ``window``.
    """
    if set(components) != set(weights):
        raise ValueError("components and weights must name the same components")
    if any(w < 0 for w in weights.values()):
        raise ValueError("component weights must be non-negative")

    common = None
    for sig in components.values():
        common = sig.index if common is None else common.intersection(sig.index)
    common = pd.DatetimeIndex(sorted(common))
    if len(common) < 2:
        raise ValueError("Components share no usable date window")
    data = {t: df.loc[df.index.isin(common)] for t, df in prices.items()}
    # keep only tickers present in every component's columns to avoid NaN pricing
    tickers = [c for c in set().union(*(set(s.columns) for s in components.values())) if c in data]

    comp_results: dict[str, BacktestResult] = {}
    for name, sig in components.items():
        s = sig.loc[common, [c for c in sig.columns if c in tickers]].reindex(columns=tickers).fillna(0.0)
        comp_results[name] = backtest(data, s, costs=costs, initial_capital=initial_capital)

    kept, dropped = correlation_filter(comp_results, max_corr)
    wsum = sum(weights[n] for n in kept)
    if wsum <= 0:
        raise ValueError("All components were filtered out; nothing left to assemble")
    weights_used = {n: weights[n] / wsum for n in kept}

    combined = None
    for name in kept:
        s = components[name].loc[common, [c for c in components[name].columns if c in tickers]]
        s = s.reindex(columns=tickers).fillna(0.0) * weights_used[name]
        combined = s if combined is None else combined.add(s, fill_value=0.0)
    combined = combined.fillna(0.0)

    result = backtest(data, combined, costs=costs, initial_capital=initial_capital)
    benchmark = buy_and_hold(data, benchmark_ticker, costs=costs, initial_capital=initial_capital)
    return {
        "equity": result.equity,
        "trades": result.trades,
        "metrics": result.metrics,
        "signals": combined,
        "component_metrics": {n: r.metrics for n, r in comp_results.items()},
        "component_equity": {n: r.equity for n, r in comp_results.items()},
        "components_kept": sorted(kept),
        "dropped": dropped,
        "weights_used": weights_used,
        "benchmark_metrics": benchmark.metrics,
        "benchmark_equity": benchmark.equity,
        "window": (common[0].strftime("%Y-%m-%d"), common[-1].strftime("%Y-%m-%d")),
    }
