"""Walk-forward sleeve repick: simulate running the selection process through history.

Methodology (per the system design):
  * At each rebalance date D, the sleeve is picked using ONLY data in
    [D - selection_lookback_years, D). Nothing at or after D may influence the
    pick -- enforced by strict slicing, and trapped by tests.
  * The picks are traded for one quarter ([D, next_D)); then we step forward.
  * A kill rule drops a held pick at rebalance if its live drawdown since
    inception exceeds ``kill_dd_multiple`` x its worst backtested drawdown at
    selection time. Every kill is logged with its reason.

Performance design:
  * Strategy signals are *causal* (signal at close t depends only on data <= t),
    so all (strategy, params) signal series are precomputed ONCE on the full
    dataset and then sliced per window. Precomputation never leaks the future
    into signal[t]; it only avoids recomputing the same causal indicators.
  * The stitched sleeve signal series is backtested in ONE engine run, so
    quarter-boundary transitions pay exact costs instead of stitching artifacts.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..criteria import PassCriteria, evaluate
from ..engine import Costs, backtest, buy_and_hold, compute_metrics
from ..strategies.core import NON_EQUITY_TICKERS


def _canonical(params: dict) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in params.items()))


def _unrepr(v: str):
    """Inverse of repr() for the JSON-ish scalars stored in canonical keys."""
    try:
        return json.loads(v)
    except Exception:
        s = v.strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            return s[1:-1]
        return s


@dataclass
class WalkForwardConfig:
    start: str = "2018-01-01"
    rebalance: str = "quarterly"  # only "quarterly" is supported
    selection_lookback_years: int = 5
    top_k: int = 3
    kill_dd_multiple: float = 2.0
    warmup_days: int = 300

    @classmethod
    def from_dict(cls, d: dict) -> "WalkForwardConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def quarter_rebalance_dates(index: pd.DatetimeIndex, start: str) -> list[pd.Timestamp]:
    """Quarter starts in [start, index.max()], snapped forward to trading days."""
    idx = pd.DatetimeIndex(sorted(index))
    raw = pd.date_range(start=pd.Timestamp(start), end=idx.max(), freq="QS")
    out: list[pd.Timestamp] = []
    for d in raw:
        snapped = idx[idx >= d.normalize()]
        if len(snapped) == 0:
            continue
        out.append(snapped[0])
    # Drop duplicates and any date too close to the end to trade a quarter.
    out = sorted(set(out))
    return [d for d in out if d < idx.max() - pd.Timedelta(days=5)]


def selection_window(D: pd.Timestamp, lookback_years: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return [sel_start, D): the selection window for rebalance date D.

    Strictly before D -- the single most important invariant in this module.
    """
    sel_start = D - pd.DateOffset(years=lookback_years)
    if not sel_start < D:
        raise ValueError("selection window must end strictly before the rebalance date")
    return sel_start, D


def should_kill(live_dd: float, selection_dd: float, kill_multiple: float) -> tuple[bool, str]:
    """Kill rule: drop a held pick whose live drawdown exceeds
    ``kill_multiple`` x its worst backtested drawdown at selection."""
    threshold = kill_multiple * selection_dd
    if live_dd > threshold:
        return True, (
            f"live drawdown {live_dd:.3f} exceeded {kill_multiple}x "
            f"selection max DD {selection_dd:.3f} (threshold {threshold:.3f})"
        )
    return False, ""


def precompute_signals(
    strategy_fns: dict[str, object],
    grid: dict[str, dict[str, list]],
    data: dict[str, pd.DataFrame],
    universe: list[str] | None = None,
) -> dict[tuple[str, tuple], pd.DataFrame]:
    """Compute signal series once for every (strategy, param combo).

    Returns {(strategy_name, canonical_params): signals DataFrame}.
    Signal[t] is causal by construction (strategies only use data <= t), so
    slicing these series per window cannot introduce lookahead.
    """
    import itertools

    combos: dict[tuple[str, tuple], pd.DataFrame] = {}
    for name, fn in strategy_fns.items():
        param_grid = grid[name]
        keys = list(param_grid.keys())
        for combo in itertools.product(*(param_grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            key = (name, _canonical(params))
            if universe is not None:
                signals = fn(data, universe=universe, **params)
            else:
                signals = fn(data, **params)
            combos[key] = signals
    return combos


def _slice_data(data: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp):
    """Slice every frame to [start, end) with per-frame label slicing.

    Per-frame (not a shared boolean mask) so tickers with slightly different
    trading calendars can't raise alignment errors.
    """
    out = {}
    for t, df in data.items():
        out[t] = df.loc[(df.index >= start) & (df.index < end)]
    return out


def run_walkforward(
    data: dict[str, pd.DataFrame],
    strategy_fns: dict[str, object],
    grid: dict[str, dict[str, list]],
    costs: Costs,
    criteria: PassCriteria,
    wf_cfg: WalkForwardConfig,
    benchmark_ticker: str = "QQQ",
    progress: bool = True,
) -> dict:
    """Run the full walk-forward simulation. Returns a results dict with
    ``signals`` (stitched sleeve signals), ``equity``, ``trades``, ``metrics``,
    ``benchmark_metrics``, ``picks`` (per-rebalance log), ``family_weights``,
    and ``runtime_s``.
    """
    t0 = time.time()
    full_index = pd.DatetimeIndex(sorted(next(iter(data.values())).index))
    wf_start = pd.Timestamp(wf_cfg.start)
    if wf_start < full_index[0]:
        raise ValueError(f"walkforward.start {wf_cfg.start} predates available data {full_index[0].date()}")

    sleeve_universe = [t for t in data if t not in NON_EQUITY_TICKERS]

    if progress:
        print("Precomputing strategy signals (causal; sliced per window later)...")
    pre = precompute_signals(strategy_fns, grid, data, universe=sleeve_universe)
    if progress:
        print(f"  {len(pre)} (strategy, params) series precomputed")

    dates = quarter_rebalance_dates(full_index, wf_cfg.start)
    dates = [d for d in dates if d >= wf_start]
    if len(dates) < 1:
        raise ValueError("Need at least 1 rebalance date for a walk-forward run")
    if progress:
        print(f"Rebalance dates: {len(dates)} quarterly steps from {dates[0].date()} to {dates[-1].date()}")

    stitched = pd.DataFrame(0.0, index=full_index[full_index >= wf_start], columns=sleeve_universe)
    family_w = pd.DataFrame(0.0, index=stitched.index, columns=sorted(strategy_fns))
    n_picks_s = pd.Series(0, index=stitched.index, name="n_picks", dtype=int)
    picks_log: list[dict] = []
    holdings: dict[tuple[str, tuple], dict] = {}  # key -> {params, inception, selection_max_dd, ...}

    for i, D in enumerate(dates):
        next_D = dates[i + 1] if i + 1 < len(dates) else full_index.max() + pd.Timedelta(days=1)
        sel_start, sel_end = selection_window(D, wf_cfg.selection_lookback_years)

        # --- selection on strictly pre-D data ---
        data_sel = _slice_data(data, sel_start, sel_end)
        bench_sel = buy_and_hold(data_sel, benchmark_ticker, costs=costs)
        scored: list[dict] = []
        for key, sig_full in pre.items():
            name, ckey = key
            params = {k: _unrepr(v) for k, v in ckey}
            sig = sig_full.loc[(sig_full.index >= sel_start) & (sig_full.index < sel_end)]
            if sig.empty:
                continue
            res = backtest(data_sel, sig, costs=costs)
            passed, _reasons = evaluate(res, bench_sel, criteria=criteria)
            m = res.metrics
            scored.append(
                {
                    "key": key, "strategy": name, "params": params,
                    "sharpe": m["sharpe"], "cagr": m["cagr"],
                    "max_dd": m["max_drawdown"], "n_trades": m["num_trades"],
                    "passed": passed,
                }
            )
        passers = sorted([s for s in scored if s["passed"]], key=lambda s: -s["sharpe"])

        # --- kill rule on currently held picks (live DD since inception) ---
        kills: list[dict] = []
        killed_keys: set[tuple[str, tuple]] = set()
        for key, info in list(holdings.items()):
            inception = info["inception"]
            sig = pre[key].loc[(pre[key].index >= inception) & (pre[key].index < D)]
            data_live = _slice_data(data, inception, D)
            res = backtest(data_live, sig, costs=costs)
            live_dd = res.metrics["max_drawdown"]
            kill, reason = should_kill(live_dd, info["selection_max_dd"], wf_cfg.kill_dd_multiple)
            if kill:
                kills.append(
                    {
                        "strategy": info["strategy"], "params": info["params"],
                        "reason": reason, "live_dd": round(live_dd, 4),
                        "threshold": round(wf_cfg.kill_dd_multiple * info["selection_max_dd"], 4),
                        "inception": inception.strftime("%Y-%m-%d"),
                    }
                )
                killed_keys.add(key)
                del holdings[key]

        # --- repick: top_k passers not killed this round ---
        new_holdings: dict[tuple[str, tuple], dict] = {}
        for s in passers:
            if len(new_holdings) >= wf_cfg.top_k:
                break
            if s["key"] in killed_keys:
                continue
            key = s["key"]
            prev = holdings.get(key)
            new_holdings[key] = {
                "strategy": s["strategy"],
                "params": s["params"],
                "inception": prev["inception"] if prev else D,
                "selection_max_dd": s["max_dd"],
                "selection_sharpe": s["sharpe"],
                "selection_cagr": s["cagr"],
            }
        dropped_out = [
            {"strategy": holdings[k]["strategy"], "params": holdings[k]["params"]}
            for k in holdings if k not in new_holdings
        ]
        holdings = new_holdings

        # --- trade the quarter with the new holdings ---
        qmask = (stitched.index >= D) & (stitched.index < next_D)
        n_hold = len(holdings)
        if n_hold:
            for key in holdings:
                pmask = (pre[key].index >= D) & (pre[key].index < next_D)
                # .add aligns on index+columns; never rely on positional order.
                stitched.loc[qmask] = stitched.loc[qmask].add(pre[key].loc[pmask] / n_hold, fill_value=0.0)
            for name in family_w.columns:
                fam_n = sum(1 for k in holdings if k[0] == name)
                family_w.loc[qmask, name] = fam_n / n_hold
            n_picks_s.loc[qmask] = n_hold

        picks_log.append(
            {
                "date": D.strftime("%Y-%m-%d"),
                "selection_window": [sel_start.strftime("%Y-%m-%d"), sel_end.strftime("%Y-%m-%d")],
                "n_tested": len(scored),
                "n_passed": len(passers),
                "picks": [
                    {
                        "strategy": v["strategy"], "params": v["params"],
                        "selection_sharpe": round(v["selection_sharpe"], 4),
                        "selection_cagr": round(v["selection_cagr"], 4),
                        "selection_max_dd": round(v["selection_max_dd"], 4),
                        "inception": v["inception"].strftime("%Y-%m-%d"),
                    }
                    for v in holdings.values()
                ],
                "kills": kills,
                "dropped_out": dropped_out,
                "n_picks": n_hold,
            }
        )
        if progress:
            pick_names = [f"{v['strategy']}" for v in holdings.values()]
            print(
                f"  {D.date()}: tested={len(scored)} passed={len(passers)} "
                f"picks={n_hold} {pick_names} kills={len(kills)}"
            )

    # --- one continuous backtest of the stitched sleeve (exact boundary costs) ---
    trade_data = _slice_data(data, wf_start, full_index.max() + pd.Timedelta(days=1))
    sleeve_res = backtest(trade_data, stitched, costs=costs)
    bench_full = buy_and_hold(trade_data, benchmark_ticker, costs=costs)

    runtime_s = time.time() - t0
    return {
        "signals": stitched,
        "equity": sleeve_res.equity,
        "trades": sleeve_res.trades,
        "metrics": sleeve_res.metrics,
        "benchmark_metrics": bench_full.metrics,
        "benchmark_equity": bench_full.equity,
        "picks": picks_log,
        "family_weights": family_w,
        "n_picks_per_day": n_picks_s,
        "runtime_s": runtime_s,
        "n_rebalances": len(dates),
    }


def save_outputs(result: dict, results_dir: str | Path) -> None:
    """Write walkforward_equity.csv, walkforward_signals.parquet, walkforward_picks.jsonl."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    eq = pd.DataFrame({"date": result["equity"].index, "equity": result["equity"].values})
    fw = result["family_weights"].copy()
    fw.columns = [f"w_{c}" for c in fw.columns]
    out = eq.join(fw.reset_index(drop=True))
    out["n_picks"] = result["n_picks_per_day"].to_numpy()
    out.to_csv(results_dir / "walkforward_equity.csv", index=False)
    result["signals"].to_parquet(results_dir / "walkforward_signals.parquet")
    with open(results_dir / "walkforward_picks.jsonl", "w", encoding="utf-8") as f:
        for entry in result["picks"]:
            f.write(json.dumps(entry, default=str) + "\n")


def print_summary(result: dict) -> None:
    m, b = result["metrics"], result["benchmark_metrics"]
    print("\n=== Walk-forward sleeve ===")
    print(f"Rebalances: {result['n_rebalances']}  Runtime: {result['runtime_s']:.0f}s")
    print(f"Sleeve : CAGR={m['cagr']:.3f} Sharpe={m['sharpe']:.2f} "
          f"maxDD={m['max_drawdown']:.3f} trades={m['num_trades']} exposure={m['exposure_pct']:.0f}%")
    print(f"QQQ BH : CAGR={b['cagr']:.3f} Sharpe={b['sharpe']:.2f} "
          f"maxDD={b['max_drawdown']:.3f} trades={b['num_trades']}")
    kills = sum(len(p["kills"]) for p in result["picks"])
    print(f"Kills over the run: {kills}")
