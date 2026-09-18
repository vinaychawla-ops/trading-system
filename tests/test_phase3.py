"""Phase 3 tests: validation discipline, walk-forward no-lookahead traps,
kill rule, correlation filter, and risk sizing math."""
import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from trading_system import synthetic
from trading_system.criteria import PassCriteria, evaluate
from trading_system.engine import BacktestResult, Costs, backtest, compute_metrics
from trading_system.portfolio import assemble, correlation_filter
from trading_system.research.validate import (
    STRATEGY_REGISTRY,
    check_train_test_separation,
    validate_candidates,
)
from trading_system.research.walkforward import (
    WalkForwardConfig,
    quarter_rebalance_dates,
    run_walkforward,
    selection_window,
    should_kill,
)
from trading_system.risk import (
    RiskConfig,
    apply_risk_limits,
    backtest_with_risk,
    size_position,
)
from trading_system.strategies.sleeve import SLEEVE_STRATEGIES


# ---------------------------------------------------------------- train/test separation

def test_check_train_test_separation_ok():
    check_train_test_separation("2010-01-01", "2022-12-31", "2023-01-01")  # no raise


def test_check_train_test_separation_overlap_rejected():
    with pytest.raises(ValueError, match="OVERLAP"):
        check_train_test_separation("2010-01-01", "2023-06-01", "2023-01-01")
    with pytest.raises(ValueError, match="OVERLAP"):
        check_train_test_separation("2010-01-01", "2023-01-01", "2023-01-01")  # touching counts


def _tiny_candidates():
    return pd.DataFrame(
        [
            {
                "strategy": "core_rotation",
                "params": json.dumps({"risky": "QQQ", "safe": "GLD", "ma": 200}),
                "n_trades": 4, "cagr": 0.1, "sharpe": 1.0, "max_dd": 0.05,
                "passed": True, "fail_reasons": "[]",
            }
        ]
    )


def test_validate_scores_strictly_on_test_window():
    """Trap: the scored signals handed to the engine must start >= test_start,
    even though indicator warmup data exists before it."""
    data = synthetic.random_walk_prices(
        ["QQQ", "GLD"], "2022-06-01", "2023-06-30", seed=3
    )
    seen_min_dates = []
    real_backtest = backtest

    def spy_backtest(prices, signals, **kw):
        seen_min_dates.append(pd.Timestamp(signals.index.min()))
        return real_backtest(prices, signals, **kw)

    with mock.patch("trading_system.research.validate.backtest", side_effect=spy_backtest):
        df = validate_candidates(
            _tiny_candidates(), data, "2023-01-01", Costs(), PassCriteria()
        )
    assert len(df) == 1
    assert set(df.columns) >= {"train_sharpe", "test_sharpe", "test_cagr", "test_max_dd", "survived"}
    assert seen_min_dates, "expected the engine to be called"
    assert all(d >= pd.Timestamp("2023-01-01") for d in seen_min_dates)


def test_validate_unknown_strategy_rejected():
    bad = _tiny_candidates()
    bad.loc[0, "strategy"] = "nope_not_real"
    data = synthetic.random_walk_prices(["QQQ", "GLD"], "2022-06-01", "2023-06-30", seed=3)
    with pytest.raises(KeyError):
        validate_candidates(bad, data, "2023-01-01", Costs(), PassCriteria())


def test_validate_empty_candidates():
    data = synthetic.random_walk_prices(["QQQ", "GLD"], "2022-06-01", "2023-06-30", seed=3)
    out = validate_candidates(pd.DataFrame(), data, "2023-01-01", Costs(), PassCriteria())
    assert out.empty


# ---------------------------------------------------------------- walk-forward lookahead traps

def _spike_data():
    """Synthetic data with two traps: a +50% one-day spike mid-quarter
    (2020-05-15) and a +50% spike exactly on a rebalance date (2020-07-01)."""
    data = synthetic.random_walk_prices(["AAA", "AAB", "AAC", "QQQ"], "2019-01-01", "2021-12-31", seed=11)
    for day in ("2020-05-15", "2020-07-01"):
        ts = pd.Timestamp(day)
        df = data["AAA"]
        df.loc[ts, "Close"] *= 1.5
        df.loc[ts, "High"] = max(df.loc[ts, "High"], df.loc[ts, "Close"])
    return data


def _wf_mini(data, **over):
    grid = {
        "trend_pullback": {"lookback": [50], "rsi_window": [2], "rsi_thresh": [30],
                           "hold_days": [5], "max_positions": [2]},
        "low_range_close": {"lookback": [50], "range_window": [10], "pct": [0.25],
                            "hold_days": [5], "max_positions": [2]},
        "quiet_pullback": {"lookback": [50], "pullback_days": [3], "range_atr_frac": [1.0],
                           "hold_days": [5], "max_positions": [2]},
    }
    cfg = dict(start="2020-01-01", selection_lookback_years=1, top_k=2,
               kill_dd_multiple=2.0, warmup_days=30)
    cfg.update(over)
    wf = WalkForwardConfig(**cfg)
    crit = PassCriteria(min_trades=1, min_sharpe=-99.0, max_drawdown=99.0,
                        must_beat_benchmark=False)
    return run_walkforward(data, SLEEVE_STRATEGIES, grid, Costs(), crit, wf, progress=False)


def test_selection_window_ends_strictly_before_D():
    D = pd.Timestamp("2020-07-01")
    s, e = selection_window(D, 5)
    assert e == D
    assert s < e


def test_quarter_rebalance_dates_are_trading_days():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    dates = quarter_rebalance_dates(idx, "2020-01-01")
    assert all(d in idx for d in dates)
    assert dates[0] == pd.Timestamp("2020-01-01")
    assert all(dates[i] < dates[i + 1] for i in range(len(dates) - 1))


def test_walkforward_future_spike_cannot_change_earlier_picks():
    """Behavioral lookahead trap: picks made at D=2020-01-01 must be identical
    whether or not later data contains huge spikes. Any leak of post-D data
    into selection would change the picks."""
    full = _spike_data()
    truncated = {t: df.loc[df.index < pd.Timestamp("2020-01-16")] for t, df in full.items()}
    r_full = _wf_mini(full)
    r_trunc = _wf_mini(truncated)
    picks_full = r_full["picks"][0]
    picks_trunc = r_trunc["picks"][0]
    assert picks_full["date"] == picks_trunc["date"] == "2020-01-01"
    assert picks_full["picks"] == picks_trunc["picks"], (
        "picks at 2020-01-01 changed when future spikes were added -> selection leaked the future"
    )


def test_walkforward_selection_excludes_spike_on_rebalance_day():
    """A spike exactly ON the rebalance date D must not enter the selection
    window [D - lookback, D): logged selection metrics must match a clean
    recomputation on strictly pre-D data."""
    data = _spike_data()
    result = _wf_mini(data)
    entry = next(p for p in result["picks"] if p["date"] == "2020-07-01")
    assert entry["picks"], "expected at least one pick at 2020-07-01"
    pick = entry["picks"][0]
    D = pd.Timestamp("2020-07-01")
    sel_start = D - pd.DateOffset(years=1)
    fn = SLEEVE_STRATEGIES[pick["strategy"]]
    universe = [t for t in data if t not in ("QQQ", "GLD", "SPY")]
    sig = fn(data, universe=universe, **pick["params"])
    sig = sig.loc[(sig.index >= sel_start) & (sig.index < D)]
    data_sel = {t: df.loc[(df.index >= sel_start) & (df.index < D)] for t, df in data.items()}
    res = backtest(data_sel, sig, costs=Costs())
    # picks log rounds to 4 decimals; allow for that.
    assert res.metrics["sharpe"] == pytest.approx(entry["picks"][0]["selection_sharpe"], abs=5e-5)
    assert res.metrics["max_drawdown"] == pytest.approx(pick["selection_max_dd"], abs=5e-5)


# ---------------------------------------------------------------- kill rule

def test_should_kill_fires_at_multiple():
    kill, reason = should_kill(0.25, 0.10, 2.0)
    assert kill and "0.250" in reason
    kill, _ = should_kill(0.15, 0.10, 2.0)
    assert not kill
    kill, _ = should_kill(0.20, 0.10, 2.0)  # exactly at threshold -> not killed
    assert not kill


# ---------------------------------------------------------------- correlation filter

def _result_from_rets(rets):
    idx = pd.bdate_range("2020-01-01", periods=len(rets))
    eq = pd.Series(100.0 * np.cumprod(1 + np.asarray(rets)), index=idx)
    return BacktestResult(equity=eq, trades=pd.DataFrame(), metrics=compute_metrics(eq))


def test_correlation_filter_drops_weaker_of_identical_pair():
    rng = np.random.default_rng(0)
    rets_a = rng.normal(0.001, 0.01, 200)
    rets_b = rets_a - 0.0002  # identical correlation (1.0), strictly worse Sharpe
    rets_c = np.random.default_rng(1).normal(0.001, 0.01, 200)
    comps = {"A": _result_from_rets(rets_a), "B": _result_from_rets(rets_b), "C": _result_from_rets(rets_c)}
    kept, dropped = correlation_filter(comps, 0.7)
    assert set(kept) == {"A", "C"}
    assert len(dropped) == 1 and dropped[0]["dropped"] == "B" and dropped[0]["kept"] == "A"
    assert dropped[0]["correlation"] == pytest.approx(1.0)
    assert "0.7" in dropped[0]["reason"]


def test_correlation_filter_keeps_uncorrelated():
    rng = np.random.default_rng(2)
    comps = {
        "A": _result_from_rets(rng.normal(0.001, 0.01, 200)),
        "B": _result_from_rets(rng.normal(0.001, 0.01, 200)),
    }
    kept, dropped = correlation_filter(comps, 0.7)
    assert set(kept) == {"A", "B"} and dropped == []


def test_assemble_combines_core_and_sleeve():
    data = synthetic.random_walk_prices(["QQQ", "GLD", "AAA", "AAB"], "2020-01-01", "2020-12-31", seed=5)
    idx = data["QQQ"].index
    core_sig = pd.DataFrame(0.0, index=idx, columns=["QQQ", "GLD"])
    core_sig["QQQ"] = 1.0
    sleeve_sig = pd.DataFrame(0.0, index=idx, columns=["AAA", "AAB"])
    sleeve_sig["AAA"] = 0.5
    out = assemble(data, {"core": core_sig, "sleeve": sleeve_sig},
                   {"core": 0.7, "sleeve": 0.3}, costs=Costs())
    assert out["weights_used"] == {"core": pytest.approx(0.7), "sleeve": pytest.approx(0.3)}
    assert len(out["equity"]) == len(idx)
    assert out["window"] == ("2020-01-01", "2020-12-31")
    assert set(out["component_metrics"]) == {"core", "sleeve"}


# ---------------------------------------------------------------- risk sizing

def test_size_position_math():
    assert size_position(100.0, 0.08, 500.0) == pytest.approx(62.5)


def test_size_position_rejects_bad_inputs():
    with pytest.raises(ValueError):
        size_position(0.0, 0.08, 500.0)
    with pytest.raises(ValueError):
        size_position(100.0, 0.0, 500.0)
    with pytest.raises(ValueError):
        size_position(100.0, 0.08, -1.0)


def test_apply_risk_limits_caps_exposure():
    cfg = RiskConfig(capital_at_risk=10_000.0, leverage_ceiling=1.0)
    out = apply_risk_limits(np.array([0.6, 0.4]), 100_000.0, cfg)
    assert out["capped"]
    assert out["exposure"] == pytest.approx(10_000.0)
    assert out["borrowed"] == 0.0


def test_borrowing_cost_reduces_returns():
    """Levered 1.5x with a 5% borrow rate must underperform the same levered
    book with borrowing free -- the cost is real."""
    data = synthetic.random_walk_prices(["AAA"], "2020-01-01", "2020-12-31", seed=9)
    idx = data["AAA"].index
    sig = pd.DataFrame({"AAA": np.full(len(idx), 1.5)}, index=idx)

    def run(rate):
        cfg = RiskConfig(capital_at_risk=10**12, leverage_ceiling=2.0, borrow_rate_annual=rate)
        res, stats = backtest_with_risk(data, sig, cfg, costs=Costs(0, 0))
        return res, stats

    res_free, stats_free = run(0.0)
    res_paid, stats_paid = run(0.05)
    assert stats_paid["total_borrow_cost"] > 0
    assert stats_free["total_borrow_cost"] == 0
    assert res_paid.metrics["total_return"] < res_free.metrics["total_return"]


def test_backtest_with_risk_caps_days():
    data = synthetic.random_walk_prices(["AAA"], "2020-01-01", "2020-03-31", seed=9)
    idx = data["AAA"].index
    sig = pd.DataFrame({"AAA": np.ones(len(idx))}, index=idx)
    cfg = RiskConfig(capital_at_risk=10_000.0, leverage_ceiling=1.0)
    res, stats = backtest_with_risk(data, sig, cfg, costs=Costs(0, 0))
    # Day 0 has zero effective weight (next-open execution); every later day capped.
    assert stats["capped_days"] == len(idx) - 1
    assert (res.equity > 0).all()


def test_metrics_key_parity_engine_vs_risk():
    # Regression: scripts/assemble.py crashed with KeyError 'exposure_pct'
    # because backtest_with_risk metrics lacked the key the plain engine sets.
    data = synthetic.random_walk_prices(["AAA"], "2020-01-01", "2020-03-31", seed=9)
    idx = data["AAA"].index
    sig = pd.DataFrame({"AAA": np.ones(len(idx))}, index=idx)
    plain = backtest(data, sig, costs=Costs(0, 0))
    cfg = RiskConfig(capital_at_risk=10**12, leverage_ceiling=1.0)
    risk_res, _ = backtest_with_risk(data, sig, cfg, costs=Costs(0, 0))
    assert "exposure_pct" in risk_res.metrics
    assert risk_res.metrics["exposure_pct"] == pytest.approx(plain.metrics["exposure_pct"])
    assert risk_res.metrics["exposure_pct"] > 0


def _result_with(sharpe, total_return, max_dd=0.10, n_trades=100):
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    return BacktestResult(
        equity=pd.Series(np.linspace(100.0, 110.0, 10), index=idx),
        trades=pd.DataFrame(),
        metrics={"sharpe": sharpe, "total_return": total_return,
                 "max_drawdown": max_dd, "num_trades": n_trades},
    )


def test_beat_benchmark_on_sharpe():
    # Higher Sharpe but lower raw return than the benchmark: passes under the
    # Sharpe rule (chosen 2026-09-18), fails under the old total-return rule.
    cand = _result_with(sharpe=0.90, total_return=0.50)
    bench = _result_with(sharpe=0.75, total_return=4.74)
    crit = PassCriteria(min_trades=1, min_sharpe=0.0, max_drawdown=0.15,
                        must_beat_benchmark=True, beat_benchmark_on="sharpe")
    passed, reasons = evaluate(cand, bench, crit)
    assert passed, reasons

    crit_tr = PassCriteria(min_trades=1, min_sharpe=0.0, max_drawdown=0.15,
                           must_beat_benchmark=True, beat_benchmark_on="total_return")
    passed_tr, reasons_tr = evaluate(cand, bench, crit_tr)
    assert not passed_tr
    assert any("total_return" in r for r in reasons_tr)


def test_beat_benchmark_sharpe_failure_names_metric():
    cand = _result_with(sharpe=0.60, total_return=0.50)
    bench = _result_with(sharpe=0.75, total_return=4.74)
    crit = PassCriteria(min_trades=1, min_sharpe=0.0, max_drawdown=0.15,
                        must_beat_benchmark=True, beat_benchmark_on="sharpe")
    passed, reasons = evaluate(cand, bench, crit)
    assert not passed
    assert any(r.startswith("sharpe") for r in reasons)


def test_beat_benchmark_on_rejects_bad_value():
    with pytest.raises(ValueError):
        PassCriteria(beat_benchmark_on="cagr")


def test_benchmark_gate_off_passes_below_benchmark_sharpe():
    # 2026-09-18: benchmark is informational, not a gate (per Vin). A candidate
    # below the benchmark Sharpe must pass when must_beat_benchmark=False.
    cand = _result_with(sharpe=0.60, total_return=0.50)
    bench = _result_with(sharpe=1.43, total_return=4.74)
    crit = PassCriteria(min_trades=1, min_sharpe=0.5, max_drawdown=0.30,
                        must_beat_benchmark=False, beat_benchmark_on="sharpe")
    passed, reasons = evaluate(cand, bench, crit)
    assert passed, reasons


def test_validate_records_benchmark_metric_informational():
    # The benchmark comparison must still be reported in validation.csv even
    # when it is not a gate, so the information isn't lost.
    data = synthetic.random_walk_prices(
        ["QQQ", "GLD"], "2022-06-01", "2023-06-30", seed=3
    )
    crit = PassCriteria(min_trades=1, min_sharpe=0.0, max_drawdown=0.30,
                        must_beat_benchmark=False, beat_benchmark_on="sharpe")
    df = validate_candidates(_tiny_candidates(), data, "2023-01-01", Costs(), crit)
    assert len(df) == 1
    assert "benchmark_sharpe" in df.columns
    assert df["benchmark_sharpe"].iloc[0] > 0
