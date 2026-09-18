"""Engine tests: no-lookahead enforcement, costs, metrics math, splitters."""
import numpy as np
import pandas as pd
import pytest

from trading_system.engine import (
    Costs,
    backtest,
    compute_metrics,
    train_test_split,
    walk_forward_splits,
)


def _alternating_flat_intraday(n=20):
    """All price movement happens overnight: open[t] == close[t], and closes
    alternate 100, 101, 100, 101, ... (zero drift overall).

    A signal that peeks at close[t+1] while deciding at close t cannot profit:
    the engine executes at the next open, which already includes the predicted
    move -- so the cheat systematically buys *after* up-moves and sells *after*
    down-moves, i.e. it buys high and sells low.
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100.0 + (np.arange(n) % 2), index=idx)
    df = pd.DataFrame(
        {
            "Open": close.values,
            "High": close.values,
            "Low": close.values,
            "Close": close.values,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    return {"AAA": df}


def _cheat_signals(prices):
    """Go long at close t iff tomorrow's close is higher (peeks at the future)."""
    close = prices["AAA"]["Close"]
    cheat = (close.shift(-1) > close).fillna(False).astype(float)
    return pd.DataFrame({"AAA": cheat})


def test_no_lookahead_cheating_signal_cannot_profit():
    prices = _alternating_flat_intraday()
    result = backtest(prices, _cheat_signals(prices), costs=Costs(0, 0))
    # A lookahead-violating engine (executing at close t) would earn > 0 here.
    assert result.metrics["total_return"] < 1e-9


def test_no_lookahead_with_costs_is_negative():
    prices = _alternating_flat_intraday()
    result = backtest(prices, _cheat_signals(prices), costs=Costs())
    assert result.metrics["total_return"] < 0  # punished, plus costs


def test_signal_fn_only_sees_data_up_to_t():
    """Strategy discipline pattern: build signals one day at a time from a
    data_up_to(t) view and confirm it matches the vectorized equivalent."""
    from trading_system.indicators import sma

    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(100 + np.sin(np.arange(60) / 5.0) * 5 + np.arange(60) * 0.1, index=idx)
    df = pd.DataFrame(
        {
            "Open": close.values,
            "High": close.values + 0.5,
            "Low": close.values - 0.5,
            "Close": close.values,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    prices = {"AAA": df}

    def data_up_to(t):
        return {"AAA": df.loc[:t]}

    # Loop version: decide at each close using only history up to t.
    loop_w = []
    for t in idx:
        view = data_up_to(t)["AAA"]["Close"]
        loop_w.append(1.0 if len(view) >= 20 and view.iloc[-1] > sma(view, 20).iloc[-1] else 0.0)
    loop_signals = pd.DataFrame({"AAA": loop_w}, index=idx)

    # Vectorized version over the full frame (still causal: rolling only looks back).
    vec = (close > sma(close, 20)).fillna(False).astype(float)
    vec_signals = pd.DataFrame({"AAA": vec.values}, index=idx)

    pd.testing.assert_frame_equal(loop_signals, vec_signals)
    result = backtest(prices, loop_signals, costs=Costs(0, 0))
    assert len(result.equity) == len(idx)


def test_costs_reduce_total_return():
    idx = pd.bdate_range("2021-01-01", periods=60)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 60))), index=idx)
    df = pd.DataFrame(
        {
            "Open": close.values,
            "High": close.values * 1.01,
            "Low": close.values * 0.99,
            "Close": close.values,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    prices = {"AAA": df}
    # Flip between 0 and 1 every 5 days to force many rebalances.
    w = (np.arange(60) // 5) % 2
    signals = pd.DataFrame({"AAA": w.astype(float)}, index=idx)
    free = backtest(prices, signals, costs=Costs(0, 0))
    costly = backtest(prices, signals, costs=Costs())
    assert costly.metrics["num_trades"] > 0
    assert costly.metrics["total_return"] < free.metrics["total_return"]


def test_max_drawdown_math_on_known_curve():
    idx = pd.bdate_range("2022-01-01", periods=4)
    equity = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx)
    m = compute_metrics(equity)
    assert m["max_drawdown"] == pytest.approx(0.25)  # (120-90)/120
    assert m["total_return"] == pytest.approx(0.10)


def test_train_test_split():
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=10))
    train, test = train_test_split(dates, "2020-01-08")
    assert (train < pd.Timestamp("2020-01-08")).all()
    assert (test >= pd.Timestamp("2020-01-08")).all()
    assert len(train) + len(test) == 10


def test_walk_forward_splits_cover_without_overlap():
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=400))
    splits = list(
        walk_forward_splits(dates, initial_train_days=200, step_days=50, horizon_days=50)
    )
    assert len(splits) > 1
    for train, test in splits:
        assert train.max() < test.min()
        assert len(train) > 0 and len(test) > 0


def test_dust_trades_not_executed_or_recorded():
    """Regression: after a rotation, the cost used to leave a tiny negative
    cash balance that the rebalancer chased with converging sub-dollar
    'trades' every following day. Sub-minimum trades are now skipped."""
    idx = pd.bdate_range("2021-01-01", periods=60)

    def _flat(px):
        s = pd.Series(px, index=idx)
        return pd.DataFrame(
            {"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1_000_000}
        )

    prices = {"AAA": _flat(100.0), "BBB": _flat(200.0)}
    w = pd.DataFrame(0.0, index=idx, columns=["AAA", "BBB"])
    w.loc[idx[:30], "AAA"] = 1.0
    w.loc[idx[30:], "BBB"] = 1.0
    result = backtest(prices, w, costs=Costs())
    # Buy AAA, realize the $60 entry cost the next day, then on the rotation
    # sell AAA + buy BBB and realize those costs the next day: 5 real trades,
    # and no converging dust cascade afterwards.
    assert len(result.trades) == 5
    notionals = result.trades["shares"] * result.trades["price"]
    assert (notionals >= 1.0).all()
    assert result.trades["date"].max() == idx[32]
