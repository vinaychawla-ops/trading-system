"""Strategy tests: core rotation behavior and sleeve signal structure."""
import numpy as np
import pandas as pd

from trading_system.strategies.core import core_rotation_signal
from trading_system.strategies.sleeve import (
    low_range_close,
    quiet_pullback,
    trend_pullback,
)
from trading_system.synthetic import random_walk_prices


def _bars(close_values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(close_values))
    close = pd.Series(np.asarray(close_values, dtype=float), index=idx)
    return pd.DataFrame(
        {
            "Open": close.values,
            "High": close.values * 1.005,
            "Low": close.values * 0.995,
            "Close": close.values,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def test_core_rotation_all_above_ma_holds_risky():
    qqq = _bars(100 + np.arange(300))  # always above its SMA200
    gld = _bars(np.full(300, 150.0))
    signals = core_rotation_signal({"QQQ": qqq, "GLD": gld})
    assert (signals["QQQ"] == 1.0).all()
    assert (signals["GLD"] == 0.0).all()


def test_core_rotation_below_ma_holds_safe():
    qqq = _bars(300 - np.arange(300))  # always below its SMA200 (after warmup)
    gld = _bars(np.full(300, 150.0))
    signals = core_rotation_signal({"QQQ": qqq, "GLD": gld})
    tail = signals.iloc[200:]
    assert (tail["GLD"] == 1.0).all()
    assert (tail["QQQ"] == 0.0).all()
    # Weights always sum to 1.
    assert (signals.sum(axis=1) == 1.0).all()


def test_sleeve_signal_structure():
    data = random_walk_prices(["AAA", "BBB", "CCC", "QQQ", "GLD"], "2020-01-01", "2021-06-30")
    for fn in (trend_pullback, low_range_close, quiet_pullback):
        signals = fn(data, max_positions=2)
        # Benchmark tickers excluded from the sleeve universe.
        assert set(signals.columns) == {"AAA", "BBB", "CCC"}
        assert ((signals >= 0.0) & (signals <= 1.0)).all().all()
        assert (signals.sum(axis=1) <= 1.0 + 1e-9).all()
        # At most max_positions names held on any day, equally weighted.
        active = (signals > 0).sum(axis=1)
        assert (active <= 2).all()
        nz = signals.stack()
        nz = nz[nz > 0]
        assert ((nz - 0.5).abs() < 1e-9).all()  # 1/max_positions each
