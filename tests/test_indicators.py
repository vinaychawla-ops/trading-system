"""Indicator tests."""
import numpy as np
import pandas as pd

from trading_system.indicators import atr, rsi, sma


def test_sma_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = sma(s, 2)
    assert out.iloc[0] != out.iloc[0]  # NaN during warmup
    assert out.iloc[-1] == 3.5


def test_rsi_monotonic_up_is_100():
    close = pd.Series(100.0 + np.arange(50, dtype=float))
    out = rsi(close, 2)
    assert out.iloc[-1] == 100.0


def test_rsi_monotonic_down_is_0():
    close = pd.Series(100.0 - np.arange(50, dtype=float))
    out = rsi(close, 2)
    assert out.iloc[-1] == 0.0


def test_atr_positive_and_finite():
    idx = pd.bdate_range("2020-01-01", periods=30)
    rng = np.random.default_rng(1)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 30)), index=idx)
    high = close + 1.0
    low = close - 1.0
    out = atr(high, low, close, 14)
    assert (out.dropna() > 0).all()
