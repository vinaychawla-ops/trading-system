"""Sleeve strategies: short-horizon tactical trades on individual stocks.

All strategies are long-only and equal-weighted with a cap on simultaneous
positions (``max_positions``) and at most one position per ticker. Signals are
target weights *decided at each day's close*; the engine makes them effective at
the next open, so no lookahead is possible.

When ``universe`` is None, all tickers in ``prices`` are used except the
benchmark / safe-haven tickers (QQQ, GLD, SPY).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr, rsi, sma
from .core import NON_EQUITY_TICKERS


def _universe(prices: dict[str, pd.DataFrame], universe: list[str] | None) -> list[str]:
    tickers = list(universe) if universe else [t for t in prices if t not in NON_EQUITY_TICKERS]
    missing = [t for t in tickers if t not in prices]
    if missing:
        raise KeyError(f"tickers missing from prices: {missing}")
    return tickers


def _common_index(
    prices: dict[str, pd.DataFrame], tickers: list[str]
) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """Align all tickers to the first ticker's calendar (forward-filled)."""
    idx = prices[tickers[0]].index
    frames = {}
    for t in tickers:
        frames[t] = prices[t].reindex(idx).ffill()
    return idx, frames


def _entries_to_signals(
    entries: pd.DataFrame, hold_days: int, max_positions: int
) -> pd.DataFrame:
    """Turn boolean entry flags (decided at close t) into target-weight signals.

    An entry at close t keeps the position active for closes t..t+hold_days-1.
    When more than ``max_positions`` are active, the most recently entered are
    kept (ties broken by ticker name for determinism). Each chosen position gets
    weight 1/max_positions, so total sleeve weight never exceeds 1.
    """
    if hold_days < 1:
        raise ValueError("hold_days must be >= 1")
    if max_positions < 1:
        raise ValueError("max_positions must be >= 1")
    entries = entries.fillna(False).astype(bool)
    signals = pd.DataFrame(0.0, index=entries.index, columns=entries.columns)
    last_entry = pd.Series(-1, index=entries.columns)  # integer position of last entry
    arr = entries.to_numpy()
    cols = list(entries.columns)
    weight = 1.0 / max_positions
    for i in range(len(entries)):
        entered = arr[i]
        for j, flag in enumerate(entered):
            if flag:
                last_entry.iloc[j] = i
        active = [cols[j] for j in range(len(cols)) if i - last_entry.iloc[j] < hold_days]
        active.sort(key=lambda t: (-last_entry[t], t))
        for t in active[:max_positions]:
            signals.iat[i, signals.columns.get_loc(t)] = weight
    return signals


def trend_pullback(
    prices: dict[str, pd.DataFrame],
    universe: list[str] | None = None,
    lookback: int = 200,
    rsi_window: int = 5,
    rsi_thresh: float = 30.0,
    hold_days: int = 10,
    max_positions: int = 5,
) -> pd.DataFrame:
    """Buy pullbacks in uptrends: Close > SMA(lookback) and RSI < rsi_thresh."""
    tickers = _universe(prices, universe)
    idx, frames = _common_index(prices, tickers)
    entries = pd.DataFrame(False, index=idx, columns=tickers)
    for t in tickers:
        close = frames[t]["Close"]
        uptrend = close > sma(close, lookback)
        oversold = rsi(close, rsi_window) < rsi_thresh
        entries[t] = (uptrend & oversold).fillna(False)
    return _entries_to_signals(entries, hold_days, max_positions)


def low_range_close(
    prices: dict[str, pd.DataFrame],
    universe: list[str] | None = None,
    lookback: int = 200,
    range_window: int = 10,
    pct: float = 0.25,
    hold_days: int = 10,
    max_positions: int = 5,
) -> pd.DataFrame:
    """Buy when an uptrending stock closes in the bottom ``pct`` of its
    ``range_window``-day high/low range."""
    tickers = _universe(prices, universe)
    idx, frames = _common_index(prices, tickers)
    entries = pd.DataFrame(False, index=idx, columns=tickers)
    for t in tickers:
        df = frames[t]
        close, high, low = df["Close"], df["High"], df["Low"]
        uptrend = close > sma(close, lookback)
        roll_high = high.rolling(range_window, min_periods=range_window).max()
        roll_low = low.rolling(range_window, min_periods=range_window).min()
        rng = (roll_high - roll_low).replace(0.0, np.nan)
        position_in_range = (close - roll_low) / rng
        cheap = position_in_range < pct
        entries[t] = (uptrend & cheap).fillna(False)
    return _entries_to_signals(entries, hold_days, max_positions)


def quiet_pullback(
    prices: dict[str, pd.DataFrame],
    universe: list[str] | None = None,
    lookback: int = 200,
    pullback_days: int = 3,
    range_atr_frac: float = 0.5,
    hold_days: int = 10,
    max_positions: int = 5,
) -> pd.DataFrame:
    """Buy ``pullback_days`` consecutive lower closes in an uptrend, where each
    day's range is quiet (daily range < ``range_atr_frac`` * ATR(14))."""
    tickers = _universe(prices, universe)
    idx, frames = _common_index(prices, tickers)
    entries = pd.DataFrame(False, index=idx, columns=tickers)
    for t in tickers:
        df = frames[t]
        close, high, low = df["Close"], df["High"], df["Low"]
        uptrend = close > sma(close, lookback)
        lower = close.diff() < 0
        consec_lower = lower.rolling(pullback_days, min_periods=pullback_days).sum() == pullback_days
        quiet = (high - low) < range_atr_frac * atr(high, low, close, 14)
        quiet_all = quiet.rolling(pullback_days, min_periods=pullback_days).sum() == pullback_days
        entries[t] = (uptrend & consec_lower & quiet_all).fillna(False)
    return _entries_to_signals(entries, hold_days, max_positions)


SLEEVE_STRATEGIES = {
    "trend_pullback": trend_pullback,
    "low_range_close": low_range_close,
    "quiet_pullback": quiet_pullback,
}
