"""Core strategy: the always-invested layer of the portfolio.

Holds 100% of the core allocation in a risky asset when it is above its
long moving average, otherwise 100% in a safe-haven asset. Switches only a
few times a year by construction.
"""
from __future__ import annotations

import pandas as pd

from ..indicators import sma

#: Tickers excluded from the default sleeve universe (benchmarks / safe havens).
NON_EQUITY_TICKERS = ("QQQ", "GLD", "SPY")


def core_rotation_signal(
    prices: dict[str, pd.DataFrame],
    risky: str = "QQQ",
    safe: str = "GLD",
    ma: int = 200,
) -> pd.DataFrame:
    """Target weights decided at each day's close.

    100% ``risky`` when its close is above its ``ma``-day SMA, else 100% ``safe``.
    While the SMA is still warming up (NaN) the strategy defaults to ``risky``.
    """
    for name in (risky, safe):
        if name not in prices:
            raise KeyError(f"core_rotation_signal needs {name!r} in prices")
    close = prices[risky]["Close"]
    trend = close > sma(close, ma)
    trend = trend | sma(close, ma).isna()  # warmup -> stay in risky
    weights = pd.DataFrame(0.0, index=close.index, columns=[risky, safe])
    weights[risky] = trend.astype(float)
    weights[safe] = (~trend).astype(float)
    return weights
