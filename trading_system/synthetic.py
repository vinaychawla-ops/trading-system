"""Deterministic synthetic OHLCV data for tests and smoke runs.

Geometric Brownian motion with a fixed seed. Never used for real research.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk_prices(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 7,
    start_price: float = 100.0,
    drift: float = 0.0005,
    vol: float = 0.02,
) -> dict[str, pd.DataFrame]:
    """Generate business-day OHLCV frames per ticker (deterministic via seed)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    out: dict[str, pd.DataFrame] = {}
    for k, ticker in enumerate(tickers):
        rets = rng.normal(drift, vol, len(idx))
        close = start_price * (1 + 0.1 * k) * np.exp(np.cumsum(rets))
        open_ = close * (1 + rng.normal(0, 0.002, len(idx)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(idx))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(idx))))
        volume = rng.integers(1_000_000, 10_000_000, len(idx))
        out[ticker] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=idx,
        )
    return out
