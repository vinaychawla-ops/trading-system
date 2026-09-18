"""Daily OHLCV data layer (yfinance) with on-disk parquet caching.

Never invents data: if a download fails after retries, it raises.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(cache_dir: str | Path, ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace(":", "_")
    return Path(cache_dir) / f"{safe}.parquet"


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    # yfinance may return MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Download for {ticker} missing columns: {missing}")
    df = df[COLUMNS].copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise RuntimeError(f"No usable rows for {ticker} after cleaning")
    return df


def _download(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
            )
            return _normalize(df, ticker)
        except Exception as exc:  # noqa: BLE001 - retry on any download failure
            last_err = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {ticker} after {retries} attempts: {last_err}")


def get_daily_bars(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/cache",
) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for each ticker, using the parquet cache when possible.

    Cache hit rule: if ``cache_dir/<ticker>.parquet`` exists and covers the full
    requested [start, end] range, the cached slice is used and no download happens.
    Otherwise the ticker is (re)downloaded and the fresh data is *merged* into the
    existing cache (union of trading days) -- the cache only ever grows, so a
    request for a shorter window (e.g. the 2010-2022 train window ending on a
    Saturday) can never truncate newer cached data.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = _cache_path(cache, ticker)
        df: pd.DataFrame | None = None
        if path.exists():
            cached = pd.read_parquet(path)
            cached.index = pd.DatetimeIndex(pd.to_datetime(cached.index))
            if cached.index.min() <= start_ts and cached.index.max() >= end_ts:
                df = cached
        if df is None:
            df = _download(ticker, start, end)
            if path.exists():
                # Merge instead of overwrite: a shorter-window request must not
                # destroy newer data already in the cache.
                old = pd.read_parquet(path)
                old.index = pd.DatetimeIndex(pd.to_datetime(old.index))
                df = pd.concat([old, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_parquet(path)
        out[ticker] = df.loc[start_ts:end_ts]
        if out[ticker].empty:
            raise RuntimeError(f"No rows for {ticker} in requested range {start}..{end}")
    return out
