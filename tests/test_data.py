"""Data layer tests (offline: cache-hit path and input validation only)."""
import numpy as np
import pandas as pd
import pytest

from trading_system.data import _normalize, get_daily_bars


def _fake_frame(start="2023-01-02", periods=10):
    idx = pd.bdate_range(start, periods=periods)
    close = 100 + np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def test_cache_hit_avoids_download(tmp_path):
    df = _fake_frame()
    (tmp_path).mkdir(exist_ok=True)
    df.to_parquet(tmp_path / "AAA.parquet")
    out = get_daily_bars(["AAA"], "2023-01-03", "2023-01-10", cache_dir=tmp_path)
    assert list(out) == ["AAA"]
    assert len(out["AAA"]) == 6  # sliced to the requested range
    assert list(out["AAA"].columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_normalize_rejects_empty():
    with pytest.raises(RuntimeError):
        _normalize(pd.DataFrame(), "AAA")


def test_cache_merge_does_not_truncate_on_shorter_request(tmp_path, monkeypatch):
    """Regression: a shorter-window request must not truncate newer cached data.

    2026-09-18: run_search requests end=2022-12-31 (a Saturday); the cache max
    is 2022-12-30 (Friday) so the hit-check misses, and the old overwrite logic
    then truncated 2023-2026 data for 37/51 tickers. The merge logic must keep
    the newer rows.
    """
    import trading_system.data as data_mod

    # Cache holds the "full" history through 2023.
    full = _fake_frame(start="2023-01-02", periods=20)
    (tmp_path).mkdir(exist_ok=True)
    full.to_parquet(tmp_path / "AAA.parquet")

    # A download for the shorter window returns only the early rows.
    short = _fake_frame(start="2023-01-02", periods=10)
    monkeypatch.setattr(data_mod, "_download", lambda *a, **k: short)

    out = get_daily_bars(["AAA"], "2023-01-02", "2023-01-13", cache_dir=tmp_path)
    assert len(out["AAA"]) == 10  # the requested slice is still exact

    healed = pd.read_parquet(tmp_path / "AAA.parquet")
    assert healed.index.max() == full.index.max()  # newer rows survived
    assert len(healed) == 20
