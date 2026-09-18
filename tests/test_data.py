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
