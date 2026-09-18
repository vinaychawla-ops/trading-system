"""Ledger tests: append/read roundtrip."""
import pytest

from trading_system.research.ledger import append_test, read_ledger


def test_ledger_roundtrip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    record = {
        "strategy": "trend_pullback",
        "params": {"lookback": 50, "rsi_window": 2},
        "n_trades": 42,
        "cagr": 0.123,
        "sharpe": 0.9,
        "max_dd": 0.1,
        "passed": True,
        "fail_reasons": [],
    }
    append_test(record, path=path)
    append_test({**record, "passed": False, "fail_reasons": ["sharpe 0.10 < min_sharpe 0.5"]}, path=path)
    df = read_ledger(path)
    assert len(df) == 2
    assert df.iloc[0]["strategy"] == "trend_pullback"
    assert df.iloc[0]["params"] == {"lookback": 50, "rsi_window": 2}
    assert bool(df.iloc[0]["passed"]) is True
    assert df.iloc[1]["fail_reasons"] == ["sharpe 0.10 < min_sharpe 0.5"]
    assert "timestamp" in df.columns


def test_ledger_missing_file_returns_empty(tmp_path):
    df = read_ledger(tmp_path / "nope.jsonl")
    assert df.empty


def test_ledger_rejects_incomplete_record(tmp_path):
    with pytest.raises(ValueError):
        append_test({"strategy": "x"}, path=tmp_path / "ledger.jsonl")
