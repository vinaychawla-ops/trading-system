"""Tests for the Phase 4 dashboard module (hand-crafted deterministic prices)."""
import pandas as pd

from trading_system import dashboard as dash
from trading_system.strategies.core import core_rotation_signal


def _frame(px, idx):
    s = pd.Series(px, index=idx, dtype=float)
    return pd.DataFrame(
        {"Open": s, "High": s * 1.01, "Low": s * 0.99, "Close": s, "Volume": 1_000_000}
    )


def _prices(up_days=300, down_days=0):
    idx = pd.bdate_range("2020-01-01", periods=up_days + down_days)
    qqq = [100.0 + i * 0.5 for i in range(up_days)]
    last = qqq[-1]
    for _ in range(down_days):
        last *= 0.985
        qqq.append(last)
    return {"QQQ": _frame(qqq, idx), "GLD": _frame([150.0] * len(idx), idx)}


def test_summarize_hold_when_above_ma():
    summary = dash.summarize_core_signal(_prices())
    assert summary["action"] == "HOLD"
    assert summary["position"] == "QQQ"
    assert summary["ma_distance_pct"] > 5.0
    assert summary["rotation_watch"] is False


def test_summarize_rotate_after_crash():
    prices = _prices(down_days=60)
    # Slice the fixture so the last two bars straddle the QQQ -> GLD flip.
    signals = core_rotation_signal(prices)
    pos = signals.idxmax(axis=1)
    flip = pos[pos != pos.shift(1)].index[1]
    keep = pos.index.get_loc(flip) + 1  # end exactly on the flip bar: ROTATE day
    prices = {t: df.iloc[:keep] for t, df in prices.items()}
    summary = dash.summarize_core_signal(prices)
    assert summary["action"] == "ROTATE"
    assert summary["rotate_from"] == "QQQ"
    assert summary["rotate_to"] == "GLD"
    assert summary["ma_distance_pct"] < 0


def test_rotation_watch_near_ma():
    idx = pd.bdate_range("2021-01-01", periods=51)
    qqq = [100.0] * 50 + [101.0]  # ends ~0.8% above its 5-day MA
    prices = {"QQQ": _frame(qqq, idx), "GLD": _frame([150.0] * 51, idx)}
    summary = dash.summarize_core_signal(prices, ma=5)
    assert summary["rotation_watch"] is True
    assert abs(summary["ma_distance_pct"]) < 2.0


def test_rotation_history_lists_flips():
    prices = _prices(down_days=60)
    signals = core_rotation_signal(prices)
    equity = pd.Series(range(len(signals)), index=signals.index, dtype=float) + 100.0
    hist = dash.rotation_history(signals, equity)
    assert list(hist.columns) == ["date", "from", "to", "days_held", "holding_return"]
    assert len(hist) >= 1
    first = hist.iloc[0]
    assert (first["from"], first["to"]) == ("QQQ", "GLD")
    assert first["days_held"] > 0


def test_current_drawdown():
    eq = pd.Series([100.0, 120.0, 90.0, 100.0])
    assert dash.current_drawdown(eq) == 1.0 - 100.0 / 120.0


def test_svg_chart_renders():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2022-01-01", periods=3))
    svg = dash._svg_line_chart([("Test", s)], title="Hello")
    assert "<svg" in svg and "Hello" in svg and "Test" in svg


def test_render_html_contains_signal():
    prices = _prices()
    signals = core_rotation_signal(prices)
    equity = pd.Series(
        [100_000.0 + i * 10 for i in range(len(signals))], index=signals.index
    )
    summary = dash.summarize_core_signal(prices)
    rotations = dash.rotation_history(signals, equity)
    html_page = dash.render_dashboard_html(
        summary=summary,
        equity=equity,
        rotations=rotations,
        metrics={"cagr": 0.10, "sharpe": 1.0, "max_drawdown": 0.05, "num_trades": 4},
        drawdown_cap=0.30,
        asof_label="2022-01-01 close",
    )
    assert "HOLD QQQ" in html_page
    assert "Methodology" in html_page
    assert "30%" in html_page


def test_generate_dashboard_html_offline_with_prices():
    # The shared local/Modal code path, with synthetic prices so no download.
    html_page, summary = dash.generate_dashboard_html(prices=_prices())
    assert summary["action"] == "HOLD"
    assert summary["position"] == "QQQ"
    assert "HOLD QQQ" in html_page
    assert "Core signal dashboard" in html_page
