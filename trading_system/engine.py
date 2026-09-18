"""No-lookahead daily backtesting engine.

Execution model (the core no-lookahead guarantee):
  * ``signals`` holds *target position weights* decided at each day's CLOSE.
  * A weight decided at close t becomes effective at the NEXT trading day's OPEN
    (``signals.shift(1)``). A strategy therefore can never profit from information
    that only exists after the close.
  * Rebalances trade at the next open price. Fractional shares are allowed.

Cost model:
  * Every rebalance pays ``(commission_bps + slippage_bps) / 10000`` on the total
    traded notional (buys + sells) at that rebalance. There are no partial fills
    and no market-impact model beyond the slippage term.

Weight handling: long-only weights are clipped to [0, 1]; if a day's weights sum
above 1 they are scaled down proportionally. Missing prices are forward-filled
(then back-filled); weights are forced to 0 wherever a price was originally NaN.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Costs:
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    min_trade_notional: float = 1.0
    """Trades smaller than this notional (dust from cost-chasing on a fully
    invested portfolio) are not executed or recorded: no broker fills them,
    and they only inflate trade counts."""

    @property
    def rate(self) -> float:
        """Total cost as a fraction of traded notional."""
        return (self.commission_bps + self.slippage_bps) / 10_000.0


@dataclass
class BacktestResult:
    equity: pd.Series  # portfolio value at each day's close
    trades: pd.DataFrame  # date, ticker, side, shares, price
    metrics: dict = field(default_factory=dict)


def compute_metrics(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    """Performance metrics from a daily equity curve.

    win_rate / avg_win / avg_loss are computed on *daily* returns (not round
    trips), since this engine does not track closed round-trip trades.
    """
    m: dict = {}
    n = len(equity)
    if n < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "num_trades": 0 if trades is None else len(trades),
            "exposure_pct": 0.0,
        }
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total_return = end / start - 1.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = (end / start) ** (1.0 / years) - 1.0
    rets = equity.pct_change().dropna()
    vol = float(rets.std())
    sharpe = float(rets.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else 0.0
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max.replace(0, np.nan)
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0
    calmar = float(cagr / max_dd) if max_dd > 0 else 0.0
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    m.update(
        {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar": float(calmar),
            "win_rate": float(len(wins) / len(rets)) if len(rets) else 0.0,
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "num_trades": 0 if trades is None else len(trades),
        }
    )
    return m


def backtest(
    prices: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    costs: Costs = Costs(),
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Run the daily backtest.

    Args:
        prices: ticker -> DataFrame with at least Open and Close columns.
        signals: DataFrame indexed by date, columns = tickers, values = target
            weights decided at that day's close (long-only 0..1).
        costs: trading cost assumptions.
        initial_capital: starting cash.
    """
    tickers = [c for c in signals.columns if c in prices]
    if not tickers:
        raise ValueError("No signal columns match tickers in `prices`")
    idx = signals.index
    opens = pd.DataFrame({t: prices[t]["Open"] for t in tickers}).reindex(idx)
    closes = pd.DataFrame({t: prices[t]["Close"] for t in tickers}).reindex(idx)

    weights = signals[tickers].astype(float).clip(0.0, 1.0)
    weights = weights.where(~opens.isna(), 0.0)  # never hold what we can't price
    row_sum = weights.sum(axis=1)
    over = row_sum > 1.0
    weights.loc[over] = weights.loc[over].div(row_sum[over], axis=0)

    opens = opens.ffill().bfill()
    closes = closes.ffill().bfill()

    # No-lookahead: weights decided at close t-1 take effect at open t.
    effective = weights.shift(1).fillna(0.0)

    o = opens.to_numpy(dtype=float)
    c = closes.to_numpy(dtype=float)
    w = effective.to_numpy(dtype=float)
    cost_rate = costs.rate

    shares = np.zeros(len(tickers))
    cash = float(initial_capital)
    equity_vals = np.empty(len(idx))
    trade_rows: list[dict] = []

    for i in range(len(idx)):
        oi, ci = o[i], c[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            v_open = cash + float(shares @ oi)
            target = np.where(oi > 0, w[i] * v_open / oi, 0.0)
        delta = target - shares
        # Skip dust: sub-minimum trades are neither executed nor recorded.
        delta = np.where(np.abs(delta) * oi > costs.min_trade_notional, delta, 0.0)
        traded_notional = float(np.abs(delta) @ oi)
        trade_cost = traded_notional * cost_rate
        shares = shares + delta
        cash = v_open - float(shares @ oi) - trade_cost
        equity_vals[i] = cash + float(shares @ ci)

        nz = np.abs(delta) > 1e-9
        for j in np.flatnonzero(nz):
            trade_rows.append(
                {
                    "date": idx[i],
                    "ticker": tickers[j],
                    "side": "BUY" if delta[j] > 0 else "SELL",
                    "shares": float(abs(delta[j])),
                    "price": float(oi[j]),
                }
            )

    equity = pd.Series(equity_vals, index=idx, name="equity")
    trades = pd.DataFrame(trade_rows, columns=["date", "ticker", "side", "shares", "price"])
    metrics = compute_metrics(equity, trades)
    metrics["exposure_pct"] = float(effective.sum(axis=1).clip(0.0, 1.0).mean() * 100.0)
    return BacktestResult(equity=equity, trades=trades, metrics=metrics)


def buy_and_hold(
    prices: dict[str, pd.DataFrame],
    ticker: str,
    costs: Costs = Costs(),
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Buy-and-hold benchmark for one ticker (invests at the first open)."""
    idx = prices[ticker].index
    signals = pd.DataFrame(1.0, index=idx, columns=[ticker])
    return backtest(prices, signals, costs=costs, initial_capital=initial_capital)


def train_test_split(
    dates: pd.DatetimeIndex, test_start: str
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split a date index into train (< test_start) and test (>= test_start)."""
    cut = pd.Timestamp(test_start)
    dates = pd.DatetimeIndex(dates)
    return dates[dates < cut], dates[dates >= cut]


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    initial_train_days: int = 756,
    step_days: int = 126,
    horizon_days: int = 126,
    expanding: bool = True,
):
    """Yield (train_dates, test_dates) windows for walk-forward analysis.

    Simple calendar-day windowing: each fold trains on ``initial_train_days``
    (or everything since the start when ``expanding``) and tests the next
    ``horizon_days``; the window then advances by ``step_days``.
    """
    dates = pd.DatetimeIndex(sorted(dates))
    start = dates[0]
    train_end = start + pd.Timedelta(days=initial_train_days)
    while True:
        test_end = train_end + pd.Timedelta(days=horizon_days)
        train_dates = dates[dates < train_end] if expanding else dates[
            (dates >= train_end - pd.Timedelta(days=initial_train_days)) & (dates < train_end)
        ]
        test_dates = dates[(dates >= train_end) & (dates < test_end)]
        if len(test_dates) == 0 or len(train_dates) == 0:
            break
        yield train_dates, test_dates
        train_end = train_end + pd.Timedelta(days=step_days)
