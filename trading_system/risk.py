"""Risk sizing: the three user levers applied to strategy weights.

The user supplies:
  * ``max_loss_per_trade`` (dollars) -- the most one trade may lose.
  * ``capital_at_risk`` (dollars) -- cap on total market exposure.
  * ``leverage_ceiling`` (e.g. 1.0 = no leverage) -- max exposure as a multiple
    of equity; borrowing above 1.0x accrues ``borrow_rate_annual`` on the
    borrowed portion.

Strategies in this repo don't emit stop-losses, so ``size_position`` uses a
configurable ``default_stop_pct`` (documented assumption: 8%). Position sizing
is primarily a live-trading (dashboard phase) concern; ``backtest_with_risk``
replays history with the exposure cap and borrowing costs applied so the
assemble summary can report capped days and borrowing drag honestly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import BacktestResult, Costs, compute_metrics

TRADING_DAYS = 252


@dataclass
class RiskConfig:
    max_loss_per_trade: float = 500.0
    capital_at_risk: float = 80_000.0
    leverage_ceiling: float = 1.0  # 1.0 = no leverage
    borrow_rate_annual: float = 0.05
    default_stop_pct: float = 0.08  # assumption: strategies have no native stops

    def __post_init__(self):
        if self.max_loss_per_trade <= 0:
            raise ValueError("max_loss_per_trade must be > 0")
        if self.capital_at_risk <= 0:
            raise ValueError("capital_at_risk must be > 0")
        if self.leverage_ceiling < 1.0:
            raise ValueError("leverage_ceiling must be >= 1.0 (use capital_at_risk to invest less)")
        if not 0.0 <= self.borrow_rate_annual:
            raise ValueError("borrow_rate_annual must be >= 0")
        if not 0.0 < self.default_stop_pct < 1.0:
            raise ValueError("default_stop_pct must be in (0, 1)")


def size_position(entry_price: float, stop_pct: float, max_loss_per_trade: float) -> float:
    """Shares to buy so a ``stop_pct`` adverse move loses ``max_loss_per_trade``.

    shares = max_loss / (entry_price * stop_pct).
    Example: entry $100, 8% stop, $500 max loss -> 62.5 shares.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    if not 0.0 < stop_pct < 1.0:
        raise ValueError("stop_pct must be in (0, 1)")
    if max_loss_per_trade <= 0:
        raise ValueError("max_loss_per_trade must be > 0")
    return max_loss_per_trade / (entry_price * stop_pct)


def apply_risk_limits(
    weights: np.ndarray, equity: float, cfg: RiskConfig
) -> dict:
    """Scale one day's target weights to the risk limits.

    * Exposure cap = min(capital_at_risk, leverage_ceiling * equity).
    * Weights are scaled down proportionally when gross exposure exceeds the cap.
    * Anything above 1.0x equity counts as borrowed.
    """
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("apply_risk_limits expects long-only (non-negative) weights")
    if equity <= 0:
        return {
            "weights": np.zeros_like(w), "exposure": 0.0, "borrowed": 0.0,
            "capped": True, "cap": 0.0, "scale": 0.0,
        }
    gross = float(w.sum() * equity)
    cap = min(cfg.capital_at_risk, cfg.leverage_ceiling * equity)
    scale = min(1.0, cap / gross) if gross > 0 else 1.0
    scaled = w * scale
    exposure = float(scaled.sum() * equity)
    borrowed = max(0.0, exposure - equity)
    return {
        "weights": scaled, "exposure": exposure, "borrowed": borrowed,
        "capped": bool(scale < 1.0 - 1e-12), "cap": cap, "scale": float(scale),
    }


def backtest_with_risk(
    prices: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    risk_cfg: RiskConfig,
    costs: Costs = Costs(),
    initial_capital: float = 100_000.0,
) -> tuple[BacktestResult, dict]:
    """Replay a backtest with per-day risk limits and borrowing costs.

    Same execution model as ``engine.backtest`` (signal at close t -> trade at
    open t+1, costs on traded notional), except:
      * each day's target weights pass through ``apply_risk_limits``;
      * the borrowed portion accrues ``borrow_rate_annual / 252`` per day,
        deducted from cash.

    Returns (BacktestResult, stats) where stats has ``capped_days``,
    ``total_borrow_cost`` and ``max_single_name_dollars``.
    """
    tickers = [c for c in signals.columns if c in prices]
    if not tickers:
        raise ValueError("No signal columns match tickers in `prices`")
    idx = signals.index
    opens = pd.DataFrame({t: prices[t]["Open"] for t in tickers}).reindex(idx)
    closes = pd.DataFrame({t: prices[t]["Close"] for t in tickers}).reindex(idx)

    weights = signals[tickers].astype(float).clip(0.0, risk_cfg.leverage_ceiling)
    weights = weights.where(~opens.isna(), 0.0)
    opens = opens.ffill().bfill()
    closes = closes.ffill().bfill()
    effective = weights.shift(1).fillna(0.0)

    o = opens.to_numpy(dtype=float)
    c = closes.to_numpy(dtype=float)
    w = effective.to_numpy(dtype=float)
    cost_rate = costs.rate
    daily_borrow = risk_cfg.borrow_rate_annual / TRADING_DAYS

    shares = np.zeros(len(tickers))
    cash = float(initial_capital)
    equity_vals = np.empty(len(idx))
    limited_rows = np.zeros((len(idx), len(tickers)))
    trade_rows: list[dict] = []
    capped_days = 0
    total_borrow_cost = 0.0
    max_single_name = 0.0
    broke = False

    for i in range(len(idx)):
        oi, ci = o[i], c[i]
        v_open = cash + float(shares @ oi)
        if v_open <= 0:
            broke = True
            equity_vals[i:] = 0.0
            break
        lim = apply_risk_limits(w[i], v_open, risk_cfg)
        if lim["capped"]:
            capped_days += 1
        limited_rows[i] = lim["weights"]
        with np.errstate(divide="ignore", invalid="ignore"):
            target = np.where(oi > 0, lim["weights"] * v_open / oi, 0.0)
        delta = target - shares
        traded_notional = float(np.abs(delta) @ oi)
        trade_cost = traded_notional * cost_rate
        borrow_cost = lim["borrowed"] * daily_borrow
        total_borrow_cost += borrow_cost
        cash = v_open - float(target @ oi) - trade_cost - borrow_cost
        shares = target
        equity_vals[i] = cash + float(shares @ ci)
        max_single_name = max(max_single_name, float(np.max(shares * ci)) if len(shares) else 0.0)

        nz = np.abs(delta) > 1e-9
        for j in np.flatnonzero(nz):
            trade_rows.append(
                {
                    "date": idx[i], "ticker": tickers[j],
                    "side": "BUY" if delta[j] > 0 else "SELL",
                    "shares": float(abs(delta[j])), "price": float(oi[j]),
                }
            )

    equity = pd.Series(equity_vals, index=idx, name="equity")
    trades = pd.DataFrame(trade_rows, columns=["date", "ticker", "side", "shares", "price"])
    metrics = compute_metrics(equity, trades)
    # Same key the plain engine sets: average gross exposure actually held
    # (post risk limits; can exceed 100% when leverage_ceiling > 1).
    metrics["exposure_pct"] = float(limited_rows.sum(axis=1).mean() * 100.0)
    stats = {
        "capped_days": capped_days,
        "total_borrow_cost": float(total_borrow_cost),
        "max_single_name_dollars": float(max_single_name),
        "bankrupt": broke,
    }
    return BacktestResult(equity=equity, trades=trades, metrics=metrics), stats
