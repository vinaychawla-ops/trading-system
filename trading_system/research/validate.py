"""Out-of-sample validation: re-run search candidates on sealed test data.

Discipline rules (the whole point of this module):
  * Candidates in ``results/candidates.csv`` were selected on TRAIN data only.
  * Validation scores them on the TEST window [test_start, data end) only.
  * A warmup buffer of pre-test data is used *solely* to seed causal
    indicators (SMA/RSI/ATR) at the start of the window. Indicators are causal,
    so a signal at test date t depends only on data <= t; the warmup data was
    already seen by the search and cannot leak future information. The scored
    equity curve starts strictly at >= test_start.
  * ``check_train_test_separation`` refuses overlapping train/test ranges.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from ..criteria import PassCriteria, evaluate
from ..engine import Costs, backtest, buy_and_hold
from ..strategies.core import core_rotation_signal
from ..strategies.sleeve import SLEEVE_STRATEGIES

STRATEGY_REGISTRY = {"core_rotation": core_rotation_signal, **SLEEVE_STRATEGIES}

#: Calendar-day buffer before test_start used only for indicator warmup.
DEFAULT_WARMUP_DAYS = 300


def check_train_test_separation(train_start: str, train_end: str, test_start: str) -> None:
    """Raise if the train range overlaps the test range.

    The search may only see data < test_start. Any overlap means the
    "out-of-sample" claim is void.
    """
    tr_s, tr_e, te_s = pd.Timestamp(train_start), pd.Timestamp(train_end), pd.Timestamp(test_start)
    if not tr_s < tr_e:
        raise ValueError(f"train_start {train_start} must be before train_end {train_end}")
    if not tr_e < te_s:
        raise ValueError(
            f"TRAIN/TEST OVERLAP: train_end {train_end} is not before test_start {test_start}. "
            "Candidates selected on this train range cannot be validated on this test range."
        )


def download_window(
    tickers: list[str],
    test_start: str,
    end: str,
    cache_dir: str | Path,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    downloader=None,
) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    """Download [test_start - warmup, end]; return (data, test_start).

    ``downloader`` is injectable for tests: fn(tickers, start, end, cache_dir).
    The scoring window always starts at >= test_start -- warmup rows exist only
    so causal indicators are defined on day one of the test window.
    """
    from ..data import get_daily_bars

    downloader = downloader or get_daily_bars
    te_s = pd.Timestamp(test_start)
    dl_start = (te_s - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    data = downloader(tickers, dl_start, end, cache_dir=cache_dir)
    return data, te_s


def _scoring_slice(data: dict[str, pd.DataFrame], test_start: pd.Timestamp):
    """Slice every frame to the scoring window; refuse anything earlier."""
    out = {}
    for t, df in data.items():
        sl = df.loc[df.index >= test_start]
        if sl.empty:
            raise ValueError(f"No test-window rows for {t} at/after {test_start.date()}")
        out[t] = sl
    return out


def validate_candidates(
    candidates: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    test_start: str | pd.Timestamp,
    costs: Costs,
    criteria: PassCriteria,
) -> pd.DataFrame:
    """Re-run every candidate on the sealed test window.

    Args:
        candidates: rows with ``strategy``, ``params`` (JSON), and train metrics
            (``sharpe``, ``cagr``, ``max_dd``, ``n_trades``).
        data: full frames covering [test_start - warmup, end].
        test_start: first scored date; anything earlier is warmup only.
    Returns:
        DataFrame with train metrics, test metrics, and ``survived`` bool.
    """
    te_s = pd.Timestamp(test_start)
    if candidates.empty:
        return pd.DataFrame()

    test_data = _scoring_slice(data, te_s)
    if "QQQ" not in test_data:
        raise KeyError("Test data must include QQQ (benchmark)")
    benchmark = buy_and_hold(test_data, "QQQ", costs=costs)

    rows = []
    for _, row in candidates.iterrows():
        name = row["strategy"]
        if name not in STRATEGY_REGISTRY:
            raise KeyError(f"Unknown strategy in candidates.csv: {name!r}")
        params = json.loads(row["params"]) if isinstance(row["params"], str) else dict(row["params"])
        signals = STRATEGY_REGISTRY[name](data, **params)  # causal; computed on full incl. warmup
        signals = signals.loc[signals.index >= te_s]  # score strictly on the test window
        if signals.empty:
            raise ValueError(f"No signal rows in test window for candidate {name} {params}")
        result = backtest(test_data, signals, costs=costs)
        passed, reasons = evaluate(result, benchmark, criteria=criteria)
        m = result.metrics
        rows.append(
            {
                "strategy": name,
                "params": json.dumps(params, sort_keys=True),
                "train_sharpe": float(row["sharpe"]),
                "train_cagr": float(row["cagr"]),
                "train_max_dd": float(row["max_dd"]),
                "train_n_trades": int(row["n_trades"]),
                "test_sharpe": float(m["sharpe"]),
                "test_cagr": float(m["cagr"]),
                "test_max_dd": float(m["max_drawdown"]),
                "test_n_trades": int(m["num_trades"]),
                "survived": bool(passed),
                "fail_reasons": "; ".join(reasons),
            }
        )
    out = pd.DataFrame(rows).sort_values("test_sharpe", ascending=False).reset_index(drop=True)
    out["test_window_start"] = te_s.strftime("%Y-%m-%d")
    return out


def save_validation(df: pd.DataFrame, path: str | Path = "results/validation.csv") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    n_surv = int(df["survived"].sum()) if n else 0
    print(f"\nValidated {n} candidates on sealed test data: {n_surv} survived, {n - n_surv} failed.")
    if n:
        show = df[
            ["strategy", "train_sharpe", "test_sharpe", "test_cagr", "test_max_dd", "survived"]
        ].copy()
        print(show.to_string(index=False))
        if n_surv:
            print("\nSurvivors:")
            surv = df[df["survived"]]
            print(surv[["strategy", "test_sharpe", "test_cagr", "test_max_dd", "params"]].to_string(index=False))
