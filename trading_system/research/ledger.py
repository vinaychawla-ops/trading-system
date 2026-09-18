"""Append-only JSONL ledger of every backtest the search loop runs.

Each line is one tested candidate: timestamp, strategy, params, key metrics,
pass/fail and the reasons. The ledger is the audit trail -- it lets you see
what was tried, what was cut, and reproduce any result.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNS = [
    "timestamp",
    "strategy",
    "params",
    "n_trades",
    "cagr",
    "sharpe",
    "max_dd",
    "passed",
    "fail_reasons",
]

REQUIRED_KEYS = {"strategy", "params", "n_trades", "cagr", "sharpe", "max_dd", "passed"}


def append_test(record: dict, path: str | Path = "results/ledger.jsonl") -> None:
    """Append one candidate record to the JSONL ledger."""
    missing = REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"ledger record missing keys: {sorted(missing)}")
    entry = dict(record)
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    entry.setdefault("fail_reasons", [])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_ledger(path: str | Path = "results/ledger.jsonl") -> pd.DataFrame:
    """Read the ledger into a DataFrame (empty DataFrame if no ledger yet)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[COLUMNS]
