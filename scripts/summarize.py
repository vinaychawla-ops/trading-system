#!/usr/bin/env python3
"""Print the ranked candidate table from a search run.

Usage:
    python scripts/summarize.py
    python scripts/summarize.py --results results/candidates.csv --top 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/candidates.csv")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"No candidates file at {path} -- run scripts/run_search.py first.")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("No passing candidates.")
        return
    df = df.head(args.top)
    cols = ["strategy", "sharpe", "cagr", "max_dd", "n_trades", "params"]
    show = df[[c for c in cols if c in df.columns]].copy()
    if "params" in show.columns:
        show["params"] = show["params"].apply(
            lambda s: json.dumps(json.loads(s), separators=(",", ":")) if isinstance(s, str) else s
        )
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
