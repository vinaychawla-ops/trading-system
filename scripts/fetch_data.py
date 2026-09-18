#!/usr/bin/env python3
"""Download (and cache) daily bars for the config universe.

Usage:
    python scripts/fetch_data.py --config configs/growth_daily.yaml
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from trading_system.data import get_daily_bars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    universe = cfg["profile"]["universe"]
    start = cfg["train_start"]
    end = date.today().isoformat()
    data = get_daily_bars(universe, start, end, cache_dir=cfg.get("cache_dir", "data/cache"))

    print(f"Fetched {len(data)} tickers, {start}..{end}")
    for ticker, df in data.items():
        print(f"  {ticker:8s} rows={len(df):5d}  {df.index[0].date()} -> {df.index[-1].date()}")


if __name__ == "__main__":
    main()
