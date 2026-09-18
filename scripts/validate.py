#!/usr/bin/env python3
"""Validate search candidates on sealed out-of-sample test data.

Reads results/candidates.csv (selected on TRAIN data), re-runs every candidate
on [test_start, today), and writes results/validation.csv with train vs test
metrics side by side.

Usage:
    python scripts/validate.py --config configs/growth_daily.yaml
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from trading_system.criteria import PassCriteria
from trading_system.engine import Costs
from trading_system.research.validate import (
    DEFAULT_WARMUP_DAYS,
    check_train_test_separation,
    download_window,
    print_summary,
    save_validation,
    validate_candidates,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_start, train_end, test_start = cfg["train_start"], cfg["train_end"], cfg["test_start"]
    check_train_test_separation(train_start, train_end, test_start)
    print(f"Train/test separation OK: train [{train_start}, {train_end}] test from {test_start}")

    results_dir = Path(cfg.get("results_dir", "results"))
    cand_path = results_dir / "candidates.csv"
    if not cand_path.exists():
        raise FileNotFoundError(f"{cand_path} not found -- run scripts/run_search.py first.")
    candidates = pd.read_csv(cand_path)
    print(f"Loaded {len(candidates)} candidates from {cand_path}")

    warmup = cfg.get("validate", {}).get("warmup_days", DEFAULT_WARMUP_DAYS)
    end = date.today().isoformat()
    data, te_s = download_window(
        cfg["profile"]["universe"], test_start, end,
        cache_dir=cfg.get("cache_dir", "data/cache"), warmup_days=warmup,
    )
    print(f"Test data: {min(d.index[0] for d in data.values()).date()} -> "
          f"{max(d.index[-1] for d in data.values()).date()} "
          f"(scoring from {te_s.date()}, warmup buffer before that)")

    costs = Costs(**cfg.get("costs", {}))
    criteria = PassCriteria(**cfg.get("criteria", {}))
    df = validate_candidates(candidates, data, test_start, costs, criteria)
    out_path = results_dir / "validation.csv"
    save_validation(df, out_path)
    print(f"Wrote {out_path}")
    print_summary(df)


if __name__ == "__main__":
    main()
