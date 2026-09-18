"""Strategy search: exhaustive grid search + evolutionary mutation search.

Every tested candidate is appended to the ledger, pass or fail. Searches run on
TRAIN data only; out-of-sample validation on sealed data is a later phase.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from ..criteria import PassCriteria, evaluate
from ..engine import BacktestResult, Costs, backtest
from .ledger import append_test


def _canonical(params: dict) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in params.items()))


def _test_candidate(
    strategy_fn,
    strategy_name: str,
    params: dict,
    data: dict[str, pd.DataFrame],
    costs: Costs,
    criteria: PassCriteria,
    benchmark_result: BacktestResult,
    ledger_path: str | Path,
) -> dict:
    signals = strategy_fn(data, **params)
    result = backtest(data, signals, costs=costs)
    passed, reasons = evaluate(result, benchmark_result, criteria=criteria)
    m = result.metrics
    record = {
        "strategy": strategy_name,
        "params": dict(params),
        "n_trades": int(m["num_trades"]),
        "cagr": float(m["cagr"]),
        "sharpe": float(m["sharpe"]),
        "max_dd": float(m["max_drawdown"]),
        "passed": bool(passed),
        "fail_reasons": list(reasons),
    }
    append_test(record, path=ledger_path)
    return record


def grid_search(
    strategy_fn,
    strategy_name: str,
    param_grid: dict[str, list],
    data: dict[str, pd.DataFrame],
    costs: Costs,
    criteria: PassCriteria,
    benchmark_result: BacktestResult,
    ledger_path: str | Path = "results/ledger.jsonl",
) -> pd.DataFrame:
    """Test every combination in ``param_grid``; return all results as a DataFrame."""
    keys = list(param_grid.keys())
    records = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        records.append(
            _test_candidate(
                strategy_fn, strategy_name, params, data, costs,
                criteria, benchmark_result, ledger_path,
            )
        )
    return pd.DataFrame(records)


def _perturb(params: dict, rng: np.random.Generator, perturb: float) -> dict:
    child = {}
    for k, v in params.items():
        if isinstance(v, bool):
            child[k] = v
        elif isinstance(v, int):
            new = int(round(v * rng.uniform(1 - perturb, 1 + perturb)))
            child[k] = max(1, new)
        elif isinstance(v, float):
            new = v * rng.uniform(1 - perturb, 1 + perturb)
            child[k] = max(new, 1e-9)
        else:
            child[k] = v
    return child


def mutate_search(
    strategy_fn,
    strategy_name: str,
    top_params: list[dict],
    data: dict[str, pd.DataFrame],
    costs: Costs,
    criteria: PassCriteria,
    benchmark_result: BacktestResult,
    n_rounds: int = 3,
    n_children: int = 8,
    perturb: float = 0.25,
    top_k: int = 3,
    seed: int = 42,
    ledger_path: str | Path = "results/ledger.jsonl",
) -> pd.DataFrame:
    """Evolve the best parameter sets: perturb numeric params, re-test, repeat.

    Each round takes the current top-``top_k`` by Sharpe, spawns ``n_children``
    perturbed children per parent, tests them all (ledger records everything),
    and re-ranks. Identical parameter sets are never re-tested. Deterministic
    for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    seen = {_canonical(p) for p in top_params}
    parents = list(top_params)
    all_records: list[dict] = []
    for _ in range(n_rounds):
        children: list[dict] = []
        for parent in parents:
            for _ in range(n_children):
                child = _perturb(parent, rng, perturb)
                key = _canonical(child)
                if key not in seen:
                    seen.add(key)
                    children.append(child)
        for child in children:
            all_records.append(
                _test_candidate(
                    strategy_fn, strategy_name, child, data, costs,
                    criteria, benchmark_result, ledger_path,
                )
            )
        if all_records:
            ranked = pd.DataFrame(all_records).sort_values("sharpe", ascending=False)
            parents = list(ranked.head(top_k)["params"])
    return pd.DataFrame(all_records)


def rank_passers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to passing candidates, ranked by Sharpe descending."""
    if df.empty:
        return df
    return df[df["passed"]].sort_values("sharpe", ascending=False).reset_index(drop=True)


def save_candidates(df: pd.DataFrame, path: str | Path = "results/candidates.csv") -> None:
    """Write ranked passing candidates to CSV (params serialized as JSON)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "params" in out.columns:
        out["params"] = out["params"].apply(lambda p: json.dumps(p, sort_keys=True))
    out.to_csv(path, index=False)
