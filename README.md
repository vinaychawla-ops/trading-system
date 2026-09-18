# trading-system

A daily-bar backtesting and strategy-research toolkit for US stocks/ETFs, built
around a **core + sleeve** portfolio architecture:

- **Core** — the always-invested layer. Example: hold QQQ when it is above its
  200-day moving average, otherwise hold GLD. Switches only a few times a year.
- **Sleeve** — short-horizon tactical strategies (pullback families) traded on
  individual large-cap stocks, re-picked from a researched bench.

This repo covers **Phases 1–2**: data, backtesting engine, pass criteria, the
strategy library, and the search loop. No live trading, no broker connections —
research and backtesting only.

> **Disclaimer — not financial advice.** This software is for research and
> education. Backtested performance is not a guarantee of future results;
> strategies decay, costs and market impact differ in live trading, and you can
> lose money. Paper-trade any candidate for at least a month or two before
> considering real capital, and never trade money you cannot afford to lose.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

All scripts run from the repo root.

```bash
# 1. Download daily bars for the config universe (cached as parquet in data/cache)
python scripts/fetch_data.py --config configs/growth_daily.yaml

# 2. Run the strategy search on TRAIN data (writes results/ledger.jsonl + results/candidates.csv)
python scripts/run_search.py --config configs/growth_daily.yaml

# 3. Print the ranked candidate table
python scripts/summarize.py
python scripts/summarize.py --top 10

# Fast smoke test (synthetic data, no downloads)
python scripts/run_search.py --config configs/smoke.yaml --smoke
```

Run the test suite:

```bash
pytest
```

## Project map

```
trading_system/
  data.py            # yfinance daily bars + parquet cache (3 retries, never invents data)
  engine.py          # no-lookahead daily backtester, costs, metrics, splitters
  criteria.py        # PassCriteria + evaluate() -- write rules BEFORE testing
  indicators.py      # sma, rsi (Wilder), atr (Wilder) -- all causal
  synthetic.py       # deterministic random-walk bars for tests/smoke runs
  strategies/
    core.py          # core_rotation_signal: QQQ > SMA200 ? QQQ : GLD
    sleeve.py        # trend_pullback, low_range_close, quiet_pullback
  research/
    ledger.py        # append-only JSONL audit trail of every tested candidate
    search.py        # grid_search + mutate_search (evolutionary perturbation)
configs/
  growth_daily.yaml  # full profile: 40-ticker universe, 2010-2022 train, costs, criteria, grids
  smoke.yaml         # tiny config for fast smoke tests
scripts/
  fetch_data.py      # download + cache the universe
  run_search.py      # grid + mutate search on train data -> candidates.csv
  summarize.py       # print ranked candidates
tests/               # pytest suite (no-lookahead, costs, metrics, strategies, ledger, data)
data/cache/          # downloaded parquet bars (git-ignored)
results/             # ledger.jsonl, candidates.csv (git-ignored)
```

## The research loop

1. **Profile** (`configs/*.yaml`) — goal, universe, train/test dates, costs, criteria,
   and parameter grids. Small grids keep runs fast.
2. **Data** — `get_daily_bars()` caches per-ticker parquet; cache is reused when it
   covers the requested range.
3. **Criteria first** — `PassCriteria` (min trades, min Sharpe, max drawdown,
   must-beat-QQQ, correlation cap) is fixed before any test is run.
4. **Engine** — signals hold *target weights decided at the close*; positions take
   effect at the *next open*. Costs (commission + slippage in bps) apply to traded
   notional on every rebalance.
5. **Search** — `grid_search` exhausts the grid on train data; `mutate_search`
   perturbs the top parameter sets for a few evolutionary rounds. Everything is
   appended to `results/ledger.jsonl`, so winners can be audited and losers studied.
6. **Candidates** — passing strategies ranked by Sharpe land in
   `results/candidates.csv`.

The 2023+ data is sealed off from the search (see `test_start` in the config);
out-of-sample validation and walk-forward sleeve re-picking arrive in Phase 3.

## Config reference

| Key | Meaning |
|---|---|
| `profile.universe` | Tickers to download (must include `QQQ` for the benchmark; sleeve strategies auto-exclude `QQQ`/`GLD`/`SPY`) |
| `train_start` / `train_end` | Date range the search is allowed to see |
| `test_start` | Sealed data start (Phase 3 validation) |
| `cache_dir` / `results_dir` | Where bars and outputs live |
| `costs.commission_bps` / `costs.slippage_bps` | Per-rebalance cost on traded notional |
| `criteria.*` | `min_trades`, `min_sharpe`, `max_drawdown` (fraction), `must_beat_benchmark`, `max_pairwise_corr` |
| `param_grids.<strategy>` | Parameter name -> list of values to exhaust |
| `mutate.*` | `n_rounds`, `n_children`, `perturb` (fraction), `top_k`, `seed` |

## Key assumptions

- Daily bars only; decisions at the close execute at the next open (no intraday).
- Long-only, fractional shares, no leverage, no shorting in this phase.
- Costs are a flat bps charge on traded notional — no partial fills, no market impact model.
- Corporate actions: prices come from yfinance unadjusted OHLC; use with care around splits/dividends for single-name backtests over long horizons.
- The benchmark is QQQ buy-and-hold over the same train window, run through the same engine (so it pays the same entry cost).
