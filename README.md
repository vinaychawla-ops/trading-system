# trading-system

A daily-bar backtesting and strategy-research toolkit for US stocks/ETFs, built
around a **core + sleeve** portfolio architecture:

- **Core** — the always-invested layer. Example: hold QQQ when it is above its
  200-day moving average, otherwise hold GLD. Switches only a few times a year.
- **Sleeve** — short-horizon tactical strategies (pullback families) traded on
  individual large-cap stocks, re-picked from a researched bench.

This repo covers **Phases 1–4**: data, backtesting engine, pass criteria, the
strategy library, the search loop, out-of-sample validation, walk-forward
sleeve re-picking, portfolio assembly, risk sizing, and the daily-signal
dashboard. No live trading, no broker connections — research and backtesting
only. Trade execution stays manual: the dashboard tells you what to do.

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

# 4. Validate candidates on sealed test data (train vs test metrics -> results/validation.csv)
python scripts/validate.py --config configs/growth_daily.yaml

# 5. Walk-forward sleeve repick (quarterly selection on pre-rebalance data only)
python scripts/walkforward.py --config configs/growth_daily.yaml

# 6. Assemble core+sleeve portfolio with correlation filter + risk overlay
python scripts/assemble.py --config configs/growth_daily.yaml
python scripts/assemble.py --config configs/growth_daily.yaml --sleeve-source validated

# 7. Phase 4: generate the daily-signal dashboard for the validated core
#    (self-contained results/dashboard.html — open it in any browser)
python scripts/make_dashboard.py --config configs/growth_daily.yaml

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
    validate.py      # out-of-sample validation on sealed test data
    walkforward.py   # quarterly sleeve repick on strictly pre-rebalance data
  portfolio.py       # core+sleeve assembly with correlation filter
  risk.py            # position sizing + risk-limited backtest with borrow costs
  dashboard.py       # Phase 4: core signal summary + self-contained HTML dashboard
configs/
  growth_daily.yaml  # full profile: 40-ticker universe, 2010-2022 train, costs, criteria, grids
  smoke.yaml         # tiny config for fast smoke tests
scripts/
  fetch_data.py      # download + cache the universe
  run_search.py      # grid + mutate search on train data -> candidates.csv
  summarize.py       # print ranked candidates
  validate.py        # re-run candidates on sealed test data -> validation.csv
  walkforward.py     # quarterly sleeve repick -> walkforward_{equity,signals,picks}
  assemble.py        # core+sleeve portfolio + risk overlay -> portfolio_summary.txt
  make_dashboard.py  # Phase 4: refresh bars + render results/dashboard.html
tests/               # pytest suite (no-lookahead, costs, metrics, strategies, ledger, data,
                     # validation discipline, walk-forward traps, kill rule, risk math)
data/cache/          # downloaded parquet bars (git-ignored)
results/             # ledger.jsonl, candidates.csv, validation.csv, walk-forward + portfolio outputs (git-ignored)
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
7. **Validation** — `scripts/validate.py` re-runs every candidate on the sealed
   2023+ window (same criteria, QQQ benchmark over the same window) and writes
   `results/validation.csv` with train vs test metrics and a `survived` flag.
   A warmup buffer of pre-2023 data seeds the causal indicators; only dates
   >= `test_start` are scored, and `check_train_test_separation` refuses any
   train/test overlap.
8. **Walk-forward** — `scripts/walkforward.py` simulates running the selection
   process through history: each quarter it re-runs the (small) grid on the
   trailing 5 years of data *strictly before* the rebalance date, holds the top
   3 passers for one quarter, and applies the kill rule (live drawdown >
   2x the selection max drawdown -> dropped, logged with reason). The stitched
   sleeve is backtested in one engine run so quarter boundaries pay exact
   costs. Outputs: `walkforward_equity.csv`, `walkforward_signals.parquet`
   (reused by assembly), `walkforward_picks.jsonl` (every rebalance's picks,
   kills, and selection metrics).
9. **Assembly** — `scripts/assemble.py` combines core (70%) + sleeve (30%):
   each component is backtested standalone, the correlation filter drops the
   weaker member of any pair with |corr| > 0.7, weights renormalize, and the
   combination is backtested once. A risk overlay replays the same portfolio
   with the exposure cap (`capital_at_risk`), leverage ceiling, and borrowing
   costs applied, reporting capped days and borrow drag.

The 2023+ data was sealed off from the search (see `test_start` in the config);
Phases 1–4 are complete.

## Phase 4: daily-signal dashboard

`scripts/make_dashboard.py` refreshes QQQ/GLD bars, recomputes the validated
core rotation signal (100% QQQ above its 200-day MA, else 100% GLD) through the
latest close, paper-tracks the model portfolio, and writes a single
self-contained `results/dashboard.html` (hand-rolled SVG charts, no new
dependencies). Open it in any browser whenever you want the signal — no server,
no accounts. The signal at close *t* is for the *next* open, matching the
backtest's no-lookahead convention. You place the trades yourself; the model
portfolio assumes every signal was followed, so reconcile it with your actual
holdings before acting.

Engine note: `Costs.min_trade_notional` ($1 default) skips sub-dollar "dust"
trades the rebalancer used to chase after costs on a fully-invested portfolio.
It changes backtested economics not at all (core sealed metrics identical to
6dp) but trade counts are now honest.

## Automated dashboard on Modal

`modal_app.py` hosts the dashboard so nobody has to run anything manually:

- `refresh_dashboard` — scheduled weekdays at 22:00 UTC (after the US close):
  refreshes QQQ/GLD bars and rebuilds the HTML on a persistent Modal volume
  (the parquet cache lives there too, so refreshes stay incremental).
- `dashboard` — public web endpoint serving the latest generated HTML.

It runs the exact same `trading_system/dashboard.py::generate_dashboard_html`
code path as `scripts/make_dashboard.py` (verified byte-identical output).
Deploy from the repo root with `modal deploy modal_app.py`; trigger a one-off
refresh with `modal run modal_app.py::refresh_dashboard`.

## Config reference

| Key | Meaning |
|---|---|
| `profile.universe` | Tickers to download (must include `QQQ` for the benchmark; sleeve strategies auto-exclude `QQQ`/`GLD`/`SPY`) |
| `train_start` / `train_end` | Date range the search is allowed to see |
| `test_start` | Sealed data start (Phase 3 validation) |
| `cache_dir` / `results_dir` | Where bars and outputs live |
| `costs.commission_bps` / `costs.slippage_bps` | Per-rebalance cost on traded notional |
| `criteria.*` | `min_trades`, `min_sharpe`, `max_drawdown` (fraction), `must_beat_benchmark`, `beat_benchmark_on` (`sharpe` or `total_return`), `max_pairwise_corr` |
| `param_grids.<strategy>` | Parameter name -> list of values to exhaust |
| `mutate.*` | `n_rounds`, `n_children`, `perturb` (fraction), `top_k`, `seed` |
| `validate.warmup_days` | Calendar-day buffer before `test_start` for indicator warmup (never scored) |
| `walkforward.*` | `start`, `rebalance: quarterly`, `selection_lookback_years`, `top_k`, `kill_dd_multiple`, `warmup_days`, `walkforward_grid` (small per-rebalance grid) |
| `portfolio.*` | `core_weight`, `sleeve_weight`, `max_pairwise_corr`, `sleeve_source` (`walkforward` or `validated`) |
| `core.*` | `risky`, `safe`, `ma` for the core rotation |
| `risk.*` | `max_loss_per_trade` ($), `capital_at_risk` ($), `leverage_ceiling`, `borrow_rate_annual`, `default_stop_pct` |

## Key assumptions

- Daily bars only; decisions at the close execute at the next open (no intraday).
- Long-only, fractional shares, no leverage by default (`risk.leverage_ceiling: 1.0`).
- Costs are a flat bps charge on traded notional — no partial fills, no market impact model.
- Corporate actions: prices come from yfinance unadjusted OHLC; use with care around splits/dividends for single-name backtests over long horizons.
- The benchmark is QQQ buy-and-hold over the same window, run through the same engine (so it pays the same entry cost).
- **Walk-forward honesty rules:** selection at rebalance date D uses only data in
  `[D - 5y, D)`; signals are precomputed once because they are causal
  (signal[t] depends only on data <= t), then sliced per window — precomputation
  is a performance optimization, not a lookahead. The kill rule compares a
  pick's live drawdown since inception against 2x its selection-time max
  drawdown; kills happen only at rebalance dates.
- **Validation honesty rules:** the test window is never touched during search;
  indicator warmup uses pre-test data the search already saw, and scoring starts
  strictly at `test_start`. Overlapping train/test ranges are refused outright.
- **Risk model:** strategies emit no stops, so position sizing assumes a default
  8% stop (`risk.default_stop_pct`). Borrowing above 1.0x equity accrues the
  annual borrow rate daily on the borrowed portion.
