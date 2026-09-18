"""Modal deployment for the Phase 4 core-signal dashboard.

Two pieces, both running the exact same generation code as the local script
(`trading_system/dashboard.py::generate_dashboard_html`):

- ``refresh_dashboard`` — scheduled weekdays at 22:00 UTC (6pm EDT / 5pm EST,
  after the US close). Refreshes QQQ/GLD bars, rebuilds the dashboard HTML,
  and stores it on the Modal volume. The parquet cache also lives on the
  volume, so daily refreshes stay incremental.
- ``dashboard`` — public web endpoint serving the latest generated HTML.
  Loads instantly; no per-visit regeneration.

Deploy from the repo root (the image bakes in local paths)::

    modal deploy modal_app.py

Trigger a one-off refresh (e.g. right after deploying)::

    modal run modal_app.py::refresh_dashboard
"""

from __future__ import annotations

import modal

APP_NAME = "core-signal-dashboard"
VOLUME_NAME = "trading-system-data"
MOUNT = "/data"
CACHE_DIR = f"{MOUNT}/cache"
DASHBOARD_PATH = f"{MOUNT}/dashboard.html"
CONFIG_PATH = "/root/config.yaml"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("yfinance", "pandas", "numpy", "pyarrow", "pyyaml", "fastapi")
    .add_local_dir("trading_system", remote_path="/root/trading_system", copy=True)
    .add_local_file("configs/growth_daily.yaml", remote_path=CONFIG_PATH, copy=True)
)


def _generate() -> dict:
    """Run the shared dashboard pipeline; write HTML to the volume."""
    import sys
    from pathlib import Path

    import yaml

    sys.path.insert(0, "/root")
    from trading_system import dashboard as dash
    from trading_system.engine import Costs

    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    costs = Costs(**cfg.get("costs", {}))
    drawdown_cap = float(cfg["criteria"]["max_drawdown"])

    html_page, summary = dash.generate_dashboard_html(
        cache_dir=CACHE_DIR,
        costs=costs,
        drawdown_cap=drawdown_cap,
    )
    Path(DASHBOARD_PATH).write_text(html_page, encoding="utf-8")
    volume.commit()
    return summary


@app.function(
    image=image,
    volumes={MOUNT: volume},
    schedule=modal.Cron("0 22 * * 1-5"),  # weekdays, after the US close
    timeout=600,
)
def refresh_dashboard():
    summary = _generate()
    action = summary["action"]
    sig = (
        f"HOLD {summary['position']}"
        if action == "HOLD"
        else f"ROTATE {summary['rotate_from']} -> {summary['rotate_to']}"
    )
    print(f"Signal for next open: {sig}")
    print(f"QQQ {summary['ma_distance_pct']:+.2f}% vs 200-day MA")
    print(f"Dashboard written to {DASHBOARD_PATH}")


@app.function(image=image, volumes={MOUNT: volume})
@modal.fastapi_endpoint(method="GET")
def dashboard():
    from pathlib import Path

    from fastapi.responses import HTMLResponse, PlainTextResponse

    path = Path(DASHBOARD_PATH)
    if not path.exists():
        return PlainTextResponse(
            "Dashboard not generated yet — it builds automatically after the "
            "next US close.",
            status_code=503,
        )
    return HTMLResponse(path.read_text(encoding="utf-8"))
