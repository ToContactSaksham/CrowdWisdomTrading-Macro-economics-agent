"""Central configuration for the macro-economics agent pipeline.

All run parameters live here (or are overridden through environment variables /
the CLI) so that every method step in the README can be stated with its exact
parameters: lookback window, time zones, walk-forward horizon, model seeds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
DB_PATH = PROCESSED_DIR / "macro_agent.sqlite"

for _d in (RAW_DIR, PROCESSED_DIR, SAMPLES_DIR, OUTPUT_DIR, CHART_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


@dataclass
class Settings:
    # ---- time window -----------------------------------------------------
    # Run date is pinned so results are reproducible; override with RUN_DATE.
    run_date: date = field(
        default_factory=lambda: date.fromisoformat(_env("RUN_DATE", date.today().isoformat()))
    )
    lookback_days: int = int(_env("LOOKBACK_DAYS", "180"))
    forecast_horizon_days: int = int(_env("FORECAST_HORIZON_DAYS", "7"))  # one trading week

    # ---- time zones --------------------------------------------------------
    source_tz: str = "America/New_York"  # calendars publish in US/Eastern
    store_tz: str = "UTC"

    # ---- external APIs -----------------------------------------------------
    apify_token: str | None = _env("APIFY_TOKEN")
    # Default actor: Apify's official generic web scraper, driven by a page
    # function (see ingestion/apify_calendar.py). Any calendar actor that returns
    # date/time/event/actual/forecast can be plugged in via APIFY_ACTOR_ID.
    apify_actor_id: str = _env("APIFY_ACTOR_ID", "apify/web-scraper")
    tavily_api_key: str | None = _env("TAVILY_API_KEY")
    tavily_max_results: int = int(_env("TAVILY_MAX_RESULTS", "10"))
    tavily_queries: List[str] = field(
        default_factory=lambda: [
            "US CPI inflation release latest",
            "FOMC meeting decision interest rates latest",
            "US nonfarm payrolls jobs report latest",
            "Federal Reserve rate outlook markets",
            "US economic outlook recession risk latest",
        ]
    )

    # ---- trading logs -------------------------------------------------------
    # Path to a real trading log CSV (schema in README). If missing, a synthetic
    # log is generated so the pipeline is runnable end to end.
    trading_log_path: Path = Path(_env("TRADING_LOG_PATH", str(RAW_DIR / "trading_logs.csv")))
    synthetic_seed: int = int(_env("SYNTHETIC_SEED", "42"))

    # ---- modelling -----------------------------------------------------------
    target: str = _env("TARGET", "pnl")  # "pnl" or "win_rate"
    min_train_days: int = int(_env("MIN_TRAIN_DAYS", "60"))
    step_days: int = int(_env("STEP_DAYS", "7"))
    purge_days: int = 0  # target is realised strictly after the feature date -> no overlap
    top_k: int = int(_env("TOP_K", "5"))
    model_seed: int = 7

    # ---- macro sentiment index -------------------------------------------
    surprise_halflife_days: float = 10.0  # EWMA half-life for surprise index
    news_halflife_days: float = 3.0

    @property
    def start_date(self) -> date:
        return self.run_date - timedelta(days=self.lookback_days)

    @property
    def offline(self) -> bool:
        return _env("OFFLINE", "0") == "1"


settings = Settings()
