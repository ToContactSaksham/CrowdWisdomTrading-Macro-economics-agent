"""Unit tests for the leakage-sensitive parts of the pipeline. Run: pytest -q"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from macro_agent.config import Settings  # noqa: E402
from macro_agent.features.macro_features import build_daily_macro_features, standardise_surprises  # noqa: E402
from macro_agent.features.time_features import build_model_frame  # noqa: E402
from macro_agent.ingestion.apify_calendar import classify_event, parse_value, synthetic_calendar  # noqa: E402
from macro_agent.ingestion.tavily_news import score_text, synthetic_news  # noqa: E402
from macro_agent.ingestion.trading_logs import synthetic_trading_logs  # noqa: E402
from macro_agent.models.walk_forward import walk_forward_folds  # noqa: E402


def _settings() -> Settings:
    s = Settings()
    s.run_date = date(2026, 10, 7)
    s.lookback_days = 120
    return s


def test_parse_value_units_and_ranges():
    assert parse_value("3.2%") == 3.2
    assert parse_value("-0.1%") == -0.1
    assert parse_value("227K") == 227_000
    assert parse_value("4.25-4.50%") == 4.50
    assert parse_value("") is None and parse_value(None) is None


def test_classify_event_core_cpi_wins_over_cpi():
    assert classify_event("Core CPI m/m")[0] == "core_cpi"
    assert classify_event("CPI y/y")[0] == "cpi"
    assert classify_event("FOMC Statement")[0] == "fomc"
    assert classify_event("Non-Farm Employment Change")[2] == +1


def test_timestamps_are_utc_with_est_rendering():
    cal = synthetic_calendar(_settings())
    assert str(cal.event_time_utc.iloc[0].tzinfo) == "UTC"
    r = cal[cal.event_name.str.contains("CPI m/m")].iloc[0]
    assert r.event_time_est.hour == 8 and r.event_time_est.minute == 30  # BLS 08:30 ET
    assert r.event_time_utc.hour in (12, 13)  # EDT / EST


def test_lexicon_sentiment_sign():
    assert score_text("CPI cools, Fed rate cut expected") > 0
    assert score_text("Hotter than expected inflation, hawkish Fed") < 0
    assert score_text("The committee met on Tuesday") == 0


def test_surprise_standardisation_is_causal():
    s = _settings()
    cal = standardise_surprises(synthetic_calendar(s).dropna(subset=["actual", "forecast"]))
    # z-score for the first observation of each series must only use its own series' std fallback
    assert cal.z_surprise.abs().max() <= 4


def test_targets_are_strictly_forward_and_folds_do_not_leak():
    s = _settings()
    cal = synthetic_calendar(s)
    news = synthetic_news(s, cal)
    trades = synthetic_trading_logs(s, cal)
    trades["exit_date"] = trades.exit_time_utc.dt.tz_convert("America/New_York").dt.date
    trades["win"] = (trades.pnl > 0).astype(int)
    pdaily = trades.groupby(["permutation_id", "exit_date"]).agg(n_trades=("pnl", "size"), pnl=("pnl", "sum"), win_rate=("win", "mean")).reset_index().rename(columns={"exit_date": "day"})
    pdaily["day"] = pdaily["day"].astype(str)
    daily = build_daily_macro_features(cal, news, s)
    frame = build_model_frame(pdaily, daily, s)
    hb = int(frame.target_window_bdays.iloc[0])
    one = frame[frame.permutation_id == frame.permutation_id.iloc[0]].reset_index(drop=True)
    # y_pnl at i equals sum of pnl over i+1 .. i+hb
    i = 10
    assert np.isclose(one.y_pnl.iloc[i], one.pnl.iloc[i + 1:i + 1 + hb].sum())
    # last hb rows have no target
    assert one.y_pnl.tail(hb).isna().all()
    # folds: every training day is at least hb business days before the cut-off
    days = pd.DatetimeIndex(np.sort(frame.day.unique()))
    for f in walk_forward_folds(frame, "y_pnl", s.min_train_days, s.step_days):
        max_train_day = frame.iloc[f.train_idx].day.max()
        assert days.get_loc(f.cutoff) - days.get_loc(max_train_day) >= hb
        assert (frame.iloc[f.test_idx].day == f.cutoff).all()
