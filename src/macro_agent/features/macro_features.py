"""Daily macro-state features from the event calendar and the news feed.

Everything is computed at the *calendar-day* grain in US/Eastern so it can be
joined to trades by realisation date. No value on day D uses information released
after D (EWMAs are causal, "days_to_next_*" uses the *scheduled* calendar which is
known in advance).

Macro sentiment index
---------------------
    surprise_index_ewm   = EWMA_{hl=10d}( sum_over_events_on_day( z_surprise * direction ) )
    news_sentiment_ewm   = EWMA_{hl=3d}( mean lexicon sentiment of the day's articles )
    macro_sentiment_index = 0.6 * z(surprise_index_ewm) + 0.4 * z(news_sentiment_ewm)

where z(.) standardises using the full-window mean/std (reported in the README; a
purely causal variant would use an expanding window - see ``causal_z``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings

KEY_EVENTS = {"fomc": "fomc", "cpi": "cpi", "nfp": "nfp"}


def standardise_surprises(cal: pd.DataFrame) -> pd.DataFrame:
    cal = cal.copy()
    cal["surprise"] = cal["actual"] - cal["forecast"]
    # within-series z-score using an *expanding* std so day D only uses history <= D
    cal = cal.sort_values("event_time_utc")
    grp = cal.groupby("event_key")["surprise"]
    exp_mean = grp.transform(lambda s: s.expanding().mean().shift(1).fillna(0.0))
    exp_std = grp.transform(lambda s: s.expanding().std(ddof=0).shift(1))
    # first observations have no history: fall back to the series' own full-window std
    full_std = grp.transform(lambda s: s.std(ddof=0))
    exp_std = exp_std.fillna(full_std).replace(0, np.nan).fillna(1.0)
    cal["z_surprise"] = ((cal["surprise"] - exp_mean) / exp_std).clip(-4, 4)
    cal["signed_z"] = cal["z_surprise"] * cal["direction"]
    return cal


def _days_to_from(days: pd.DatetimeIndex, event_days: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ev = np.sort(pd.to_datetime(event_days.unique()).values.astype("datetime64[D]"))
    d = days.values.astype("datetime64[D]")
    to_next = np.full(len(d), np.nan)
    since_last = np.full(len(d), np.nan)
    if len(ev) == 0:
        return to_next, since_last
    idx = np.searchsorted(ev, d, side="left")
    has_next = idx < len(ev)
    to_next[has_next] = (ev[idx[has_next]] - d[has_next]).astype(int)
    idx_prev = np.searchsorted(ev, d, side="right") - 1
    has_prev = idx_prev >= 0
    since_last[has_prev] = (d[has_prev] - ev[idx_prev[has_prev]]).astype(int)
    return to_next, since_last


def build_daily_macro_features(calendar: pd.DataFrame, news: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    days = pd.date_range(settings.start_date, settings.run_date, freq="D")
    out = pd.DataFrame(index=days)
    out.index.name = "day"

    cal = standardise_surprises(calendar.dropna(subset=["actual", "forecast"]))
    cal["day"] = pd.to_datetime(cal["event_date"])
    g = cal.groupby("day")
    out["n_events"] = calendar.assign(day=pd.to_datetime(calendar["event_date"])).groupby("day").size()
    out["n_high_impact"] = calendar[calendar.importance.str.lower().str.contains("high", na=False)] \
        .assign(day=pd.to_datetime(calendar["event_date"])).groupby("day").size()
    out["surprise_sum_signed"] = g["signed_z"].sum()
    out["surprise_abs_mean"] = g["z_surprise"].apply(lambda s: s.abs().mean())
    for cat, col in [("inflation", "surprise_inflation"), ("employment", "surprise_employment"),
                     ("growth", "surprise_growth"), ("monetary_policy", "surprise_policy")]:
        out[col] = cal[cal.category == cat].groupby("day")["signed_z"].sum()
    out = out.fillna(0.0)
    out["surprise_index_ewm"] = out["surprise_sum_signed"].ewm(halflife=settings.surprise_halflife_days).mean()
    out["shock_index_ewm"] = out["surprise_abs_mean"].ewm(halflife=5).mean()

    if news is not None and not news.empty:
        nd = news.assign(day=pd.to_datetime(news["published_date"])).groupby("day")
        out["news_count"] = nd.size()
        out["news_sentiment_mean"] = nd["sentiment"].mean()
    else:
        out["news_count"] = 0
        out["news_sentiment_mean"] = 0.0
    out["news_count"] = out["news_count"].fillna(0).astype(int)
    out["news_sentiment_mean"] = out["news_sentiment_mean"].fillna(0.0)
    out["news_sentiment_ewm"] = out["news_sentiment_mean"].ewm(halflife=settings.news_halflife_days).mean()

    out["macro_sentiment_index"] = 0.6 * causal_z(out["surprise_index_ewm"]) + 0.4 * causal_z(out["news_sentiment_ewm"])

    for key, name in KEY_EVENTS.items():
        ev_days = calendar.loc[calendar.event_key.isin([key] + (["core_cpi"] if key == "cpi" else [])), "event_date"]
        to_next, since_last = _days_to_from(days, pd.Series(ev_days))
        out[f"days_to_next_{name}"] = pd.Series(to_next, index=days).fillna(45).astype(int)
        out[f"days_since_last_{name}"] = pd.Series(since_last, index=days).fillna(45).astype(int)

    out = out.reset_index()
    out["day"] = out["day"].dt.strftime("%Y-%m-%d")
    return out


def causal_z(s: pd.Series, min_periods: int = 10) -> pd.Series:
    """Expanding-window z-score so the index on day D uses only data up to D."""
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=0).replace(0, np.nan)
    return ((s - mu) / sd).fillna(0.0).clip(-4, 4)
