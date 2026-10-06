"""Historical trading-log ingestion.

Expected CSV schema (one row per closed trade)::

    trade_id, strategy, permutation_id, lookback, risk_mult, session,
    symbol, side, entry_time_utc, exit_time_utc, pnl

* ``permutation_id`` identifies a strategy parameter set ("permutation"); if the
  column is absent it is built as ``f"{strategy}|lb{lookback}|rm{risk_mult}|{session}"``.
* ``entry_time_utc`` / ``exit_time_utc`` are ISO-8601 (naive timestamps are
  assumed to be UTC). The exit time is the time at which P&L is realised and is
  the timestamp used for all joins and for the walk-forward target.

If the file does not exist a synthetic log is generated so the pipeline can run
end to end. The synthetic generator plants realistic structure that a model
*should* be able to learn: trend strategies profit from large macro surprises,
mean-reversion strategies profit from calm regimes, breakout strategies profit on
high-impact event days, plus day-of-week seasonality and heavy-tailed noise.
Rows are tagged ``source="synthetic_fallback"``.
"""
from __future__ import annotations

import itertools
import logging
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import Settings

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

REQUIRED = {"strategy", "entry_time_utc", "exit_time_utc", "pnl"}

STRATEGIES = ["trend", "mean_reversion", "breakout"]
LOOKBACKS = [20, 50]
RISK_MULTS = [1.0, 2.0]
SESSIONS = ["us_open", "us_close"]  # us_open 09:30-12:00 ET, us_close 13:00-16:00 ET


def permutation_grid() -> pd.DataFrame:
    rows = [
        {"strategy": s, "lookback": lb, "risk_mult": rm, "session": ss,
         "permutation_id": f"{s}|lb{lb}|rm{rm:g}|{ss}"}
        for s, lb, rm, ss in itertools.product(STRATEGIES, LOOKBACKS, RISK_MULTS, SESSIONS)
    ]
    return pd.DataFrame(rows)


def load_trading_logs(settings: Settings, calendar: pd.DataFrame) -> pd.DataFrame:
    path = settings.trading_log_path
    if path.exists():
        df = pd.read_csv(path)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"trading log {path} missing columns {missing}")
        for c in ("entry_time_utc", "exit_time_utc"):
            df[c] = pd.to_datetime(df[c], utc=True)
        if "permutation_id" not in df.columns:
            df["permutation_id"] = (df["strategy"].astype(str) + "|lb" + df["lookback"].astype(str)
                                    + "|rm" + df["risk_mult"].astype(str) + "|" + df["session"].astype(str))
        df["source"] = "user_log"
        log.info("Loaded %d trades from %s", len(df), path)
    else:
        log.warning("No trading log at %s -> generating synthetic log", path)
        df = synthetic_trading_logs(settings, calendar)
        df.to_csv(path, index=False)
    df["win"] = (df["pnl"] > 0).astype(int)
    df["exit_date"] = df["exit_time_utc"].dt.tz_convert(ET).dt.date
    return df.sort_values("exit_time_utc").reset_index(drop=True)


def _daily_regime(calendar: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per-day latent drivers derived from the calendar: surprise shock & event flag."""
    days = pd.date_range(settings.start_date, settings.run_date, freq="D").date
    cal = calendar.dropna(subset=["actual", "forecast"]).copy()
    # standardise surprise within event series
    cal["surprise"] = cal["actual"] - cal["forecast"]
    cal["z"] = cal.groupby("event_key")["surprise"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    cal["signed_z"] = cal["z"] * cal["direction"]
    agg = cal.groupby("event_date").agg(abs_shock=("z", lambda s: float(np.abs(s).mean())),
                                        signed=("signed_z", "sum"),
                                        high=("importance", lambda s: int((s == "high").any())))
    reg = pd.DataFrame({"day": days}).set_index("day").join(agg).fillna(0.0)
    # macro "mood" = EWMA of signed surprises (what the model will reconstruct)
    reg["mood"] = reg["signed"].ewm(halflife=10).mean()
    reg["shock_ewm"] = reg["abs_shock"].ewm(halflife=5).mean()
    return reg


def synthetic_trading_logs(settings: Settings, calendar: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(settings.synthetic_seed + 2)
    grid = permutation_grid()
    regime = _daily_regime(calendar, settings)
    recs = []
    tid = 0
    for day, r in regime.iterrows():
        if day.weekday() >= 5:
            continue
        dow = day.weekday()
        for _, p in grid.iterrows():
            n_trades = int(rng.integers(2, 5))
            # expected edge in $ per trade for this permutation on this day
            if p.strategy == "trend":
                edge = 18 * r.shock_ewm + 10 * abs(r.mood) - 6
            elif p.strategy == "mean_reversion":
                edge = 14 - 16 * r.shock_ewm - 8 * abs(r.mood)
            else:  # breakout
                edge = 25 * r.high + 6 * r.abs_shock - 8
            edge += {20: 2.0, 50: -2.0}[p.lookback] * np.sign(r.mood)  # short lookback likes positive mood
            edge += 4.0 if (p.session == "us_close" and dow == 4) else 0.0  # Friday close effect
            edge -= 3.0 if (p.session == "us_open" and dow == 0) else 0.0  # Monday open drag
            scale = 40 * p.risk_mult * (1 + 0.8 * r.shock_ewm)
            edge *= p.risk_mult
            for _ in range(n_trades):
                if p.session == "us_open":
                    t0 = (datetime.combine(day, time(9, 30), tzinfo=ET) + timedelta(minutes=int(rng.integers(0, 150))))
                else:
                    t0 = (datetime.combine(day, time(13, 0), tzinfo=ET) + timedelta(minutes=int(rng.integers(0, 150))))
                hold = int(rng.integers(5, 60))
                t1 = t0 + timedelta(minutes=hold)
                pnl = edge + scale * rng.standard_t(df=4) / np.sqrt(2)
                tid += 1
                recs.append({
                    "trade_id": tid, "strategy": p.strategy, "permutation_id": p.permutation_id,
                    "lookback": p.lookback, "risk_mult": p.risk_mult, "session": p.session,
                    "symbol": rng.choice(["ES", "NQ", "6E", "ZN"]), "side": rng.choice(["long", "short"]),
                    "entry_time_utc": t0.astimezone(UTC), "exit_time_utc": t1.astimezone(UTC),
                    "pnl": round(float(pnl), 2), "source": "synthetic_fallback",
                })
    df = pd.DataFrame(recs)
    log.info("Generated %d synthetic trades across %d permutations", len(df), grid.shape[0])
    return df
