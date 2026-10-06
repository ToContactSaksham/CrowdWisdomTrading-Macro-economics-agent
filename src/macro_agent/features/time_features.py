"""Time-based and lagged-performance features at the (permutation, day) grain,
plus the forward-looking target used by the walk-forward model.

Feature groups
--------------
time        : day_of_week, day_of_month, week_of_month, month, is_month_end,
              is_quarter_end, is_fomc_week, is_nfp_week, is_cpi_week
macro       : everything in daily_macro_features (joined by day)
perf (lag)  : rolling 5/10/20-day mean pnl, 20-day win rate, 20-day pnl std,
              5-day trade count, pnl momentum (5d mean - 20d mean) -- all shifted
              so that day D uses realised results up to and including D
permutation : strategy (categorical), lookback, risk_mult, session (categorical)
interaction : strategy x shock_index, strategy x macro_sentiment_index

Target
------
    y_pnl(D)      = sum of pnl over trading days D+1 .. D+H  (H = forecast_horizon_days)
    y_win_rate(D) = trade-weighted win rate over the same window
The target window starts strictly after D, so no purge gap is required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings

CATEGORICAL = ["strategy", "session"]


def _calendar_grid(perm_daily: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Dense (permutation x business day) grid so lags/targets have regular spacing."""
    bdays = pd.bdate_range(settings.start_date, settings.run_date)
    perms = sorted(perm_daily.permutation_id.unique())
    grid = pd.MultiIndex.from_product([perms, bdays], names=["permutation_id", "day"]).to_frame(index=False)
    pdaily = perm_daily.copy()
    pdaily["day"] = pd.to_datetime(pdaily["day"])
    g = grid.merge(pdaily, on=["permutation_id", "day"], how="left")
    g["n_trades"] = g["n_trades"].fillna(0).astype(int)
    g["pnl"] = g["pnl"].fillna(0.0)
    g["win_rate"] = g["win_rate"].fillna(np.nan)
    return g


def build_model_frame(perm_daily: pd.DataFrame, daily_macro: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    g = _calendar_grid(perm_daily, settings).sort_values(["permutation_id", "day"]).reset_index(drop=True)

    # --- permutation attributes parsed from the id (works for user logs too) ---
    parts = g["permutation_id"].str.split("|", expand=True)
    g["strategy"] = parts[0]
    g["lookback"] = pd.to_numeric(parts[1].str.replace("lb", "", regex=False), errors="coerce").fillna(0)
    g["risk_mult"] = pd.to_numeric(parts[2].str.replace("rm", "", regex=False), errors="coerce").fillna(1.0)
    g["session"] = parts[3].fillna("all")

    # --- time features ------------------------------------------------------
    d = g["day"]
    g["day_of_week"] = d.dt.dayofweek
    g["day_of_month"] = d.dt.day
    g["week_of_month"] = (d.dt.day - 1) // 7 + 1
    g["month"] = d.dt.month
    g["is_month_end"] = (d.dt.days_in_month - d.dt.day <= 3).astype(int)
    g["is_quarter_end"] = ((d.dt.month % 3 == 0) & (d.dt.days_in_month - d.dt.day <= 5)).astype(int)

    # --- macro join -----------------------------------------------------------
    m = daily_macro.copy()
    m["day"] = pd.to_datetime(m["day"])
    g = g.merge(m, on="day", how="left")
    macro_cols = [c for c in m.columns if c != "day"]
    g[macro_cols] = g[macro_cols].ffill().fillna(0.0)
    g["is_fomc_week"] = (g["days_to_next_fomc"] <= 4).astype(int)
    g["is_nfp_week"] = (g["days_to_next_nfp"] <= 4).astype(int)
    g["is_cpi_week"] = (g["days_to_next_cpi"] <= 4).astype(int)

    # --- lagged performance (causal) --------------------------------------
    gp = g.groupby("permutation_id", group_keys=False)
    for w in (5, 10, 20):
        g[f"pnl_mean_{w}d"] = gp["pnl"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    g["pnl_std_20d"] = gp["pnl"].transform(lambda s: s.rolling(20, min_periods=3).std()).fillna(0.0)
    g["win_rate_20d"] = gp["win_rate"].transform(lambda s: s.rolling(20, min_periods=1).mean()).fillna(0.5)
    g["trades_5d"] = gp["n_trades"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    g["pnl_momentum"] = g["pnl_mean_5d"] - g["pnl_mean_20d"]
    g["sharpe_20d"] = (g["pnl_mean_20d"] / g["pnl_std_20d"].replace(0, np.nan)).fillna(0.0).clip(-5, 5)

    # --- interactions -----------------------------------------------------
    for s in g["strategy"].unique():
        ind = (g["strategy"] == s).astype(float)
        g[f"{s}_x_shock"] = ind * g["shock_index_ewm"]
        g[f"{s}_x_sentiment"] = ind * g["macro_sentiment_index"]

    # --- forward targets ----------------------------------------------------
    H = settings.forecast_horizon_days
    # H calendar days ~= H*5/7 business days; use business-day count
    hb = max(1, int(round(H * 5 / 7)))
    g["y_pnl"] = gp["pnl"].transform(lambda s: s[::-1].rolling(hb, min_periods=hb).sum()[::-1].shift(-1))
    wins = (g["win_rate"].fillna(0) * g["n_trades"])
    g["_wins"] = wins
    fw = g.groupby("permutation_id", group_keys=False)["_wins"].transform(lambda s: s[::-1].rolling(hb, min_periods=1).sum()[::-1].shift(-1))
    fn = gp["n_trades"].transform(lambda s: s[::-1].rolling(hb, min_periods=1).sum()[::-1].shift(-1))
    g["y_win_rate"] = (fw / fn.replace(0, np.nan))
    g = g.drop(columns=["_wins"])
    g["target_window_bdays"] = hb
    return g


def feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {"permutation_id", "day", "n_trades", "pnl", "win_rate", "y_pnl", "y_win_rate",
               "target_window_bdays"}
    return [c for c in frame.columns if c not in exclude]
