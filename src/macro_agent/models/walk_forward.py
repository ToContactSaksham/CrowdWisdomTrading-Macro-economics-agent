"""Walk-forward (anchored / expanding-window) validation.

For fold k with cut-off business day T_k:

    train rows : all (permutation, day) rows with day <= T_k - hb, where hb is the
                 target window length in business days -- so every training target
                 is fully realised *before* T_k (no leakage from the future).
    test rows  : the rows for day T_k (one per permutation). Their features are
                 known at the close of T_k; their target is realised over
                 T_k+1 .. T_k+hb.

T_k advances by ``step_days`` calendar days. The first cut-off is
``start_date + min_train_days``. The model is re-fit from scratch at every fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd


@dataclass
class Fold:
    k: int
    cutoff: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_folds(frame: pd.DataFrame, target: str, min_train_days: int, step_days: int) -> Iterator[Fold]:
    days = np.sort(frame["day"].unique())
    days = pd.DatetimeIndex(days)
    hb = int(frame["target_window_bdays"].iloc[0])
    first_cut = days[0] + pd.Timedelta(days=min_train_days)
    cut_days = [d for d in days if d >= first_cut]
    # keep only cut-offs whose own target is realised within the sample
    pos = {d: i for i, d in enumerate(days)}
    k = 0
    last_taken = None
    for d in cut_days:
        if last_taken is not None and (d - last_taken).days < step_days:
            continue
        if pos[d] + hb >= len(days):
            break
        train_days = days[: pos[d] - hb + 1]
        train_mask = frame["day"].isin(train_days) & frame[target].notna()
        test_mask = (frame["day"] == d) & frame[target].notna()
        if train_mask.sum() < 50 or test_mask.sum() == 0:
            continue
        yield Fold(k, d, np.where(train_mask)[0], np.where(test_mask)[0])
        k += 1
        last_taken = d


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_pred) == 0 or np.std(y_true) == 0:
        return np.nan
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def evaluate_fold(test: pd.DataFrame, pred: np.ndarray, target: str, top_k: int) -> dict:
    y = test[target].to_numpy()
    order = np.argsort(-pred)
    top = order[:top_k]
    trailing = np.argsort(-test["pnl_mean_20d"].to_numpy())[:top_k]
    return {
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
        "spearman_ic": spearman_ic(y, pred),
        "hit_rate_sign": float(np.mean(np.sign(y) == np.sign(pred))) if target == "y_pnl" else np.nan,
        "topk_realised": float(np.mean(y[top])),
        "all_realised": float(np.mean(y)),
        "trailing_topk_realised": float(np.mean(y[trailing])),
        "n_test": int(len(y)),
    }
