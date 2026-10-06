"""Model definitions, walk-forward run and final next-period forecast.

Models compared (all re-fit at every fold):

* ``lgbm``     : LightGBM regressor, 300 trees, lr 0.03, num_leaves 15,
                 min_child_samples 20, feature_fraction 0.8, bagging 0.8, seed 7
* ``ridge``    : Ridge(alpha=10) on standardised numeric + one-hot categoricals
* ``trailing`` : naive baseline, prediction = hb * pnl_mean_20d (or win_rate_20d)

The target is selected with ``settings.target`` ("pnl" -> y_pnl, "win_rate" ->
y_win_rate).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import OUTPUT_DIR, Settings
from ..features.time_features import CATEGORICAL, feature_columns
from .walk_forward import evaluate_fold, walk_forward_folds

log = logging.getLogger(__name__)


def target_column(settings: Settings) -> str:
    return "y_pnl" if settings.target == "pnl" else "y_win_rate"


@dataclass
class ModelSpec:
    name: str

    def fit_predict(self, Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, settings: Settings):
        raise NotImplementedError


class LGBMSpec(ModelSpec):
    def __init__(self):
        super().__init__("lgbm")
        self.last_model = None

    def fit_predict(self, Xtr, ytr, Xte, settings):
        import lightgbm as lgb

        Xtr = Xtr.copy(); Xte = Xte.copy()
        for c in CATEGORICAL:
            cats = pd.CategoricalDtype(sorted(set(Xtr[c]) | set(Xte[c])))
            Xtr[c] = Xtr[c].astype(cats); Xte[c] = Xte[c].astype(cats)
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=settings.model_seed, verbose=-1,
        )
        model.fit(Xtr, ytr, categorical_feature=CATEGORICAL)
        self.last_model = model
        return model.predict(Xte)


class RidgeSpec(ModelSpec):
    def __init__(self):
        super().__init__("ridge")

    def fit_predict(self, Xtr, ytr, Xte, settings):
        num = [c for c in Xtr.columns if c not in CATEGORICAL]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
        pipe = Pipeline([("pre", pre), ("ridge", Ridge(alpha=10.0))])
        pipe.fit(Xtr, ytr)
        return pipe.predict(Xte)


class TrailingSpec(ModelSpec):
    def __init__(self):
        super().__init__("trailing")

    def fit_predict(self, Xtr, ytr, Xte, settings):
        hb = int(Xte["target_window_bdays"].iloc[0]) if "target_window_bdays" in Xte else 5
        if settings.target == "pnl":
            return Xte["pnl_mean_20d"].to_numpy() * hb
        return Xte["win_rate_20d"].to_numpy()


def run_walk_forward(frame: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fold_metrics, fold_predictions) for all models."""
    target = target_column(settings)
    feats = feature_columns(frame)
    specs = [LGBMSpec(), RidgeSpec(), TrailingSpec()]
    metrics, preds = [], []
    folds = list(walk_forward_folds(frame, target, settings.min_train_days, settings.step_days))
    log.info("Walk-forward: %d folds, %d features, target=%s", len(folds), len(feats), target)
    for fold in folds:
        tr, te = frame.iloc[fold.train_idx], frame.iloc[fold.test_idx]
        for spec in specs:
            Xte = te[feats + ["target_window_bdays"]] if spec.name == "trailing" else te[feats]
            p = spec.fit_predict(tr[feats], tr[target].to_numpy(), Xte, settings)
            m = evaluate_fold(te, p, target, settings.top_k)
            m.update({"fold": fold.k, "cutoff": fold.cutoff.date().isoformat(), "model": spec.name,
                      "n_train": len(tr)})
            metrics.append(m)
            preds.append(pd.DataFrame({"fold": fold.k, "cutoff": fold.cutoff, "model": spec.name,
                                       "permutation_id": te["permutation_id"].values,
                                       "y_true": te[target].values, "y_pred": p}))
    return pd.DataFrame(metrics), pd.concat(preds, ignore_index=True)


def summarise_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    agg = fold_metrics.groupby("model").agg(
        folds=("fold", "count"), rmse=("rmse", "mean"), mae=("mae", "mean"),
        spearman_ic_mean=("spearman_ic", "mean"), spearman_ic_std=("spearman_ic", "std"),
        ic_positive_share=("spearman_ic", lambda s: float((s > 0).mean())),
        hit_rate_sign=("hit_rate_sign", "mean"),
        topk_realised_mean=("topk_realised", "mean"), all_realised_mean=("all_realised", "mean"),
        trailing_topk_realised_mean=("trailing_topk_realised", "mean"),
    )
    agg["ic_t_stat"] = agg["spearman_ic_mean"] / (agg["spearman_ic_std"] / np.sqrt(agg["folds"]))
    return agg.reset_index()


def fit_final_and_forecast(frame: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LightGBM on every realised row and forecast the next period from the last day."""
    target = target_column(settings)
    feats = feature_columns(frame)
    train = frame[frame[target].notna()]
    last_day = frame["day"].max()
    latest = frame[frame["day"] == last_day]
    spec = LGBMSpec()
    pred = spec.fit_predict(train[feats], train[target].to_numpy(), latest[feats], settings)
    out = latest[["permutation_id", "strategy", "lookback", "risk_mult", "session", "pnl_mean_20d",
                  "win_rate_20d", "sharpe_20d"]].copy()
    out["forecast_" + target] = pred
    out["as_of_day"] = last_day.date().isoformat()
    hb = int(frame["target_window_bdays"].iloc[0])
    out["forecast_window"] = f"next {hb} business days after {last_day.date().isoformat()}"
    out = out.sort_values("forecast_" + target, ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    imp = pd.DataFrame({"feature": feats, "gain": spec.last_model.booster_.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    return out, imp


def save_model_outputs(fold_metrics, fold_preds, summary, forecast, importances, settings: Settings) -> None:
    fold_metrics.to_csv(OUTPUT_DIR / "walk_forward_fold_metrics.csv", index=False)
    fold_preds.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "model_metrics_summary.csv", index=False)
    forecast.to_csv(OUTPUT_DIR / "next_period_permutation_forecast.csv", index=False)
    importances.to_csv(OUTPUT_DIR / "feature_importance_lgbm.csv", index=False)
    (OUTPUT_DIR / "model_metrics.json").write_text(json.dumps({
        "target": target_column(settings),
        "forecast_horizon_days": settings.forecast_horizon_days,
        "min_train_days": settings.min_train_days,
        "step_days": settings.step_days,
        "top_k": settings.top_k,
        "models": summary.set_index("model").round(4).to_dict(orient="index"),
    }, indent=2, default=float))
