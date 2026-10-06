"""Visual output: macro-sentiment-over-time charts and the permutation matrix.

Palette: a validated colour-blind-safe set (blue / orange / aqua / yellow /
magenta / green / violet / red) with a blue<->red diverging ramp and a neutral
grey midpoint for signed quantities. Single-axis charts only - two measures of
different scale are shown as stacked small multiples, never as a dual axis.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from ..config import CHART_DIR  # noqa: E402

log = logging.getLogger(__name__)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1c1c"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlesize": 11, "legend.frameon": False,
})


def _save(fig, name: str) -> Path:
    p = CHART_DIR / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", p)
    return p


def _month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


# ---------------------------------------------------------------------------
def chart_sentiment_timeline(daily: pd.DataFrame, calendar: pd.DataFrame) -> Path:
    d = daily.copy(); d["day"] = pd.to_datetime(d["day"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    ax = axes[0]
    ax.axhline(0, color=BASE, lw=1)
    ax.fill_between(d.day, d.macro_sentiment_index, 0, where=d.macro_sentiment_index >= 0, color=SERIES[0], alpha=0.18, lw=0)
    ax.fill_between(d.day, d.macro_sentiment_index, 0, where=d.macro_sentiment_index < 0, color=SERIES[7], alpha=0.18, lw=0)
    ax.plot(d.day, d.macro_sentiment_index, color=SERIES[0], lw=2, label="Macro sentiment index (z)")
    for key, col, lab in [("fomc", SERIES[6], "FOMC"), ("cpi", SERIES[1], "CPI"), ("nfp", SERIES[2], "NFP")]:
        days = pd.to_datetime(calendar.loc[calendar.event_key == key, "event_date"].unique())
        days = [x for x in days if d.day.min() <= x <= d.day.max()]
        for i, x in enumerate(days):
            ax.axvline(x, color=col, lw=1, ls="--", alpha=0.7, label=lab if i == 0 else None)
    ax.set_title("Macro sentiment index over time (positive = risk-on / dovish surprises)")
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    ax.set_ylabel("z-score")

    ax = axes[1]
    ax.axhline(0, color=BASE, lw=1)
    ax.bar(d.day, d.surprise_sum_signed, color=np.where(d.surprise_sum_signed >= 0, SERIES[0], SERIES[7]), width=1.0)
    ax.plot(d.day, d.surprise_index_ewm, color=INK, lw=1.6, label="EWMA (half-life 10d)")
    ax.set_title("Daily signed data surprises (sum of z-scored actual - forecast, signed by market direction)")
    ax.legend(loc="upper left", fontsize=9); ax.set_ylabel("signed z")

    ax = axes[2]
    ax.axhline(0, color=BASE, lw=1)
    ax.bar(d.day, d.news_sentiment_mean, color=np.where(d.news_sentiment_mean >= 0, SERIES[2], SERIES[1]), width=1.0, alpha=0.8)
    ax.plot(d.day, d.news_sentiment_ewm, color=INK, lw=1.6, label="EWMA (half-life 3d)")
    ax.set_title("Daily news sentiment (lexicon score, -1 hawkish/negative .. +1 dovish/positive)")
    ax.legend(loc="lower left", fontsize=9); ax.set_ylabel("score")
    _month_axis(ax)
    fig.align_ylabels(axes)
    return _save(fig, "01_macro_sentiment_timeline.png")


def chart_surprise_heatmap(daily: pd.DataFrame) -> Path:
    d = daily.copy(); d["day"] = pd.to_datetime(d["day"])
    d["week"] = d["day"].dt.to_period("W-SUN").dt.start_time
    cols = ["surprise_inflation", "surprise_employment", "surprise_growth", "surprise_policy"]
    labels = ["Inflation", "Employment", "Growth", "Monetary policy"]
    wk = d.groupby("week")[cols].sum()
    wk["Total"] = wk.sum(axis=1)
    M = wk.T.to_numpy()
    lim = max(1e-6, np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(12, 3.6))
    im = ax.imshow(M, aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
    ax.set_yticks(range(len(labels) + 1)); ax.set_yticklabels(labels + ["Total"])
    ax.set_xticks(range(len(wk.index))); ax.set_xticklabels([w.strftime("%d %b") for w in wk.index], rotation=90, fontsize=8)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) > 0.6 * lim:
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=7, color="white")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("signed surprise z (weekly sum)")
    ax.set_title("Weekly macro surprise matrix by category (blue = risk-positive, red = risk-negative)")
    return _save(fig, "02_weekly_surprise_matrix.png")


def chart_sentiment_vs_strategy(frame: pd.DataFrame) -> Path:
    f = frame.copy()
    f["week"] = f["day"].dt.to_period("W-SUN").dt.start_time
    wk_sent = f.groupby("week")["macro_sentiment_index"].mean()
    wk_pnl = f.groupby(["week", "strategy"])["pnl"].sum().unstack()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    ax = axes[0]
    ax.axhline(0, color=BASE, lw=1)
    ax.plot(wk_sent.index, wk_sent.values, color=SERIES[0], lw=2, marker="o", ms=4)
    ax.set_title("Weekly mean macro sentiment index"); ax.set_ylabel("z-score")
    ax = axes[1]
    ax.axhline(0, color=BASE, lw=1)
    for i, s in enumerate(wk_pnl.columns):
        ax.plot(wk_pnl.index, wk_pnl[s].values, color=SERIES[i % len(SERIES)], lw=2, label=s.replace("_", " "))
        ax.annotate(s.replace("_", " "), (wk_pnl.index[-1], wk_pnl[s].values[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color=INK2, va="center")
    ax.set_title("Weekly realised P&L by strategy family (sum over permutations)"); ax.set_ylabel("P&L")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    _month_axis(ax)
    return _save(fig, "03_sentiment_vs_strategy_pnl.png")


def chart_permutation_matrix(fold_preds: pd.DataFrame, model: str = "lgbm") -> Path:
    p = fold_preds[fold_preds.model == model]
    pred = p.pivot(index="permutation_id", columns="cutoff", values="y_pred")
    true = p.pivot(index="permutation_id", columns="cutoff", values="y_true")
    # order permutations by mean realised target
    order = true.mean(axis=1).sort_values(ascending=False).index
    pred, true = pred.loc[order], true.loc[order]
    lim = float(np.nanmax(np.abs(np.concatenate([pred.to_numpy().ravel(), true.to_numpy().ravel()]))))
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.32 * len(order) + 2.2), sharey=True)
    for ax, M, title in [(axes[0], pred, f"Predicted next-period target ({model})"), (axes[1], true, "Realised next-period target")]:
        im = ax.imshow(M.to_numpy(), aspect="auto", cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim))
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([pd.Timestamp(c).strftime("%d %b") for c in M.columns], rotation=90, fontsize=8)
        ax.set_title(title); ax.grid(False)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([s.replace("|", " · ") for s in order], fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01); cb.set_label("target value")
    fig.suptitle("Strategy-permutation matrix across walk-forward folds (rows sorted by mean realised target)", fontsize=12, fontweight="semibold", color=INK)
    return _save(fig, "04_permutation_matrix_pred_vs_realised.png")


def chart_walk_forward_performance(fold_metrics: pd.DataFrame, top_k: int) -> Path:
    fm = fold_metrics.copy(); fm["cutoff"] = pd.to_datetime(fm["cutoff"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax = axes[0]
    ax.axhline(0, color=BASE, lw=1)
    series = {"lgbm": ("topk_realised", SERIES[0], f"Model (LightGBM) top-{top_k}"),
              "ridge": ("topk_realised", SERIES[6], f"Ridge top-{top_k}"),
              "trailing": ("topk_realised", SERIES[1], f"Trailing-mean top-{top_k}")}
    for m, (col, c, lab) in series.items():
        s = fm[fm.model == m].set_index("cutoff")[col].cumsum()
        ax.plot(s.index, s.values, color=c, lw=2, label=lab)
    s = fm[fm.model == "lgbm"].set_index("cutoff")["all_realised"].cumsum()
    ax.plot(s.index, s.values, color=MUTED, lw=2, ls="--", label="All permutations (equal weight)")
    ax.set_title(f"Walk-forward: cumulative realised next-period target of the top-{top_k} permutations selected at each cut-off")
    ax.legend(loc="upper left", fontsize=9); ax.set_ylabel("cumulative target")
    ax = axes[1]
    ax.axhline(0, color=BASE, lw=1)
    for i, m in enumerate(["lgbm", "ridge", "trailing"]):
        s = fm[fm.model == m].set_index("cutoff")["spearman_ic"]
        ax.plot(s.index, s.values, color=[SERIES[0], SERIES[6], SERIES[1]][i], lw=1.8, marker="o", ms=3, label=m)
    ax.set_title("Per-fold Spearman rank IC between predicted and realised target across permutations")
    ax.set_ylabel("IC"); ax.legend(loc="upper left", ncol=3, fontsize=9)
    _month_axis(ax)
    return _save(fig, "05_walk_forward_performance.png")


def chart_forecast_bar(forecast: pd.DataFrame, target_col: str) -> Path:
    f = forecast.sort_values(target_col)
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(f) + 1.5))
    colors = np.where(f[target_col] >= 0, SERIES[0], SERIES[7])
    ax.barh([s.replace("|", " · ") for s in f.permutation_id], f[target_col], color=colors, height=0.7)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_title(f"Next-period forecast per permutation ({target_col}), as of {f.as_of_day.iloc[0]}")
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, "06_next_period_forecast.png")


def chart_feature_importance(imp: pd.DataFrame, n: int = 20) -> Path:
    top = imp.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.3 * n + 1.2))
    ax.barh(top.feature, top.gain / top.gain.sum(), color=SERIES[0], height=0.7)
    ax.set_title("LightGBM feature importance (share of total gain, final model)")
    ax.set_xlabel("share of gain")
    return _save(fig, "07_feature_importance.png")


def write_gallery(paths: list[Path]) -> Path:
    items = "\n".join(f'<figure><img src="charts/{p.name}" alt="{p.stem}"><figcaption>{p.stem}</figcaption></figure>' for p in paths)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Macro agent charts</title>
<style>body{{font-family:system-ui,sans-serif;background:{SURFACE};color:{INK};max-width:1200px;margin:24px auto;padding:0 16px}}
img{{width:100%;border:1px solid {GRID};border-radius:6px}}figure{{margin:0 0 28px}}figcaption{{color:{INK2};font-size:13px;margin-top:6px}}</style></head>
<body><h1>CrowdWisdomTrading macro agent - chart gallery</h1>{items}</body></html>"""
    p = CHART_DIR.parent / "charts_gallery.html"
    p.write_text(html)
    return p
