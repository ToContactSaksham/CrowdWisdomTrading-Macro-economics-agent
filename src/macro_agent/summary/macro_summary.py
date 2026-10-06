"""One-page macro-economic outlook summary generated from the pipeline data.

The text is templated from the computed features (no LLM call) so that every
statement is traceable to a number in the database. Wording is deliberately
measured: directions ("firmer", "softer") are only used when the underlying
30-day aggregate is at least 0.5 standard-deviation units from zero.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import OUTPUT_DIR, Settings


def _tone(x: float, pos: str, neg: str, flat: str = "broadly in line with consensus", thr: float = 0.5) -> str:
    if x > thr:
        return pos
    if x < -thr:
        return neg
    return flat


def build_macro_summary(daily: pd.DataFrame, calendar: pd.DataFrame, news: pd.DataFrame,
                        forecast: pd.DataFrame, summary_metrics: pd.DataFrame, settings: Settings) -> str:
    d = daily.copy(); d["day"] = pd.to_datetime(d["day"])
    last = d.iloc[-1]
    last30 = d[d.day > d.day.max() - pd.Timedelta(days=30)]
    prev30 = d[(d.day <= d.day.max() - pd.Timedelta(days=30)) & (d.day > d.day.max() - pd.Timedelta(days=60))]
    cal = calendar.copy()
    cal["event_date"] = pd.to_datetime(cal["event_date"])
    synthetic = (cal["source"] == "synthetic_fallback").any() or (news["source"] == "synthetic_fallback").any()

    def cat_sum(df, col):
        return float(df[col].sum())

    infl, emp, grow, pol = (cat_sum(last30, c) for c in ["surprise_inflation", "surprise_employment", "surprise_growth", "surprise_policy"])
    infl_p, emp_p, grow_p = (cat_sum(prev30, c) for c in ["surprise_inflation", "surprise_employment", "surprise_growth"])

    # latest prints of the key series
    def latest(key):
        r = cal[(cal.event_key == key) & cal.actual.notna()].sort_values("event_time_utc")
        return r.iloc[-1] if len(r) else None

    def fmt(r, unit=""):
        if r is None:
            return "n/a"
        s = f"{r.actual:g}{unit} vs {r.forecast:g}{unit} consensus"
        if r.previous is not None and not pd.isna(r.previous):
            s += f" (prior {r.previous:g}{unit})"
        return s + f", {r.event_date.date().isoformat()}"

    cpi, fomc, nfp, ur, pce = (latest(k) for k in ["cpi", "fomc", "nfp", "unemployment_rate", "pce"])

    # upcoming scheduled events (from calendar conventions / scraped future rows)
    today = pd.Timestamp(settings.run_date)
    next_fomc = today + pd.Timedelta(days=int(last.days_to_next_fomc))
    next_cpi = today + pd.Timedelta(days=int(last.days_to_next_cpi))
    next_nfp = today + pd.Timedelta(days=int(last.days_to_next_nfp))

    # news
    recent_news = news[pd.to_datetime(news.published_date) > today - pd.Timedelta(days=14)]
    top_news = recent_news.sort_values(["relevance", "published_utc"], ascending=False).drop_duplicates("title").head(6)
    news_tone = float(last.news_sentiment_ewm)

    # model view
    tgt = [c for c in forecast.columns if c.startswith("forecast_")][0]
    fam = forecast.groupby("strategy")[tgt].mean().sort_values(ascending=False)
    top = forecast.head(settings.top_k)
    lg = summary_metrics.set_index("model").loc["lgbm"]
    tr = summary_metrics.set_index("model").loc["trailing"]

    idx_now = float(last.macro_sentiment_index)
    idx_30 = float(last30.macro_sentiment_index.mean())
    idx_prev = float(prev30.macro_sentiment_index.mean()) if len(prev30) else np.nan
    trend_word = "improved" if idx_30 > idx_prev + 0.2 else ("deteriorated" if idx_30 < idx_prev - 0.2 else "was little changed")

    lines = []
    lines.append(f"# Macro-economic outlook summary — as of {settings.run_date.isoformat()}")
    lines.append("")
    lines.append(f"*Window: {settings.start_date.isoformat()} to {settings.run_date.isoformat()} ({settings.lookback_days} days). "
                 f"Calendar source: {cal.source.iloc[0]}; news source: {news.source.iloc[0]}. "
                 f"Sentiment index = 0.6·z(EWMA₁₀ signed surprises) + 0.4·z(EWMA₃ news sentiment).*")
    if synthetic:
        lines.append("")
        lines.append("> **Data caveat.** Part or all of the inputs below are synthetic fallback data generated because no Apify/Tavily "
                     "credentials were available at run time. The schedule and timing conventions are real; the printed values are not. "
                     "Re-run with `APIFY_TOKEN` and `TAVILY_API_KEY` set to replace them with live data.")
    lines.append("")
    lines.append("## 1. Where the macro sentiment index stands")
    lines.append(f"- Current reading: **{idx_now:+.2f}σ** ({_tone(idx_now, 'risk-positive', 'risk-negative', 'neutral', 0.5)}). "
                 f"30-day mean {idx_30:+.2f}σ versus {idx_prev:+.2f}σ in the prior 30 days — the index {trend_word}.")
    lines.append(f"- Surprise component (EWMA): {last.surprise_index_ewm:+.2f}; news component (EWMA): {news_tone:+.2f} "
                 f"({_tone(news_tone, 'dovish / constructive headline tone', 'hawkish / cautious headline tone', 'mixed headline tone', 0.15)}).")
    lines.append(f"- Data-shock intensity (EWMA of |surprise|): {last.shock_index_ewm:.2f} — "
                 f"{'elevated' if last.shock_index_ewm > d.shock_index_ewm.quantile(0.75) else 'subdued' if last.shock_index_ewm < d.shock_index_ewm.quantile(0.25) else 'around its window median'} relative to the window.")
    lines.append("")
    lines.append("## 2. Inflation")
    lines.append(f"- Last 30 days of inflation prints were {_tone(-infl, 'firmer than consensus (negative for risk)', 'softer than consensus (positive for risk)')} "
                 f"(signed surprise sum {infl:+.1f}; prior 30d {infl_p:+.1f}).")
    lines.append(f"- Latest CPI: {fmt(cpi, '%')}. Latest core PCE: {fmt(pce, '%')}.")
    lines.append("")
    lines.append("## 3. Labour market")
    lines.append(f"- Employment data over the last 30 days were {_tone(emp, 'stronger than consensus', 'weaker than consensus')} "
                 f"(signed surprise sum {emp:+.1f}; prior 30d {emp_p:+.1f}).")
    lines.append(f"- Latest payrolls: {fmt(nfp)}. Unemployment rate: {fmt(ur, '%')}.")
    lines.append("")
    lines.append("## 4. Growth and activity")
    lines.append(f"- Growth/activity releases were {_tone(grow, 'better than consensus', 'worse than consensus')} "
                 f"(signed surprise sum {grow:+.1f}; prior 30d {grow_p:+.1f}).")
    lines.append("")
    lines.append("## 5. Monetary policy")
    lines.append(f"- Latest FOMC decision: {fmt(fomc, '%')} (upper bound of target range). Policy surprise sum last 30d: {pol:+.1f}.")
    lines.append(f"- Next scheduled: FOMC {next_fomc.date().isoformat()} ({int(last.days_to_next_fomc)}d), "
                 f"CPI {next_cpi.date().isoformat()} ({int(last.days_to_next_cpi)}d), payrolls {next_nfp.date().isoformat()} ({int(last.days_to_next_nfp)}d).")
    lines.append("")
    lines.append("## 6. Recent headlines (Tavily)")
    if top_news.empty:
        lines.append("- No articles in the last 14 days.")
    for _, r in top_news.iterrows():
        lines.append(f"- {pd.Timestamp(r.published_utc).strftime('%d %b')}: {r.title} (sentiment {r.sentiment:+.2f})")
    lines.append("")
    lines.append("## 7. Implications for strategy permutations (model view)")
    lines.append(f"- Walk-forward LightGBM: mean Spearman IC {lg.spearman_ic_mean:+.3f} (t = {lg.ic_t_stat:.2f}, positive in {lg.ic_positive_share:.0%} of {int(lg.folds)} folds); "
                 f"top-{settings.top_k} realised mean {lg.topk_realised_mean:+.1f} vs {lg.all_realised_mean:+.1f} for all permutations and {tr.topk_realised_mean:+.1f} for the trailing-mean baseline.")
    lines.append("- Family ranking for the next period: " + ", ".join(f"{k.replace('_', ' ')} ({v:+.1f})" for k, v in fam.items()) + ".")
    lines.append("- Top permutations: " + "; ".join(f"{r.permutation_id} ({r[tgt]:+.1f})" for _, r in top.iterrows()) + ".")
    lines.append("")
    lines.append("## 8. Risks to the view")
    lines.append("- The model is trained on roughly four months of post-warm-up history; IC estimates have wide confidence bands and the ranking should be treated as a tilt, not a signal to concentrate.")
    lines.append("- Event-time alignment assumes calendar times are US/Eastern; a feed in another zone would shift same-day features by one day around late-evening releases.")
    lines.append("- Lexicon news sentiment ignores negation and context; it is a coarse proxy for headline tone.")
    text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "macro_outlook_summary.md").write_text(text)
    return text
