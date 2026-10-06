#!/usr/bin/env python3
"""End-to-end pipeline: ingest -> database -> features -> walk-forward model -> charts -> summary.

Usage:
    python run_pipeline.py                 # uses APIFY_TOKEN / TAVILY_API_KEY if set, else fallbacks
    OFFLINE=1 python run_pipeline.py       # force fallbacks (reproducible)
    RUN_DATE=2026-10-07 TARGET=win_rate python run_pipeline.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from macro_agent.config import OUTPUT_DIR, settings  # noqa: E402
from macro_agent.db import database as db  # noqa: E402
from macro_agent.features.macro_features import build_daily_macro_features  # noqa: E402
from macro_agent.features.time_features import build_model_frame  # noqa: E402
from macro_agent.ingestion.apify_calendar import fetch_calendar  # noqa: E402
from macro_agent.ingestion.tavily_news import fetch_news  # noqa: E402
from macro_agent.ingestion.trading_logs import load_trading_logs  # noqa: E402
from macro_agent.models.train import (fit_final_and_forecast, run_walk_forward, save_model_outputs,  # noqa: E402
                                      summarise_metrics, target_column)
from macro_agent.summary.macro_summary import build_macro_summary  # noqa: E402
from macro_agent.viz import charts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")


def main() -> None:
    t0 = time.time()
    log.info("Run date %s, window %s -> %s", settings.run_date, settings.start_date, settings.run_date)

    # 1. Ingestion -----------------------------------------------------------
    calendar = fetch_calendar(settings)
    news = fetch_news(settings, calendar)
    trades = load_trading_logs(settings, calendar)
    log.info("calendar=%d events, news=%d articles, trades=%d", len(calendar), len(news), len(trades))

    # 2. Database ------------------------------------------------------------
    con = db.connect()
    db.replace_table(con, "macro_events", calendar)
    db.replace_table(con, "macro_news", news)
    db.replace_table(con, "trades", trades)
    perm_daily = db.perm_daily_from_trades(con)

    # 3. Features --------------------------------------------------------------
    daily = build_daily_macro_features(calendar, news, settings)
    db.replace_table(con, "daily_macro_features", daily)
    daily.to_csv(OUTPUT_DIR / "daily_macro_features.csv", index=False)
    joined = db.joined_training_frame(con)  # the SQL join (saved for inspection)
    joined.to_csv(OUTPUT_DIR / "joined_perm_daily_macro.csv", index=False)
    frame = build_model_frame(perm_daily, daily, settings)
    frame.to_csv(OUTPUT_DIR / "model_frame.csv", index=False)
    log.info("model frame: %d rows x %d cols", *frame.shape)

    # 4. Model + walk-forward --------------------------------------------------
    fold_metrics, fold_preds = run_walk_forward(frame, settings)
    summary = summarise_metrics(fold_metrics)
    forecast, importances = fit_final_and_forecast(frame, settings)
    save_model_outputs(fold_metrics, fold_preds, summary, forecast, importances, settings)
    log.info("\n%s", summary.round(3).to_string(index=False))

    # 5. Charts ------------------------------------------------------------------
    paths = [
        charts.chart_sentiment_timeline(daily, calendar),
        charts.chart_surprise_heatmap(daily),
        charts.chart_sentiment_vs_strategy(frame),
        charts.chart_permutation_matrix(fold_preds, "lgbm"),
        charts.chart_walk_forward_performance(fold_metrics, settings.top_k),
        charts.chart_forecast_bar(forecast, "forecast_" + target_column(settings)),
        charts.chart_feature_importance(importances),
    ]
    charts.write_gallery(paths)

    # 6. Summary ------------------------------------------------------------------
    text = build_macro_summary(daily, calendar, news, forecast, summary, settings)
    log.info("Summary written (%d words)", len(text.split()))
    con.close()
    log.info("Done in %.1fs. Outputs in %s", time.time() - t0, OUTPUT_DIR)


if __name__ == "__main__":
    main()
