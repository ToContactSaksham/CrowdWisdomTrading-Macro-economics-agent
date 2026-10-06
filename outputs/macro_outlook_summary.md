# Macro-economic outlook summary — as of 2026-10-07

*Window: 2026-04-10 to 2026-10-07 (180 days). Calendar source: synthetic_fallback; news source: synthetic_fallback. Sentiment index = 0.6·z(EWMA₁₀ signed surprises) + 0.4·z(EWMA₃ news sentiment).*

> **Data caveat.** Part or all of the inputs below are synthetic fallback data generated because no Apify/Tavily credentials were available at run time. The schedule and timing conventions are real; the printed values are not. Re-run with `APIFY_TOKEN` and `TAVILY_API_KEY` set to replace them with live data.

## 1. Where the macro sentiment index stands
- Current reading: **-0.55σ** (risk-negative). 30-day mean +0.96σ versus +0.57σ in the prior 30 days — the index improved.
- Surprise component (EWMA): +0.34; news component (EWMA): -0.44 (hawkish / cautious headline tone).
- Data-shock intensity (EWMA of |surprise|): 0.28 — subdued relative to the window.

## 2. Inflation
- Last 30 days of inflation prints were softer than consensus (positive for risk) (signed surprise sum +13.0; prior 30d +5.0).
- Latest CPI: 0.2% vs 0.2% consensus (prior 0.2%), 2026-09-10. Latest core PCE: 0.1% vs 0.2% consensus (prior 0.4%), 2026-09-25.

## 3. Labour market
- Employment data over the last 30 days were weaker than consensus (signed surprise sum -1.7; prior 30d +4.9).
- Latest payrolls: 56000 vs 150000 consensus (prior 177000), 2026-10-02. Unemployment rate: 4.2% vs 4.2% consensus (prior 4%), 2026-10-02.

## 4. Growth and activity
- Growth/activity releases were better than consensus (signed surprise sum +5.7; prior 30d +1.3).

## 5. Monetary policy
- Latest FOMC decision: 4% vs 3.75% consensus (prior 3.75%), 2026-09-16 (upper bound of target range). Policy surprise sum last 30d: -0.2.
- Next scheduled: FOMC 2026-10-28 (21d), CPI 2026-10-16 (9d), payrolls 2026-11-06 (30d).

## 6. Recent headlines (Tavily)
- 07 Oct: Jobs report misses; unemployment rises to multi-year high (sentiment -0.90)
- 07 Oct: Sticky core inflation keeps Fed cautious, officials say (sentiment -0.80)
- 06 Oct: Inflation runs hotter than expected; traders pare back rate-cut bets (sentiment -0.90)
- 06 Oct: ISM manufacturing slips further into contraction (sentiment -1.00)
- 05 Oct: Stagflation fears grow after weak GDP and hot prices (sentiment -0.90)
- 05 Oct: Fed holds rates steady, signals higher for longer stance (sentiment -0.65)

## 7. Implications for strategy permutations (model view)
- Walk-forward LightGBM: mean Spearman IC +0.139 (t = 3.32, positive in 82% of 17 folds); top-5 realised mean +96.7 vs +60.8 for all permutations and +112.3 for the trailing-mean baseline.
- Family ranking for the next period: trend (+142.8), mean reversion (+86.4), breakout (+57.7).
- Top permutations: trend|lb50|rm2|us_close (+293.8); mean_reversion|lb50|rm2|us_close (+266.4); trend|lb50|rm2|us_open (+246.6); breakout|lb50|rm2|us_close (+206.9); mean_reversion|lb20|rm2|us_close (+203.7).

## 8. Risks to the view
- The model is trained on roughly four months of post-warm-up history; IC estimates have wide confidence bands and the ranking should be treated as a tilt, not a signal to concentrate.
- Event-time alignment assumes calendar times are US/Eastern; a feed in another zone would shift same-day features by one day around late-evening releases.
- Lexicon news sentiment ignores negation and context; it is a coarse proxy for headline tone.
