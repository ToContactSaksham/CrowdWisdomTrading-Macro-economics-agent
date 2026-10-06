# CrowdWisdomTrading — Macro-economics Agent

Quantitative Data Scientist intern assessment. A Python pipeline that

1. scrapes a macro-economic event calendar for the last 180 days with **Apify** and pulls recent macro news with **Tavily**,
2. joins the events, news and historical trading logs in a **SQLite** database,
3. engineers time-based and macro features and writes a **one-page macro outlook**,
4. trains a model to forecast next-period P&L (or win rate) for every **strategy permutation** under **walk-forward validation**, and
5. renders a **visual matrix** of the permutations plus charts of how macro sentiment changes through time.

```
python run_pipeline.py            # live: uses APIFY_TOKEN / TAVILY_API_KEY from the environment
OFFLINE=1 python run_pipeline.py  # reproducible run on the built-in fallbacks (what ships in outputs/)
python -m pytest -q tests         # 6 tests: parsing, time zones, causality of features, fold leakage
```

Runtime for the offline run: ~8 s on a laptop. Python 3.11+, `pip install -r requirements.txt`.

---

## 1. Repository layout

```
run_pipeline.py                     end-to-end driver (ingest -> DB -> features -> model -> charts -> summary)
src/macro_agent/
  config.py                         every run parameter in one dataclass (env-overridable)
  ingestion/apify_calendar.py       Apify actor run + page function, normalisation, synthetic fallback
  ingestion/tavily_news.py          Tavily news search, lexicon sentiment, synthetic fallback
  ingestion/trading_logs.py         trading-log loader (CSV schema below) + synthetic log generator
  db/schema.sql, db/database.py     SQLite schema, loaders, SQL aggregation and join view
  features/macro_features.py        daily macro state: surprises, EWMAs, sentiment index, event distances
  features/time_features.py         (permutation x day) frame: time, lagged-performance, interactions, targets
  models/walk_forward.py            anchored walk-forward fold generator + per-fold metrics
  models/train.py                   LightGBM / Ridge / trailing-mean baseline, final forecast, exports
  viz/charts.py                     7 charts + HTML gallery
  summary/macro_summary.py          templated one-page outlook from the computed numbers
tests/test_pipeline.py
data/raw/                           raw pulls & fallbacks (apify_calendar_raw.json, tavily_news_raw.json, trading_logs.csv)
data/processed/macro_agent.sqlite   the database
outputs/                            everything the assessment asks for (see §7)
```

## 2. Data ingestion

### 2.1 Macro calendar via Apify (`ingestion/apify_calendar.py`)

* **Actor**: `apify/web-scraper` (Apify's official actor) with a custom `pageFunction` that walks `tr.calendar__row` on the ForexFactory monthly calendar pages covering the window (`https://www.forexfactory.com/calendar?month=<mon>.<yyyy>`, one URL per month from the window start to 60 days after the run date, so the forward schedule is also captured). Apify proxy enabled, 3 retries. Set `APIFY_ACTOR_ID` to use any dedicated calendar actor instead; its items go through the same normaliser, which accepts `date/time/event/actual/forecast/previous/impact/currency` or an ISO `timestamp` field.
* **Fields extracted**: event date & time, currency, event name, impact, actual, forecast, previous. Only `USD` rows are kept (CPI, core CPI, PPI, PCE, FOMC statement/rate decision, non-farm payrolls, unemployment rate, average hourly earnings, weekly jobless claims, JOLTS, retail sales, GDP, ISM manufacturing/services, UoM and Conference Board confidence). `classify_event()` maps names to a taxonomy key, a category (`inflation / employment / growth / monetary_policy / sentiment`) and a **market direction** (+1 if a positive surprise is risk-positive, e.g. payrolls; −1 if risk-negative, e.g. CPI, unemployment, jobless claims, fed funds).
* **Value parsing**: `parse_value()` handles `3.2%`, `-0.1%`, `227K`, `1.8B` and ranges such as `4.25-4.50%` (upper bound kept). Units stay per-series because surprises are standardised within a series.
* **Time alignment**: ForexFactory renders times in the viewer's zone; anonymous Apify sessions get the site default, US/Eastern. Each row is localised to `America/New_York`, converted to UTC, and stored as both `event_time_utc` and `event_time_est` (DST-aware via `zoneinfo`). The ET calendar date is the join key to trades. "All Day/Tentative" rows are pinned to 09:00 ET.
* **Fallback**: if `APIFY_TOKEN` is missing, `OFFLINE=1`, or the run fails/returns < 20 usable rows, `synthetic_calendar()` builds the calendar from the *real* publication conventions (BLS/BEA 08:30 ET, ISM 10:00 ET, FOMC statements 14:00 ET on the published 2025/2026 meeting dates; NFP first Friday, CPI second week, claims every Thursday, PCE last Friday, …). Actual/forecast values are synthetic draws around a slow-moving latent inflation/growth pressure (so surprises autocorrelate as in real data) and every row is tagged `source = "synthetic_fallback"`. **The shipped `outputs/` were produced with this fallback** because no API credentials were available in the build environment; re-running with keys replaces them with live data without any code change.

### 2.2 Macro news via Tavily (`ingestion/tavily_news.py`)

* `TavilyClient.search(query, topic="news", search_depth="advanced", days=180, max_results=10)` for five queries (CPI, FOMC, payrolls, Fed outlook, recession risk). Results are de-duplicated by URL, restricted to the window, and cached to `data/raw/tavily_news_raw.json`.
* **Sentiment**: a transparent finance lexicon (`score_text`) returns −1…+1 (hawkish/weak → negative, dovish/strong → positive). It is deliberately simple and replaceable (FinBERT/LLM) — its limits are listed in the summary's risk section.
* Fallback: templated headlines (1–3 per day) whose tone follows the sign of the week's calendar surprises; tagged `synthetic_fallback`.

### 2.3 Trading logs (`ingestion/trading_logs.py`)

Place your log at `data/raw/trading_logs.csv` (or set `TRADING_LOG_PATH`). One row per closed trade:

| column | meaning |
|---|---|
| `trade_id` | unique id |
| `strategy`, `lookback`, `risk_mult`, `session` | the parameter set; `permutation_id` = `strategy|lb{lookback}|rm{risk_mult}|{session}` is built if absent |
| `symbol`, `side` | optional |
| `entry_time_utc`, `exit_time_utc` | ISO-8601; naive values are treated as UTC |
| `pnl` | realised P&L in account currency |

Trades are attributed to the **ET date of `exit_time_utc`** (when P&L is realised). No log was supplied with the assessment, so the shipped run uses a synthetic 24-permutation log (3 strategy families × lookback {20, 50} × risk multiplier {1, 2} × session {US open, US close}; 9 305 trades) with planted, learnable structure: trend permutations gain in high-shock regimes, mean-reversion in calm regimes, breakout on high-impact event days, plus Monday-open drag, Friday-close lift and Student-t(4) noise. `data/samples/trading_logs_schema_example.csv` shows the schema.

## 3. Database (`db/`)

SQLite at `data/processed/macro_agent.sqlite`, tables `macro_events`, `macro_news`, `trades`, `daily_macro_features`, `perm_daily` and the view `v_trades_macro` (every trade joined to the macro state of its realisation day). Trades are aggregated **in SQL** to the `(permutation_id, day)` grain (`perm_daily_from_trades`), and the modelling table is the SQL `LEFT JOIN` of `perm_daily` with `daily_macro_features` on `day` (`outputs/joined_perm_daily_macro.csv`). All timestamps are stored as ISO-8601 UTC strings with an `_est` rendering alongside.

## 4. Feature engineering (`features/`)

**Daily macro state (one row per calendar day, ET)** — all causal:

* `surprise = actual − forecast`, z-scored **within each event series with an expanding mean/std** (first prints fall back to the series' full-window std), clipped to ±4, then signed by the market direction. Daily sums overall and per category (`surprise_inflation / employment / growth / policy`), `surprise_abs_mean`, `n_events`, `n_high_impact`.
* `surprise_index_ewm` = EWMA(half-life 10 d) of the signed daily sum; `shock_index_ewm` = EWMA(half-life 5 d) of mean |z|.
* `news_sentiment_mean` (daily), `news_sentiment_ewm` = EWMA(half-life 3 d), `news_count`.
* **Macro sentiment index** = 0.6 · z(surprise_index_ewm) + 0.4 · z(news_sentiment_ewm), z computed with an expanding window (min 10 days) so day D uses only data ≤ D.
* Scheduled-event distances from the forward calendar: `days_to_next_{fomc,cpi,nfp}`, `days_since_last_{…}`.

**(Permutation × business day) modelling frame** (`build_model_frame`, 3 096 rows × 47 features in the shipped run):

* time: `day_of_week, day_of_month, week_of_month, month, is_month_end, is_quarter_end, is_fomc_week, is_nfp_week, is_cpi_week`
* macro: all daily columns above, forward-filled onto business days
* lagged performance (rolling, inclusive of day D): `pnl_mean_{5,10,20}d, pnl_std_20d, win_rate_20d, trades_5d, pnl_momentum (5d−20d), sharpe_20d`
* permutation attributes: `strategy` (categorical), `lookback`, `risk_mult`, `session` (categorical)
* interactions: `{strategy}_x_shock`, `{strategy}_x_sentiment`

**Targets**: with H = `FORECAST_HORIZON_DAYS` = 7 calendar days → hb = round(7·5/7) = **5 business days**:
`y_pnl(D) = Σ pnl over D+1 … D+5`, `y_win_rate(D)` = trade-weighted win rate over the same window. The window starts strictly after D, so no purge gap is needed (asserted in `tests/`).

## 5. Model and walk-forward validation (`models/`)

* **Anchored (expanding-window) walk-forward.** Cut-offs T_k start at `start_date + 60 days` and advance every 7 calendar days (17 folds: 2026-06-09 → 2026-09-29). At each fold the model is re-fit from scratch on every row whose **target window is fully realised before T_k** (day ≤ T_k − 5 business days; 912 rows in the first fold, 2 832 in the last) and predicts the 24 rows of day T_k, which are scored against their realised D+1…D+5 outcome.
* **Models** (identical features): `lgbm` — LightGBM regressor, 300 trees, learning-rate 0.03, 15 leaves, min 20 samples/leaf, 80 % row & column subsampling, λ₂ = 1, seed 7; `ridge` — Ridge(α = 10) on standardised numerics + one-hot categoricals; `trailing` — naive baseline, forecast = 5 × `pnl_mean_20d`.
* **Metrics per fold**: RMSE, MAE, Spearman rank IC across the 24 permutations, sign hit-rate, and the realised mean target of the **top-5 permutations selected by the model** versus the equal-weight average of all permutations and the trailing-mean top-5.

Shipped results (`outputs/model_metrics.json`, synthetic data, target = `y_pnl`):

| model | mean IC | IC t-stat | IC > 0 share | RMSE | top-5 realised | all-perm realised | trailing top-5 |
|---|---|---|---|---|---|---|---|
| LightGBM | 0.140 | 3.32 | 82 % | 342.8 | 96.7 | 60.8 | 112.3 |
| Ridge | 0.166 | 4.22 | 88 % | 332.2 | 141.0 | 60.8 | 112.3 |
| trailing mean | 0.112 | 2.15 | 65 % | 352.7 | 112.3 | 60.8 | — |

Reading: both learned models rank permutations better than the trailing-mean baseline (positive IC in most folds, t > 3), but with ~17 weekly folds the bands are wide; on this sample the regularised linear model is the better selector and LightGBM's top-5 did **not** beat the trailing baseline on realised P&L. That is reported as is; the final forecast file uses LightGBM because it can represent the strategy × regime interactions the synthetic log contains, and both models' fold-level predictions are exported so the choice can be revisited on real logs. `TARGET=win_rate` reruns everything for the win-rate target.

**Output**: `outputs/next_period_permutation_forecast.csv` — the forecast and rank of every permutation for the 5 business days after the run date, with its trailing 20-day stats for context; `feature_importance_lgbm.csv`.

## 6. Visual matrix output (`viz/charts.py`, `outputs/charts/`)

| file | what it shows |
|---|---|
| `01_macro_sentiment_timeline.png` | macro sentiment index through time with FOMC/CPI/NFP markers; signed daily surprises + EWMA; daily news sentiment + EWMA |
| `02_weekly_surprise_matrix.png` | week × category matrix of signed surprises (blue risk-positive, red risk-negative, grey = none) |
| `03_sentiment_vs_strategy_pnl.png` | weekly sentiment next to weekly realised P&L per strategy family |
| `04_permutation_matrix_pred_vs_realised.png` | **the permutation matrix**: predicted vs realised next-period target for every permutation at every walk-forward cut-off |
| `05_walk_forward_performance.png` | cumulative realised target of each selector's top-5 vs the all-permutation average; per-fold IC |
| `06_next_period_forecast.png` | ranked forecast bar chart for the next period |
| `07_feature_importance.png` | LightGBM gain share |

`outputs/charts_gallery.html` collects them on one page. Colours are a colour-blind-safe categorical set with a blue↔red diverging ramp; no dual axes.

## 7. Deliverables checklist

| requirement | where |
|---|---|
| Apify-based 180-day macro calendar (timestamps UTC/EST, names, actual, forecast) | `ingestion/apify_calendar.py`, table `macro_events`, `data/raw/apify_calendar_raw.json` (live) / `macro_calendar_fallback.csv` |
| Recent macro news via Tavily | `ingestion/tavily_news.py`, table `macro_news` |
| Join in a database | `db/schema.sql`, `data/processed/macro_agent.sqlite`, `outputs/joined_perm_daily_macro.csv` |
| Time-based feature engineering | `features/time_features.py`, `outputs/model_frame.csv` |
| 1-page macro outlook | `outputs/macro_outlook_summary.md` |
| Predictive model per permutation + walk-forward validation | `models/`, `outputs/walk_forward_*.csv`, `model_metrics.json`, `next_period_permutation_forecast.csv` |
| Charts of macro sentiment over time + visual matrix | `outputs/charts/*.png`, `charts_gallery.html` |

## 8. Assumptions and limitations

* Calendar times are assumed to be US/Eastern at the source; a feed in another zone would need `source_tz` changed in `config.py`.
* The synthetic fallbacks exist so the pipeline is runnable and testable without credentials; their values carry no information about the real economy and every output produced from them says so. Live credentials swap them out automatically.
* Lexicon sentiment ignores negation and context. News publication timestamps from Tavily can be missing; such articles are dated to the run date.
* With 180 days of data the walk-forward has 17 weekly folds; IC statistics are indicative only, and the permutation ranking should be read as a tilt, not a concentrated allocation signal.
