-- SQLite schema for the macro-economics agent.
-- All timestamps are stored as ISO-8601 strings in UTC; *_est columns hold the
-- America/New_York rendering for human inspection / session logic.

CREATE TABLE IF NOT EXISTS macro_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time_utc  TEXT NOT NULL,
    event_time_est  TEXT NOT NULL,
    event_date      TEXT NOT NULL,          -- ET calendar date (YYYY-MM-DD)
    event_name      TEXT NOT NULL,
    event_key       TEXT NOT NULL,          -- taxonomy key (cpi, fomc, nfp, ...)
    category        TEXT NOT NULL,          -- inflation / employment / growth / monetary_policy / sentiment
    direction       INTEGER NOT NULL,       -- +1 growth-positive surprise, -1 risk-negative, 0 ambiguous
    country         TEXT NOT NULL,
    actual          REAL,
    forecast        REAL,
    previous        REAL,
    importance      TEXT,
    source          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_macro_events_date ON macro_events(event_date);

CREATE TABLE IF NOT EXISTS macro_news (
    news_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    published_utc   TEXT NOT NULL,
    published_date  TEXT NOT NULL,
    title           TEXT,
    url             TEXT,
    content         TEXT,
    query           TEXT,
    relevance       REAL,
    sentiment       REAL,                   -- lexicon score in [-1, 1]
    source          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_macro_news_date ON macro_news(published_date);

CREATE TABLE IF NOT EXISTS trades (
    trade_id        INTEGER PRIMARY KEY,
    strategy        TEXT NOT NULL,
    permutation_id  TEXT NOT NULL,
    lookback        INTEGER,
    risk_mult       REAL,
    session         TEXT,
    symbol          TEXT,
    side            TEXT,
    entry_time_utc  TEXT NOT NULL,
    exit_time_utc   TEXT NOT NULL,
    exit_date       TEXT NOT NULL,          -- ET date on which P&L is realised
    pnl             REAL NOT NULL,
    win             INTEGER NOT NULL,
    source          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_trades_perm_date ON trades(permutation_id, exit_date);

-- Daily macro state (one row per calendar day) produced by feature engineering.
CREATE TABLE IF NOT EXISTS daily_macro_features (
    day                     TEXT PRIMARY KEY,
    n_events                INTEGER,
    n_high_impact           INTEGER,
    surprise_sum_signed     REAL,
    surprise_abs_mean       REAL,
    surprise_inflation      REAL,
    surprise_employment     REAL,
    surprise_growth         REAL,
    surprise_policy         REAL,
    surprise_index_ewm      REAL,           -- EWMA(half-life 10d) of signed surprise
    shock_index_ewm         REAL,           -- EWMA(half-life 5d) of |surprise|
    news_count              INTEGER,
    news_sentiment_mean     REAL,
    news_sentiment_ewm      REAL,           -- EWMA(half-life 3d)
    macro_sentiment_index   REAL,           -- 0.6*z(surprise_index_ewm) + 0.4*z(news_sentiment_ewm)
    days_to_next_fomc       INTEGER,
    days_to_next_cpi        INTEGER,
    days_to_next_nfp        INTEGER,
    days_since_last_fomc    INTEGER,
    days_since_last_cpi     INTEGER,
    days_since_last_nfp     INTEGER
);

-- Daily per-permutation aggregates (the modelling grain).
CREATE TABLE IF NOT EXISTS perm_daily (
    permutation_id  TEXT NOT NULL,
    day             TEXT NOT NULL,
    n_trades        INTEGER,
    pnl             REAL,
    win_rate        REAL,
    PRIMARY KEY (permutation_id, day)
);

-- Join view: every trade enriched with the macro state of its realisation day.
CREATE VIEW IF NOT EXISTS v_trades_macro AS
SELECT t.*, f.surprise_index_ewm, f.shock_index_ewm, f.news_sentiment_ewm,
       f.macro_sentiment_index, f.n_high_impact, f.days_to_next_fomc, f.days_to_next_cpi, f.days_to_next_nfp
FROM trades t
LEFT JOIN daily_macro_features f ON f.day = t.exit_date;
