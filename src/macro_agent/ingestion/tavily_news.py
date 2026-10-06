"""Recent macro-economic news via Tavily, with a lexicon sentiment score.

Primary path: ``TavilyClient.search(topic="news")`` for each query in
``settings.tavily_queries``, restricted to the lookback window. Results are
cached to ``data/raw/tavily_news_raw.json``.

Fallback (no ``TAVILY_API_KEY`` / ``OFFLINE=1`` / API error): a synthetic set of
headlines is generated from templates spread across the window and tagged
``source="synthetic_fallback"``.

Sentiment is scored with a small, transparent finance lexicon (no external
model), returning a value in [-1, 1] where +1 is risk-positive / dovish and -1
is risk-negative / hawkish. The lexicon is deliberately simple so the method can
be stated exactly; swap ``score_text`` for FinBERT or an LLM if desired.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import RAW_DIR, Settings

log = logging.getLogger(__name__)
UTC = ZoneInfo("UTC")

POSITIVE = {
    "cools": 1.0, "cooling": 1.0, "eases": 1.0, "easing": 1.0, "slows": 0.6, "slowing inflation": 1.0,
    "rate cut": 1.0, "cuts rates": 1.0, "cut rates": 1.0, "dovish": 1.0, "soft landing": 1.0,
    "beats": 0.8, "beat expectations": 1.0, "stronger than expected": 0.8, "resilient": 0.8,
    "rally": 0.8, "record high": 0.6, "growth accelerates": 1.0, "hiring rebounds": 0.8,
    "disinflation": 1.0, "below forecast inflation": 1.0, "optimism": 0.6, "upbeat": 0.6,
}
NEGATIVE = {
    "hot": -0.8, "hotter than expected": -1.0, "reaccelerat": -1.0, "sticky": -0.8, "stubborn": -0.8,
    "hawkish": -1.0, "rate hike": -1.0, "hikes rates": -1.0, "higher for longer": -1.0, "holds rates": -0.3,
    "misses": -0.8, "missed expectations": -1.0, "weaker than expected": -0.8, "slump": -1.0,
    "recession": -1.0, "layoffs": -0.8, "contraction": -1.0, "sell-off": -1.0, "selloff": -1.0,
    "tariff": -0.5, "stagflation": -1.0, "jobless claims rise": -0.8, "unemployment rises": -1.0,
    "downgrade": -0.8, "uncertainty": -0.5, "volatility": -0.4,
}


def score_text(text: str) -> float:
    """Lexicon score in [-1, 1]; 0 when no lexicon term is present."""
    t = text.lower()
    score, hits = 0.0, 0
    for lex in (POSITIVE, NEGATIVE):
        for term, w in lex.items():
            n = len(re.findall(re.escape(term), t))
            if n:
                score += w * n
                hits += n
    if hits == 0:
        return 0.0
    return float(np.clip(score / hits, -1, 1))


def run_tavily(settings: Settings) -> list[dict[str, Any]]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    items: list[dict[str, Any]] = []
    for q in settings.tavily_queries:
        resp = client.search(
            query=q,
            topic="news",
            search_depth="advanced",
            max_results=settings.tavily_max_results,
            days=settings.lookback_days,
            include_answer=False,
        )
        for r in resp.get("results", []):
            r["query"] = q
            items.append(r)
        log.info("Tavily '%s' -> %d results", q, len(resp.get("results", [])))
    (RAW_DIR / "tavily_news_raw.json").write_text(json.dumps(items, indent=1, default=str))
    return items


def normalise_news(items: list[dict[str, Any]], settings: Settings, source: str) -> pd.DataFrame:
    recs = []
    for it in items:
        pub = it.get("published_date") or it.get("published") or it.get("date")
        try:
            ts = pd.to_datetime(pub, utc=True)
        except Exception:  # noqa: BLE001
            ts = pd.NaT
        if pd.isna(ts):
            ts = pd.Timestamp(settings.run_date, tz="UTC")
        title = it.get("title") or ""
        content = it.get("content") or it.get("snippet") or ""
        recs.append({
            "published_utc": ts,
            "published_date": ts.tz_convert("America/New_York").date(),
            "title": title,
            "url": it.get("url", ""),
            "content": content[:1000],
            "query": it.get("query", ""),
            "relevance": float(it.get("score", 0.0) or 0.0),
            "sentiment": score_text(f"{title}. {content}"),
            "source": source,
        })
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["url"]) if df["url"].astype(bool).any() else df
    df = df[(df.published_date >= settings.start_date) & (df.published_date <= settings.run_date)]
    return df.sort_values("published_utc").reset_index(drop=True)


_TEMPLATES = [
    ("CPI cools more than expected in {m}, bolstering case for Fed rate cut", 0.9),
    ("Inflation runs hotter than expected; traders pare back rate-cut bets", -0.9),
    ("Fed holds rates steady, signals higher for longer stance", -0.6),
    ("Fed cuts rates by 25bp, cites cooling labour market", 0.7),
    ("Nonfarm payrolls beat expectations as hiring rebounds", 0.8),
    ("Jobs report misses; unemployment rises to multi-year high", -0.9),
    ("Jobless claims rise for third straight week", -0.6),
    ("Retail sales stronger than expected, consumer resilient", 0.7),
    ("ISM manufacturing slips further into contraction", -0.7),
    ("Services PMI beats as demand holds up", 0.6),
    ("Tariff uncertainty weighs on business investment outlook", -0.6),
    ("Markets rally to record high on soft-landing optimism", 0.8),
    ("Sticky core inflation keeps Fed cautious, officials say", -0.7),
    ("Core PCE eases, disinflation trend intact", 0.8),
    ("Stagflation fears grow after weak GDP and hot prices", -1.0),
    ("Consumer sentiment upbeat as gasoline prices fall", 0.5),
]


def synthetic_news(settings: Settings, calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Headlines spread across the window; tone follows the calendar surprise sign when available."""
    rng = np.random.default_rng(settings.synthetic_seed + 1)
    items = []
    day = settings.start_date
    while day <= settings.run_date:
        n = rng.integers(1, 4)
        bias = 0.0
        if calendar is not None and not calendar.empty:
            recent = calendar[(calendar.event_date <= day) & (calendar.event_date >= day - timedelta(days=5))]
            if not recent.empty:
                s = ((recent.actual - recent.forecast) * recent.direction)
                bias = float(np.sign(s.sum())) * 0.6
        for _ in range(n):
            # choose template whose tone agrees with bias with prob 0.7
            tones = np.array([t[1] for t in _TEMPLATES])
            w = np.where(np.sign(tones) == np.sign(bias), 0.7, 0.3) if bias else np.ones(len(tones))
            idx = rng.choice(len(_TEMPLATES), p=w / w.sum())
            title = _TEMPLATES[idx][0].format(m=day.strftime("%B"))
            ts = datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(hours=int(rng.integers(11, 22)))
            items.append({"title": title, "url": f"https://example.invalid/news/{day.isoformat()}/{idx}",
                          "content": title, "published_date": ts.isoformat(), "score": 0.5, "query": "synthetic"})
        day += timedelta(days=1)
    return normalise_news(items, settings, source="synthetic_fallback")


def fetch_news(settings: Settings, calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    if not settings.offline and settings.tavily_api_key:
        try:
            items = run_tavily(settings)
            df = normalise_news(items, settings, source="tavily")
            if len(df) >= 5:
                log.info("Using Tavily news: %d articles", len(df))
                return df
            log.warning("Tavily returned too few dated articles (%d); falling back", len(df))
        except Exception as exc:  # noqa: BLE001
            log.warning("Tavily failed (%s); using synthetic fallback news", exc)
    else:
        log.warning("No TAVILY_API_KEY / OFFLINE=1 -> using synthetic fallback news")
    df = synthetic_news(settings, calendar)
    df.to_csv(RAW_DIR / "macro_news_fallback.csv", index=False)
    return df
