"""Macro-economic calendar ingestion via Apify.

Primary path
------------
Runs an Apify actor and normalises the returned dataset into a tidy table:

    event_time_utc | event_time_est | event_date | event_name | category |
    country | actual | forecast | previous | importance | source

The default actor is Apify's official ``apify/web-scraper`` driven by the page
function below, pointed at the ForexFactory monthly calendar pages that cover the
lookback window. Any other calendar actor can be used by setting
``APIFY_ACTOR_ID``; its items are passed through the same ``normalise_items``
function, which tolerates the common field names (date/time/event/actual/
forecast/previous/impact/currency).

Fallback path
-------------
If no ``APIFY_TOKEN`` is set, ``OFFLINE=1`` is set, or the actor run fails, a
synthetic calendar is generated from the *known publication schedules* of the
major US releases (CPI, PPI, FOMC, NFP, retail sales, GDP, PCE, ISM PMIs, weekly
jobless claims). Timestamps follow the real release-time conventions
(08:30 ET for BLS/BEA data, 14:00 ET for FOMC statements, 10:00 ET for ISM), but
the actual/forecast *values* are synthetic and the rows are tagged
``source="synthetic_fallback"`` so they are never mistaken for real data.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import RAW_DIR, Settings

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------------------
# Event taxonomy. ``direction`` encodes how a positive surprise (actual > forecast)
# is read by risk markets: +1 = growth-positive (good for risk), -1 = inflation /
# weakness (bad for risk), 0 = ambiguous.
# ---------------------------------------------------------------------------
EVENT_TAXONOMY: dict[str, dict[str, Any]] = {
    "cpi": {"category": "inflation", "direction": -1, "patterns": [r"\bcpi\b", r"consumer price"]},
    "core_cpi": {"category": "inflation", "direction": -1, "patterns": [r"core cpi"]},
    "ppi": {"category": "inflation", "direction": -1, "patterns": [r"\bppi\b", r"producer price"]},
    "pce": {"category": "inflation", "direction": -1, "patterns": [r"\bpce\b"]},
    "fomc": {"category": "monetary_policy", "direction": -1, "patterns": [r"fomc", r"federal funds", r"fed interest rate", r"rate decision"]},
    "nfp": {"category": "employment", "direction": +1, "patterns": [r"non-?farm", r"payroll"]},
    "unemployment_rate": {"category": "employment", "direction": -1, "patterns": [r"unemployment rate"]},
    "jobless_claims": {"category": "employment", "direction": -1, "patterns": [r"jobless claims", r"initial claims"]},
    "avg_hourly_earnings": {"category": "employment", "direction": -1, "patterns": [r"hourly earnings"]},
    "retail_sales": {"category": "growth", "direction": +1, "patterns": [r"retail sales"]},
    "gdp": {"category": "growth", "direction": +1, "patterns": [r"\bgdp\b"]},
    "ism_manufacturing": {"category": "growth", "direction": +1, "patterns": [r"ism manufacturing"]},
    "ism_services": {"category": "growth", "direction": +1, "patterns": [r"ism services", r"ism non-manufacturing"]},
    "consumer_confidence": {"category": "sentiment", "direction": +1, "patterns": [r"consumer confidence", r"consumer sentiment", r"michigan"]},
    "jolts": {"category": "employment", "direction": +1, "patterns": [r"jolts", r"job openings"]},
}


def classify_event(name: str) -> tuple[str, str, int]:
    """Return (event_key, category, direction) for a raw calendar event name."""
    n = name.lower()
    # core cpi must win over cpi
    for key in ["core_cpi"] + [k for k in EVENT_TAXONOMY if k != "core_cpi"]:
        for pat in EVENT_TAXONOMY[key]["patterns"]:
            if re.search(pat, n):
                return key, EVENT_TAXONOMY[key]["category"], EVENT_TAXONOMY[key]["direction"]
    return "other", "other", 0


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_value(raw: Any) -> float | None:
    """Parse '3.2%', '227K', '-0.1%', '4.25-4.50%', '1.8B' into a float.

    Units are kept per-event (percent stays percent, K stays thousands) because
    surprises are standardised *within* an event series downstream, so the unit
    cancels. For ranges (fed funds target) the upper bound is used.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if s in ("", "-", "—", "n/a", "N/A", "None"):
        return None
    s_clean = s.replace(",", "")
    # a range like "4.25-4.50%" -> split on an interior hyphen and take the upper bound
    rng_match = re.match(r"^\s*([-+]?\d*\.?\d+)\s*[-–]\s*(\d*\.?\d+)", s_clean)
    nums = [rng_match.group(2)] if rng_match else _NUM_RE.findall(s_clean)
    if not nums:
        return None
    val = float(nums[-1])
    mult = 1.0
    if re.search(r"\d\s*[kK]\b", s):
        mult = 1e3
    elif re.search(r"\d\s*[mM]\b", s):
        mult = 1e6
    elif re.search(r"\d\s*[bB]\b", s):
        mult = 1e9
    return val * mult


# ---------------------------------------------------------------------------
# Apify page function for apify/web-scraper (ForexFactory monthly calendar)
# ---------------------------------------------------------------------------
PAGE_FUNCTION = r"""
async function pageFunction(context) {
    const { $, request, log } = context;
    const rows = [];
    let currentDate = null;
    let currentTime = null;
    $('tr.calendar__row').each((_, tr) => {
        const row = $(tr);
        const d = row.find('td.calendar__date').text().trim();
        if (d) currentDate = d;
        const t = row.find('td.calendar__time').text().trim();
        if (t) currentTime = t;
        const currency = row.find('td.calendar__currency').text().trim();
        const event = row.find('td.calendar__event').text().trim();
        if (!event) return;
        const impactEl = row.find('td.calendar__impact span');
        rows.push({
            url: request.url,
            date: currentDate,
            time: currentTime,
            currency,
            event,
            impact: impactEl.attr('title') || impactEl.attr('class') || '',
            actual: row.find('td.calendar__actual').text().trim(),
            forecast: row.find('td.calendar__forecast').text().trim(),
            previous: row.find('td.calendar__previous').text().trim(),
        });
    });
    log.info(`Scraped ${rows.length} rows from ${request.url}`);
    return rows;
}
"""


def _month_urls(start: date, end: date) -> list[dict[str, str]]:
    urls = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        urls.append({"url": f"https://www.forexfactory.com/calendar?month={cur.strftime('%b').lower()}.{cur.year}"})
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return urls


def run_apify_actor(settings: Settings) -> list[dict[str, Any]]:
    """Run the configured Apify actor and return raw dataset items."""
    from apify_client import ApifyClient  # imported lazily so offline runs need no token

    client = ApifyClient(settings.apify_token)
    if settings.apify_actor_id == "apify/web-scraper":
        run_input = {
            "startUrls": _month_urls(settings.start_date, settings.run_date + timedelta(days=60)),
            "pageFunction": PAGE_FUNCTION,
            "proxyConfiguration": {"useApifyProxy": True},
            "maxRequestRetries": 3,
            "maxPagesPerCrawl": 12,
        }
    else:
        run_input = {"from": settings.start_date.isoformat(), "to": settings.run_date.isoformat(), "countries": ["US"]}
    log.info("Starting Apify actor %s", settings.apify_actor_id)
    run = client.actor(settings.apify_actor_id).call(run_input=run_input, timeout_secs=900)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    log.info("Apify returned %d items", len(items))
    (RAW_DIR / "apify_calendar_raw.json").write_text(json.dumps(items, indent=1, default=str))
    return items


def _parse_ff_datetime(date_str: str, time_str: str, year_hint: int) -> datetime | None:
    """ForexFactory shows e.g. date='Fri Apr 10' time='8:30am' in the viewer TZ.

    Anonymous viewers get the site default (US Eastern), so we localise to ET.
    'All Day' / 'Tentative' rows are pinned to 09:00 ET.
    """
    if not date_str:
        return None
    m = re.search(r"([A-Za-z]{3})\s+(\d{1,2})", date_str)
    if not m:
        return None
    try:
        d = datetime.strptime(f"{m.group(1)} {m.group(2)} {year_hint}", "%b %d %Y").date()
    except ValueError:
        return None
    tm = re.match(r"(\d{1,2}):(\d{2})(am|pm)", (time_str or "").lower())
    if tm:
        hh, mm = int(tm.group(1)) % 12, int(tm.group(2))
        if tm.group(3) == "pm":
            hh += 12
        t = time(hh, mm)
    else:
        t = time(9, 0)
    return datetime.combine(d, t, tzinfo=ET)


def normalise_items(items: Iterable[dict[str, Any]], settings: Settings) -> pd.DataFrame:
    """Normalise heterogeneous actor output to the canonical schema."""
    recs = []
    for it in items:
        name = it.get("event") or it.get("title") or it.get("name") or ""
        if not name:
            continue
        currency = (it.get("currency") or it.get("country") or "USD").upper()
        if currency not in ("USD", "US", "UNITED STATES"):
            continue
        ts = it.get("timestamp") or it.get("datetime") or it.get("date_utc")
        if ts:
            dt = pd.to_datetime(ts, utc=True).to_pydatetime()
        else:
            dt = _parse_ff_datetime(it.get("date", ""), it.get("time", ""), settings.run_date.year)
            if dt is None:
                continue
            # cross-year guard: a month page from the previous year
            if dt.date() > settings.run_date + timedelta(days=90):
                dt = dt.replace(year=dt.year - 1)
        dt_utc = dt.astimezone(UTC)
        key, cat, direction = classify_event(name)
        recs.append(
            {
                "event_time_utc": dt_utc,
                "event_time_est": dt_utc.astimezone(ET),
                "event_date": dt_utc.astimezone(ET).date(),
                "event_name": name,
                "event_key": key,
                "category": cat,
                "direction": direction,
                "country": "US",
                "actual": parse_value(it.get("actual")),
                "forecast": parse_value(it.get("forecast")),
                "previous": parse_value(it.get("previous")),
                "importance": str(it.get("impact") or it.get("importance") or ""),
                "source": f"apify:{settings.apify_actor_id}",
            }
        )
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df = df[(df.event_date >= settings.start_date) & (df.event_date <= settings.run_date + timedelta(days=60))]
    return df.sort_values("event_time_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic fallback built from real release-schedule conventions
# ---------------------------------------------------------------------------
# FOMC statement dates (second day of each 2026 meeting, 14:00 ET). Source: Fed
# published 2026 meeting calendar.
FOMC_2026 = [date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
             date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9)]
FOMC_2025 = [date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
             date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10)]


def _first_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _nth_weekday(y: int, m: int, weekday: int, n: int) -> date:
    d = date(y, m, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _months_between(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        yield cur.year, cur.month
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)


def synthetic_calendar(settings: Settings) -> pd.DataFrame:
    rng = np.random.default_rng(settings.synthetic_seed)
    start, run = settings.start_date, settings.run_date
    # extend 60 days past the run date so "days_to_next_*" features see the
    # forward schedule; future rows carry no actual value.
    end = run + timedelta(days=60)
    rows: list[dict[str, Any]] = []

    # Latent macro regime: slow-moving inflation & growth pressure that makes
    # surprises autocorrelated (as in real data) rather than white noise.
    n_days = (end - start).days + 1
    infl_pressure = np.cumsum(rng.normal(0, 0.04, n_days)); infl_pressure -= infl_pressure.mean()
    growth_pressure = np.cumsum(rng.normal(0, 0.05, n_days)); growth_pressure -= growth_pressure.mean()

    def pressure(d: date, arr) -> float:
        return float(arr[min(max((d - start).days, 0), n_days - 1)])

    def add(d: date, t: time, name: str, forecast: float, noise: float, pres: float, unit_round: int):
        if not (start <= d <= end):
            return
        actual = round(forecast + noise * rng.normal() + pres, unit_round)
        rows.append({"date": d, "time": t, "event": name, "actual": actual, "forecast": round(forecast, unit_round),
                     "previous": round(forecast + noise * rng.normal(), unit_round)})

    for y, m in _months_between(start - timedelta(days=31), end):
        # CPI: ~10th-15th, 08:30 ET; headline m/m and core m/m, y/y
        cpi_day = _nth_weekday(y, m, 1, 2) + timedelta(days=int(rng.integers(0, 4)))  # 2nd week Tue..Fri
        ip = pressure(cpi_day, infl_pressure)
        add(cpi_day, time(8, 30), "CPI m/m", 0.25, 0.08, 0.6 * ip, 1)
        add(cpi_day, time(8, 30), "Core CPI m/m", 0.25, 0.07, 0.5 * ip, 1)
        add(cpi_day, time(8, 30), "CPI y/y", 2.9, 0.08, 1.2 * ip, 1)
        # PPI next business day
        add(cpi_day + timedelta(days=1), time(8, 30), "PPI m/m", 0.2, 0.15, 0.5 * ip, 1)
        # Employment: NFP first Friday 08:30 ET
        nfp = _first_friday(y, m)
        gp = pressure(nfp, growth_pressure)
        add(nfp, time(8, 30), "Non-Farm Employment Change", 150e3, 55e3, 400e3 * gp, -3)
        add(nfp, time(8, 30), "Unemployment Rate", 4.2, 0.08, -0.5 * gp, 1)
        add(nfp, time(8, 30), "Average Hourly Earnings m/m", 0.3, 0.1, 0.3 * ip, 1)
        # JOLTS first Tuesday 10:00
        add(_nth_weekday(y, m, 1, 1), time(10, 0), "JOLTS Job Openings", 7.3e6, 0.25e6, 1.5e6 * gp, -4)
        # Retail sales ~ mid month 08:30
        add(_nth_weekday(y, m, 1, 3) + timedelta(days=int(rng.integers(0, 3))), time(8, 30), "Retail Sales m/m", 0.3, 0.3, 1.0 * gp, 1)
        # ISM PMIs: 1st business day (mfg) and 3rd business day (services), 10:00
        add(_nth_weekday(y, m, 0, 1) if date(y, m, 1).weekday() >= 5 else date(y, m, 1), time(10, 0), "ISM Manufacturing PMI", 49.0, 1.0, 5.0 * gp, 1)
        add(date(y, m, 1) + timedelta(days=2 + (date(y, m, 1).weekday() >= 4) * 2), time(10, 0), "ISM Services PMI", 52.0, 1.2, 5.0 * gp, 1)
        # PCE last Friday 08:30
        last = (date(y, m, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        last_fri = last - timedelta(days=(last.weekday() - 4) % 7)
        add(last_fri, time(8, 30), "Core PCE Price Index m/m", 0.25, 0.07, 0.5 * pressure(last_fri, infl_pressure), 1)
        # GDP advance/second/third ~ last Thursday
        add(last_fri - timedelta(days=1), time(8, 30), "Advance GDP q/q", 2.0, 0.6, 3.0 * gp, 1)
        # UMich sentiment prelim 2nd Friday 10:00
        add(_first_friday(y, m) + timedelta(days=7), time(10, 0), "Prelim UoM Consumer Sentiment", 60.0, 2.5, 10.0 * gp, 1)
        add(_nth_weekday(y, m, 1, 4), time(10, 0), "CB Consumer Confidence", 95.0, 3.0, 12.0 * gp, 1)

    # Weekly initial jobless claims every Thursday 08:30
    d = start + timedelta(days=(3 - start.weekday()) % 7)
    while d <= end:
        add(d, time(8, 30), "Unemployment Claims", 225e3, 9e3, -40e3 * pressure(d, growth_pressure), -3)
        d += timedelta(days=7)

    # FOMC statements 14:00 ET; target rate upper bound (level), forecast = consensus
    rate = 3.75
    for fd in sorted(FOMC_2025 + FOMC_2026):
        if start <= fd <= end:
            cut = rng.random() < 0.35
            forecast = rate - (0.25 if cut else 0.0)
            surprise = rng.choice([0.0, 0.0, 0.0, -0.25, 0.25], p=[0.7, 0.1, 0.1, 0.05, 0.05])
            actual = forecast + surprise
            rows.append({"date": fd, "time": time(14, 0), "event": "FOMC Statement / Federal Funds Rate",
                         "actual": actual, "forecast": forecast, "previous": rate})
            rate = actual

    recs = []
    for r in rows:
        dt_et = datetime.combine(r["date"], r["time"], tzinfo=ET)
        key, cat, direction = classify_event(r["event"])
        recs.append({
            "event_time_utc": dt_et.astimezone(UTC), "event_time_est": dt_et, "event_date": r["date"],
            "event_name": r["event"], "event_key": key, "category": cat, "direction": direction,
            "country": "US", "actual": r["actual"] if r["date"] <= run else None,
            "forecast": r["forecast"], "previous": r["previous"],
            "importance": "high" if key in ("cpi", "core_cpi", "fomc", "nfp", "unemployment_rate", "pce", "gdp") else "medium",
            "source": "synthetic_fallback",
        })
    df = pd.DataFrame(recs).sort_values("event_time_utc").reset_index(drop=True)
    return df


def fetch_calendar(settings: Settings) -> pd.DataFrame:
    """Return the macro calendar for [start_date, run_date], via Apify if possible."""
    if not settings.offline and settings.apify_token:
        try:
            items = run_apify_actor(settings)
            df = normalise_items(items, settings)
            if len(df) >= 20:
                log.info("Using Apify calendar: %d US events", len(df))
                return df
            log.warning("Apify returned too few usable rows (%d); falling back", len(df))
        except Exception as exc:  # noqa: BLE001 - any failure -> fallback, but recorded
            log.warning("Apify run failed (%s); using synthetic fallback calendar", exc)
    else:
        log.warning("No APIFY_TOKEN / OFFLINE=1 -> using synthetic fallback calendar")
    df = synthetic_calendar(settings)
    df.to_csv(RAW_DIR / "macro_calendar_fallback.csv", index=False)
    return df
