"""Thin SQLite layer: create schema, load tables, run joins."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from ..config import DB_PATH

log = logging.getLogger(__name__)
SCHEMA = Path(__file__).with_name("schema.sql")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA.read_text())
    return con


def _stringify_times(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        elif out[c].dtype == object and len(out) and hasattr(out[c].dropna().iloc[0] if out[c].notna().any() else None, "isoformat"):
            out[c] = out[c].map(lambda v: v.isoformat() if v is not None and hasattr(v, "isoformat") else v)
    return out


def replace_table(con: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    con.execute(f"DELETE FROM {name}")
    cols = [c for c in df.columns if c in _table_columns(con, name)]
    _stringify_times(df[cols]).to_sql(name, con, if_exists="append", index=False)
    con.commit()
    log.info("Loaded %d rows into %s", len(df), name)


def _table_columns(con: sqlite3.Connection, name: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({name})")}


def read_sql(con: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, con)


def perm_daily_from_trades(con: sqlite3.Connection) -> pd.DataFrame:
    """Aggregate trades to the (permutation, day) grain inside the database."""
    sql = """
    SELECT permutation_id, exit_date AS day, COUNT(*) AS n_trades,
           SUM(pnl) AS pnl, AVG(win) AS win_rate
    FROM trades GROUP BY permutation_id, exit_date ORDER BY day, permutation_id
    """
    df = read_sql(con, sql)
    replace_table(con, "perm_daily", df)
    return df


def joined_training_frame(con: sqlite3.Connection) -> pd.DataFrame:
    """perm_daily LEFT JOIN daily_macro_features — the modelling table."""
    sql = """
    SELECT p.*, f.*
    FROM perm_daily p
    LEFT JOIN daily_macro_features f ON f.day = p.day
    ORDER BY p.day, p.permutation_id
    """
    df = read_sql(con, sql)
    df = df.loc[:, ~df.columns.duplicated()]
    return df
