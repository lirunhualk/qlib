from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass
class MarketData:
    stock: pd.DataFrame
    benchmark: pd.DataFrame


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_sql(engine: Engine, name: str, sql: str, params: dict[str, int | str]) -> pd.DataFrame:
    started = time.time()
    df = pd.read_sql_query(text(sql), engine, params=params)
    _log(f"loaded {name}: {len(df):,} rows in {time.time() - started:.1f}s")
    return df


def _dedupe(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=subset, keep="last")


def latest_stock_trade_date(engine: Engine) -> int:
    df = pd.read_sql_query("SELECT MAX(dt) AS max_dt FROM ts_daily", engine)
    value = df.loc[0, "max_dt"]
    if pd.isna(value):
        raise ValueError("ts_daily has no rows")
    return int(value)


def load_stock_panel(engine: Engine, start_date: int, end_date: int) -> pd.DataFrame:
    params = {"start_date": int(start_date), "end_date": int(end_date)}
    daily = _read_sql(
        engine,
        "ts_daily",
        """
        SELECT ts_code, dt, open, close, pre_close, amount
        FROM ts_daily
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    adj = _read_sql(
        engine,
        "ts_adj_factor",
        """
        SELECT ts_code, dt, adj_factor
        FROM ts_adj_factor
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    basic = _read_sql(
        engine,
        "ts_daily_basic",
        """
        SELECT ts_code, dt, total_mv, circ_mv, turnover_rate_f, turnover_rate
        FROM ts_daily_basic
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    limit_price = _read_sql(
        engine,
        "ts_stk_limit",
        """
        SELECT ts_code, dt, up_limit, down_limit
        FROM ts_stk_limit
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    stock_basic = _read_sql(
        engine,
        "ts_stock_basic",
        """
        SELECT ts_code, dt, name, market, list_status, list_date
        FROM ts_stock_basic
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )

    daily = _dedupe(daily, ["ts_code", "dt"])
    adj = _dedupe(adj, ["ts_code", "dt"])
    basic = _dedupe(basic, ["ts_code", "dt"])
    limit_price = _dedupe(limit_price, ["ts_code", "dt"])
    stock_basic = _dedupe(stock_basic, ["ts_code", "dt"])

    started = time.time()
    panel = daily.merge(adj, on=["ts_code", "dt"], how="left", validate="one_to_one")
    panel = panel.merge(basic, on=["ts_code", "dt"], how="left", validate="one_to_one")
    panel = panel.merge(limit_price, on=["ts_code", "dt"], how="left", validate="one_to_one")
    panel = panel.merge(stock_basic, on=["ts_code", "dt"], how="left", validate="many_to_one")
    _log(f"merged stock panel: {len(panel):,} rows in {time.time() - started:.1f}s")

    if panel.empty:
        raise ValueError("No stock data loaded for the requested date range")
    return panel


def load_benchmark(engine: Engine, code: str, start_date: int, end_date: int) -> pd.DataFrame:
    params = {"code": code, "start_date": int(start_date), "end_date": int(end_date)}
    benchmark = _read_sql(
        engine,
        f"ts_index_daily[{code}]",
        """
        SELECT ts_code, dt, close
        FROM ts_index_daily
        WHERE ts_code = :code
          AND dt BETWEEN :start_date AND :end_date
        ORDER BY dt
        """,
        params,
    )
    if benchmark.empty:
        raise ValueError(f"No benchmark data found for {code}")
    benchmark = benchmark.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")
    benchmark["bench"] = pd.to_numeric(benchmark["close"], errors="coerce").pct_change()
    return benchmark[["dt", "bench"]]


def load_market_data(engine: Engine, start_date: int, test_start_date: int, end_date: int, benchmark_code: str) -> MarketData:
    stock = load_stock_panel(engine, start_date, end_date)
    benchmark = load_benchmark(engine, benchmark_code, test_start_date, end_date)
    return MarketData(stock=stock, benchmark=benchmark)

