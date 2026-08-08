from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import FUTURE_SPECS


@dataclass(frozen=True)
class MarketBundle:
    fut_code: str
    spot_code: str
    market: pd.DataFrame
    spot: pd.DataFrame


def _to_int_date(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(pd.Timestamp(value).strftime("%Y%m%d"))


class MarketDataLoader:
    def __init__(self, engine: Engine):
        self.engine = engine

    def load_market_bundle(
        self,
        fut_code: str,
        *,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
    ) -> MarketBundle:
        spec = FUTURE_SPECS[fut_code]
        start_dt = _to_int_date(start_date) or spec.first_trade_date
        end_dt = _to_int_date(end_date) or 20991231

        metadata = self._load_contract_metadata(fut_code)
        futures_daily = self._load_futures_daily(fut_code, start_dt, end_dt)
        spot = self._load_spot_daily(spec.spot_code, start_dt, end_dt)

        market = futures_daily.merge(metadata, on="ts_code", how="left", validate="many_to_one")
        market = market.merge(
            spot[
                [
                    "dt",
                    "trade_date",
                    "spot_open",
                    "spot_close",
                    "spot_return_1d",
                    "spot_intraday_return",
                ]
            ],
            on=["dt", "trade_date"],
            how="left",
            validate="many_to_one",
        )

        for col in ["dt", "list_date", "delist_date", "last_ddate"]:
            if col in market.columns:
                market[col] = pd.to_numeric(market[col], errors="coerce").astype("Int64")

        market["delist_ts"] = pd.to_datetime(market["delist_date"].astype(str), format="%Y%m%d", errors="coerce")
        market["list_ts"] = pd.to_datetime(market["list_date"].astype(str), format="%Y%m%d", errors="coerce")
        market = market[market["trade_date"].ge(market["list_ts"]) & market["trade_date"].le(market["delist_ts"])]

        market["open"] = pd.to_numeric(market["open"], errors="coerce")
        market["close"] = pd.to_numeric(market["close"], errors="coerce")
        market["vol"] = pd.to_numeric(market["vol"], errors="coerce").fillna(0.0)
        market["amount"] = pd.to_numeric(market["amount"], errors="coerce").fillna(0.0)
        market["oi"] = pd.to_numeric(market["oi"], errors="coerce").fillna(0.0)
        market = market.dropna(subset=["open", "close", "spot_open", "spot_close"])

        market.sort_values(["ts_code", "trade_date"], inplace=True)
        market["prev_close"] = market.groupby("ts_code")["close"].shift(1)
        market["trade_days_to_expiry"] = market.groupby("ts_code").cumcount(ascending=False)
        market["calendar_days_to_expiry"] = (market["delist_ts"] - market["trade_date"]).dt.days
        market["intraday_return"] = np.where(market["open"] > 0, market["close"] / market["open"] - 1.0, np.nan)

        market.sort_values(["trade_date", "delist_date", "ts_code"], inplace=True)
        market["expiry_rank"] = market.groupby("trade_date").cumcount() + 1
        market.reset_index(drop=True, inplace=True)

        return MarketBundle(fut_code=fut_code, spot_code=spec.spot_code, market=market, spot=spot)

    def _load_contract_metadata(self, fut_code: str) -> pd.DataFrame:
        sql = text(
            """
            SELECT DISTINCT b.ts_code, b.symbol, b.fut_code, b.multiplier, b.list_date, b.delist_date, b.last_ddate
            FROM ts_fut_basic b
            JOIN (
                SELECT ts_code, MAX(dt) AS max_dt
                FROM ts_fut_basic
                WHERE exchange = 'CFFEX' AND fut_code = :fut_code
                GROUP BY ts_code
            ) latest
              ON latest.ts_code = b.ts_code
             AND latest.max_dt = b.dt
            WHERE b.exchange = 'CFFEX'
              AND b.fut_code = :fut_code
            """
        )
        df = pd.read_sql(sql, self.engine, params={"fut_code": fut_code})
        if df.empty:
            raise ValueError(f"No contract metadata found for {fut_code}")
        return df.drop_duplicates(subset=["ts_code"]).copy()

    def _load_futures_daily(self, fut_code: str, start_dt: int, end_dt: int) -> pd.DataFrame:
        root_regex = rf"^{fut_code}[0-9]+$"
        sql = text(
            """
            SELECT ts_code, dt, open, high, low, close, settle, vol, amount, oi
            FROM ts_fut_daily
            WHERE ts_code LIKE :prefix
              AND SUBSTRING_INDEX(ts_code, '.', 1) REGEXP :root_regex
              AND dt BETWEEN :start_dt AND :end_dt
            """
        )
        df = pd.read_sql(
            sql,
            self.engine,
            params={"prefix": f"{fut_code}%.CFX", "root_regex": root_regex, "start_dt": start_dt, "end_dt": end_dt},
        )
        if df.empty:
            raise ValueError(f"No futures daily data found for {fut_code}")
        df["dt"] = pd.to_numeric(df["dt"], errors="coerce").astype("Int64")
        df["trade_date"] = pd.to_datetime(df["dt"].astype(str), format="%Y%m%d", errors="coerce")
        return df.dropna(subset=["trade_date"]).drop_duplicates(subset=["ts_code", "dt"]).copy()

    def _load_spot_daily(self, spot_code: str, start_dt: int, end_dt: int) -> pd.DataFrame:
        sql = text(
            """
            SELECT dt, open AS spot_open, close AS spot_close
            FROM ts_index_daily
            WHERE ts_code = :spot_code
              AND dt BETWEEN :start_dt AND :end_dt
            """
        )
        df = pd.read_sql(sql, self.engine, params={"spot_code": spot_code, "start_dt": start_dt, "end_dt": end_dt})
        if df.empty:
            raise ValueError(f"No spot index data found for {spot_code}")
        df["dt"] = pd.to_numeric(df["dt"], errors="coerce").astype("Int64")
        df["trade_date"] = pd.to_datetime(df["dt"].astype(str), format="%Y%m%d", errors="coerce")
        df["spot_open"] = pd.to_numeric(df["spot_open"], errors="coerce")
        df["spot_close"] = pd.to_numeric(df["spot_close"], errors="coerce")
        df = df.dropna(subset=["trade_date", "spot_open", "spot_close"])
        df = df.drop_duplicates(subset=["dt"]).sort_values("trade_date").reset_index(drop=True)
        df["spot_return_1d"] = df["spot_close"].pct_change()
        df["spot_return_3d"] = df["spot_close"].pct_change(3)
        df["spot_intraday_return"] = np.where(df["spot_open"] > 0, df["spot_close"] / df["spot_open"] - 1.0, np.nan)
        return df
