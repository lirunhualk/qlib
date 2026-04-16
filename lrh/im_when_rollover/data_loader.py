from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import FUTURE_SPECS


@dataclass
class MarketBundle:
    fut_code: str
    spot_code: str
    market: pd.DataFrame


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
        spot_daily = self._load_spot_daily(spec.spot_code, start_dt, end_dt)
        basis_daily = self._load_basis_daily(fut_code, start_dt, end_dt)

        market = futures_daily.merge(metadata, on="ts_code", how="left", validate="many_to_one")
        market = market.merge(spot_daily, on="dt", how="left", validate="many_to_one")
        market = market.merge(basis_daily, on=["ts_code", "dt"], how="left", validate="one_to_one")

        for col in ["dt", "list_date", "delist_date", "last_ddate"]:
            if col in market.columns:
                market[col] = pd.to_numeric(market[col], errors="coerce").astype("Int64")

        market["trade_date"] = pd.to_datetime(market["dt"].astype(str), format="%Y%m%d")
        market["delist_ts"] = pd.to_datetime(market["delist_date"].astype(str), format="%Y%m%d")
        market["list_ts"] = pd.to_datetime(market["list_date"].astype(str), format="%Y%m%d")
        market = market[market["trade_date"] >= market["list_ts"]].copy()
        market = market[market["trade_date"] <= market["delist_ts"]].copy()

        market["vwap"] = np.where(
            (market["vol"] > 0) & market["amount"].notna() & market["multiplier"].notna(),
            market["amount"] * 10000.0 / (market["vol"] * market["multiplier"]),
            market["close"],
        )
        market["vwap"] = market["vwap"].replace([np.inf, -np.inf], np.nan).fillna(market["close"])

        market.sort_values(["ts_code", "trade_date"], inplace=True)
        market["prev_close"] = market.groupby("ts_code")["close"].shift(1)
        market["trade_days_to_expiry"] = market.groupby("ts_code").cumcount(ascending=False)
        market["calendar_days_to_expiry"] = (market["delist_ts"] - market["trade_date"]).dt.days
        market["spot_close"] = market["spot_close"].astype(float)
        market["spot_open"] = market["spot_open"].astype(float)
        market["basis_points"] = market["spot_close"] - market["close"]
        market["discount_rate_spot"] = market["basis_points"] / market["spot_close"]
        market["discount_rate_fut"] = market["basis_points"] / market["close"]

        positive_days = market["calendar_days_to_expiry"].where(market["calendar_days_to_expiry"] > 0)
        market["annualized_carry"] = market["discount_rate_spot"] * 365.0 / positive_days

        market.sort_values(["trade_date", "delist_date", "ts_code"], inplace=True)
        market["expiry_rank"] = market.groupby("trade_date").cumcount() + 1
        market.reset_index(drop=True, inplace=True)

        return MarketBundle(fut_code=fut_code, spot_code=spec.spot_code, market=market)

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
            params={
                "prefix": f"{fut_code}%.CFX",
                "root_regex": root_regex,
                "start_dt": start_dt,
                "end_dt": end_dt,
            },
        )
        if df.empty:
            raise ValueError(f"No futures daily data found for {fut_code}")
        return df.drop_duplicates(subset=["ts_code", "dt"]).copy()

    def _load_basis_daily(self, fut_code: str, start_dt: int, end_dt: int) -> pd.DataFrame:
        root_regex = rf"^{fut_code}[0-9]+$"
        sql = text(
            """
            SELECT ts_code, dt, discount AS ts_discount, basis AS ts_basis
            FROM ts_fut_daily_basis
            WHERE ts_code LIKE :prefix
              AND SUBSTRING_INDEX(ts_code, '.', 1) REGEXP :root_regex
              AND dt BETWEEN :start_dt AND :end_dt
            """
        )
        df = pd.read_sql(
            sql,
            self.engine,
            params={
                "prefix": f"{fut_code}%.CFX",
                "root_regex": root_regex,
                "start_dt": start_dt,
                "end_dt": end_dt,
            },
        )
        return df.drop_duplicates(subset=["ts_code", "dt"]).copy()

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
        return df.drop_duplicates(subset=["dt"]).copy()
