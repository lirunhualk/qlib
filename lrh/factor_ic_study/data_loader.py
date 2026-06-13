from __future__ import annotations

import sys
import time

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def latest_stock_trade_date(engine: Engine) -> int:
    df = pd.read_sql_query("SELECT MAX(dt) AS max_dt FROM ts_daily", engine)
    value = df.loc[0, "max_dt"]
    if pd.isna(value):
        raise ValueError("ts_daily has no rows")
    return int(value)


def _read(engine: Engine, name: str, sql: str, params: dict[str, int]) -> pd.DataFrame:
    started = time.time()
    df = pd.read_sql_query(text(sql), engine, params=params)
    _log(f"loaded {name}: {len(df):,} rows in {time.time() - started:.1f}s")
    return df.drop_duplicates(subset=["ts_code", "dt"], keep="last") if {"ts_code", "dt"}.issubset(df.columns) else df


def load_raw_tables(engine: Engine, start_date: int, end_date: int, feature_set: str = "core") -> dict[str, pd.DataFrame]:
    params = {"start_date": int(start_date), "end_date": int(end_date)}
    tables: dict[str, pd.DataFrame] = {}
    tables["daily"] = _read(
        engine,
        "ts_daily",
        """
        SELECT ts_code, dt, open, high, low, close, pre_close, pct_chg, vol, amount
        FROM ts_daily
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    tables["adj"] = _read(
        engine,
        "ts_adj_factor",
        """
        SELECT ts_code, dt, adj_factor
        FROM ts_adj_factor
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    tables["daily_basic"] = _read(
        engine,
        "ts_daily_basic",
        """
        SELECT ts_code, dt, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb,
               ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share,
               total_mv, circ_mv
        FROM ts_daily_basic
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    tables["stk_factor"] = _read(
        engine,
        "ts_stk_factor",
        """
        SELECT ts_code, dt, macd_dif, macd_dea, macd, kdj_k, kdj_d, kdj_j,
               rsi_6, rsi_12, rsi_24, boll_upper, boll_mid, boll_lower, cci
        FROM ts_stk_factor
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    tables["moneyflow"] = _read(
        engine,
        "ts_moneyflow",
        """
        SELECT ts_code, dt, buy_sm_amount, sell_sm_amount, buy_md_amount, sell_md_amount,
               buy_lg_amount, sell_lg_amount, buy_elg_amount, sell_elg_amount,
               net_mf_vol, net_mf_amount
        FROM ts_moneyflow
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    if feature_set == "extended":
        tables["margin"] = _read(
            engine,
            "ts_margin_detail",
            """
            SELECT ts_code, dt, rzye, rqye, rzmre, rqyl, rzche, rqchl, rqmcl, rzrqye
            FROM ts_margin_detail
            WHERE dt BETWEEN :start_date AND :end_date
            """,
            params,
        )
        tables["fund_position"] = _read(
            engine,
            "ts_fp_fea",
            """
            SELECT ts_code, dt, fp_mkv, fp_fundratio, fp_stockratio
            FROM ts_fp_fea
            WHERE dt BETWEEN :start_date AND :end_date
            """,
            params,
        )
        tables["fina"] = _read(
            engine,
            "ts_fina_indicator_vip_fea",
            """
            SELECT ts_code, dt, fi_q_profit_yoy, fi_assets_turn, fi_profit_to_gr,
                   fi_roe_dt, fi_roe_yearly, fi_debt_to_assets, fi_netprofit_yoy, fi_tr_yoy
            FROM ts_fina_indicator_vip_fea
            WHERE dt BETWEEN :start_date AND :end_date
            """,
            params,
        )
        tables["pledge"] = _read(
            engine,
            "ts_pledge_detail_fea",
            """
            SELECT ts_code, dt, pledge_ratio_sum
            FROM ts_pledge_detail_fea
            WHERE dt BETWEEN :start_date AND :end_date
            """,
            params,
        )
        tables["holders"] = _read(
            engine,
            "ts_stk_holdernumber_fea",
            """
            SELECT ts_code, dt, holder_num
            FROM ts_stk_holdernumber_fea
            WHERE dt BETWEEN :start_date AND :end_date
            """,
            params,
        )
        tables["repurchase"] = _read(
            engine,
            "ts_repurchase_fea",
            """
            SELECT ts_code, dt, repurchase_amt, repurchase_amtratio
            FROM ts_repurchase_fea
            WHERE dt BETWEEN :start_date AND :end_date
            """,
            params,
        )
    elif feature_set != "core":
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    tables["limit_price"] = _read(
        engine,
        "ts_stk_limit",
        """
        SELECT ts_code, dt, up_limit, down_limit
        FROM ts_stk_limit
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    tables["stock_basic"] = _read(
        engine,
        "ts_stock_basic",
        """
        SELECT ts_code, dt, name, market, list_status, list_date
        FROM ts_stock_basic
        WHERE dt BETWEEN :start_date AND :end_date
        """,
        params,
    )
    return tables
