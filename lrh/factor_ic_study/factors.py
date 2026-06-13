from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FactorStudyConfig


@dataclass(frozen=True)
class FactorBuildResult:
    panel: pd.DataFrame
    factor_columns: list[str]
    factor_sources: pd.DataFrame


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _merge_base(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    started = time.time()
    panel = raw_tables["daily"].merge(raw_tables["adj"], on=["ts_code", "dt"], how="left", validate="one_to_one")
    for name in [
        "daily_basic",
        "stk_factor",
        "moneyflow",
        "limit_price",
        "stock_basic",
    ]:
        panel = panel.merge(raw_tables[name], on=["ts_code", "dt"], how="left", validate="one_to_one")
    for name in ["margin", "fund_position", "fina", "pledge", "holders", "repurchase"]:
        if name in raw_tables:
            panel = panel.merge(raw_tables[name], on=["ts_code", "dt"], how="left", validate="one_to_one")
    print(f"merged factor panel: {len(panel):,} rows in {time.time() - started:.1f}s", flush=True)
    return panel


def build_factor_panel(raw_tables: dict[str, pd.DataFrame], config: FactorStudyConfig) -> FactorBuildResult:
    df = _merge_base(raw_tables)
    numeric_cols = [col for col in df.columns if col not in {"ts_code", "name", "market", "list_status"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dt"] = df["dt"].astype("int32")

    df.sort_values(["ts_code", "dt"], inplace=True)
    grouped = df.groupby("ts_code", sort=False)
    df["adj_open"] = df["open"] * df["adj_factor"]
    df["adj_close"] = df["close"] * df["adj_factor"]
    df["exec_ret_fwd"] = grouped["adj_open"].shift(-2) / grouped["adj_open"].shift(-1) - 1.0
    df["cc_ret"] = grouped["adj_close"].pct_change(fill_method=None)
    df["abs_ret_over_amount"] = df["cc_ret"].abs() / df["amount"].replace(0, np.nan)

    factor_sources: list[dict[str, str]] = []

    optional_cols = [
        "rzye",
        "rqye",
        "rzmre",
        "rqyl",
        "rzche",
        "rqchl",
        "rqmcl",
        "rzrqye",
        "fp_mkv",
        "fp_fundratio",
        "fp_stockratio",
        "fi_q_profit_yoy",
        "fi_assets_turn",
        "fi_profit_to_gr",
        "fi_roe_dt",
        "fi_roe_yearly",
        "fi_debt_to_assets",
        "fi_netprofit_yoy",
        "fi_tr_yoy",
        "pledge_ratio_sum",
        "holder_num",
        "repurchase_amt",
        "repurchase_amtratio",
    ]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    def add_factor(name: str, source: str, description: str) -> None:
        if name in df.columns:
            factor_sources.append({"factor": name, "source": source, "description": description})

    # Price and volume derived factors.
    df["intraday_ret"] = _safe_div(df["close"], df["open"]) - 1.0
    df["gap_ret"] = _safe_div(df["open"], df["pre_close"]) - 1.0
    df["range_pct"] = _safe_div(df["high"], df["low"]) - 1.0
    df["close_position"] = _safe_div(df["close"] - df["low"], df["high"] - df["low"])
    df["limit_width"] = _safe_div(df["up_limit"], df["down_limit"]) - 1.0
    for window in [1, 2, 3, 5, 10, 20, 40, 60, 120]:
        df[f"ret_{window}d"] = df["adj_close"] / grouped["adj_close"].shift(window) - 1.0
        add_factor(f"ret_{window}d", "derived_price", f"{window}-day adjusted close return")
    for window in [5, 10, 20, 40, 60, 120]:
        df[f"vol_{window}d"] = (
            grouped["cc_ret"].rolling(window, min_periods=max(3, window // 2)).std().reset_index(level=0, drop=True)
            * math.sqrt(252)
        )
        df[f"amt_mean_{window}d"] = grouped["amount"].rolling(window, min_periods=max(3, window // 2)).mean().reset_index(level=0, drop=True)
        df[f"amt_std_{window}d"] = grouped["amount"].rolling(window, min_periods=max(3, window // 2)).std().reset_index(level=0, drop=True)
        df[f"turn_mean_{window}d"] = (
            grouped["turnover_rate_f"].rolling(window, min_periods=max(3, window // 2)).mean().reset_index(level=0, drop=True)
        )
        df[f"turn_std_{window}d"] = (
            grouped["turnover_rate_f"].rolling(window, min_periods=max(3, window // 2)).std().reset_index(level=0, drop=True)
        )
        df[f"amihud_{window}d"] = (
            grouped["abs_ret_over_amount"].rolling(window, min_periods=max(3, window // 2)).mean().reset_index(level=0, drop=True)
        )
        add_factor(f"vol_{window}d", "derived_price", f"{window}-day annualized volatility")
        add_factor(f"amt_mean_{window}d", "derived_liquidity", f"{window}-day average amount")
        add_factor(f"amt_std_{window}d", "derived_liquidity", f"{window}-day amount volatility")
        add_factor(f"turn_mean_{window}d", "derived_turnover", f"{window}-day average free-float turnover")
        add_factor(f"turn_std_{window}d", "derived_turnover", f"{window}-day turnover volatility")
        add_factor(f"amihud_{window}d", "derived_liquidity", f"{window}-day Amihud illiquidity proxy")

    df["amt_ratio_5_20"] = _safe_div(df["amt_mean_5d"], df["amt_mean_20d"])
    df["amt_ratio_20_60"] = _safe_div(df["amt_mean_20d"], df["amt_mean_60d"])
    df["turn_ratio_5_20"] = _safe_div(df["turn_mean_5d"], df["turn_mean_20d"])
    df["turn_ratio_20_60"] = _safe_div(df["turn_mean_20d"], df["turn_mean_60d"])
    df["log_total_mv"] = np.log(df["total_mv"].where(df["total_mv"] > 0))
    df["log_circ_mv"] = np.log(df["circ_mv"].where(df["circ_mv"] > 0))
    df["free_share_ratio"] = _safe_div(df["free_share"], df["total_share"])
    df["float_share_ratio"] = _safe_div(df["float_share"], df["total_share"])
    df["boll_position"] = _safe_div(df["adj_close"] - df["boll_lower"], df["boll_upper"] - df["boll_lower"])
    for window in [5, 10, 20, 60, 120]:
        ma = grouped["adj_close"].rolling(window, min_periods=max(3, window // 2)).mean().reset_index(level=0, drop=True)
        df[f"ma_gap_{window}d"] = df["adj_close"] / ma - 1.0
        add_factor(f"ma_gap_{window}d", "derived_price", f"adjusted close gap to {window}-day moving average")

    for col in [
        "pct_chg",
        "intraday_ret",
        "gap_ret",
        "range_pct",
        "close_position",
        "limit_width",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
        "log_total_mv",
        "log_circ_mv",
        "free_share_ratio",
        "float_share_ratio",
    ]:
        add_factor(col, "daily_basic_or_price", col)

    for col in ["macd_dif", "macd_dea", "macd", "kdj_k", "kdj_d", "kdj_j", "rsi_6", "rsi_12", "rsi_24", "cci", "boll_position"]:
        add_factor(col, "ts_stk_factor", col)

    # Moneyflow factors.
    money_amount_cols = [
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_md_amount",
        "sell_md_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
    ]
    df["moneyflow_total_amount"] = df[money_amount_cols].sum(axis=1, min_count=1)
    for prefix in ["sm", "md", "lg", "elg"]:
        df[f"net_{prefix}_amount"] = df[f"buy_{prefix}_amount"] - df[f"sell_{prefix}_amount"]
        df[f"net_{prefix}_amount_ratio"] = _safe_div(df[f"net_{prefix}_amount"], df["moneyflow_total_amount"])
        df[f"buy_{prefix}_amount_ratio"] = _safe_div(df[f"buy_{prefix}_amount"], df["moneyflow_total_amount"])
        add_factor(f"net_{prefix}_amount", "ts_moneyflow", f"net {prefix} money amount")
        add_factor(f"net_{prefix}_amount_ratio", "ts_moneyflow", f"net {prefix} money amount / total moneyflow")
        add_factor(f"buy_{prefix}_amount_ratio", "ts_moneyflow", f"buy {prefix} money amount / total moneyflow")
    df["net_mf_amount_ratio"] = _safe_div(df["net_mf_amount"], df["moneyflow_total_amount"])
    add_factor("net_mf_amount", "ts_moneyflow", "net moneyflow amount")
    add_factor("net_mf_amount_ratio", "ts_moneyflow", "net moneyflow amount / total moneyflow")
    add_factor("net_mf_vol", "ts_moneyflow", "net moneyflow volume")

    # Margin, fund ownership, financial, pledge, holder, and repurchase factors.
    df["rzye_to_mv"] = _safe_div(df["rzye"], df["total_mv"])
    df["rqye_to_mv"] = _safe_div(df["rqye"], df["total_mv"])
    df["rzrqye_to_mv"] = _safe_div(df["rzrqye"], df["total_mv"])
    df["rzmre_to_amount"] = _safe_div(df["rzmre"], df["amount"])
    df["rzche_to_amount"] = _safe_div(df["rzche"], df["amount"])
    df["margin_buy_minus_repay"] = df["rzmre"] - df["rzche"]
    for col in [
        "rzye",
        "rqye",
        "rzmre",
        "rqyl",
        "rzche",
        "rqchl",
        "rqmcl",
        "rzrqye",
        "rzye_to_mv",
        "rqye_to_mv",
        "rzrqye_to_mv",
        "rzmre_to_amount",
        "rzche_to_amount",
        "margin_buy_minus_repay",
        "fp_mkv",
        "fp_fundratio",
        "fp_stockratio",
        "fi_q_profit_yoy",
        "fi_assets_turn",
        "fi_profit_to_gr",
        "fi_roe_dt",
        "fi_roe_yearly",
        "fi_debt_to_assets",
        "fi_netprofit_yoy",
        "fi_tr_yoy",
        "pledge_ratio_sum",
        "holder_num",
        "repurchase_amt",
        "repurchase_amtratio",
    ]:
        add_factor(col, "auxiliary_daily_features", col)

    trade_dates = pd.to_datetime(df["dt"].astype(str), format="%Y%m%d", errors="coerce")
    list_dates = pd.to_datetime(df["list_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    df["age_days"] = (trade_dates - list_dates).dt.days
    name = df["name"].fillna("")
    df["amt20_for_filter"] = grouped["amount"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    df["base_ok"] = (
        df["list_status"].eq("L")
        & ~df["market"].isin(config.excluded_markets)
        & ~name.str.contains(r"ST|退", regex=True)
        & (df["age_days"] >= config.min_age_days)
        & (df["close"] >= config.min_price)
        & (df["amt20_for_filter"] >= config.min_amt20)
        & df["exec_ret_fwd"].notna()
    )

    sources = pd.DataFrame(factor_sources).drop_duplicates(subset=["factor"], keep="first")
    candidate_cols = sources["factor"].tolist()
    valid_cols = []
    mask = df["base_ok"] & (df["dt"] >= config.test_start_date)
    for col in candidate_cols:
        if col in df.columns and df.loc[mask, col].notna().sum() >= config.min_ic_obs:
            valid_cols.append(col)

    keep_cols = ["dt", "ts_code", "name", "base_ok", "exec_ret_fwd"] + valid_cols
    df = df[keep_cols].replace([np.inf, -np.inf], np.nan).sort_values(["dt", "ts_code"]).reset_index(drop=True)
    sources = sources[sources["factor"].isin(valid_cols)].reset_index(drop=True)
    return FactorBuildResult(panel=df, factor_columns=valid_cols, factor_sources=sources)
