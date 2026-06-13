from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig


@dataclass
class SelectionResult:
    weights: dict[str, float]
    selected: pd.DataFrame
    blocked_buys: int
    blocked_sells: int


@dataclass
class BacktestResult:
    config: StrategyConfig
    daily: pd.DataFrame
    holdings: pd.DataFrame
    annual_returns: pd.DataFrame
    summary: dict[str, float | int | str | None]


def prepare_features(raw: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    df = raw.copy()
    numeric_cols = [
        "open",
        "close",
        "pre_close",
        "amount",
        "adj_factor",
        "total_mv",
        "circ_mv",
        "turnover_rate_f",
        "turnover_rate",
        "up_limit",
        "down_limit",
        "list_date",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dt"] = pd.to_numeric(df["dt"], errors="coerce").astype("int32")

    df.sort_values(["ts_code", "dt"], inplace=True)
    df["adj_open"] = df["open"] * df["adj_factor"]
    df["adj_close"] = df["close"] * df["adj_factor"]
    grouped = df.groupby("ts_code", sort=False)

    df["open_ret_fwd"] = grouped["adj_open"].shift(-1) / df["adj_open"] - 1.0
    df["cc_ret"] = grouped["adj_close"].pct_change(fill_method=None)
    df["ret20"] = df["adj_close"] / grouped["adj_close"].shift(20) - 1.0
    df["ret60"] = df["adj_close"] / grouped["adj_close"].shift(60) - 1.0
    df["ret120"] = df["adj_close"] / grouped["adj_close"].shift(120) - 1.0
    df["vol60"] = (
        grouped["cc_ret"]
        .rolling(60, min_periods=40)
        .std()
        .reset_index(level=0, drop=True)
        * math.sqrt(252)
    )
    df["amt20"] = grouped["amount"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    df["turn20"] = grouped["turnover_rate_f"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)

    trade_dates = pd.to_datetime(df["dt"].astype(str), format="%Y%m%d", errors="coerce")
    list_dates = pd.to_datetime(df["list_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    df["age_days"] = (trade_dates - list_dates).dt.days

    name = df["name"].fillna("")
    market_ok = ~df["market"].isin(config.excluded_markets)
    df["base_ok"] = (
        df["list_status"].eq("L")
        & market_ok
        & ~name.str.contains(r"ST|退", regex=True)
        & (df["age_days"] >= config.min_age_days)
        & (df["close"] >= config.min_price)
        & df["open_ret_fwd"].notna()
        & df["total_mv"].notna()
        & df["amt20"].notna()
        & df["turn20"].notna()
        & df["vol60"].notna()
    )

    df.sort_values(["dt", "ts_code"], inplace=True)
    return df


def _is_up_limit(row: pd.Series, config: StrategyConfig) -> bool:
    return bool(pd.notna(row.get("up_limit")) and pd.notna(row.get("open")) and row["open"] >= row["up_limit"] * (1 - config.buy_limit_epsilon))


def _is_down_limit(row: pd.Series, config: StrategyConfig) -> bool:
    return bool(pd.notna(row.get("down_limit")) and pd.notna(row.get("open")) and row["open"] <= row["down_limit"] * (1 + config.sell_limit_epsilon))


def apply_trade_constraints(
    desired: dict[str, float],
    current: dict[str, float],
    execution_frame: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[dict[str, float], int, int]:
    """Apply open-price limit-up buy blocks and limit-down sell blocks."""

    fixed: dict[str, float] = {}
    blocked_buys = 0
    blocked_sells = 0
    quotes = execution_frame.set_index("ts_code") if "ts_code" in execution_frame.columns else execution_frame

    for code in set(desired) | set(current):
        current_weight = float(current.get(code, 0.0))
        desired_weight = float(desired.get(code, 0.0))
        if code not in quotes.index:
            continue
        row = quotes.loc[code]
        if desired_weight > current_weight and _is_up_limit(row, config):
            fixed[code] = current_weight
            blocked_buys += 1
        elif desired_weight < current_weight and _is_down_limit(row, config):
            fixed[code] = current_weight
            blocked_sells += 1

    free_desired = {code: weight for code, weight in desired.items() if code not in fixed}
    fixed_weight = sum(fixed.values())
    free_weight = sum(free_desired.values())
    available = max(0.0, 1.0 - fixed_weight)
    scale = min(1.0, available / free_weight) if free_weight > 0 else 0.0

    weights = {code: weight for code, weight in fixed.items() if weight > 1e-12}
    weights.update({code: weight * scale for code, weight in free_desired.items() if weight * scale > 1e-12})
    return weights, blocked_buys, blocked_sells


def select_portfolio(
    signal_frame: pd.DataFrame,
    execution_frame: pd.DataFrame,
    current: dict[str, float],
    config: StrategyConfig,
) -> SelectionResult:
    mask = (
        signal_frame["base_ok"]
        & (signal_frame["amt20"] >= config.min_amt20)
        & (signal_frame["ret20"] > config.min_ret20)
        & (signal_frame["ret60"] > config.min_ret60)
        & (signal_frame["vol60"] < config.max_vol60)
    )
    universe = signal_frame.loc[mask].copy()
    if universe.empty:
        return SelectionResult(weights={}, selected=universe, blocked_buys=0, blocked_sells=0)

    universe["score"] = (
        config.size_score_weight * universe["total_mv"].rank(pct=True)
        + config.turn_score_weight * universe["turn20"].rank(pct=True)
        + config.vol_score_weight * universe["vol60"].rank(pct=True)
        + config.ret20_score_weight * universe["ret20"].rank(pct=True)
    )
    selected = universe.sort_values(["score", "total_mv", "ts_code"]).head(config.holdings_count).copy()
    if selected.empty:
        return SelectionResult(weights={}, selected=selected, blocked_buys=0, blocked_sells=0)

    desired_weight = 1.0 / len(selected)
    desired = dict(zip(selected["ts_code"], [desired_weight] * len(selected), strict=False))
    weights, blocked_buys, blocked_sells = apply_trade_constraints(desired, current, execution_frame, config)
    selected["target_weight"] = selected["ts_code"].map(weights).fillna(0.0)
    selected = selected[selected["target_weight"] > 0].copy()
    return SelectionResult(weights=weights, selected=selected, blocked_buys=blocked_buys, blocked_sells=blocked_sells)


def _annualize_return(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).prod()
    return float(equity ** (252.0 / len(returns)) - 1.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1.0).min())


def _compute_annual_returns(daily: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(daily["date"].astype(str), format="%Y%m%d")
    annual = (
        daily.assign(year=dates.dt.year)
        .groupby("year", as_index=False)
        .agg(strategy_return=("return", lambda x: float((1.0 + x).prod() - 1.0)),
             benchmark_return=("bench", lambda x: float((1.0 + x.fillna(0.0)).prod() - 1.0)))
    )
    return annual


def summarize(daily: pd.DataFrame, annual_returns: pd.DataFrame, config: StrategyConfig) -> dict[str, float | int | str | None]:
    returns = daily["return"].astype(float)
    equity = (1.0 + returns).cumprod()
    bench = daily["bench"].fillna(0.0).astype(float)
    bench_equity = (1.0 + bench).cumprod()
    vol = float(returns.std() * math.sqrt(252))
    ann = _annualize_return(returns)
    bench_ann = _annualize_return(bench)
    summary: dict[str, float | int | str | None] = {
        "strategy": "smallcap_lowturn",
        "start_date": int(daily["date"].iloc[0]),
        "end_date": int(daily["date"].iloc[-1]),
        "days": int(len(daily)),
        "annualized_return": ann,
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "annualized_vol": vol,
        "sharpe": float(ann / vol) if vol > 0 else None,
        "max_drawdown": _max_drawdown(equity),
        "benchmark_code": config.benchmark_code,
        "benchmark_annualized_return": bench_ann,
        "benchmark_cumulative_return": float(bench_equity.iloc[-1] - 1.0),
        "benchmark_max_drawdown": _max_drawdown(bench_equity),
        "average_turnover": float(daily["turnover"].mean()),
        "average_holdings": float(daily["holdings"].mean()),
        "average_cash_weight": float(daily["cash_weight"].mean()),
        "total_cost": float(daily["cost"].sum()),
        "rebalance_interval": int(config.rebalance_interval),
        "holdings_count": int(config.holdings_count),
        "cost_bps": float(config.cost_bps),
        "min_amt20": float(config.min_amt20),
    }
    if not annual_returns.empty:
        summary["worst_calendar_year"] = int(annual_returns.loc[annual_returns["strategy_return"].idxmin(), "year"])
        summary["worst_calendar_year_return"] = float(annual_returns["strategy_return"].min())
    return summary


def run_backtest(raw: pd.DataFrame, benchmark: pd.DataFrame, config: StrategyConfig) -> BacktestResult:
    panel = prepare_features(raw, config)
    by_date = {int(dt): frame for dt, frame in panel.groupby("dt", sort=True)}
    trade_dates = sorted(by_date)
    if not trade_dates:
        raise ValueError("No trade dates found in prepared feature panel")
    start_idx = next((idx for idx, dt in enumerate(trade_dates) if dt >= config.test_start_date), None)
    if start_idx is None or start_idx == 0:
        raise ValueError(f"No usable test start date >= {config.test_start_date}")

    ret_by_date = {
        int(dt): frame.set_index("ts_code")["open_ret_fwd"]
        for dt, frame in panel[["dt", "ts_code", "open_ret_fwd"]].groupby("dt", sort=True)
    }
    quote_by_date = {
        int(dt): frame.set_index("ts_code")[["open", "up_limit", "down_limit"]]
        for dt, frame in panel[["dt", "ts_code", "open", "up_limit", "down_limit"]].groupby("dt", sort=True)
    }
    bench_by_date = benchmark.set_index("dt")["bench"].to_dict()

    current: dict[str, float] = {}
    daily_rows: list[dict[str, float | int]] = []
    holding_rows: list[pd.DataFrame] = []
    cost_rate = config.cost_bps / 10_000.0
    equity = 1.0

    for idx in range(start_idx, len(trade_dates) - 1):
        dt = trade_dates[idx]
        prev_dt = trade_dates[idx - 1]
        turnover = 0.0
        cost = 0.0
        blocked_buys = 0
        blocked_sells = 0

        if (idx - start_idx) % config.rebalance_interval == 0 or not current:
            selection = select_portfolio(by_date[prev_dt], quote_by_date[dt], current, config)
            target = selection.weights
            keys = set(current) | set(target)
            turnover = sum(abs(target.get(code, 0.0) - current.get(code, 0.0)) for code in keys)
            cost = turnover * cost_rate
            current = target
            blocked_buys = selection.blocked_buys
            blocked_sells = selection.blocked_sells
            if not selection.selected.empty:
                hold = selection.selected[
                    [
                        "ts_code",
                        "name",
                        "target_weight",
                        "score",
                        "total_mv",
                        "circ_mv",
                        "amt20",
                        "turn20",
                        "vol60",
                        "ret20",
                        "ret60",
                    ]
                ].copy()
                hold.insert(0, "signal_date", prev_dt)
                hold.insert(0, "rebalance_date", dt)
                holding_rows.append(hold)

        ret_series = ret_by_date[dt]
        gross_return = 0.0
        next_weights: dict[str, float] = {}
        for code, weight in current.items():
            stock_ret = ret_series.get(code, np.nan)
            if pd.isna(stock_ret):
                stock_ret = 0.0
            gross_return += weight * float(stock_ret)
            next_weights[code] = weight * (1.0 + float(stock_ret))

        net_return = gross_return - cost
        equity *= 1.0 + net_return
        denominator = max(1e-12, 1.0 + net_return)
        current = {code: weight / denominator for code, weight in next_weights.items() if weight > 1e-12}
        cash_weight = max(0.0, 1.0 - sum(current.values()))

        daily_rows.append(
            {
                "date": dt,
                "signal_date": prev_dt,
                "return": net_return,
                "gross_return": gross_return,
                "bench": float(bench_by_date.get(dt, 0.0) or 0.0),
                "cost": cost,
                "turnover": turnover,
                "holdings": len(current),
                "cash_weight": cash_weight,
                "blocked_buys": blocked_buys,
                "blocked_sells": blocked_sells,
                "equity": equity,
            }
        )

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise ValueError("Backtest produced no daily rows")
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    daily["bench_equity"] = (1.0 + daily["bench"].fillna(0.0)).cumprod()
    daily["bench_drawdown"] = daily["bench_equity"] / daily["bench_equity"].cummax() - 1.0

    holdings = pd.concat(holding_rows, ignore_index=True) if holding_rows else pd.DataFrame()
    annual_returns = _compute_annual_returns(daily)
    summary = summarize(daily, annual_returns, config)
    summary.update({f"config_{key}": value for key, value in asdict(config).items() if isinstance(value, (int, float, str, type(None)))})
    return BacktestResult(config=config, daily=daily, holdings=holdings, annual_returns=annual_returns, summary=summary)
