from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import FactorStudyConfig


@dataclass(frozen=True)
class EvaluationResult:
    daily_ic: pd.DataFrame
    factor_metrics: pd.DataFrame
    correlation_summary: pd.DataFrame


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    return float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    vol = returns.std() * math.sqrt(252)
    ann = annualized_return(returns)
    return float(ann / vol) if vol and not np.isnan(vol) else float("nan")


def _portfolio_turnover(previous: set[str], current: set[str], count: int) -> float:
    if not current:
        return 0.0
    if not previous:
        return 1.0
    common = len(previous & current)
    return max(0.0, 2.0 * (1.0 - common / max(1, count)))


def compute_daily_ic(panel: pd.DataFrame, factor_columns: list[str], config: FactorStudyConfig) -> pd.DataFrame:
    rows: list[pd.Series] = []
    work = panel[(panel["base_ok"]) & (panel["dt"] >= config.test_start_date)].copy()
    for dt, group in work.groupby("dt", sort=True):
        if len(group) < config.min_ic_obs:
            continue
        y = group["exec_ret_fwd"].rank(method="average")
        ranks = group[factor_columns].rank(method="average")
        ic = ranks.corrwith(y, method="pearson")
        ic.name = int(dt)
        rows.append(ic)
    if not rows:
        raise ValueError("No daily IC rows were produced")
    daily_ic = pd.DataFrame(rows)
    daily_ic.index.name = "dt"
    return daily_ic.sort_index()


def summarize_ic(daily_ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in daily_ic.columns:
        s = daily_ic[factor].dropna()
        if s.empty:
            continue
        std = s.std()
        t_stat = float(s.mean() / (std / math.sqrt(len(s)))) if std > 0 else float("nan")
        p_value = float(2.0 * stats.t.sf(abs(t_stat), len(s) - 1)) if len(s) > 1 and not np.isnan(t_stat) else float("nan")
        rows.append(
            {
                "factor": factor,
                "ic_days": int(len(s)),
                "mean_ic": float(s.mean()),
                "abs_mean_ic": float(abs(s.mean())),
                "std_ic": float(std),
                "ic_ir": float(s.mean() / std * math.sqrt(252)) if std > 0 else float("nan"),
                "abs_ic_ir": float(abs(s.mean()) / std * math.sqrt(252)) if std > 0 else float("nan"),
                "positive_ic_rate": float((s > 0).mean()),
                "ic_t_stat": t_stat,
                "ic_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def _backtest_one_factor(panel: pd.DataFrame, factor: str, config: FactorStudyConfig) -> dict[str, float | int | str]:
    work = panel[panel["dt"] >= config.test_start_date].copy()
    signal = work[work["base_ok"] & work[factor].notna()]
    grouped_signal = {int(dt): g for dt, g in signal.groupby("dt", sort=True)}
    grouped_returns = {int(dt): g.set_index("ts_code")["exec_ret_fwd"] for dt, g in work.groupby("dt", sort=True)}
    dates = sorted(grouped_returns)
    rebal_dates = [dt for dt in dates if dt in grouped_signal and len(grouped_signal[dt]) >= config.min_strategy_obs]
    rebal_dates = rebal_dates[:: config.rebalance_interval]
    if not rebal_dates:
        return {"factor": factor}

    high_returns: list[float] = []
    low_returns: list[float] = []
    spread_returns: list[float] = []
    return_dates: list[int] = []
    high_prev: set[str] = set()
    low_prev: set[str] = set()
    cost_rate = config.cost_bps / 10_000.0
    date_to_idx = {dt: idx for idx, dt in enumerate(dates)}

    for reb_idx, reb_dt in enumerate(rebal_dates):
        group = grouped_signal[reb_dt].sort_values([factor, "ts_code"])
        low_codes = set(group.head(config.holdings_count)["ts_code"])
        high_codes = set(group.tail(config.holdings_count)["ts_code"])
        next_reb_dt = rebal_dates[reb_idx + 1] if reb_idx + 1 < len(rebal_dates) else None
        start = date_to_idx[reb_dt]
        end = date_to_idx[next_reb_dt] if next_reb_dt is not None else len(dates)
        high_turnover = _portfolio_turnover(high_prev, high_codes, config.holdings_count)
        low_turnover = _portfolio_turnover(low_prev, low_codes, config.holdings_count)
        high_prev = high_codes
        low_prev = low_codes
        for pos, dt in enumerate(dates[start:end]):
            returns = grouped_returns[dt]
            high_ret = returns.reindex(list(high_codes)).dropna().mean()
            low_ret = returns.reindex(list(low_codes)).dropna().mean()
            if pd.isna(high_ret) or pd.isna(low_ret):
                continue
            if pos == 0:
                high_ret -= high_turnover * cost_rate
                low_ret -= low_turnover * cost_rate
            high_returns.append(float(high_ret))
            low_returns.append(float(low_ret))
            spread_returns.append(float(high_ret - low_ret))
            return_dates.append(dt)

    high = pd.Series(high_returns, index=return_dates, dtype=float)
    low = pd.Series(low_returns, index=return_dates, dtype=float)
    spread = pd.Series(spread_returns, index=return_dates, dtype=float)
    if len(high) < config.min_strategy_obs:
        return {"factor": factor}
    return {
        "factor": factor,
        "strategy_days": int(len(high)),
        "high_ann": annualized_return(high),
        "low_ann": annualized_return(low),
        "spread_ann": annualized_return(spread),
        "high_max_dd": max_drawdown(high),
        "low_max_dd": max_drawdown(low),
        "spread_max_dd": max_drawdown(spread),
        "high_sharpe": sharpe(high),
        "low_sharpe": sharpe(low),
        "spread_sharpe": sharpe(spread),
    }


def backtest_factor_strategies(panel: pd.DataFrame, factor_columns: list[str], config: FactorStudyConfig) -> pd.DataFrame:
    rows = []
    for idx, factor in enumerate(factor_columns, start=1):
        if idx % 10 == 0 or idx == 1:
            print(f"backtesting factor {idx}/{len(factor_columns)}: {factor}", flush=True)
        rows.append(_backtest_one_factor(panel, factor, config))
    return pd.DataFrame(rows)


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    data = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return float("nan"), float("nan"), int(len(data))
    if method == "pearson":
        r, p = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])
    else:
        r, p = stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])
    return float(r), float(p), int(len(data))


def build_correlation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    tests = [
        ("mean_ic", "spread_ann", "signed IC vs high-minus-low strategy annualized return"),
        ("mean_ic", "high_ann", "signed IC vs high-factor long-only annualized return"),
        ("abs_mean_ic", "chosen_ann", "absolute IC vs IC-direction long-only annualized return"),
        ("abs_ic_ir", "chosen_ann", "absolute ICIR vs IC-direction long-only annualized return"),
        ("abs_mean_ic", "abs_spread_ann", "absolute IC vs absolute high-minus-low annualized return"),
    ]
    rows = []
    for x_col, y_col, description in tests:
        for method in ["pearson", "spearman"]:
            r, p, n = _safe_corr(metrics[x_col], metrics[y_col], method)
            rows.append({"x": x_col, "y": y_col, "method": method, "correlation": r, "p_value": p, "n": n, "description": description})
    return pd.DataFrame(rows)


def evaluate(panel: pd.DataFrame, factor_columns: list[str], config: FactorStudyConfig) -> EvaluationResult:
    daily_ic = compute_daily_ic(panel, factor_columns, config)
    ic_metrics = summarize_ic(daily_ic)
    strategy_metrics = backtest_factor_strategies(panel, factor_columns, config)
    metrics = ic_metrics.merge(strategy_metrics, on="factor", how="left")
    metrics["chosen_direction"] = np.where(metrics["mean_ic"] >= 0, "high", "low")
    metrics["chosen_ann"] = np.where(metrics["mean_ic"] >= 0, metrics["high_ann"], metrics["low_ann"])
    metrics["chosen_max_dd"] = np.where(metrics["mean_ic"] >= 0, metrics["high_max_dd"], metrics["low_max_dd"])
    metrics["chosen_sharpe"] = np.where(metrics["mean_ic"] >= 0, metrics["high_sharpe"], metrics["low_sharpe"])
    metrics["abs_spread_ann"] = metrics["spread_ann"].abs()
    corr = build_correlation_summary(metrics)
    return EvaluationResult(daily_ic=daily_ic, factor_metrics=metrics, correlation_summary=corr)

