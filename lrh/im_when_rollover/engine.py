from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategySpec
from .strategies import select_target_contract


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps_per_side: float = 1.0
    signal_lag_days: int = 1
    execution_price: str = "open"

    @property
    def cost_rate_per_side(self) -> float:
        return self.cost_bps_per_side / 10000.0


def _safe_return(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator - 1.0)


def run_strategy_backtest(
    market: pd.DataFrame,
    strategy: StrategySpec,
    config: BacktestConfig,
) -> pd.DataFrame:
    if market.empty:
        raise ValueError("Market data is empty")
    if config.execution_price != "open":
        raise ValueError(f"Unsupported execution price: {config.execution_price}")
    if config.signal_lag_days != 1:
        raise ValueError("Only T-1 signal lag is supported in this backtest engine")

    prepared = market.copy()
    prepared.sort_values(["trade_date", "delist_date", "ts_code"], inplace=True)
    snapshots = {
        trade_date: frame.set_index("ts_code", drop=False).copy()
        for trade_date, frame in prepared.groupby("trade_date", sort=True)
    }
    trade_dates = sorted(snapshots)
    if len(trade_dates) <= config.signal_lag_days:
        return pd.DataFrame()

    current_contract: str | None = None
    nav = 1.0
    benchmark_nav = 1.0
    rows: list[dict] = []

    for idx in range(config.signal_lag_days, len(trade_dates)):
        signal_date = trade_dates[idx - config.signal_lag_days]
        trade_date = trade_dates[idx]
        signal_snapshot = snapshots[signal_date]
        exec_snapshot = snapshots[trade_date]

        try:
            target_contract = select_target_contract(strategy, signal_snapshot, current_contract)
        except ValueError:
            if current_contract is None:
                continue
            if current_contract not in exec_snapshot.index:
                break
            target_contract = current_contract

        if target_contract not in exec_snapshot.index:
            if current_contract is None:
                continue
            if current_contract not in exec_snapshot.index:
                break
            target_contract = current_contract

        target_row = exec_snapshot.loc[target_contract]
        signal_row = signal_snapshot.loc[target_contract] if target_contract in signal_snapshot.index else None

        rolled = False
        turnover = 0.0
        strategy_return_gross = 0.0
        strategy_return_net = 0.0
        transaction_cost = 0.0
        contract_before_trade = current_contract
        cost_sides = 0
        lifecycle_stage = "hold"

        if current_contract is None:
            lifecycle_stage = "enter"
            turnover = 1.0
            cost_sides = 1
            entry_open = float(target_row["open"])
            close_price = float(target_row["close"])
            gross_factor = close_price / entry_open
            net_factor = gross_factor * (1.0 - config.cost_rate_per_side)
            strategy_return_gross = float(gross_factor - 1.0)
            strategy_return_net = float(net_factor - 1.0)
            transaction_cost = float(strategy_return_gross - strategy_return_net)
            current_contract = target_contract
        else:
            if current_contract not in exec_snapshot.index:
                raise KeyError(f"{strategy.name} cannot find current contract {current_contract} on {trade_date:%Y-%m-%d}")

            held_row = exec_snapshot.loc[current_contract]
            prev_close = float(held_row["prev_close"]) if pd.notna(held_row["prev_close"]) else np.nan

            if target_contract == current_contract:
                lifecycle_stage = "hold"
                strategy_return_gross = _safe_return(float(held_row["close"]), prev_close)
                strategy_return_net = strategy_return_gross
            else:
                lifecycle_stage = "roll"
                rolled = True
                turnover = 2.0
                cost_sides = 2
                old_open = float(held_row["open"])
                new_open = float(target_row["open"])
                new_close = float(target_row["close"])
                gross_factor = old_open / prev_close * new_close / new_open
                net_factor = gross_factor * (1.0 - config.cost_rate_per_side) ** 2
                strategy_return_gross = float(gross_factor - 1.0)
                strategy_return_net = float(net_factor - 1.0)
                transaction_cost = float(strategy_return_gross - strategy_return_net)
                current_contract = target_contract

        if pd.isna(strategy_return_gross) or pd.isna(strategy_return_net):
            continue

        exec_row = exec_snapshot.loc[current_contract]
        spot_prev_close = float(exec_row["spot_close"] if idx == 0 else snapshots[trade_dates[idx - 1]].iloc[0]["spot_close"])
        spot_open = float(exec_row["spot_open"])
        spot_close = float(exec_row["spot_close"])
        if lifecycle_stage == "enter":
            benchmark_return = _safe_return(spot_close, spot_open)
        else:
            benchmark_return = _safe_return(spot_close, spot_prev_close)
        if pd.isna(benchmark_return):
            benchmark_return = 0.0

        nav *= 1.0 + strategy_return_net
        benchmark_nav *= 1.0 + benchmark_return
        excess_return_net = np.nan
        excess_return_gross = np.nan
        if 1.0 + benchmark_return != 0:
            excess_return_net = (1.0 + strategy_return_net) / (1.0 + benchmark_return) - 1.0
            excess_return_gross = (1.0 + strategy_return_gross) / (1.0 + benchmark_return) - 1.0

        held_signal_row = signal_snapshot.loc[contract_before_trade] if contract_before_trade in signal_snapshot.index else None

        rows.append(
            {
                "signal_date": signal_date,
                "trade_date": trade_date,
                "strategy": strategy.name,
                "strategy_name": strategy.display_name,
                "signal_lag_days": config.signal_lag_days,
                "execution_price": config.execution_price,
                "contract_before_trade": contract_before_trade,
                "signal_target_contract": target_contract,
                "holding_contract": current_contract,
                "rolled": rolled,
                "lifecycle_stage": lifecycle_stage,
                "turnover": turnover,
                "cost_sides": cost_sides,
                "transaction_cost": transaction_cost,
                "strategy_return_gross": strategy_return_gross,
                "strategy_return_net": strategy_return_net,
                "benchmark_return": benchmark_return,
                "excess_return_net": excess_return_net,
                "excess_return_gross": excess_return_gross,
                "nav": nav,
                "benchmark_nav": benchmark_nav,
                "relative_nav": nav / benchmark_nav if benchmark_nav else np.nan,
                "holding_open": float(exec_row["open"]),
                "holding_close": float(exec_row["close"]),
                "holding_prev_close": float(exec_row["prev_close"]) if pd.notna(exec_row["prev_close"]) else np.nan,
                "holding_vwap": float(exec_row["vwap"]),
                "holding_trade_days_to_expiry": int(exec_row["trade_days_to_expiry"]),
                "holding_annualized_carry": float(exec_row["annualized_carry"]),
                "signal_holding_trade_days_to_expiry": (
                    int(signal_row["trade_days_to_expiry"]) if signal_row is not None else np.nan
                ),
                "signal_holding_annualized_carry": (
                    float(signal_row["annualized_carry"]) if signal_row is not None else np.nan
                ),
                "old_contract_open": (
                    float(exec_snapshot.loc[contract_before_trade, "open"])
                    if contract_before_trade in exec_snapshot.index
                    else np.nan
                ),
                "old_contract_close": (
                    float(exec_snapshot.loc[contract_before_trade, "close"])
                    if contract_before_trade in exec_snapshot.index
                    else np.nan
                ),
                "old_contract_prev_close": (
                    float(exec_snapshot.loc[contract_before_trade, "prev_close"])
                    if contract_before_trade in exec_snapshot.index and pd.notna(exec_snapshot.loc[contract_before_trade, "prev_close"])
                    else np.nan
                ),
                "old_contract_signal_days_to_expiry": (
                    int(held_signal_row["trade_days_to_expiry"]) if held_signal_row is not None else np.nan
                ),
                "new_contract_open": float(target_row["open"]),
                "new_contract_close": float(target_row["close"]),
                "spot_open": spot_open,
                "spot_close": spot_close,
                "spot_prev_close": spot_prev_close,
            }
        )

    return pd.DataFrame(rows)
