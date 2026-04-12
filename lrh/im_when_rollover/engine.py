from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategySpec
from .strategies import select_target_contract


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps_per_side: float = 1.0

    @property
    def cost_rate_per_side(self) -> float:
        return self.cost_bps_per_side / 10000.0


def run_strategy_backtest(
    market: pd.DataFrame,
    strategy: StrategySpec,
    config: BacktestConfig,
) -> pd.DataFrame:
    if market.empty:
        raise ValueError("Market data is empty")

    prepared = market.copy()
    prepared.sort_values(["trade_date", "delist_date", "ts_code"], inplace=True)
    snapshots = {
        trade_date: frame.set_index("ts_code", drop=False).copy()
        for trade_date, frame in prepared.groupby("trade_date", sort=True)
    }
    benchmark_close = prepared.groupby("trade_date")["spot_close"].first().sort_index()
    benchmark_return = benchmark_close.pct_change().fillna(0.0)

    current_contract: str | None = None
    nav = 1.0
    benchmark_nav = 1.0
    rows: list[dict] = []

    for trade_date, snapshot in snapshots.items():
        bench_ret = float(benchmark_return.loc[trade_date])
        try:
            target_contract = select_target_contract(strategy, snapshot, current_contract)
        except ValueError:
            if current_contract is None:
                continue
            if current_contract not in snapshot.index:
                break
            target_contract = current_contract
        target_row = snapshot.loc[target_contract]

        rolled = False
        turnover = 0.0
        strategy_ret = 0.0
        contract_before_trade = current_contract

        if current_contract is None:
            current_contract = target_contract
        else:
            if current_contract not in snapshot.index:
                raise KeyError(f"{strategy.name} cannot find current contract {current_contract} on {trade_date:%Y-%m-%d}")

            held_row = snapshot.loc[current_contract]
            prev_close = held_row["prev_close"]
            if pd.isna(prev_close) or prev_close <= 0:
                strategy_ret = 0.0
            elif target_contract == current_contract:
                strategy_ret = float(held_row["close"] / prev_close - 1.0)
            else:
                rolled = True
                turnover = 2.0
                sell_price = float(held_row["vwap"])
                buy_price = float(target_row["vwap"])
                gross_factor = sell_price / float(prev_close) * float(target_row["close"]) / buy_price
                net_factor = gross_factor * (1.0 - config.cost_rate_per_side) ** 2
                strategy_ret = float(net_factor - 1.0)
                current_contract = target_contract

        nav *= 1.0 + strategy_ret
        benchmark_nav *= 1.0 + bench_ret
        excess_ret = np.nan
        if 1.0 + bench_ret != 0:
            excess_ret = (1.0 + strategy_ret) / (1.0 + bench_ret) - 1.0

        rows.append(
            {
                "trade_date": trade_date,
                "strategy": strategy.name,
                "strategy_name": strategy.display_name,
                "contract_before_trade": contract_before_trade,
                "holding_contract": current_contract,
                "target_contract": target_contract,
                "rolled": rolled,
                "turnover": turnover,
                "strategy_return": strategy_ret,
                "benchmark_return": bench_ret,
                "excess_return": excess_ret,
                "nav": nav,
                "benchmark_nav": benchmark_nav,
                "relative_nav": nav / benchmark_nav if benchmark_nav else np.nan,
                "holding_close": float(snapshot.loc[current_contract, "close"]),
                "holding_vwap": float(snapshot.loc[current_contract, "vwap"]),
                "holding_trade_days_to_expiry": int(snapshot.loc[current_contract, "trade_days_to_expiry"]),
                "holding_annualized_carry": float(snapshot.loc[current_contract, "annualized_carry"]),
                "spot_close": float(snapshot.loc[current_contract, "spot_close"]),
                "old_contract_trade_days_to_expiry": (
                    int(snapshot.loc[contract_before_trade, "trade_days_to_expiry"])
                    if contract_before_trade in snapshot.index
                    else np.nan
                ),
                "old_contract_vwap": float(snapshot.loc[contract_before_trade, "vwap"])
                if contract_before_trade in snapshot.index
                else np.nan,
                "new_contract_vwap": float(target_row["vwap"]),
                "new_contract_close": float(target_row["close"]),
            }
        )

    return pd.DataFrame(rows)
