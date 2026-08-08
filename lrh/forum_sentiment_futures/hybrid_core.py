from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendCoreConfig:
    trend_horizons: tuple[tuple[int, int], ...] = ((20, 5), (60, 10), (120, 20))
    minimum_positive_horizons: int = 2
    volatility_days: int = 20
    target_volatility: float = 0.10
    max_position_size: float = 0.50
    rebalance_interval_days: int = 5
    position_step: float = 0.10
    commission_bps_per_side: float = 1.0

    @property
    def cost_rate_per_side(self) -> float:
        return self.commission_bps_per_side / 10000.0

    def to_dict(self) -> dict:
        return asdict(self)


def _rounded_position(values: pd.Series, step: float) -> pd.Series:
    if step <= 0:
        raise ValueError("position_step must be positive")
    return (values / step).round() * step


def run_trend_core(
    continuous_futures: pd.DataFrame,
    config: TrendCoreConfig,
) -> pd.DataFrame:
    """Build a long/flat, volatility-targeted core from prior-close information.

    The input return is the same-contract close-to-close gross return from the
    continuous-futures benchmark. A row's position uses indicators shifted by
    one trading day, so no current-day return enters the signal.
    """

    required = {
        "fut_code",
        "trade_date",
        "target_contract",
        "strategy_return_gross",
    }
    missing = required.difference(continuous_futures.columns)
    if missing:
        raise ValueError(f"Continuous-futures data missing columns: {sorted(missing)}")
    if config.rebalance_interval_days <= 0:
        raise ValueError("rebalance_interval_days must be positive")
    if not 0 < config.max_position_size <= 1:
        raise ValueError("max_position_size must be in (0, 1]")
    if not config.trend_horizons:
        raise ValueError("trend_horizons must not be empty")
    if not 1 <= config.minimum_positive_horizons <= len(config.trend_horizons):
        raise ValueError("minimum_positive_horizons must be valid for trend_horizons")

    source = continuous_futures.copy()
    source["trade_date"] = pd.to_datetime(source["trade_date"]).dt.normalize()
    source = source.sort_values(["fut_code", "trade_date"]).reset_index(drop=True)
    frames: list[pd.DataFrame] = []

    for fut_code, code_frame in source.groupby("fut_code", sort=False):
        frame = code_frame.copy().reset_index(drop=True)
        market_return = pd.to_numeric(
            frame["strategy_return_gross"], errors="coerce"
        ).fillna(0.0)
        synthetic_price = (1.0 + market_return).cumprod()

        known_price = synthetic_price.shift(1)
        horizon_signals: list[pd.Series] = []
        horizon_features: dict[str, pd.Series] = {}
        for moving_average_days, momentum_days in config.trend_horizons:
            moving_average = known_price.rolling(
                moving_average_days,
                min_periods=moving_average_days,
            ).mean()
            momentum = known_price / known_price.shift(momentum_days) - 1.0
            signal = known_price.gt(moving_average) & momentum.gt(0.0)
            horizon_signals.append(signal.fillna(False))
            horizon_features[f"core_ma_{moving_average_days}d"] = moving_average
            horizon_features[f"core_momentum_{momentum_days}d"] = momentum
        positive_horizons = pd.concat(horizon_signals, axis=1).sum(axis=1)
        realized_volatility = (
            market_return.shift(1)
            .rolling(
                config.volatility_days,
                min_periods=config.volatility_days,
            )
            .std(ddof=0)
            * np.sqrt(252.0)
        )
        trend_is_positive = positive_horizons.ge(config.minimum_positive_horizons)
        risk_size = (config.target_volatility / realized_volatility).clip(
            lower=0.0,
            upper=config.max_position_size,
        )
        desired_position = pd.Series(
            np.where(trend_is_positive, risk_size, 0.0),
            index=frame.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], np.nan)
        desired_position = _rounded_position(
            desired_position.fillna(0.0), config.position_step
        ).clip(lower=0.0, upper=config.max_position_size)

        rebalance = pd.Series(
            np.arange(len(frame)) % config.rebalance_interval_days == 0,
            index=frame.index,
        )
        position = desired_position.where(rebalance).ffill().fillna(0.0)
        previous_position = position.shift(1).fillna(0.0)
        contract_roll = frame["target_contract"].ne(frame["target_contract"].shift(1))
        turnover = (position - previous_position).abs()
        turnover.loc[contract_roll] = (
            position.loc[contract_roll].abs() + previous_position.loc[contract_roll].abs()
        )
        transaction_cost = turnover * config.cost_rate_per_side
        gross_return = position * market_return
        net_return = gross_return - transaction_cost

        frame["core_signal_price"] = known_price
        for column, values in horizon_features.items():
            frame[column] = values
        frame["core_positive_horizons"] = positive_horizons
        frame["core_realized_volatility"] = realized_volatility
        frame["core_trend_positive"] = trend_is_positive.fillna(False)
        frame["core_rebalance"] = rebalance
        frame["core_desired_position"] = desired_position
        frame["core_position"] = position
        frame["core_contract_roll"] = contract_roll
        frame["core_turnover"] = turnover
        frame["core_transaction_cost"] = transaction_cost
        frame["core_return_gross"] = gross_return
        frame["core_return_net"] = net_return
        frame["core_trade_count"] = turnover.gt(0.0).astype(int)
        frame["core_nav"] = (1.0 + net_return).cumprod()
        frame["fut_code"] = fut_code
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def combine_core_and_intraday(daily: pd.DataFrame) -> pd.DataFrame:
    required = {
        "core_return_gross",
        "core_transaction_cost",
        "strategy_return_gross",
        "transaction_cost",
        "core_trade_count",
        "trade_count",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Hybrid daily data missing columns: {sorted(missing)}")

    result = daily.copy()
    result["intraday_overlay_return_gross"] = result["strategy_return_gross"]
    result["intraday_overlay_transaction_cost"] = result["transaction_cost"]
    result["intraday_overlay_return_net"] = result["strategy_return_net"]
    result["hybrid_return_gross"] = (
        result["core_return_gross"] + result["intraday_overlay_return_gross"]
    )
    result["hybrid_transaction_cost"] = (
        result["core_transaction_cost"] + result["intraday_overlay_transaction_cost"]
    )
    result["hybrid_return_net"] = (
        result["hybrid_return_gross"] - result["hybrid_transaction_cost"]
    )
    result["hybrid_trade_count"] = (
        result["core_trade_count"].fillna(0).astype(int)
        + result["trade_count"].fillna(0).astype(int)
    )
    result["hybrid_turnover"] = result["core_turnover"] + result["turnover"]
    result["hybrid_nav"] = result.groupby("fut_code")["hybrid_return_net"].transform(
        lambda values: (1.0 + values).cumprod()
    )
    return result
