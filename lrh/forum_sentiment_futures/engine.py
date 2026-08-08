from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategySpec


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps_per_side: float = 1.0
    signal_lag_days: int = 1

    @property
    def cost_rate_per_side(self) -> float:
        return self.cost_bps_per_side / 10000.0


def select_dominant_contract(snapshot: pd.DataFrame) -> str:
    eligible_mask = snapshot["trade_days_to_expiry"] > 1
    if "calendar_days_to_expiry" in snapshot.columns:
        eligible_mask = eligible_mask | (snapshot["calendar_days_to_expiry"] > 2)
    eligible = snapshot[eligible_mask].copy()
    if eligible.empty:
        raise ValueError("No eligible futures contract with more than one trading day to expiry")
    eligible["liquidity"] = eligible["amount"].fillna(0.0) + eligible["vol"].fillna(0.0)
    eligible = eligible.sort_values(["liquidity", "oi", "delist_date"], ascending=[False, False, True])
    return str(eligible.iloc[0]["ts_code"])


def _finite(row: pd.Series, column: str) -> float | None:
    value = row.get(column, np.nan)
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def _long_condition(row: pd.Series, spec: StrategySpec) -> bool:
    panic_z = _finite(row, "panic_z")
    activity = _finite(row, "activity_score")
    rank_improvement = _finite(row, "rank_improvement_z")
    prev_activity = _finite(row, "prev_activity_score")
    forum_panic = _finite(row, "forum_panic_z")
    forum_excitement = _finite(row, "forum_excitement_z")
    if panic_z is None or activity is None:
        return False
    fading = False
    if prev_activity is not None and rank_improvement is not None:
        fading = prev_activity > spec.excited_entry_z and rank_improvement < spec.fade_z
    if forum_panic is not None and forum_excitement is not None:
        forum_confirmation = forum_panic >= spec.forum_panic_entry_z or activity <= spec.low_activity_z
        return (
            panic_z >= spec.panic_entry_z
            and forum_confirmation
            and forum_excitement <= spec.max_forum_excitement_z
            and not fading
        )
    return panic_z >= spec.panic_entry_z and activity <= spec.low_activity_z and not fading


def _short_condition(row: pd.Series, spec: StrategySpec) -> bool:
    panic_z = _finite(row, "panic_z")
    activity = _finite(row, "activity_score")
    momentum = _finite(row, spec.momentum_column)
    forum_excitement = _finite(row, "forum_excitement_z")
    if panic_z is None or activity is None or momentum is None:
        return False
    if forum_excitement is not None:
        return (
            activity >= 0.0
            and forum_excitement >= spec.excited_entry_z
            and abs(panic_z) <= spec.calm_abs_panic_z
            and momentum <= 0.0
        )
    return activity >= spec.excited_entry_z and abs(panic_z) <= spec.calm_abs_panic_z and momentum <= 0.0


def select_position(spec: StrategySpec, signal_row: pd.Series | None) -> int:
    if spec.kind in {"always_long", "always_hold"}:
        return 1
    if signal_row is None or not bool(signal_row.get("has_sentiment", False)):
        return 0
    if spec.kind == "panic_long":
        return 1 if _long_condition(signal_row, spec) else 0
    if spec.kind == "excitement_short":
        return -1 if _short_condition(signal_row, spec) else 0
    if spec.kind == "dual":
        if _long_condition(signal_row, spec):
            return 1
        if _short_condition(signal_row, spec):
            return -1
        return 0
    raise ValueError(f"Unsupported strategy kind: {spec.kind}")


def run_signal_backtest(
    market: pd.DataFrame,
    signal_features: pd.DataFrame,
    strategy: StrategySpec,
    config: BacktestConfig,
) -> pd.DataFrame:
    if config.signal_lag_days != 1:
        raise ValueError("Only one-trading-day signal lag is supported")
    if market.empty:
        raise ValueError("Market data is empty")

    prepared = market.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"])
    prepared = prepared.sort_values(["trade_date", "delist_date", "ts_code"])
    snapshots = {
        trade_date: frame.set_index("ts_code", drop=False).copy()
        for trade_date, frame in prepared.groupby("trade_date", sort=True)
    }
    trade_dates = sorted(snapshots)

    features = signal_features.copy()
    features["trade_date"] = pd.to_datetime(features["trade_date"])
    feature_rows = {row.trade_date: row for row in features.itertuples(index=False)}

    nav = 1.0
    benchmark_nav = 1.0
    rows: list[dict] = []
    previous_target_contract: str | None = None

    for idx in range(config.signal_lag_days, len(trade_dates)):
        signal_date = trade_dates[idx - config.signal_lag_days]
        trade_date = trade_dates[idx]
        signal_snapshot = snapshots[signal_date]
        exec_snapshot = snapshots[trade_date]
        signal_row_tuple = feature_rows.get(signal_date)
        signal_row = pd.Series(signal_row_tuple._asdict()) if signal_row_tuple is not None else None
        position = select_position(strategy, signal_row)

        target_contract = None
        contract_open = np.nan
        contract_close = np.nan
        strategy_return_gross = 0.0
        strategy_return_net = 0.0
        transaction_cost = 0.0
        turnover = 0.0

        if position != 0:
            try:
                target_contract = select_dominant_contract(signal_snapshot)
            except ValueError:
                target_contract = select_dominant_contract(exec_snapshot)
            if target_contract not in exec_snapshot.index:
                target_contract = select_dominant_contract(exec_snapshot)

            exec_row = exec_snapshot.loc[target_contract]
            contract_open = float(exec_row["open"])
            contract_close = float(exec_row["close"])
            if strategy.kind == "always_hold":
                previous_close = float(exec_row["prev_close"])
                if not np.isfinite(previous_close) or previous_close <= 0:
                    raise ValueError(f"Missing previous close for continuous holding: {target_contract}")
                contract_open = previous_close
                strategy_return_gross = float(contract_close / previous_close - 1.0)
                turnover = 1.0 if previous_target_contract is None else float(
                    2.0 if target_contract != previous_target_contract else 0.0
                )
                previous_target_contract = target_contract
            else:
                futures_intraday_return = contract_close / contract_open - 1.0
                strategy_return_gross = float(position * futures_intraday_return)
                turnover = float(abs(position) * 2.0)
            transaction_cost = float(turnover * config.cost_rate_per_side)
            strategy_return_net = strategy_return_gross - transaction_cost
        else:
            exec_row = exec_snapshot.iloc[0]

        benchmark_return = float(exec_row.get("spot_return_1d", exec_row["spot_intraday_return"]))
        nav *= 1.0 + strategy_return_net
        benchmark_nav *= 1.0 + benchmark_return

        feature_values = signal_row.to_dict() if signal_row is not None else {}
        rows.append(
            {
                "fut_code": str(exec_row["fut_code"]),
                "signal_date": signal_date,
                "trade_date": trade_date,
                "strategy": strategy.name,
                "strategy_name": strategy.display_name,
                "position": position,
                "target_contract": target_contract,
                "contract_open": contract_open,
                "contract_close": contract_close,
                "strategy_return_gross": strategy_return_gross,
                "transaction_cost": transaction_cost,
                "strategy_return_net": strategy_return_net,
                "benchmark_return": benchmark_return,
                "excess_return_net": strategy_return_net - benchmark_return,
                "turnover": turnover,
                "nav": nav,
                "benchmark_nav": benchmark_nav,
                "relative_nav": nav / benchmark_nav if benchmark_nav else np.nan,
                "signal_activity_score": feature_values.get("activity_score", np.nan),
                "signal_panic_z": feature_values.get("panic_z", np.nan),
                "signal_rank_improvement_z": feature_values.get("rank_improvement_z", np.nan),
                "signal_new_fan_z": feature_values.get("new_fan_z", np.nan),
                "signal_forum_panic_z": feature_values.get("forum_panic_z", np.nan),
                "signal_forum_excitement_z": feature_values.get("forum_excitement_z", np.nan),
                "signal_post_count": feature_values.get("post_count", np.nan),
                "signal_emoji_count": feature_values.get("emoji_count", np.nan),
                "signal_spot_return_3d": feature_values.get("spot_return_3d", np.nan),
                "signal_fading_excitement": bool(feature_values.get("fading_excitement", False)),
                "sentiment_date": feature_values.get("sentiment_date", pd.NaT),
            }
        )

    return pd.DataFrame(rows)
