from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class XueqiuSignalConfig:
    signal_start_slot: int = 1
    signal_end_slot: int = 12
    min_activity_z: float = 0.50
    min_author_breadth_z: float = -0.50
    panic_level_z: float = 0.90
    panic_velocity_z: float = 1.00
    euphoria_level_z: float = 0.90
    euphoria_velocity_z: float = 1.00
    min_market_move: float = 0.003
    long_stabilization_bar_return: float = -0.0005
    short_stall_bar_return: float = 0.0005
    max_realized_volatility: float = 0.35
    min_relative_activity_z: float = -0.75
    min_signal_score: float = 3.00

    def to_dict(self) -> dict:
        return asdict(self)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def apply_xueqiu_signal(
    features: pd.DataFrame,
    config: XueqiuSignalConfig | None = None,
) -> pd.DataFrame:
    selected = config or XueqiuSignalConfig()
    result = features.copy()
    activity = _numeric(result, "activity_z")
    breadth = _numeric(result, "xq_author_breadth_z")
    bearish = _numeric(result, "xq_bearish_z")
    bullish = _numeric(result, "xq_bullish_z")
    velocity = _numeric(result, "xq_sentiment_velocity_z")
    market_panic = _numeric(result, "market_panic_z")
    day_return = _numeric(result, "day_return")
    bar_return = _numeric(result, "bar_return")
    realized_vol = _numeric(result, "realized_vol_annualized")
    bar_slot = _numeric(result, "bar_slot")
    relative_activity = _numeric(result, "xq_relative_activity_z").fillna(0.0)
    quality = result.get(
        "xq_signal_quality_ok", pd.Series(False, index=result.index)
    ).fillna(False)
    can_signal = result.get(
        "can_signal", pd.Series(True, index=result.index)
    ).fillna(False)

    common = (
        quality
        & can_signal
        & bar_slot.between(selected.signal_start_slot, selected.signal_end_slot)
        & activity.ge(selected.min_activity_z)
        & breadth.ge(selected.min_author_breadth_z)
        & relative_activity.ge(selected.min_relative_activity_z)
        & realized_vol.le(selected.max_realized_volatility)
    )

    long_score = (
        bearish.clip(lower=0.0)
        + (-velocity).clip(lower=0.0)
        + market_panic.clip(lower=0.0)
        + activity.clip(lower=0.0)
    )
    short_score = (
        bullish.clip(lower=0.0)
        + velocity.clip(lower=0.0)
        + (-market_panic).clip(lower=0.0)
        + activity.clip(lower=0.0)
    )
    long_condition = (
        common
        & bearish.ge(selected.panic_level_z)
        & velocity.le(-selected.panic_velocity_z)
        & day_return.le(-selected.min_market_move)
        & bar_return.ge(selected.long_stabilization_bar_return)
        & long_score.ge(selected.min_signal_score)
    )
    short_condition = (
        common
        & bullish.ge(selected.euphoria_level_z)
        & velocity.ge(selected.euphoria_velocity_z)
        & day_return.ge(selected.min_market_move)
        & bar_return.le(selected.short_stall_bar_return)
        & short_score.ge(selected.min_signal_score)
    )

    signal = pd.Series(0, index=result.index, dtype="int8")
    signal.loc[long_condition & ~short_condition] = 1
    signal.loc[short_condition & ~long_condition] = -1
    both = long_condition & short_condition
    signal.loc[both & long_score.ge(short_score)] = 1
    signal.loc[both & short_score.gt(long_score)] = -1
    result["xueqiu_signal"] = signal
    result["xueqiu_long_score"] = long_score
    result["xueqiu_short_score"] = short_score
    chosen_score = np.where(signal.gt(0), long_score, np.where(signal.lt(0), short_score, 0.0))
    result["xueqiu_signal_confidence"] = np.clip(
        (chosen_score - selected.min_signal_score) / 3.0, 0.0, 1.0
    )
    reason = np.full(len(result), "no_signal", dtype=object)
    reason[~quality.to_numpy(dtype=bool)] = "insufficient_point_in_time_posts"
    reason[long_condition.to_numpy(dtype=bool)] = "broad_panic_with_price_stabilization"
    reason[short_condition.to_numpy(dtype=bool)] = "broad_euphoria_with_price_stall"
    reason[both.to_numpy(dtype=bool)] = "conflicting_extremes_score_selected"
    result["xueqiu_signal_reason"] = reason
    return result
