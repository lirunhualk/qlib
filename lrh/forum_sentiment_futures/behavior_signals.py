from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd

from .intraday_engine import IntradayStrategyConfig


@dataclass(frozen=True)
class BehaviorSignalConfig:
    family: str
    feature_column: str
    feature_threshold: float
    direction: int
    signal_start_slot: int
    signal_end_slot: int
    event_column: str | None = None
    event_threshold: float = 0.0
    min_day_return: float = -np.inf
    max_day_return: float = np.inf
    min_bar_return: float = -np.inf
    max_bar_return: float = np.inf
    min_attention_persistence: float = 0.0
    require_moderate_volatility: bool = False
    max_holding_bars: int = 3
    stop_loss: float = 0.006
    take_profit: float = 0.012
    target_volatility: float = 0.05
    max_position_size: float = 0.35
    min_position_size: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)

    def execution_config(self) -> IntradayStrategyConfig:
        return IntradayStrategyConfig(
            max_holding_bars=self.max_holding_bars,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            target_volatility=self.target_volatility,
            max_position_size=self.max_position_size,
            min_position_size=self.min_position_size,
            max_trades_per_day=1,
            cooldown_bars=1,
            daily_loss_limit=0.008,
            commission_bps_per_side=1.0,
            slippage_bps_per_side=1.0,
            allow_long=self.direction > 0,
            allow_short=self.direction < 0,
        )


def augment_cross_board_features(features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fut_code",
        "bar_end",
        "attention_cumulative_z",
        "activity_z",
        "special_attention_cumulative_z",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Cross-board feature input missing columns: {sorted(missing)}")

    source = features.copy()
    source["bar_end"] = pd.to_datetime(source["bar_end"])

    def high_share(values: pd.Series) -> float:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return float(numeric.ge(0.75).mean()) if not numeric.empty else np.nan

    cross = source.groupby("bar_end", as_index=False).agg(
        cross_board_count=("fut_code", "nunique"),
        cross_attention_mean_z=("attention_cumulative_z", "mean"),
        cross_activity_mean_z=("activity_z", "mean"),
        cross_special_mean_z=("special_attention_cumulative_z", "mean"),
        cross_attention_breadth=("attention_cumulative_z", high_share),
        cross_special_breadth=("special_attention_cumulative_z", high_share),
    )
    result = source.merge(cross, on="bar_end", how="left", validate="many_to_one")
    result["current_symbol_event_count"] = result[
        [
            "exclamation_count",
            "question_count",
            "directional_sign_count",
            "punctuation_run_count",
        ]
    ].sum(axis=1)
    result["current_special_event_count"] = result[
        [
            "concise_extreme_count",
            "sarcasm_count",
            "polarity_conflict_count",
            "rhetorical_extreme_count",
        ]
    ].sum(axis=1)
    result["current_reaction_event_count"] = (
        result["current_symbol_event_count"]
        + result["concise_extreme_count"]
        + result["panic_event_count"]
        + result["excitement_event_count"]
    )
    return result


def apply_behavior_signal(
    features: pd.DataFrame,
    config: BehaviorSignalConfig,
) -> pd.DataFrame:
    required = {
        "can_signal",
        "bar_slot",
        "day_return",
        "bar_return",
        "attention_persistence_60m",
        config.feature_column,
    }
    if config.event_column:
        required.add(config.event_column)
    if config.require_moderate_volatility:
        required.add("moderate_volatility_regime")
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Behavior signal input missing columns: {sorted(missing)}")
    if config.direction not in {-1, 1}:
        raise ValueError("Behavior signal direction must be -1 or 1")

    result = features.copy()
    mask = (
        result["can_signal"].fillna(False)
        & result["bar_slot"].between(
            config.signal_start_slot,
            config.signal_end_slot,
        )
        & pd.to_numeric(result[config.feature_column], errors="coerce").ge(
            config.feature_threshold
        )
        & pd.to_numeric(result["day_return"], errors="coerce").between(
            config.min_day_return,
            config.max_day_return,
        )
        & pd.to_numeric(result["bar_return"], errors="coerce").between(
            config.min_bar_return,
            config.max_bar_return,
        )
        & pd.to_numeric(
            result["attention_persistence_60m"], errors="coerce"
        ).ge(config.min_attention_persistence)
    )
    if config.event_column:
        mask &= pd.to_numeric(
            result[config.event_column], errors="coerce"
        ).gt(config.event_threshold)
    if config.require_moderate_volatility:
        mask &= result["moderate_volatility_regime"].fillna(False)

    result["behavior_feature_value"] = pd.to_numeric(
        result[config.feature_column], errors="coerce"
    )
    result["behavior_signal"] = np.where(mask, config.direction, 0).astype(int)
    return result


def build_behavior_candidates() -> list[BehaviorSignalConfig]:
    candidates: list[BehaviorSignalConfig] = []

    for threshold, max_day_return, holding, persistence in product(
        (1.75, 2.25),
        (-0.003, 0.0),
        (2, 3, 4),
        (0.0, 2.0),
    ):
        candidates.append(
            BehaviorSignalConfig(
                family="cross_attention_rebound_long",
                feature_column="cross_attention_mean_z",
                feature_threshold=threshold,
                direction=1,
                signal_start_slot=4,
                signal_end_slot=7,
                event_column="post_count",
                max_day_return=max_day_return,
                min_attention_persistence=persistence,
                max_holding_bars=holding,
            )
        )

    for threshold, slots, require_stabilization, holding in product(
        (1.25, 1.75, 2.25),
        ((4, 7), (4, 13)),
        (False, True),
        (2, 3),
    ):
        candidates.append(
            BehaviorSignalConfig(
                family="concise_extreme_reversal_long",
                feature_column="concise_extreme_z",
                feature_threshold=threshold,
                direction=1,
                signal_start_slot=slots[0],
                signal_end_slot=slots[1],
                event_column="concise_extreme_count",
                max_day_return=-0.003,
                min_bar_return=0.0 if require_stabilization else -np.inf,
                max_holding_bars=holding,
            )
        )

    for threshold, holding in product((1.25, 1.75, 2.25), (3, 4)):
        candidates.append(
            BehaviorSignalConfig(
                family="special_attention_reversal_long",
                feature_column="special_attention_cumulative_z",
                feature_threshold=threshold,
                direction=1,
                signal_start_slot=4,
                signal_end_slot=7,
                event_column="current_special_event_count",
                max_day_return=-0.003,
                max_holding_bars=holding,
            )
        )
        candidates.append(
            BehaviorSignalConfig(
                family="sarcasm_bottom_long",
                feature_column="sarcasm_z",
                feature_threshold=threshold,
                direction=1,
                signal_start_slot=4,
                signal_end_slot=13,
                event_column="sarcasm_count",
                max_day_return=-0.003,
                min_bar_return=0.0,
                max_holding_bars=holding,
            )
        )

    for threshold, slots, holding in product(
        (1.75, 2.25),
        ((4, 7), (8, 13)),
        (3, 4),
    ):
        candidates.append(
            BehaviorSignalConfig(
                family="symbol_exhaustion_short",
                feature_column="symbol_intensity_z",
                feature_threshold=threshold,
                direction=-1,
                signal_start_slot=slots[0],
                signal_end_slot=slots[1],
                event_column="current_symbol_event_count",
                min_day_return=-0.004,
                max_day_return=0.004,
                max_holding_bars=holding,
            )
        )

    for threshold in (0.50, 0.75):
        candidates.append(
            BehaviorSignalConfig(
                family="cross_breadth_followthrough_short",
                feature_column="cross_attention_breadth",
                feature_threshold=threshold,
                direction=-1,
                signal_start_slot=8,
                signal_end_slot=13,
                event_column="post_count",
                max_day_return=-0.003,
                max_holding_bars=4,
            )
        )
        candidates.append(
            BehaviorSignalConfig(
                family="late_1445_followthrough_short",
                feature_column="cross_attention_breadth",
                feature_threshold=threshold,
                direction=-1,
                signal_start_slot=14,
                signal_end_slot=14,
                event_column="post_count",
                max_day_return=-0.003,
                max_holding_bars=1,
                stop_loss=0.004,
                take_profit=0.006,
            )
        )

    for holding in (3, 4):
        candidates.append(
            BehaviorSignalConfig(
                family="author_concentration_short",
                feature_column="user_concentration_z",
                feature_threshold=0.75,
                direction=-1,
                signal_start_slot=4,
                signal_end_slot=7,
                event_column="post_count",
                min_day_return=-0.004,
                max_day_return=0.004,
                max_holding_bars=holding,
            )
        )

    for threshold, holding, moderate, direction in product(
        (1.75, 2.25),
        (2, 3),
        (False, True),
        (-1, 1),
    ):
        candidates.append(
            BehaviorSignalConfig(
                family=(
                    "reaction_speed_reversal_long"
                    if direction > 0
                    else "reaction_speed_reversal_short"
                ),
                feature_column="reaction_speed_magnitude_z",
                feature_threshold=threshold,
                direction=direction,
                signal_start_slot=4,
                signal_end_slot=13,
                event_column="current_reaction_event_count",
                min_day_return=0.003 if direction < 0 else -np.inf,
                max_day_return=-0.003 if direction > 0 else np.inf,
                require_moderate_volatility=moderate,
                max_holding_bars=holding,
            )
        )

    for holding, moderate, direction in product(
        (2, 3),
        (False, True),
        (-1, 1),
    ):
        candidates.append(
            BehaviorSignalConfig(
                family=(
                    "activity_speed_reversal_long"
                    if direction > 0
                    else "activity_speed_reversal_short"
                ),
                feature_column="activity_speed_z",
                feature_threshold=2.25,
                direction=direction,
                signal_start_slot=4,
                signal_end_slot=13,
                event_column="post_count",
                min_day_return=0.003 if direction < 0 else -np.inf,
                max_day_return=-0.003 if direction > 0 else np.inf,
                require_moderate_volatility=moderate,
                max_holding_bars=holding,
            )
        )

    for holding, moderate, direction in product(
        (3, 4),
        (False, True),
        (-1, 1),
    ):
        candidates.append(
            BehaviorSignalConfig(
                family=(
                    "positive_expectation_trend_long"
                    if direction > 0
                    else "negative_expectation_trend_short"
                ),
                feature_column=(
                    "positive_expectation_speed_z"
                    if direction > 0
                    else "negative_expectation_speed_z"
                ),
                feature_threshold=1.75,
                direction=direction,
                signal_start_slot=4,
                signal_end_slot=13,
                event_column=(
                    "positive_expectation_count"
                    if direction > 0
                    else "negative_expectation_count"
                ),
                min_day_return=0.002 if direction > 0 else -np.inf,
                max_day_return=-0.002 if direction < 0 else np.inf,
                require_moderate_volatility=moderate,
                max_holding_bars=holding,
            )
        )

    return candidates
