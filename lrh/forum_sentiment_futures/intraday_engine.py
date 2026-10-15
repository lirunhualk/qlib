from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntradayStrategyConfig:
    forum_panic_threshold: float = 0.8
    market_panic_threshold: float = 0.5
    long_score_threshold: float = 0.9
    forum_excitement_threshold: float = 1.0
    short_score_threshold: float = 0.9
    calm_market_abs_z: float = 1.2
    max_holding_bars: int = 4
    stop_loss: float = 0.007
    take_profit: float = 0.010
    target_volatility: float = 0.10
    max_position_size: float = 1.0
    min_position_size: float = 0.15
    max_trades_per_day: int = 2
    cooldown_bars: int = 2
    daily_loss_limit: float = 0.012
    commission_bps_per_side: float = 1.0
    slippage_bps_per_side: float = 1.0
    allow_long: bool = True
    allow_short: bool = True
    long_signal_start_slot: int = 0
    long_signal_end_slot: int = 14
    long_max_momentum_30m: float = -0.005
    long_require_new_relevant_token: bool = False
    signal_start_slot: int = 0
    signal_end_slot: int = 14
    require_new_relevant_token: bool = False
    min_activity_z: float = -99.0
    max_activity_z: float = 99.0
    min_day_return: float = -99.0
    max_day_return: float = 99.0

    @property
    def total_cost_rate_per_side(self) -> float:
        return (self.commission_bps_per_side + self.slippage_bps_per_side) / 10000.0

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(row: pd.Series, column: str) -> float | None:
    value = row.get(column, np.nan)
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def select_intraday_signal(row: pd.Series, config: IntradayStrategyConfig) -> int:
    if not bool(row.get("can_signal", False)):
        return 0
    forum_panic = _finite(row, "forum_panic_z")
    forum_excitement = _finite(row, "forum_excitement_z")
    market_panic = _finite(row, "market_panic_z")
    momentum = _finite(row, "momentum_30m")
    long_score = _finite(row, "long_signal_score")
    short_score = _finite(row, "short_signal_score")
    activity = _finite(row, "activity_z")
    day_return = _finite(row, "day_return")
    bar_slot = _finite(row, "bar_slot")
    if None in {
        forum_panic,
        forum_excitement,
        market_panic,
        momentum,
        long_score,
        short_score,
        activity,
        day_return,
        bar_slot,
    }:
        return 0
    panic_token_count = float(row.get("panic_token_count", 0.0) or 0.0)
    excitement_token_count = float(row.get("excitement_token_count", 0.0) or 0.0)
    long_event_ok = not config.long_require_new_relevant_token or panic_token_count > 0
    short_event_ok = not config.require_new_relevant_token or excitement_token_count > 0
    long_context_ok = config.long_signal_start_slot <= bar_slot <= config.long_signal_end_slot
    short_context_ok = (
        config.signal_start_slot <= bar_slot <= config.signal_end_slot
        and config.min_activity_z <= activity <= config.max_activity_z
        and config.min_day_return <= day_return <= config.max_day_return
    )

    long_condition = (
        config.allow_long
        and long_context_ok
        and long_event_ok
        and forum_panic >= config.forum_panic_threshold
        and market_panic >= config.market_panic_threshold
        and momentum <= config.long_max_momentum_30m
        and long_score >= config.long_score_threshold
    )
    short_condition = (
        config.allow_short
        and short_context_ok
        and short_event_ok
        and forum_excitement >= config.forum_excitement_threshold
        and abs(market_panic) <= config.calm_market_abs_z
        and momentum <= 0.0
        and short_score >= config.short_score_threshold
    )
    if long_condition and short_condition:
        return 1 if long_score >= short_score else -1
    if long_condition:
        return 1
    if short_condition:
        return -1
    return 0


def _position_size(row: pd.Series, config: IntradayStrategyConfig) -> float:
    realized_vol = _finite(row, "realized_vol_annualized")
    if realized_vol is None or realized_vol <= 0:
        return 0.0
    raw_size = config.target_volatility / realized_vol
    if raw_size < config.min_position_size:
        return 0.0
    return float(min(raw_size, config.max_position_size))


def _exit_from_bar(
    row: pd.Series,
    *,
    direction: int,
    entry_price: float,
    held_bars: int,
    is_last_bar: bool,
    config: IntradayStrategyConfig,
) -> tuple[float | None, str | None]:
    if direction > 0:
        stop_price = entry_price * (1.0 - config.stop_loss)
        take_price = entry_price * (1.0 + config.take_profit)
        if float(row["open"]) <= stop_price:
            return float(row["open"]), "stop_gap"
        if float(row["open"]) >= take_price:
            return take_price, "take_gap"
        if float(row["low"]) <= stop_price:
            return stop_price, "stop_loss"
        if float(row["high"]) >= take_price:
            return take_price, "take_profit"
    else:
        stop_price = entry_price * (1.0 + config.stop_loss)
        take_price = entry_price * (1.0 - config.take_profit)
        if float(row["open"]) >= stop_price:
            return float(row["open"]), "stop_gap"
        if float(row["open"]) <= take_price:
            return take_price, "take_gap"
        if float(row["high"]) >= stop_price:
            return stop_price, "stop_loss"
        if float(row["low"]) <= take_price:
            return take_price, "take_profit"
    if held_bars >= config.max_holding_bars:
        return float(row["close"]), "max_holding"
    if is_last_bar:
        return float(row["close"]), "market_close"
    return None, None


def run_intraday_backtest(
    features: pd.DataFrame,
    config: IntradayStrategyConfig,
    *,
    signal_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "fut_code",
        "trade_date",
        "bar_end",
        "target_contract",
        "open",
        "high",
        "low",
        "close",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Intraday features missing columns: {sorted(missing)}")
    if signal_column is not None and signal_column not in features.columns:
        raise ValueError(f"Intraday features missing signal column: {signal_column}")
    if features["fut_code"].nunique() > 1:
        raise ValueError("Run the intraday engine separately for each futures product")

    source = features.copy()
    source["trade_date"] = pd.to_datetime(source["trade_date"]).dt.normalize()
    source["bar_end"] = pd.to_datetime(source["bar_end"])
    source = source.sort_values(["trade_date", "bar_end"]).reset_index(drop=True)

    daily_rows: list[dict] = []
    trade_rows: list[dict] = []
    nav = 1.0
    for trade_date, day in source.groupby("trade_date", sort=True):
        day = day.reset_index(drop=True)
        pending_direction = 0
        pending_signal: dict | None = None
        direction = 0
        position_size = 0.0
        entry_price = np.nan
        entry_time = pd.NaT
        entry_bar_index = -1
        entry_signal: dict = {}
        cooldown = 0
        day_trade_count = 0
        day_gross_nav = 1.0
        day_net_nav = 1.0
        day_cost = 0.0
        day_turnover = 0.0
        long_trades = 0
        short_trades = 0
        sizes: list[float] = []

        for bar_index, row in day.iterrows():
            is_last_bar = bar_index == len(day) - 1
            if cooldown > 0:
                cooldown -= 1

            if (
                direction == 0
                and pending_direction != 0
                and cooldown == 0
                and day_trade_count < config.max_trades_per_day
                and day_net_nav - 1.0 > -config.daily_loss_limit
            ):
                position_size = float((pending_signal or {}).get("position_size", 0.0))
                if position_size > 0:
                    direction = pending_direction
                    entry_price = float(row["open"])
                    entry_time = row.get(
                        "bar_start", row["bar_end"] - pd.Timedelta(minutes=15)
                    )
                    entry_bar_index = bar_index
                    entry_signal = pending_signal or {}
                    day_trade_count += 1
                    sizes.append(position_size)
                    if direction > 0:
                        long_trades += 1
                    else:
                        short_trades += 1
                pending_direction = 0
                pending_signal = None

            if direction != 0:
                held_bars = bar_index - entry_bar_index + 1
                exit_price, exit_reason = _exit_from_bar(
                    row,
                    direction=direction,
                    entry_price=entry_price,
                    held_bars=held_bars,
                    is_last_bar=is_last_bar,
                    config=config,
                )
                if exit_price is not None:
                    gross_return = direction * (exit_price / entry_price - 1.0) * position_size
                    turnover = 2.0 * position_size
                    transaction_cost = turnover * config.total_cost_rate_per_side
                    net_return = gross_return - transaction_cost
                    day_gross_nav += gross_return
                    day_net_nav += net_return
                    day_cost += transaction_cost
                    day_turnover += turnover
                    trade_rows.append(
                        {
                            "fut_code": row["fut_code"],
                            "trade_date": trade_date,
                            "target_contract": row["target_contract"],
                            "direction": direction,
                            "position_size": position_size,
                            "signal_time": entry_signal.get("signal_time", pd.NaT),
                            "entry_time": entry_time,
                            "entry_bar_end": day.iloc[entry_bar_index]["bar_end"],
                            "exit_time": row["bar_end"],
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "holding_bars": held_bars,
                            "exit_reason": exit_reason,
                            "gross_return": gross_return,
                            "transaction_cost": transaction_cost,
                            "net_return": net_return,
                            "signal_forum_panic_z": entry_signal.get("forum_panic_z", np.nan),
                            "signal_forum_excitement_z": entry_signal.get(
                                "forum_excitement_z", np.nan
                            ),
                            "signal_market_panic_z": entry_signal.get("market_panic_z", np.nan),
                            "signal_activity_z": entry_signal.get("activity_z", np.nan),
                            "signal_long_score": entry_signal.get("long_signal_score", np.nan),
                            "signal_short_score": entry_signal.get("short_signal_score", np.nan),
                            "signal_source": entry_signal.get("signal_source", "legacy"),
                            "signal_behavior_feature": entry_signal.get(
                                "behavior_feature", np.nan
                            ),
                            "signal_cross_attention_mean_z": entry_signal.get(
                                "cross_attention_mean_z", np.nan
                            ),
                            "signal_cross_attention_breadth": entry_signal.get(
                                "cross_attention_breadth", np.nan
                            ),
                        }
                    )
                    direction = 0
                    position_size = 0.0
                    entry_price = np.nan
                    entry_time = pd.NaT
                    entry_bar_index = -1
                    entry_signal = {}
                    cooldown = config.cooldown_bars

            if (
                direction == 0
                and pending_direction == 0
                and not is_last_bar
                and cooldown == 0
                and day_trade_count < config.max_trades_per_day
                and day_net_nav - 1.0 > -config.daily_loss_limit
            ):
                if signal_column is None:
                    signal = select_intraday_signal(row, config)
                else:
                    raw_signal = pd.to_numeric(
                        pd.Series([row.get(signal_column, 0)]), errors="coerce"
                    ).iloc[0]
                    signal = (
                        int(np.sign(raw_signal))
                        if pd.notna(raw_signal) and np.isfinite(float(raw_signal))
                        else 0
                    )
                    if (signal > 0 and not config.allow_long) or (
                        signal < 0 and not config.allow_short
                    ):
                        signal = 0
                if signal != 0:
                    pending_direction = signal
                    pending_signal = {
                        "signal_time": row["bar_end"],
                        "position_size": _position_size(row, config),
                        "forum_panic_z": row.get("forum_panic_z", np.nan),
                        "forum_excitement_z": row.get("forum_excitement_z", np.nan),
                        "market_panic_z": row.get("market_panic_z", np.nan),
                        "activity_z": row.get("activity_z", np.nan),
                        "long_signal_score": row.get("long_signal_score", np.nan),
                        "short_signal_score": row.get("short_signal_score", np.nan),
                        "signal_source": signal_column or "legacy",
                        "behavior_feature": row.get("behavior_feature_value", np.nan),
                        "cross_attention_mean_z": row.get(
                            "cross_attention_mean_z", np.nan
                        ),
                        "cross_attention_breadth": row.get(
                            "cross_attention_breadth", np.nan
                        ),
                    }

        strategy_return_gross = day_gross_nav - 1.0
        strategy_return_net = day_net_nav - 1.0
        benchmark_return = float(day["close"].iloc[-1] / day["open"].iloc[0] - 1.0)
        nav *= 1.0 + strategy_return_net
        daily_rows.append(
            {
                "fut_code": day["fut_code"].iloc[0],
                "trade_date": trade_date,
                "strategy": "intraday_forum_risk_controlled",
                "strategy_name": "Intraday forum sentiment, risk controlled",
                "strategy_return_gross": strategy_return_gross,
                "transaction_cost": day_cost,
                "strategy_return_net": strategy_return_net,
                "benchmark_return": benchmark_return,
                "excess_return_net": strategy_return_net - benchmark_return,
                "turnover": day_turnover,
                "trade_count": day_trade_count,
                "long_trades": long_trades,
                "short_trades": short_trades,
                "average_position_size": float(np.mean(sizes)) if sizes else 0.0,
                "nav": nav,
            }
        )
    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows)


def performance_metrics(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {
            "observations": 0,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "total_return": np.nan,
            "trade_count": 0,
        }
    returns = pd.to_numeric(daily["strategy_return_net"], errors="coerce").fillna(0.0)
    nav = (1.0 + returns).cumprod()
    observations = len(returns)
    total_return = float(nav.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (252.0 / observations) - 1.0)
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252.0)) if returns.std(ddof=0) > 0 else 0.0
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return {
        "observations": observations,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "total_return": total_return,
        "trade_count": int(daily.get("trade_count", pd.Series(dtype=float)).sum()),
    }
