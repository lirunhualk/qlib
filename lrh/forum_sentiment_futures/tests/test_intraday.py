from __future__ import annotations

import unittest

import pandas as pd

from lrh.forum_sentiment_futures.behavior_signals import (
    BehaviorSignalConfig,
    apply_behavior_signal,
    augment_cross_board_features,
)
from lrh.forum_sentiment_futures.intraday_engine import (
    IntradayStrategyConfig,
    run_intraday_backtest,
)
from lrh.forum_sentiment_futures.intraday_features import build_intraday_features
from lrh.forum_sentiment_futures.intraday_sources import (
    EXPECTED_15M_TIMES,
    assemble_dominant_intraday,
)


def _day_bars(date: str, fut_code: str = "IF", contract: str = "IF2601.CFX") -> pd.DataFrame:
    rows = []
    for index, bar_time in enumerate(sorted(EXPECTED_15M_TIMES)):
        price = 100.0 + index * 0.1
        rows.append(
            {
                "fut_code": fut_code,
                "selection_date": pd.Timestamp(date) - pd.Timedelta(days=1),
                "trade_date": pd.Timestamp(date),
                "target_contract": contract,
                "ts_code": contract,
                "bar_end": pd.Timestamp(f"{date} {bar_time}:00"),
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price + 0.1,
                "volume": 1000.0,
                "oi": 2000.0,
            }
        )
    return pd.DataFrame(rows)


class IntradayFeatureTest(unittest.TestCase):
    def test_post_is_mapped_to_current_bar_close_then_traded_later(self) -> None:
        bars = pd.concat(
            [_day_bars("2026-01-05"), _day_bars("2026-01-06")], ignore_index=True
        )
        posts = pd.DataFrame(
            [
                {
                    "board_code": "zssh000300",
                    "post_id": "p1",
                    "publish_time": pd.Timestamp("2026-01-06 09:40:00"),
                    "user_id": "u1",
                    "title": "暴跌恐慌[哭]",
                }
            ]
        )
        features = build_intraday_features(
            bars,
            posts,
            board_code="zssh000300",
            baseline_window_days=2,
            baseline_min_days=1,
            volatility_window_bars=16,
        )
        first_bar = features[features["bar_end"] == pd.Timestamp("2026-01-06 09:45:00")].iloc[0]
        self.assertEqual(int(first_bar["post_count"]), 1)
        self.assertEqual(int(first_bar["panic_token_count"]), 2)
        self.assertEqual(int(first_bar["panic_face_count"]), 1)
        self.assertEqual(int(first_bar["panic_event_count"]), 3)

    def test_incomplete_intraday_day_is_excluded(self) -> None:
        mapping = pd.DataFrame(
            [
                {
                    "fut_code": "IF",
                    "selection_date": pd.Timestamp("2026-01-04"),
                    "trade_date": pd.Timestamp("2026-01-05"),
                    "target_contract": "IF2601.CFX",
                }
            ]
        )
        bars = _day_bars("2026-01-05").drop(index=0)
        bars = bars[
            ["ts_code", "bar_end", "trade_date", "open", "high", "low", "close", "volume", "oi"]
        ]
        result, coverage = assemble_dominant_intraday(mapping, bars)
        self.assertTrue(result.empty)
        self.assertFalse(bool(coverage["complete"].iloc[0]))
        self.assertEqual(int(coverage["observed_bars"].iloc[0]), 15)

    def test_bracket_face_is_diagnostic_not_a_tradable_token(self) -> None:
        bars = pd.concat(
            [_day_bars("2026-01-05"), _day_bars("2026-01-06")], ignore_index=True
        )
        posts = pd.DataFrame(
            [
                {
                    "board_code": "zssh000300",
                    "post_id": "p1",
                    "publish_time": pd.Timestamp("2026-01-06 10:01:00"),
                    "user_id": "u1",
                    "title": "[火箭]",
                }
            ]
        )
        features = build_intraday_features(
            bars,
            posts,
            board_code="zssh000300",
            baseline_window_days=2,
            baseline_min_days=1,
            volatility_window_bars=16,
        )
        signal_bar = features[
            features["bar_end"] == pd.Timestamp("2026-01-06 10:15:00")
        ].iloc[0]

        self.assertEqual(int(signal_bar["excitement_token_count"]), 0)
        self.assertEqual(int(signal_bar["excitement_face_count"]), 1)
        self.assertEqual(float(signal_bar["forum_excitement_raw"]), 0.0)
        self.assertGreater(float(signal_bar["forum_excitement_event_raw"]), 0.0)

    def test_behavior_features_capture_concise_symbols_and_sarcasm(self) -> None:
        bars = pd.concat(
            [_day_bars("2026-01-05"), _day_bars("2026-01-06")], ignore_index=True
        )
        posts = pd.DataFrame(
            [
                {
                    "board_code": "zssh000300",
                    "post_id": "p1",
                    "publish_time": pd.Timestamp("2026-01-06 10:01:00"),
                    "user_id": "u1",
                    "title": "哈哈，稳了！！",
                }
            ]
        )
        features = build_intraday_features(
            bars,
            posts,
            board_code="zssh000300",
            baseline_window_days=2,
            baseline_min_days=1,
            volatility_window_bars=16,
        )
        signal_bar = features[
            features["bar_end"] == pd.Timestamp("2026-01-06 10:15:00")
        ].iloc[0]
        self.assertEqual(int(signal_bar["sarcasm_count"]), 2)
        self.assertEqual(int(signal_bar["exclamation_count"]), 2)
        self.assertEqual(int(signal_bar["punctuation_run_count"]), 1)
        self.assertEqual(int(signal_bar["concise_extreme_count"]), 1)
        self.assertGreater(float(signal_bar["reaction_speed_magnitude_raw"]), 0.0)

    def test_cross_board_breadth_and_behavior_mask(self) -> None:
        rows = []
        for fut_code, attention in [("IF", 2.0), ("IC", 1.0), ("IH", -0.5)]:
            rows.append(
                {
                    "fut_code": fut_code,
                    "bar_end": pd.Timestamp("2026-01-06 10:45:00"),
                    "attention_cumulative_z": attention,
                    "activity_z": attention,
                    "special_attention_cumulative_z": attention,
                    "exclamation_count": 0,
                    "question_count": 0,
                    "directional_sign_count": 0,
                    "punctuation_run_count": 0,
                    "concise_extreme_count": 1,
                    "sarcasm_count": 0,
                    "polarity_conflict_count": 0,
                    "rhetorical_extreme_count": 0,
                    "panic_event_count": 0,
                    "excitement_event_count": 0,
                    "can_signal": True,
                    "bar_slot": 4,
                    "day_return": -0.004,
                    "bar_return": 0.001,
                    "attention_persistence_60m": 2.0,
                }
            )
        features = augment_cross_board_features(pd.DataFrame(rows))
        self.assertAlmostEqual(float(features["cross_attention_breadth"].iloc[0]), 2 / 3)
        config = BehaviorSignalConfig(
            family="test",
            feature_column="cross_attention_mean_z",
            feature_threshold=0.75,
            direction=1,
            signal_start_slot=4,
            signal_end_slot=7,
            event_column="current_special_event_count",
            max_day_return=-0.003,
        )
        signaled = apply_behavior_signal(features, config)
        self.assertTrue(signaled["behavior_signal"].eq(1).all())


class IntradayEngineTest(unittest.TestCase):
    def test_signal_executes_at_next_bar_open_with_risk_controls(self) -> None:
        times = pd.to_datetime(
            [
                "2026-01-05 09:45:00",
                "2026-01-05 10:00:00",
                "2026-01-05 10:15:00",
                "2026-01-05 10:30:00",
            ]
        )
        features = pd.DataFrame(
            {
                "fut_code": "IF",
                "trade_date": pd.Timestamp("2026-01-05"),
                "bar_end": times,
                "target_contract": "IF2601.CFX",
                "open": [99.5, 100.0, 101.0, 101.0],
                "high": [100.0, 101.2, 101.2, 101.2],
                "low": [99.0, 99.8, 100.8, 100.8],
                "close": [99.5, 101.0, 101.0, 101.0],
                "can_signal": [True, True, True, False],
                "forum_panic_z": [2.0, 0.0, 0.0, 0.0],
                "forum_excitement_z": [0.0, 0.0, 0.0, 0.0],
                "market_panic_z": [1.5, 0.0, 0.0, 0.0],
                "momentum_30m": [-0.01, 0.0, 0.0, 0.0],
                "long_signal_score": [2.0, 0.0, 0.0, 0.0],
                "short_signal_score": [0.0, 0.0, 0.0, 0.0],
                "activity_z": [0.0, 0.0, 0.0, 0.0],
                "day_return": [-0.01, 0.0, 0.0, 0.0],
                "bar_slot": [0, 1, 2, 3],
                "realized_vol_annualized": [0.1, 0.1, 0.1, 0.1],
            }
        )
        config = IntradayStrategyConfig(
            forum_panic_threshold=1.0,
            market_panic_threshold=1.0,
            long_score_threshold=1.0,
            take_profit=0.01,
            commission_bps_per_side=1.0,
            slippage_bps_per_side=1.0,
        )
        daily, trades = run_intraday_backtest(features, config)
        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(trade["signal_time"], pd.Timestamp("2026-01-05 09:45:00"))
        self.assertEqual(trade["entry_time"], pd.Timestamp("2026-01-05 09:45:00"))
        self.assertEqual(trade["entry_bar_end"], pd.Timestamp("2026-01-05 10:00:00"))
        self.assertEqual(trade["exit_reason"], "take_profit")
        self.assertAlmostEqual(float(trade["gross_return"]), 0.01)
        self.assertAlmostEqual(float(trade["net_return"]), 0.0096)
        self.assertAlmostEqual(float(daily["strategy_return_net"].iloc[0]), 0.0096)

    def test_preset_signal_reuses_next_bar_execution(self) -> None:
        features = pd.DataFrame(
            {
                "fut_code": "IF",
                "trade_date": pd.Timestamp("2026-01-05"),
                "bar_end": pd.to_datetime(
                    [
                        "2026-01-05 09:45:00",
                        "2026-01-05 10:00:00",
                    ]
                ),
                "target_contract": "IF2601.CFX",
                "open": [100.0, 100.0],
                "high": [100.2, 101.2],
                "low": [99.8, 99.8],
                "close": [100.0, 101.0],
                "behavior_signal": [1, 0],
                "realized_vol_annualized": [0.1, 0.1],
            }
        )
        config = IntradayStrategyConfig(
            max_holding_bars=1,
            target_volatility=0.05,
            max_position_size=0.5,
            min_position_size=0.1,
            commission_bps_per_side=1.0,
            slippage_bps_per_side=1.0,
        )
        daily, trades = run_intraday_backtest(
            features,
            config,
            signal_column="behavior_signal",
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["entry_time"], pd.Timestamp("2026-01-05 09:45:00"))
        self.assertAlmostEqual(float(trades.iloc[0]["position_size"]), 0.5)
        self.assertAlmostEqual(float(daily.iloc[0]["strategy_return_net"]), 0.0048)


if __name__ == "__main__":
    unittest.main()
