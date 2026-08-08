from __future__ import annotations

import unittest

import pandas as pd

from lrh.forum_sentiment_futures.config import StrategySpec
from lrh.forum_sentiment_futures.engine import BacktestConfig, run_signal_backtest, select_position


def _market() -> pd.DataFrame:
    rows = []
    previous_close = None
    for date, open_price, close_price in [
        ("2026-01-01", 100.0, 101.0),
        ("2026-01-02", 102.0, 105.0),
        ("2026-01-05", 105.0, 103.0),
    ]:
        rows.append(
            {
                "fut_code": "IF",
                "trade_date": pd.Timestamp(date),
                "ts_code": "IF2601.CFX",
                "delist_date": 20260116,
                "open": open_price,
                "close": close_price,
                "prev_close": previous_close,
                "vol": 1000,
                "amount": 1000,
                "oi": 1000,
                "trade_days_to_expiry": 10,
                "spot_return_1d": 0.005,
                "spot_intraday_return": close_price / open_price - 1.0,
            }
        )
        previous_close = close_price
    return pd.DataFrame(rows)


class EngineTest(unittest.TestCase):
    def test_panic_signal_trades_next_open_to_close(self) -> None:
        signals = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2026-01-01"),
                    "has_sentiment": True,
                    "panic_z": 1.5,
                    "activity_score": -0.5,
                    "rank_improvement_z": 0.0,
                    "prev_activity_score": 0.0,
                    "spot_return_3d": -0.01,
                    "new_fan_z": 0.0,
                    "sentiment_date": pd.Timestamp("2026-01-01"),
                    "fading_excitement": False,
                },
                {
                    "trade_date": pd.Timestamp("2026-01-02"),
                    "has_sentiment": True,
                    "panic_z": 0.0,
                    "activity_score": 2.0,
                    "rank_improvement_z": 0.0,
                    "prev_activity_score": 0.0,
                    "spot_return_3d": -0.01,
                    "new_fan_z": 0.0,
                    "sentiment_date": pd.Timestamp("2026-01-02"),
                    "fading_excitement": False,
                },
            ]
        )
        spec = StrategySpec("panic_reversal_long", "Panic reversal long", "panic_long")
        result = run_signal_backtest(_market(), signals, spec, BacktestConfig(cost_bps_per_side=1.0))

        first_trade = result[result["trade_date"] == pd.Timestamp("2026-01-02")].iloc[0]
        expected = 105.0 / 102.0 - 1.0 - 2 * 0.0001
        self.assertEqual(first_trade["position"], 1)
        self.assertAlmostEqual(first_trade["strategy_return_net"], expected, places=10)

        second_trade = result[result["trade_date"] == pd.Timestamp("2026-01-05")].iloc[0]
        self.assertEqual(second_trade["position"], 0)
        self.assertAlmostEqual(first_trade["benchmark_return"], 0.005)

    def test_short_condition_requires_calm_market(self) -> None:
        spec = StrategySpec("excitement_short", "Excitement short", "excitement_short")
        row = pd.Series(
            {
                "has_sentiment": True,
                "panic_z": 0.2,
                "activity_score": 1.4,
                "spot_return_3d": -0.01,
            }
        )
        self.assertEqual(select_position(spec, row), -1)
        row["panic_z"] = 2.0
        self.assertEqual(select_position(spec, row), 0)

    def test_continuous_holding_uses_close_to_close_and_roll_style_cost(self) -> None:
        signals = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2026-01-01", "2026-01-02"])}
        )
        spec = StrategySpec("always_long_continuous", "Always long continuous futures", "always_hold")
        result = run_signal_backtest(_market(), signals, spec, BacktestConfig(cost_bps_per_side=1.0))

        first_trade = result.iloc[0]
        self.assertAlmostEqual(first_trade["strategy_return_gross"], 105.0 / 101.0 - 1.0)
        self.assertAlmostEqual(first_trade["transaction_cost"], 0.0001)
        second_trade = result.iloc[1]
        self.assertAlmostEqual(second_trade["strategy_return_gross"], 103.0 / 105.0 - 1.0)
        self.assertEqual(second_trade["turnover"], 0.0)


if __name__ == "__main__":
    unittest.main()
