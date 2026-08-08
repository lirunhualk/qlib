from __future__ import annotations

import unittest

import pandas as pd

from lrh.forum_sentiment_futures.hybrid_core import (
    TrendCoreConfig,
    combine_core_and_intraday,
    run_trend_core,
)


def _continuous_returns() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=12)
    return pd.DataFrame(
        {
            "fut_code": "IF",
            "trade_date": dates,
            "target_contract": ["IF2601.CFX"] * 8 + ["IF2602.CFX"] * 4,
            "strategy_return_gross": [0.01] * len(dates),
        }
    )


class TrendCoreTest(unittest.TestCase):
    def test_position_uses_only_prior_day_returns(self) -> None:
        config = TrendCoreConfig(
            trend_horizons=((3, 2),),
            minimum_positive_horizons=1,
            volatility_days=2,
            target_volatility=0.10,
            max_position_size=0.50,
            rebalance_interval_days=1,
        )
        original = _continuous_returns()
        changed = original.copy()
        changed.loc[6, "strategy_return_gross"] = -0.25

        first = run_trend_core(original, config)
        second = run_trend_core(changed, config)

        self.assertEqual(float(first.loc[6, "core_position"]), float(second.loc[6, "core_position"]))
        self.assertNotEqual(
            float(first.loc[7, "core_signal_price"]),
            float(second.loc[7, "core_signal_price"]),
        )

    def test_roll_turnover_closes_and_reopens_position(self) -> None:
        varying_returns = _continuous_returns()
        varying_returns["strategy_return_gross"] = [0.008, 0.012] * 6
        config = TrendCoreConfig(
            trend_horizons=((3, 2),),
            minimum_positive_horizons=1,
            volatility_days=2,
            target_volatility=0.10,
            max_position_size=0.50,
            rebalance_interval_days=1,
        )
        result = run_trend_core(varying_returns, config)
        roll = result.iloc[8]

        self.assertGreater(float(roll["core_position"]), 0.0)
        self.assertAlmostEqual(
            float(roll["core_turnover"]),
            float(result.iloc[7]["core_position"] + roll["core_position"]),
        )

    def test_hybrid_return_adds_independent_component_pnl_and_costs(self) -> None:
        daily = pd.DataFrame(
            {
                "fut_code": ["IF"],
                "strategy_return_gross": [0.002],
                "transaction_cost": [0.0002],
                "strategy_return_net": [0.0018],
                "turnover": [1.0],
                "trade_count": [1],
                "core_return_gross": [0.003],
                "core_transaction_cost": [0.0001],
                "core_return_net": [0.0029],
                "core_turnover": [1.0],
                "core_trade_count": [1],
            }
        )
        result = combine_core_and_intraday(daily).iloc[0]

        self.assertAlmostEqual(float(result["hybrid_return_gross"]), 0.005)
        self.assertAlmostEqual(float(result["hybrid_transaction_cost"]), 0.0003)
        self.assertAlmostEqual(float(result["hybrid_return_net"]), 0.0047)
        self.assertEqual(int(result["hybrid_trade_count"]), 2)


if __name__ == "__main__":
    unittest.main()
