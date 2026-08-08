from __future__ import annotations

import unittest

import pandas as pd

from lrh.forum_sentiment_futures.run_behavior_update_backtest import _portfolio


class BehaviorPortfolioTest(unittest.TestCase):
    def test_portfolio_averages_returns_and_sums_trades(self) -> None:
        daily = pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2026-01-05")] * 2,
                "updated_hybrid_return_net": [0.01, -0.005],
                "hybrid_return_net": [0.008, -0.004],
                "core_return_net": [0.007, -0.003],
                "behavior_return_net": [0.002, -0.001],
                "original_daily_return": [0.0, 0.0],
                "futures_buy_hold_return": [0.02, -0.01],
                "updated_hybrid_trade_count": [2, 3],
                "hybrid_trade_count": [1, 2],
                "core_trade_count": [1, 1],
                "behavior_trade_count": [1, 1],
            }
        )
        portfolio = _portfolio(daily)
        self.assertAlmostEqual(
            float(portfolio["updated_hybrid_return_net"].iloc[0]),
            0.0025,
        )
        self.assertEqual(int(portfolio["behavior_trade_count"].iloc[0]), 2)
        self.assertEqual(int(portfolio["updated_hybrid_trade_count"].iloc[0]), 5)


if __name__ == "__main__":
    unittest.main()
