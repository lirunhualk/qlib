from __future__ import annotations

import unittest

import pandas as pd

from lrh.forum_sentiment_futures.reports import build_reports


class ReportTest(unittest.TestCase):
    def test_partial_year_returns_are_annualized(self) -> None:
        dates = pd.date_range("2026-01-05", periods=10, freq="B")
        daily = pd.DataFrame(
            {
                "fut_code": "IF",
                "strategy": "dual_forum_sentiment",
                "strategy_name": "Dual forum sentiment",
                "trade_date": dates,
                "strategy_return_net": [0.001] * 10,
                "benchmark_return": [0.0005] * 10,
                "excess_return_net": [0.0005] * 10,
                "position": [1] * 10,
                "turnover": [2.0] * 10,
                "transaction_cost": [0.0002] * 10,
                "nav": (1.0 + pd.Series([0.001] * 10)).cumprod(),
                "benchmark_nav": (1.0 + pd.Series([0.0005] * 10)).cumprod(),
                "relative_nav": (1.0 + pd.Series([0.0005] * 10)).cumprod(),
            }
        )

        annual = build_reports(daily).annual_returns.iloc[0]
        self.assertEqual(int(annual["observations"]), 10)
        self.assertGreater(annual["strategy_annualized_return"], annual["strategy_return_net"])
        self.assertGreater(annual["benchmark_annualized_return"], annual["benchmark_return"])


if __name__ == "__main__":
    unittest.main()
