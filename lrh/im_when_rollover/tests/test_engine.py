from __future__ import annotations

import unittest

import pandas as pd

from lrh.im_when_rollover.compat import qlib_style_risk_analysis
from lrh.im_when_rollover.config import StrategySpec
from lrh.im_when_rollover.engine import BacktestConfig, run_strategy_backtest
from lrh.im_when_rollover.strategies import select_target_contract


def _snapshot(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("ts_code", drop=False)


class StrategySelectionTest(unittest.TestCase):
    def test_front_roll_switches_when_days_hit_threshold(self) -> None:
        snapshot = _snapshot(
            [
                {"ts_code": "IM2404.CFX", "delist_date": 20240419, "trade_days_to_expiry": 1, "vol": 10, "oi": 10, "amount": 10, "annualized_carry": 0.1},
                {"ts_code": "IM2405.CFX", "delist_date": 20240517, "trade_days_to_expiry": 20, "vol": 8, "oi": 8, "amount": 8, "annualized_carry": 0.2},
                {"ts_code": "IM2406.CFX", "delist_date": 20240621, "trade_days_to_expiry": 40, "vol": 6, "oi": 6, "amount": 6, "annualized_carry": 0.3},
            ]
        )
        spec = StrategySpec(
            name="front_roll_1",
            display_name="近月提前1日",
            family="front_roll",
            selector="roll_chain",
            initial_rank=1,
            roll_days_before_expiry=1,
        )
        selected = select_target_contract(spec, snapshot, "IM2404.CFX")
        self.assertEqual(selected, "IM2405.CFX")

    def test_best_carry_daily_prefers_highest_carry(self) -> None:
        snapshot = _snapshot(
            [
                {"ts_code": "IF2404.CFX", "delist_date": 20240419, "trade_days_to_expiry": 10, "vol": 10, "oi": 10, "amount": 10, "annualized_carry": 0.05},
                {"ts_code": "IF2405.CFX", "delist_date": 20240517, "trade_days_to_expiry": 30, "vol": 20, "oi": 20, "amount": 20, "annualized_carry": 0.07},
                {"ts_code": "IF2406.CFX", "delist_date": 20240621, "trade_days_to_expiry": 60, "vol": 30, "oi": 30, "amount": 30, "annualized_carry": 0.06},
            ]
        )
        spec = StrategySpec("max_carry_daily", "每日重选最大年化贴水", "carry", "max_carry_daily")
        selected = select_target_contract(spec, snapshot, None)
        self.assertEqual(selected, "IF2405.CFX")


class BacktestEngineTest(unittest.TestCase):
    def test_roll_day_uses_vwap_transition(self) -> None:
        market = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2024-04-10"),
                    "ts_code": "IM2404.CFX",
                    "delist_date": 20240419,
                    "close": 100.0,
                    "prev_close": None,
                    "vwap": 100.0,
                    "trade_days_to_expiry": 2,
                    "annualized_carry": 0.08,
                    "spot_close": 101.0,
                },
                {
                    "trade_date": pd.Timestamp("2024-04-10"),
                    "ts_code": "IM2405.CFX",
                    "delist_date": 20240517,
                    "close": 103.0,
                    "prev_close": None,
                    "vwap": 103.0,
                    "trade_days_to_expiry": 20,
                    "annualized_carry": 0.07,
                    "spot_close": 101.0,
                },
                {
                    "trade_date": pd.Timestamp("2024-04-11"),
                    "ts_code": "IM2404.CFX",
                    "delist_date": 20240419,
                    "close": 101.0,
                    "prev_close": 100.0,
                    "vwap": 101.5,
                    "trade_days_to_expiry": 1,
                    "annualized_carry": 0.09,
                    "spot_close": 102.0,
                },
                {
                    "trade_date": pd.Timestamp("2024-04-11"),
                    "ts_code": "IM2405.CFX",
                    "delist_date": 20240517,
                    "close": 104.0,
                    "prev_close": 103.0,
                    "vwap": 103.5,
                    "trade_days_to_expiry": 19,
                    "annualized_carry": 0.07,
                    "spot_close": 102.0,
                },
            ]
        )
        spec = StrategySpec(
            name="front_roll_1",
            display_name="近月提前1日",
            family="front_roll",
            selector="roll_chain",
            initial_rank=1,
            roll_days_before_expiry=1,
        )
        result = run_strategy_backtest(market, spec, BacktestConfig(cost_bps_per_side=1.0))
        day2 = result[result["trade_date"] == pd.Timestamp("2024-04-11")].iloc[0]

        expected_factor = (101.5 / 100.0) * (104.0 / 103.5) * (1 - 0.0001) ** 2
        self.assertAlmostEqual(day2["strategy_return"], expected_factor - 1.0, places=10)
        self.assertTrue(bool(day2["rolled"]))

    def test_risk_analysis_product_mode(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.015, 0.0])
        report = qlib_style_risk_analysis(returns, N=252, mode="product")
        self.assertIn("annualized_return", report.index)
        self.assertLessEqual(float(report.loc["max_drawdown", "risk"]), 0.0)


if __name__ == "__main__":
    unittest.main()
