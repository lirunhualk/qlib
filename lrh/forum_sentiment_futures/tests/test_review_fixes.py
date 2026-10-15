from __future__ import annotations

import unittest

import pandas as pd

from lrh.forum_sentiment_futures.hybrid_core import TrendCoreConfig, run_trend_core
from lrh.forum_sentiment_futures.intraday_engine import IntradayStrategyConfig, performance_metrics, run_intraday_backtest
from lrh.forum_sentiment_futures.intraday_features import build_intraday_features
from lrh.forum_sentiment_futures.tests.test_intraday import _day_bars
from lrh.forum_sentiment_futures.xueqiu_strategy.features import build_xueqiu_intraday_features
from lrh.forum_sentiment_futures.xueqiu_strategy.ingest import AuthorizationMetadata, normalize_xueqiu_frame
from lrh.forum_sentiment_futures.xueqiu_strategy.run_backtest import merge_xueqiu_overlay


class ReviewedEngineTest(unittest.TestCase):
    def test_execution_bar_volatility_cannot_change_order_size(self) -> None:
        bars = _day_bars("2026-01-05").iloc[:3].copy()
        bars["signal"] = [1, 0, 0]
        bars["realized_vol_annualized"] = [0.12, 9.0, 0.12]
        _, trades = run_intraday_backtest(bars, IntradayStrategyConfig(target_volatility=0.06, max_holding_bars=1), signal_column="signal")
        self.assertAlmostEqual(trades.iloc[0].position_size, 0.5)

    def test_stop_gap_executes_at_open(self) -> None:
        bars = _day_bars("2026-01-05").iloc[:4].copy()
        bars["signal"] = [1, 0, 0, 0]
        bars["realized_vol_annualized"] = 0.1
        bars.loc[2, ["open", "high", "low", "close"]] = [95.0, 96.0, 94.0, 95.0]
        _, trades = run_intraday_backtest(bars, IntradayStrategyConfig(max_holding_bars=4), signal_column="signal")
        self.assertEqual(trades.iloc[0].exit_reason, "stop_gap")
        self.assertEqual(trades.iloc[0].exit_price, 95.0)

    def test_initial_loss_counts_in_drawdown(self) -> None:
        result = performance_metrics(pd.DataFrame({"strategy_return_net": [-0.1, 0.0]}))
        self.assertAlmostEqual(result["max_drawdown"], -0.1)

    def test_intraday_pnl_and_cost_reconcile(self) -> None:
        bars = _day_bars("2026-01-05").copy()
        bars["signal"] = 1
        bars["realized_vol_annualized"] = 0.1
        daily, trades = run_intraday_backtest(bars, IntradayStrategyConfig(max_trades_per_day=2, max_holding_bars=1, cooldown_bars=0), signal_column="signal")
        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual(daily.iloc[0].strategy_return_net, trades.net_return.sum())
        self.assertAlmostEqual(daily.iloc[0].strategy_return_gross - daily.iloc[0].transaction_cost, daily.iloc[0].strategy_return_net)


class ReviewedFeatureTest(unittest.TestCase):
    def test_overnight_posts_do_not_enter_intraday_window(self) -> None:
        bars = _day_bars("2026-01-05")
        posts = pd.DataFrame([{"board_code": "test", "post_id": "p1", "publish_time": pd.Timestamp("2026-01-04 22:00"), "user_id": "u1", "title": "暴跌"}])
        result = build_intraday_features(bars, posts, board_code="test")
        self.assertEqual(result.post_count.sum(), 0)

    def test_author_union_is_distinct_across_bars(self) -> None:
        bars = _day_bars("2026-01-05")
        posts = pd.DataFrame([{"board_code": "test", "post_id": str(i), "publish_time": pd.Timestamp(f"2026-01-05 {hour}"), "user_id": "same_user", "title": ""} for i, hour in enumerate(["09:40", "09:55", "10:10"])])
        result = build_intraday_features(bars, posts, board_code="test")
        self.assertEqual(result.iloc[2].unique_user_count_60m, 1)
        self.assertEqual(result.iloc[2].cumulative_unique_user_count, 1)

    def test_late_observed_text_waits_for_availability(self) -> None:
        bars = _day_bars("2026-01-05")
        posts = pd.DataFrame([{"fut_code": "IF", "post_id": "p1", "user_id": "u1", "created_at": pd.Timestamp("2026-01-05 09:40"), "available_at": pd.Timestamp("2026-01-05 10:02"), "text": "看多", "is_content_point_in_time": True}])
        result = build_xueqiu_intraday_features(bars, posts, fut_code="IF")
        self.assertEqual(result.iloc[0].post_count, 0)
        self.assertEqual(result.iloc[1].post_count, 0)
        self.assertEqual(result.iloc[2].post_count, 1)

    def test_reload_does_not_duplicate_title_and_handles_utc(self) -> None:
        metadata = AuthorizationMetadata("test", True, True, False, "Asia/Shanghai", "test")
        raw = pd.DataFrame([{"post_id": "p1", "created_at": "2026-01-05T02:00:00Z", "title": "标题", "text": "正文", "symbol_codes": "SH000300"}])
        first = normalize_xueqiu_frame(raw, metadata=metadata)
        second = normalize_xueqiu_frame(first, metadata=metadata)
        self.assertEqual(first.iloc[0].text, second.iloc[0].text)
        self.assertEqual(first.iloc[0].created_at, pd.Timestamp("2026-01-05 10:00"))


class ReviewedAttributionTest(unittest.TestCase):
    def test_xueqiu_returns_cannot_read_old_baseline_overlay(self) -> None:
        keys = {"fut_code": ["IF"], "trade_date": [pd.Timestamp("2026-01-05")]}
        baseline = pd.DataFrame({**keys, "core_return_gross": [0.002], "core_transaction_cost": [0.0001], "core_turnover": [0.1], "hybrid_return_net": [0.09], "strategy_return_gross": [0.09], "transaction_cost": [0.01], "strategy_return_net": [0.08], "turnover": [10.0]})
        overlay = pd.DataFrame({**keys, "strategy_return_gross": [0.003], "transaction_cost": [0.0002], "strategy_return_net": [0.0028], "turnover": [0.5], "trade_count": [1]})
        result = merge_xueqiu_overlay(baseline, overlay).iloc[0]
        self.assertAlmostEqual(result.xueqiu_hybrid_return_net, 0.0047)
        self.assertAlmostEqual(result.xueqiu_hybrid_turnover, 0.6)

    def test_open_rebalance_uses_previous_position_for_overnight(self) -> None:
        dates = pd.bdate_range("2026-01-05", periods=10)
        frame = pd.DataFrame({"fut_code": "IF", "trade_date": dates, "target_contract": "IF2601.CFX", "strategy_return_gross": [0.008, 0.012] * 5, "overnight_return": 0.04, "intraday_return": 0.0})
        config = TrendCoreConfig(trend_horizons=((3, 2),), minimum_positive_horizons=1, volatility_days=2, rebalance_interval_days=1)
        result = run_trend_core(frame, config)
        expected = result.core_position.shift(1).fillna(0) * 0.04
        pd.testing.assert_series_equal(result.core_return_gross, expected, check_names=False)


if __name__ == "__main__":
    unittest.main()
