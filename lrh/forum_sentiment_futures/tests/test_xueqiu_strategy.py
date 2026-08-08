from __future__ import annotations

import unittest

import pandas as pd

from lrh.forum_sentiment_futures.xueqiu_strategy.audit import (
    audit_backtest_data,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.features import (
    score_xueqiu_text,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.ingest import (
    AuthorizationMetadata,
    first_observed_post_snapshot,
    normalize_xueqiu_frame,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.membership import (
    assign_posts_to_futures,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.signals import (
    XueqiuSignalConfig,
    apply_xueqiu_signal,
)


class XueqiuIngestTest(unittest.TestCase):
    def test_dataseed_columns_and_html_emoji_are_normalized(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "股票代码": "SH600519, 000858.SZ",
                    "ID": 123,
                    "发帖者ID": 456,
                    "帖子发表者呢称": "测试作者",
                    "帖子内容": '暴跌<img alt="[哭泣]" src="x.png">',
                    "来源": "雪球",
                    "发帖时间": "2024/01/02 10:01:02",
                    "点赞量": "9",
                }
            ]
        )
        metadata = AuthorizationMetadata(
            provider="licensed_test",
            authorized_use=True,
            content_snapshot_at_creation=True,
            interaction_metrics_point_in_time=False,
            timezone="Asia/Shanghai",
            license_reference="unit-test",
        )
        normalized = normalize_xueqiu_frame(raw, metadata=metadata)
        row = normalized.iloc[0]
        self.assertEqual(row["post_id"], "123")
        self.assertEqual(row["symbol_codes"], "SH600519|SZ000858")
        self.assertIn("[哭泣]", row["text"])
        self.assertTrue(bool(row["is_content_point_in_time"]))
        self.assertFalse(bool(row["is_metrics_point_in_time"]))

    def test_unverified_historical_text_is_not_point_in_time(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "post_id": "p1",
                    "created_at": "2024-01-02 10:01:02",
                    "symbols": "SH000300",
                    "text": "看多",
                }
            ]
        )
        normalized = normalize_xueqiu_frame(raw)
        self.assertFalse(bool(normalized["is_content_point_in_time"].iloc[0]))
        self.assertFalse(bool(normalized["authorized_use"].iloc[0]))

    def test_first_observation_wins_over_later_interaction_snapshot(self) -> None:
        snapshots = pd.DataFrame(
            [
                {
                    "provider": "licensed",
                    "post_id": "p1",
                    "created_at": "2024-01-02 10:01:00",
                    "observed_at": "2024-01-02 10:02:00",
                    "content_observed_at": "2024-01-02 10:02:00",
                    "text": "first",
                },
                {
                    "provider": "licensed",
                    "post_id": "p1",
                    "created_at": "2024-01-02 10:01:00",
                    "observed_at": "2024-01-02 11:00:00",
                    "content_observed_at": "2024-01-02 11:00:00",
                    "text": "later edit",
                },
            ]
        )
        selected = first_observed_post_snapshot(snapshots)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected["text"].iloc[0], "first")


class XueqiuFeatureTest(unittest.TestCase):
    def test_negation_flips_direction(self) -> None:
        score = score_xueqiu_text("并不看好，今天没有利好，可能继续暴跌[哭泣]")
        self.assertEqual(score["xq_bullish_term_count"], 0.0)
        self.assertGreaterEqual(score["xq_bearish_term_count"], 3.0)

    def test_fixed_signal_requires_panic_and_price_stabilization(self) -> None:
        features = pd.DataFrame(
            [
                {
                    "activity_z": 1.2,
                    "xq_author_breadth_z": 0.0,
                    "xq_bearish_z": 1.5,
                    "xq_bullish_z": -0.5,
                    "xq_sentiment_velocity_z": -1.4,
                    "market_panic_z": 1.1,
                    "day_return": -0.006,
                    "bar_return": 0.0,
                    "realized_vol_annualized": 0.2,
                    "bar_slot": 5,
                    "xq_relative_activity_z": 0.0,
                    "xq_signal_quality_ok": True,
                    "can_signal": True,
                }
            ]
        )
        result = apply_xueqiu_signal(
            features, XueqiuSignalConfig(min_signal_score=3.0)
        )
        self.assertEqual(int(result["xueqiu_signal"].iloc[0]), 1)
        self.assertEqual(
            result["xueqiu_signal_reason"].iloc[0],
            "broad_panic_with_price_stabilization",
        )


class XueqiuMembershipAndAuditTest(unittest.TestCase):
    def test_historical_constituent_snapshot_maps_post(self) -> None:
        posts = pd.DataFrame(
            [
                {
                    "provider": "licensed",
                    "post_id": "p1",
                    "user_id": "u1",
                    "created_at": pd.Timestamp("2024-01-02 10:01:00"),
                    "symbol_codes": "SH600519",
                    "text": "暴跌",
                    "is_content_point_in_time": True,
                    "authorized_use": True,
                }
            ]
        )
        membership = pd.DataFrame(
            [
                {
                    "fut_code": "IF",
                    "snapshot_date": pd.Timestamp("2023-12-29"),
                    "symbol": "SH600519",
                    "weight": 5.0,
                }
            ]
        )
        mapped, coverage = assign_posts_to_futures(
            posts, membership, fut_codes=["IF"]
        )
        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped["fut_code"].iloc[0], "IF")
        self.assertEqual(float(mapped["constituent_weight"].iloc[0]), 5.0)
        self.assertEqual(int(coverage["mapped_posts"].iloc[0]), 1)

    def test_audit_rejects_non_point_in_time_content(self) -> None:
        posts = pd.DataFrame(
            [
                {
                    "fut_code": "IF",
                    "post_id": "p1",
                    "created_at": pd.Timestamp("2024-01-02 10:01:00"),
                    "authorized_use": True,
                    "is_content_point_in_time": False,
                    "is_metrics_point_in_time": False,
                    "membership_staleness_days": 1,
                }
            ]
        )
        calendar = pd.DataFrame(
            [{"fut_code": "IF", "trade_date": pd.Timestamp("2024-01-02")}]
        )
        audit = audit_backtest_data(
            posts,
            calendar,
            fut_codes=["IF"],
            start_date=pd.Timestamp("2024-01-02"),
            end_date=pd.Timestamp("2024-01-02"),
            min_trading_day_coverage=0.0,
            min_median_posts_per_day=0.0,
        )
        self.assertFalse(audit.eligible)
        self.assertTrue(any("事后编辑" in reason for reason in audit.reasons))


if __name__ == "__main__":
    unittest.main()
