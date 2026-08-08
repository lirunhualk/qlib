from __future__ import annotations

import unittest

import pandas as pd

from lrh.forum_sentiment_futures.features import build_basket_sentiment, build_forum_sentiment


class FeatureBuildTest(unittest.TestCase):
    def test_rank_improvement_raises_activity_score(self) -> None:
        rows = []
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        for idx, date in enumerate(dates):
            rows.append({"date": date, "symbol": "SH600000", "rank": 500 - idx * 50, "new_fan_rate": 0.2 + idx * 0.01, "loyal_fan_rate": 0.8})
            rows.append({"date": date, "symbol": "SZ000001", "rank": 800 - idx * 80, "new_fan_rate": 0.3 + idx * 0.01, "loyal_fan_rate": 0.7})
        history = pd.DataFrame(rows)

        features = build_basket_sentiment(
            history,
            fut_code="IF",
            symbols=("SH600000", "SZ000001"),
            window=3,
            min_periods=2,
        )

        self.assertIn("activity_score", features.columns)
        self.assertEqual(float(features["coverage_ratio"].iloc[-1]), 1.0)
        self.assertGreater(features["activity_score"].iloc[-1], features["activity_score"].dropna().iloc[0])

    def test_forum_titles_build_behavior_features(self) -> None:
        rows = []
        dates = pd.date_range("2026-01-01", periods=8, freq="D")
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "board_code": "zssh000300",
                    "post_id": f"normal-{idx}",
                    "user_id": f"user-{idx}",
                    "title": "指数震荡",
                }
            )
        rows.extend(
            [
                {
                    "date": dates[-1],
                    "board_code": "zssh000300",
                    "post_id": "panic-1",
                    "user_id": "panic-user-1",
                    "title": "暴跌恐慌，快逃命",
                },
                {
                    "date": dates[-1],
                    "board_code": "zssh000300",
                    "post_id": "panic-2",
                    "user_id": "panic-user-2",
                    "title": "崩盘割肉",
                },
            ]
        )

        features = build_forum_sentiment(
            pd.DataFrame(rows),
            fut_code="IF",
            board_code="zssh000300",
            window=5,
            min_periods=3,
        )

        self.assertEqual(int(features["post_count"].iloc[-1]), 3)
        self.assertEqual(int(features["panic_token_count"].iloc[-1]), 5)
        self.assertGreater(features["forum_panic_z"].iloc[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
