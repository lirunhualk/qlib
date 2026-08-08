from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..intraday_features import build_intraday_features


BULLISH_TERMS = (
    "超预期",
    "继续看好",
    "坚定看多",
    "买入",
    "加仓",
    "抄底",
    "做多",
    "看多",
    "突破",
    "反弹",
    "上涨",
    "大涨",
    "涨停",
    "新高",
    "牛市",
    "利好",
    "起飞",
    "火箭",
    "回暖",
)
BEARISH_TERMS = (
    "不及预期",
    "坚定看空",
    "卖出",
    "减仓",
    "清仓",
    "止损",
    "做空",
    "看空",
    "破位",
    "下跌",
    "大跌",
    "暴跌",
    "跌停",
    "新低",
    "熊市",
    "利空",
    "崩盘",
    "恐慌",
    "逃命",
)
NEGATIONS = (
    "不",
    "没",
    "无",
    "未",
    "别",
    "莫",
    "没有",
    "毫无",
    "并非",
    "不是",
    "绝非",
)
POSITIVE_FACE_TAGS = ("[火箭]", "[上涨]", "[牛]", "[发财]", "[鼓掌]", "[大笑]")
NEGATIVE_FACE_TAGS = ("[哭泣]", "[哭]", "[下跌]", "[心碎]", "[惊恐]", "[怒]")
AROUSAL_FACE_TAGS = (
    "[捂脸]",
    "[摊手]",
    "[笑哭]",
    "[滴汗]",
    "[惊讶]",
    "[火箭]",
    "[哭泣]",
    "[哭]",
)

TERM_PATTERN = re.compile(
    "|".join(
        re.escape(term)
        for term in sorted(
            set(BULLISH_TERMS + BEARISH_TERMS), key=len, reverse=True
        )
    )
)


def _score_directional_terms(text: str) -> tuple[int, int]:
    bullish = 0
    bearish = 0
    for match in TERM_PATTERN.finditer(text):
        term = match.group(0)
        polarity = 1 if term in BULLISH_TERMS else -1
        prefix = text[max(0, match.start() - 4) : match.start()]
        if any(prefix.endswith(negation) for negation in NEGATIONS):
            polarity *= -1
        if polarity > 0:
            bullish += 1
        else:
            bearish += 1
    bullish += sum(text.count(tag) for tag in POSITIVE_FACE_TAGS)
    bearish += sum(text.count(tag) for tag in NEGATIVE_FACE_TAGS)
    return bullish, bearish


def score_xueqiu_text(text: str) -> dict[str, float]:
    value = str(text or "")
    bullish, bearish = _score_directional_terms(value)
    arousal = sum(value.count(tag) for tag in AROUSAL_FACE_TAGS)
    exclamations = len(re.findall(r"[!！]", value))
    questions = len(re.findall(r"[?？]", value))
    directional = bullish + bearish
    return {
        "xq_bullish_term_count": float(bullish),
        "xq_bearish_term_count": float(bearish),
        "xq_directional_post_count": float(directional > 0),
        "xq_mixed_post_count": float(bullish > 0 and bearish > 0),
        "xq_emotion_marker_count": float(arousal + exclamations + questions),
    }


def _map_posts_to_next_bar(
    posts: pd.DataFrame, bars: pd.DataFrame
) -> pd.DataFrame:
    post_frame = posts.copy()
    post_frame["publish_time"] = pd.to_datetime(
        post_frame["publish_time"], errors="coerce"
    )
    post_frame = post_frame.dropna(subset=["publish_time", "post_id"])
    bar_times = (
        pd.to_datetime(bars["bar_end"])
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
    )
    post_times = post_frame["publish_time"].to_numpy(dtype="datetime64[ns]")
    locations = np.searchsorted(bar_times, post_times, side="left")
    valid = locations < len(bar_times)
    mapped = post_frame.loc[valid].copy()
    mapped["bar_end"] = pd.to_datetime(bar_times[locations[valid]])
    return mapped


def _rolling_within_day(
    values: pd.Series, dates: pd.Series, window: int
) -> pd.Series:
    return (
        pd.DataFrame({"value": values, "trade_date": dates})
        .groupby("trade_date", sort=False)["value"]
        .transform(lambda series: series.rolling(window, min_periods=1).sum())
    )


def _difference_within_day(
    values: pd.Series, dates: pd.Series
) -> pd.Series:
    return (
        pd.DataFrame({"value": values, "trade_date": dates})
        .groupby("trade_date", sort=False)["value"]
        .diff()
    )


def _causal_slot_zscore(
    values: pd.Series,
    slots: pd.Series,
    *,
    window_days: int,
    min_days: int,
) -> pd.Series:
    frame = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="coerce"), "slot": slots}
    )
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for indices in frame.groupby("slot", sort=False).groups.values():
        series = frame.loc[indices, "value"]
        history = series.shift(1)
        mean = history.rolling(window_days, min_periods=min_days).mean()
        std = (
            history.rolling(window_days, min_periods=min_days)
            .std(ddof=0)
            .replace(0, np.nan)
        )
        result.loc[indices] = ((series - mean) / std).to_numpy()
    return result.replace([np.inf, -np.inf], np.nan)


def build_xueqiu_intraday_features(
    dominant_bars: pd.DataFrame,
    mapped_posts: pd.DataFrame,
    *,
    fut_code: str,
    forum_window_bars: int = 4,
    baseline_window_days: int = 40,
    baseline_min_days: int = 20,
    volatility_window_bars: int = 80,
    min_posts_60m: int = 5,
    min_unique_users_60m: int = 3,
    min_content_coverage: float = 0.95,
) -> pd.DataFrame:
    posts = mapped_posts[mapped_posts["fut_code"].eq(fut_code)].copy()
    if posts.empty:
        raise ValueError(f"No mapped Xueqiu posts for {fut_code}")
    board_code = f"xueqiu_{fut_code}"
    posts["board_code"] = board_code
    posts["publish_time"] = pd.to_datetime(posts["created_at"], errors="coerce")
    posts["title"] = posts["text"].where(
        posts["is_content_point_in_time"].fillna(False), ""
    )

    base = build_intraday_features(
        dominant_bars,
        posts,
        board_code=board_code,
        forum_window_bars=forum_window_bars,
        baseline_window_days=baseline_window_days,
        baseline_min_days=baseline_min_days,
        volatility_window_bars=volatility_window_bars,
    )

    scored = posts.copy()
    score_frame = pd.DataFrame(
        [score_xueqiu_text(value) for value in scored["title"]],
        index=scored.index,
    )
    scored = pd.concat([scored, score_frame], axis=1)
    scored["xq_point_in_time_content_count"] = scored[
        "is_content_point_in_time"
    ].fillna(False).astype(float)
    mapped = _map_posts_to_next_bar(scored, base)
    aggregate_columns = [
        "xq_bullish_term_count",
        "xq_bearish_term_count",
        "xq_directional_post_count",
        "xq_mixed_post_count",
        "xq_emotion_marker_count",
        "xq_point_in_time_content_count",
    ]
    aggregated = (
        mapped.groupby("bar_end", as_index=False)[aggregate_columns].sum()
        if not mapped.empty
        else pd.DataFrame(columns=["bar_end", *aggregate_columns])
    )
    features = base.merge(
        aggregated, on="bar_end", how="left", validate="one_to_one"
    )
    features[aggregate_columns] = features[aggregate_columns].fillna(0.0)
    for column in aggregate_columns:
        features[f"{column}_60m"] = _rolling_within_day(
            features[column], features["trade_date"], forum_window_bars
        )

    post_denominator = features["post_count_60m"].clip(lower=1.0)
    directional_denominator = (
        features["xq_bullish_term_count_60m"]
        + features["xq_bearish_term_count_60m"]
    ).clip(lower=1.0)
    features["xq_sentiment_raw"] = (
        features["xq_bullish_term_count_60m"]
        - features["xq_bearish_term_count_60m"]
    ) / post_denominator
    features["xq_bullish_raw"] = (
        features["xq_bullish_term_count_60m"] / post_denominator
    )
    features["xq_bearish_raw"] = (
        features["xq_bearish_term_count_60m"] / post_denominator
    )
    features["xq_disagreement_raw"] = (
        2.0
        * np.minimum(
            features["xq_bullish_term_count_60m"],
            features["xq_bearish_term_count_60m"],
        )
        / directional_denominator
    )
    features["xq_emotion_intensity_raw"] = (
        features["xq_emotion_marker_count_60m"] / post_denominator
    )
    features["xq_author_breadth_raw"] = (
        features["unique_user_count_60m"] / post_denominator
    ).clip(upper=1.0)
    features["xq_content_coverage_60m"] = (
        features["xq_point_in_time_content_count_60m"] / post_denominator
    ).clip(upper=1.0)

    current_denominator = features["post_count"].clip(lower=1.0)
    features["xq_sentiment_current_raw"] = (
        features["xq_bullish_term_count"]
        - features["xq_bearish_term_count"]
    ) / current_denominator
    features["xq_sentiment_velocity_raw"] = _difference_within_day(
        features["xq_sentiment_current_raw"], features["trade_date"]
    )
    features["xq_activity_velocity_raw"] = _difference_within_day(
        features["activity_current_raw"], features["trade_date"]
    )

    zscore_map = {
        "xq_sentiment_raw": "xq_sentiment_z",
        "xq_bullish_raw": "xq_bullish_z",
        "xq_bearish_raw": "xq_bearish_z",
        "xq_disagreement_raw": "xq_disagreement_z",
        "xq_emotion_intensity_raw": "xq_emotion_intensity_z",
        "xq_author_breadth_raw": "xq_author_breadth_z",
        "xq_sentiment_velocity_raw": "xq_sentiment_velocity_z",
        "xq_activity_velocity_raw": "xq_activity_velocity_z",
    }
    for source, target in zscore_map.items():
        features[target] = _causal_slot_zscore(
            features[source],
            features["bar_slot"],
            window_days=baseline_window_days,
            min_days=baseline_min_days,
        )

    features["xq_signal_quality_ok"] = (
        features["post_count_60m"].ge(min_posts_60m)
        & features["unique_user_count_60m"].ge(min_unique_users_60m)
        & features["xq_content_coverage_60m"].ge(min_content_coverage)
        & features["xq_sentiment_z"].notna()
        & features["xq_sentiment_velocity_z"].notna()
    )
    return features


def augment_cross_product_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    grouped = result.groupby("bar_end", sort=False)
    result["xq_cross_activity_mean_z"] = grouped["activity_z"].transform("mean")
    result["xq_cross_sentiment_mean_z"] = grouped["xq_sentiment_z"].transform(
        "mean"
    )
    result["xq_relative_activity_z"] = (
        result["activity_z"] - result["xq_cross_activity_mean_z"]
    )
    result["xq_relative_sentiment_z"] = (
        result["xq_sentiment_z"] - result["xq_cross_sentiment_mean_z"]
    )
    result["xq_cross_emotion_breadth"] = grouped[
        "xq_sentiment_z"
    ].transform(lambda values: values.abs().ge(0.75).mean())
    return result
