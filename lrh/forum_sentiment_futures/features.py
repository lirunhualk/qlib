from __future__ import annotations

import re

import numpy as np
import pandas as pd


PANIC_TERMS = (
    "暴跌",
    "大跌",
    "崩盘",
    "股灾",
    "跌停",
    "割肉",
    "清仓",
    "逃命",
    "完了",
    "亏惨",
    "关灯吃面",
    "恐慌",
)
EXCITEMENT_TERMS = (
    "暴涨",
    "大涨",
    "涨停",
    "牛市",
    "起飞",
    "突破",
    "抄底",
    "买入",
    "加仓",
    "满仓",
    "冲高",
    "新高",
)
EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
BRACKET_EMOJI_PATTERN = re.compile(r"\[[^\[\]]{1,16}\]")
BRACKET_TAG_PATTERN = re.compile(r"\[([^\[\]]{1,16})\]")
EXCITEMENT_FACE_TAGS = {
    "上涨",
    "看多",
    "买入",
    "加仓",
    "满仓",
    "梭哈",
    "火箭",
    "兴奋",
    "大笑",
    "得意",
    "鼓掌",
    "赞",
    "强壮",
    "胜利",
    "发财",
    "赚大了",
    "财神",
    "牛",
}
PANIC_FACE_TAGS = {
    "下跌",
    "看空",
    "卖出",
    "减仓",
    "哭",
    "大哭",
    "惨!关灯吃面",
    "惊吓",
    "滴汗",
    "摊手",
    "捂脸",
    "叹气",
}


def rolling_zscore(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    values = pd.Series(series, dtype="float64")
    mean = values.rolling(window=window, min_periods=min_periods).mean()
    std = values.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (values - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def build_basket_sentiment(
    rank_history: pd.DataFrame,
    *,
    fut_code: str,
    symbols: tuple[str, ...],
    window: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    required = {"date", "symbol", "rank", "new_fan_rate", "loyal_fan_rate"}
    missing = required.difference(rank_history.columns)
    if missing:
        raise ValueError(f"Rank history missing columns: {sorted(missing)}")

    raw = rank_history[rank_history["symbol"].isin(symbols)].copy()
    if raw.empty:
        raise ValueError(f"No rank history available for {fut_code}")

    raw["date"] = pd.to_datetime(raw["date"])
    raw["rank"] = pd.to_numeric(raw["rank"], errors="coerce")
    raw["new_fan_rate"] = pd.to_numeric(raw["new_fan_rate"], errors="coerce")
    raw["loyal_fan_rate"] = pd.to_numeric(raw["loyal_fan_rate"], errors="coerce")
    raw = raw.dropna(subset=["date", "rank"])
    raw = raw[raw["rank"] > 0].sort_values(["symbol", "date"])
    raw["log_rank"] = np.log(raw["rank"])
    raw["rank_heat"] = -raw["log_rank"]
    raw["rank_improvement"] = raw.groupby("symbol")["log_rank"].shift(1) - raw["log_rank"]

    grouped = raw.groupby("date", as_index=False).agg(
        symbol_count=("symbol", "nunique"),
        median_rank=("rank", "median"),
        median_log_rank=("log_rank", "median"),
        active_heat_raw=("rank_heat", "median"),
        rank_improvement_raw=("rank_improvement", "median"),
        new_fan_rate=("new_fan_rate", "mean"),
        loyal_fan_rate=("loyal_fan_rate", "mean"),
    )
    grouped["fut_code"] = fut_code
    grouped["basket_size"] = len(set(symbols))
    grouped["coverage_ratio"] = grouped["symbol_count"] / grouped["basket_size"]
    grouped.sort_values("date", inplace=True)

    grouped["active_heat_z"] = rolling_zscore(grouped["active_heat_raw"], window=window, min_periods=min_periods)
    grouped["rank_improvement_z"] = rolling_zscore(
        grouped["rank_improvement_raw"], window=window, min_periods=min_periods
    )
    grouped["new_fan_z"] = rolling_zscore(grouped["new_fan_rate"], window=window, min_periods=min_periods)
    grouped["loyal_fan_z"] = rolling_zscore(grouped["loyal_fan_rate"], window=window, min_periods=min_periods)
    grouped["activity_score"] = (
        0.55 * grouped["active_heat_z"].fillna(0.0)
        + 0.25 * grouped["rank_improvement_z"].fillna(0.0)
        + 0.20 * grouped["new_fan_z"].fillna(0.0)
    )
    grouped.loc[grouped["active_heat_z"].isna(), "activity_score"] = np.nan
    grouped["prev_activity_score"] = grouped["activity_score"].shift(1)
    grouped["fading_excitement"] = (grouped["prev_activity_score"] > 1.0) & (grouped["rank_improvement_z"] < -0.8)
    return grouped.reset_index(drop=True)


def _term_count(value: str, terms: tuple[str, ...]) -> int:
    return sum(value.count(term) for term in terms)


def build_forum_sentiment(
    forum_posts: pd.DataFrame,
    *,
    fut_code: str,
    board_code: str,
    window: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    required = {"date", "board_code", "post_id", "user_id", "title"}
    missing = required.difference(forum_posts.columns)
    if missing:
        raise ValueError(f"Forum posts missing columns: {sorted(missing)}")

    raw = forum_posts[forum_posts["board_code"] == board_code].copy()
    if raw.empty:
        raise ValueError(f"No forum posts available for {fut_code} / {board_code}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw = raw.dropna(subset=["date", "post_id"])
    raw["title"] = raw["title"].fillna("").astype(str)
    raw["user_id"] = raw["user_id"].fillna("").astype(str)
    raw["panic_token_count"] = raw["title"].map(lambda value: _term_count(value, PANIC_TERMS))
    raw["excitement_token_count"] = raw["title"].map(lambda value: _term_count(value, EXCITEMENT_TERMS))
    raw["emoji_count"] = raw["title"].map(
        lambda value: len(EMOJI_PATTERN.findall(value))
        + len(BRACKET_EMOJI_PATTERN.findall(value))
    )

    daily = raw.groupby("date", as_index=True).agg(
        post_count=("post_id", "nunique"),
        unique_user_count=("user_id", lambda values: values[values.ne("")].nunique()),
        panic_token_count=("panic_token_count", "sum"),
        excitement_token_count=("excitement_token_count", "sum"),
        emoji_count=("emoji_count", "sum"),
    )
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0).rename_axis("date").reset_index()
    daily["fut_code"] = fut_code
    daily["board_code"] = board_code
    daily["source"] = "eastmoney_forum_posts"

    denominator = daily["post_count"].clip(lower=1)
    daily["panic_token_rate"] = daily["panic_token_count"] / denominator
    daily["excitement_token_rate"] = daily["excitement_token_count"] / denominator
    daily["emoji_rate"] = daily["emoji_count"] / denominator
    daily["sentiment_balance_raw"] = (
        daily["excitement_token_count"] - daily["panic_token_count"]
    ) / denominator
    daily["post_count_log"] = np.log1p(daily["post_count"])
    daily["unique_user_log"] = np.log1p(daily["unique_user_count"])
    daily["activity_change_raw"] = daily["post_count_log"].diff()

    daily["post_count_z"] = rolling_zscore(
        daily["post_count_log"], window=window, min_periods=min_periods
    )
    daily["unique_user_z"] = rolling_zscore(
        daily["unique_user_log"], window=window, min_periods=min_periods
    )
    daily["emoji_rate_z"] = rolling_zscore(
        daily["emoji_rate"], window=window, min_periods=min_periods
    )
    daily["panic_rate_z"] = rolling_zscore(
        daily["panic_token_rate"], window=window, min_periods=min_periods
    )
    daily["excitement_rate_z"] = rolling_zscore(
        daily["excitement_token_rate"], window=window, min_periods=min_periods
    )
    daily["sentiment_balance_z"] = rolling_zscore(
        daily["sentiment_balance_raw"], window=window, min_periods=min_periods
    )
    daily["activity_change_z"] = rolling_zscore(
        daily["activity_change_raw"], window=window, min_periods=min_periods
    )

    daily["activity_score"] = (
        0.60 * daily["post_count_z"].fillna(0.0)
        + 0.25 * daily["unique_user_z"].fillna(0.0)
        + 0.15 * daily["emoji_rate_z"].fillna(0.0)
    )
    daily.loc[daily["post_count_z"].isna(), "activity_score"] = np.nan
    daily["forum_panic_z"] = (
        0.70 * daily["panic_rate_z"].fillna(0.0)
        - 0.30 * daily["sentiment_balance_z"].fillna(0.0)
    )
    daily["forum_excitement_z"] = (
        0.70 * daily["excitement_rate_z"].fillna(0.0)
        + 0.30 * daily["sentiment_balance_z"].fillna(0.0)
    )
    daily.loc[daily["panic_rate_z"].isna(), "forum_panic_z"] = np.nan
    daily.loc[daily["excitement_rate_z"].isna(), "forum_excitement_z"] = np.nan

    # Compatibility aliases keep the original rank-history strategy/report contract usable.
    daily["active_heat_z"] = daily["post_count_z"]
    daily["rank_improvement_z"] = daily["activity_change_z"]
    daily["new_fan_z"] = daily["unique_user_z"]
    daily["loyal_fan_z"] = np.nan
    daily["prev_activity_score"] = daily["activity_score"].shift(1)
    daily["prev_forum_excitement_z"] = daily["forum_excitement_z"].shift(1)
    daily["fading_excitement"] = (
        (daily["prev_forum_excitement_z"] > 1.0) & (daily["activity_change_z"] < -0.8)
    )
    daily["coverage_ratio"] = daily["post_count"].gt(0).astype(float)
    return daily.reset_index(drop=True)


def build_signal_features(
    sentiment: pd.DataFrame,
    spot: pd.DataFrame,
    *,
    fut_code: str,
    market_window: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    spot_frame = spot.copy().sort_values("trade_date")
    spot_frame["panic_z"] = -rolling_zscore(
        spot_frame["spot_return_1d"], window=market_window, min_periods=min_periods
    )
    spot_frame["spot_return_5d"] = spot_frame["spot_close"].pct_change(5)
    spot_frame["spot_return_20d"] = spot_frame["spot_close"].pct_change(20)

    sent = sentiment[sentiment["fut_code"] == fut_code].copy()
    sent = sent.rename(columns={"date": "sentiment_date"}).sort_values("sentiment_date")
    if sent.empty:
        raise ValueError(f"No sentiment frame for {fut_code}")

    aligned = pd.merge_asof(
        spot_frame.sort_values("trade_date"),
        sent,
        left_on="trade_date",
        right_on="sentiment_date",
        direction="backward",
    )
    aligned["fut_code"] = fut_code
    aligned["has_sentiment"] = aligned["sentiment_date"].notna()
    aligned["fading_excitement"] = aligned["fading_excitement"].fillna(False).astype(bool)
    return aligned.sort_values("trade_date").reset_index(drop=True)
