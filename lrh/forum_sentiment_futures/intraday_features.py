from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .features import (
    BRACKET_EMOJI_PATTERN,
    BRACKET_TAG_PATTERN,
    EMOJI_PATTERN,
    EXCITEMENT_FACE_TAGS,
    EXCITEMENT_TERMS,
    PANIC_FACE_TAGS,
    PANIC_TERMS,
)

SARCASM_TERMS = (
    "呵呵",
    "哈哈",
    "笑死",
    "赢麻",
    "又赢",
    "稳了",
    "技术性调整",
    "干得漂亮",
    "不愧",
    "就这",
)
EXPECTATION_TERMS = (
    "预期",
    "预计",
    "未来",
    "即将",
    "将会",
    "目标",
    "看好",
    "趋势",
    "逻辑",
    "基本面",
    "业绩",
    "估值",
    "空间",
)
EXCLAMATION_PATTERN = re.compile(r"[!！]")
QUESTION_PATTERN = re.compile(r"[?？]")
DIRECTIONAL_SIGN_PATTERN = re.compile(r"[+＋\-－]")
PUNCTUATION_RUN_PATTERN = re.compile(r"[!！?？~～]{2,}")


def _term_count(value: str, terms: tuple[str, ...]) -> int:
    return sum(value.count(term) for term in terms)


def _causal_slot_zscore(
    values: pd.Series,
    slots: pd.Series,
    *,
    window_days: int,
    min_days: int,
) -> pd.Series:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "slot": slots})
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, indices in frame.groupby("slot", sort=False).groups.items():
        series = frame.loc[indices, "value"]
        history = series.shift(1)
        mean = history.rolling(window_days, min_periods=min_days).mean()
        std = history.rolling(window_days, min_periods=min_days).std(ddof=0).replace(0, np.nan)
        result.loc[indices] = ((series - mean) / std).to_numpy()
    return result.replace([np.inf, -np.inf], np.nan)


def _rolling_within_day(values: pd.Series, dates: pd.Series, window: int) -> pd.Series:
    return (
        pd.DataFrame({"value": values, "trade_date": dates})
        .groupby("trade_date", sort=False)["value"]
        .transform(lambda series: series.rolling(window, min_periods=1).sum())
    )


def _cumulative_within_day(values: pd.Series, dates: pd.Series) -> pd.Series:
    return (
        pd.DataFrame({"value": values, "trade_date": dates})
        .groupby("trade_date", sort=False)["value"]
        .cumsum()
    )


def _difference_within_day(values: pd.Series, dates: pd.Series) -> pd.Series:
    return (
        pd.DataFrame({"value": values, "trade_date": dates})
        .groupby("trade_date", sort=False)["value"]
        .diff()
    )


def _map_posts_to_next_bar(posts: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    post_frame = posts.copy()
    post_frame["publish_time"] = pd.to_datetime(post_frame["publish_time"], errors="coerce")
    post_frame = post_frame.dropna(subset=["publish_time", "post_id"])
    post_frame["title"] = post_frame["title"].fillna("").astype(str)
    post_frame["user_id"] = post_frame["user_id"].fillna("").astype(str)
    post_frame["panic_token_count"] = post_frame["title"].map(
        lambda value: _term_count(value, PANIC_TERMS)
    )
    post_frame["excitement_token_count"] = post_frame["title"].map(
        lambda value: _term_count(value, EXCITEMENT_TERMS)
    )
    post_frame["emoji_count"] = post_frame["title"].map(
        lambda value: len(EMOJI_PATTERN.findall(value))
        + len(BRACKET_EMOJI_PATTERN.findall(value))
    )
    post_frame["excitement_face_count"] = post_frame["title"].map(
        lambda value: sum(tag in EXCITEMENT_FACE_TAGS for tag in BRACKET_TAG_PATTERN.findall(value))
    )
    post_frame["panic_face_count"] = post_frame["title"].map(
        lambda value: sum(tag in PANIC_FACE_TAGS for tag in BRACKET_TAG_PATTERN.findall(value))
    )
    post_frame["excitement_event_count"] = (
        post_frame["excitement_token_count"] + post_frame["excitement_face_count"]
    )
    post_frame["panic_event_count"] = (
        post_frame["panic_token_count"] + post_frame["panic_face_count"]
    )
    post_frame["exclamation_count"] = post_frame["title"].map(
        lambda value: len(EXCLAMATION_PATTERN.findall(value))
    )
    post_frame["question_count"] = post_frame["title"].map(
        lambda value: len(QUESTION_PATTERN.findall(value))
    )
    post_frame["directional_sign_count"] = post_frame["title"].map(
        lambda value: len(DIRECTIONAL_SIGN_PATTERN.findall(value))
    )
    post_frame["punctuation_run_count"] = post_frame["title"].map(
        lambda value: len(PUNCTUATION_RUN_PATTERN.findall(value))
    )
    post_frame["sarcasm_count"] = post_frame["title"].map(
        lambda value: _term_count(value, SARCASM_TERMS)
    )
    post_frame["expectation_token_count"] = post_frame["title"].map(
        lambda value: _term_count(value, EXPECTATION_TERMS)
    )
    post_frame["polarity_conflict_count"] = (
        post_frame["panic_token_count"].gt(0)
        & post_frame["excitement_token_count"].gt(0)
    ).astype(int)
    post_frame["rhetorical_extreme_count"] = (
        post_frame["question_count"].gt(0)
        & (
            post_frame["panic_event_count"].gt(0)
            | post_frame["excitement_event_count"].gt(0)
            | post_frame["sarcasm_count"].gt(0)
        )
    ).astype(int)
    post_frame["title_char_count"] = post_frame["title"].str.replace(
        r"\s+", "", regex=True
    ).str.len()
    post_frame["concise_extreme_count"] = (
        post_frame["title_char_count"].between(1, 14)
        & (
            post_frame["panic_event_count"].gt(0)
            | post_frame["excitement_event_count"].gt(0)
            | post_frame["sarcasm_count"].gt(0)
            | (
                post_frame[
                    [
                        "exclamation_count",
                        "question_count",
                        "directional_sign_count",
                    ]
                ].sum(axis=1)
                >= 2
            )
        )
    ).astype(int)
    post_frame["positive_expectation_count"] = (
        post_frame["expectation_token_count"].gt(0)
        & post_frame["excitement_event_count"].gt(0)
    ).astype(int)
    post_frame["negative_expectation_count"] = (
        post_frame["expectation_token_count"].gt(0)
        & post_frame["panic_event_count"].gt(0)
    ).astype(int)

    bar_times = bars["bar_end"].drop_duplicates().sort_values().to_numpy(dtype="datetime64[ns]")
    post_times = post_frame["publish_time"].to_numpy(dtype="datetime64[ns]")
    locations = np.searchsorted(bar_times, post_times, side="left")
    valid = locations < len(bar_times)
    mapped = post_frame.loc[valid].copy()
    mapped["bar_end"] = pd.to_datetime(bar_times[locations[valid]])
    return mapped


def build_intraday_features(
    dominant_bars: pd.DataFrame,
    forum_posts: pd.DataFrame,
    *,
    board_code: str,
    forum_window_bars: int = 4,
    baseline_window_days: int = 40,
    baseline_min_days: int = 20,
    volatility_window_bars: int = 80,
) -> pd.DataFrame:
    bars = dominant_bars.copy().sort_values("bar_end").reset_index(drop=True)
    bars["bar_end"] = pd.to_datetime(bars["bar_end"])
    bars["bar_start"] = bars["bar_end"] - pd.Timedelta(minutes=15)
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    posts = forum_posts[forum_posts["board_code"] == board_code].copy()
    if posts.empty:
        raise ValueError(f"No forum posts found for board {board_code}")

    mapped_posts = _map_posts_to_next_bar(posts, bars)
    aggregated = mapped_posts.groupby("bar_end", as_index=False).agg(
        post_count=("post_id", "nunique"),
        unique_user_count=("user_id", lambda values: values[values.ne("")].nunique()),
        panic_token_count=("panic_token_count", "sum"),
        excitement_token_count=("excitement_token_count", "sum"),
        emoji_count=("emoji_count", "sum"),
        excitement_face_count=("excitement_face_count", "sum"),
        panic_face_count=("panic_face_count", "sum"),
        excitement_event_count=("excitement_event_count", "sum"),
        panic_event_count=("panic_event_count", "sum"),
        exclamation_count=("exclamation_count", "sum"),
        question_count=("question_count", "sum"),
        directional_sign_count=("directional_sign_count", "sum"),
        punctuation_run_count=("punctuation_run_count", "sum"),
        sarcasm_count=("sarcasm_count", "sum"),
        polarity_conflict_count=("polarity_conflict_count", "sum"),
        rhetorical_extreme_count=("rhetorical_extreme_count", "sum"),
        concise_extreme_count=("concise_extreme_count", "sum"),
        expectation_token_count=("expectation_token_count", "sum"),
        positive_expectation_count=("positive_expectation_count", "sum"),
        negative_expectation_count=("negative_expectation_count", "sum"),
    )
    features = bars.merge(aggregated, on="bar_end", how="left", validate="one_to_one")
    count_columns = [
        "post_count",
        "unique_user_count",
        "panic_token_count",
        "excitement_token_count",
        "emoji_count",
        "excitement_face_count",
        "panic_face_count",
        "excitement_event_count",
        "panic_event_count",
        "exclamation_count",
        "question_count",
        "directional_sign_count",
        "punctuation_run_count",
        "sarcasm_count",
        "polarity_conflict_count",
        "rhetorical_extreme_count",
        "concise_extreme_count",
        "expectation_token_count",
        "positive_expectation_count",
        "negative_expectation_count",
    ]
    features[count_columns] = features[count_columns].fillna(0.0)

    for column in count_columns:
        features[f"{column}_60m"] = _rolling_within_day(
            features[column], features["trade_date"], forum_window_bars
        )
    denominator = features["post_count_60m"].clip(lower=1.0)
    features["activity_raw"] = (
        np.log1p(features["post_count_60m"])
        + 0.35 * np.log1p(features["unique_user_count_60m"])
        + 0.15 * np.log1p(features["emoji_count_60m"])
    )
    # Narrow financial terms drive the tradable signal. Bracketed face tags are
    # retained separately for diagnostics because their historical meanings are
    # less stable and broadened tags increased false positives in training.
    features["forum_panic_raw"] = features["panic_token_count_60m"] / denominator
    features["forum_excitement_raw"] = features["excitement_token_count_60m"] / denominator
    features["forum_balance_raw"] = (
        features["excitement_token_count_60m"] - features["panic_token_count_60m"]
    ) / denominator
    features["forum_panic_event_raw"] = features["panic_event_count_60m"] / denominator
    features["forum_excitement_event_raw"] = (
        features["excitement_event_count_60m"] / denominator
    )
    features["symbol_intensity_raw"] = (
        features["exclamation_count_60m"]
        + features["question_count_60m"]
        + features["directional_sign_count_60m"]
        + 2.0 * features["punctuation_run_count_60m"]
    ) / denominator
    features["sarcasm_raw"] = features["sarcasm_count_60m"] / denominator
    features["polarity_conflict_raw"] = (
        features["polarity_conflict_count_60m"] / denominator
    )
    features["rhetorical_extreme_raw"] = (
        features["rhetorical_extreme_count_60m"] / denominator
    )
    features["concise_extreme_raw"] = (
        features["concise_extreme_count_60m"] / denominator
    )
    features["expectation_raw"] = (
        features["expectation_token_count_60m"] / denominator
    )
    features["positive_expectation_raw"] = (
        features["positive_expectation_count_60m"] / denominator
    )
    features["negative_expectation_raw"] = (
        features["negative_expectation_count_60m"] / denominator
    )
    features["user_concentration_raw"] = (
        1.0 - features["unique_user_count_60m"] / denominator
    ).clip(lower=0.0, upper=1.0)
    features["cumulative_post_count"] = _cumulative_within_day(
        features["post_count"], features["trade_date"]
    )
    features["cumulative_unique_user_count"] = _cumulative_within_day(
        features["unique_user_count"], features["trade_date"]
    )
    features["cumulative_special_event_count"] = _cumulative_within_day(
        features["concise_extreme_count"]
        + features["sarcasm_count"]
        + features["polarity_conflict_count"]
        + features["rhetorical_extreme_count"],
        features["trade_date"],
    )
    features["cumulative_expectation_count"] = _cumulative_within_day(
        features["expectation_token_count"],
        features["trade_date"],
    )
    cumulative_denominator = features["cumulative_post_count"].clip(lower=1.0)
    features["attention_cumulative_raw"] = (
        np.log1p(features["cumulative_post_count"])
        + 0.35 * np.log1p(features["cumulative_unique_user_count"])
    )
    features["special_attention_cumulative_raw"] = (
        features["cumulative_special_event_count"] / cumulative_denominator
    )
    features["expectation_cumulative_raw"] = (
        features["cumulative_expectation_count"] / cumulative_denominator
    )

    current_denominator = features["post_count"].clip(lower=1.0)
    features["activity_current_raw"] = (
        np.log1p(features["post_count"])
        + 0.35 * np.log1p(features["unique_user_count"])
        + 0.15 * np.log1p(features["emoji_count"])
    )
    for name, numerator in [
        ("panic", "panic_token_count"),
        ("excitement", "excitement_token_count"),
        ("symbol", "current_symbol_event_count"),
        ("concise", "concise_extreme_count"),
        ("expectation", "expectation_token_count"),
        ("positive_expectation", "positive_expectation_count"),
        ("negative_expectation", "negative_expectation_count"),
    ]:
        if numerator == "current_symbol_event_count":
            values = (
                features["exclamation_count"]
                + features["question_count"]
                + features["directional_sign_count"]
                + 2.0 * features["punctuation_run_count"]
            )
        else:
            values = features[numerator]
        features[f"{name}_current_raw"] = values / current_denominator

    for name in [
        "activity",
        "panic",
        "excitement",
        "symbol",
        "concise",
        "expectation",
        "positive_expectation",
        "negative_expectation",
    ]:
        features[f"{name}_speed_raw"] = _difference_within_day(
            features[f"{name}_current_raw"],
            features["trade_date"],
        )
    features["reaction_speed_magnitude_raw"] = (
        features["panic_speed_raw"].abs()
        + features["excitement_speed_raw"].abs()
        + features["symbol_speed_raw"].abs()
        + features["concise_speed_raw"].abs()
    )

    features = features.copy()
    features["bar_slot"] = features.groupby("trade_date").cumcount()
    for source, target in [
        ("activity_raw", "activity_z"),
        ("forum_panic_raw", "forum_panic_z"),
        ("forum_excitement_raw", "forum_excitement_z"),
        ("forum_balance_raw", "forum_balance_z"),
        ("forum_panic_event_raw", "forum_panic_event_z"),
        ("forum_excitement_event_raw", "forum_excitement_event_z"),
        ("symbol_intensity_raw", "symbol_intensity_z"),
        ("sarcasm_raw", "sarcasm_z"),
        ("polarity_conflict_raw", "polarity_conflict_z"),
        ("rhetorical_extreme_raw", "rhetorical_extreme_z"),
        ("concise_extreme_raw", "concise_extreme_z"),
        ("expectation_raw", "expectation_z"),
        ("positive_expectation_raw", "positive_expectation_z"),
        ("negative_expectation_raw", "negative_expectation_z"),
        ("user_concentration_raw", "user_concentration_z"),
        ("attention_cumulative_raw", "attention_cumulative_z"),
        ("special_attention_cumulative_raw", "special_attention_cumulative_z"),
        ("expectation_cumulative_raw", "expectation_cumulative_z"),
        ("activity_speed_raw", "activity_speed_z"),
        ("panic_speed_raw", "panic_speed_z"),
        ("excitement_speed_raw", "excitement_speed_z"),
        ("symbol_speed_raw", "symbol_speed_z"),
        ("concise_speed_raw", "concise_speed_z"),
        ("expectation_speed_raw", "expectation_speed_z"),
        ("positive_expectation_speed_raw", "positive_expectation_speed_z"),
        ("negative_expectation_speed_raw", "negative_expectation_speed_z"),
        ("reaction_speed_magnitude_raw", "reaction_speed_magnitude_z"),
    ]:
        features[target] = _causal_slot_zscore(
            features[source],
            features["bar_slot"],
            window_days=baseline_window_days,
            min_days=baseline_min_days,
        )
    features = features.copy()
    features["attention_persistence_60m"] = _rolling_within_day(
        features["activity_z"].ge(0.75).fillna(False).astype(float),
        features["trade_date"],
        forum_window_bars,
    )
    features["special_attention_persistence_60m"] = _rolling_within_day(
        features[
            [
                "symbol_intensity_z",
                "sarcasm_z",
                "polarity_conflict_z",
                "rhetorical_extreme_z",
                "concise_extreme_z",
            ]
        ]
        .max(axis=1)
        .ge(0.75)
        .fillna(False)
        .astype(float),
        features["trade_date"],
        forum_window_bars,
    )

    grouped = features.groupby("trade_date", sort=False)
    features["day_open"] = grouped["open"].transform("first")
    features["day_return"] = features["close"] / features["day_open"] - 1.0
    features["bar_return"] = grouped["close"].pct_change()
    first_bar = features["bar_slot"].eq(0)
    features.loc[first_bar, "bar_return"] = (
        features.loc[first_bar, "close"] / features.loc[first_bar, "open"] - 1.0
    )
    features["momentum_30m"] = grouped["close"].pct_change(2)
    features["momentum_60m"] = grouped["close"].pct_change(4)
    features["market_panic_raw"] = -features["day_return"]
    features["market_panic_z"] = _causal_slot_zscore(
        features["market_panic_raw"],
        features["bar_slot"],
        window_days=baseline_window_days,
        min_days=baseline_min_days,
    )

    intraday_returns = features["bar_return"].fillna(0.0)
    features["realized_vol_annualized"] = (
        intraday_returns.rolling(
            volatility_window_bars,
            min_periods=min(32, volatility_window_bars),
        ).std(ddof=0)
        * np.sqrt(252.0 * 16.0)
    )
    volatility_history = features["realized_vol_annualized"].shift(1)
    features["realized_volatility_q20"] = volatility_history.rolling(
        16 * 120,
        min_periods=16 * 40,
    ).quantile(0.20)
    features["realized_volatility_q70"] = volatility_history.rolling(
        16 * 120,
        min_periods=16 * 40,
    ).quantile(0.70)
    features["moderate_volatility_regime"] = features[
        "realized_vol_annualized"
    ].between(
        features["realized_volatility_q20"],
        features["realized_volatility_q70"],
    )
    features["long_signal_score"] = (
        0.55 * features["forum_panic_z"]
        + 0.45 * features["market_panic_z"]
        - 0.25 * features["forum_excitement_z"]
    )
    features["short_signal_score"] = (
        0.55 * features["forum_excitement_z"]
        + 0.25 * features["activity_z"]
        - 0.20 * features["market_panic_z"].abs()
    )
    features["can_signal"] = features["bar_slot"].between(0, 14)
    return features.sort_values(["trade_date", "bar_end"]).reset_index(drop=True)
