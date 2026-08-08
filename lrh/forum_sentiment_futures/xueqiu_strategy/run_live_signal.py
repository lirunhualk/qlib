from __future__ import annotations

import argparse
import json
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lrh.forum_sentiment_futures.config import FUTURE_SPECS
from lrh.forum_sentiment_futures.db import get_engine
from lrh.forum_sentiment_futures.xueqiu_strategy.features import (
    augment_cross_product_features,
    build_xueqiu_intraday_features,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.ingest import (
    first_observed_post_snapshot,
    load_xueqiu_export,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.membership import (
    assign_posts_to_futures,
    load_index_membership,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.signals import (
    XueqiuSignalConfig,
    apply_xueqiu_signal,
)


def _project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit one-shot intraday Xueqiu signals from authorized feed drops."
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=list(FUTURE_SPECS),
        choices=sorted(FUTURE_SPECS),
    )
    parser.add_argument(
        "--posts-file",
        default=str(_project_dir() / "data" / "xueqiu" / "live_posts.jsonl"),
    )
    parser.add_argument(
        "--authorization-metadata",
        default=str(_project_dir() / "data" / "xueqiu" / "authorization.json"),
    )
    parser.add_argument("--bars-file", required=True)
    parser.add_argument(
        "--output-file",
        default=str(_project_dir() / "data" / "xueqiu" / "latest_signal.json"),
    )
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--max-feed-lag-minutes", type=float, default=10.0)
    parser.add_argument("--max-bar-lag-minutes", type=float, default=20.0)
    parser.add_argument("--dsn", default=None)
    return parser.parse_args()


def _load_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
    else:
        frame = pd.read_csv(path)
    required = {
        "fut_code",
        "trade_date",
        "bar_end",
        "target_contract",
        "open",
        "high",
        "low",
        "close",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Live bar file missing columns: {sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    frame["bar_end"] = pd.to_datetime(frame["bar_end"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "oi"):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(
        subset=["fut_code", "trade_date", "bar_end", "open", "high", "low", "close"]
    ).sort_values(["fut_code", "bar_end"])


def _market_is_open(as_of: pd.Timestamp) -> bool:
    if as_of.weekday() >= 5:
        return False
    current = as_of.time()
    return time(9, 30) <= current <= time(11, 30) or time(13, 0) <= current <= time(15, 0)


def _file_lag_minutes(path: Path, as_of: pd.Timestamp) -> float:
    modified = (
        pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        .tz_convert("Asia/Shanghai")
        .tz_localize(None)
    )
    return max(0.0, float((as_of - modified).total_seconds() / 60.0))


def _position_size(realized_volatility: float, direction: int) -> float:
    if direction == 0 or not np.isfinite(realized_volatility) or realized_volatility <= 0:
        return 0.0
    size = float(np.clip(0.06 / realized_volatility, 0.10, 0.35))
    return direction * size


def main() -> int:
    args = parse_args()
    posts_path = Path(args.posts_file)
    authorization_path = Path(args.authorization_metadata)
    bars_path = Path(args.bars_file)
    output_path = Path(args.output_file)
    as_of = (
        pd.Timestamp(args.as_of)
        if args.as_of
        else pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    )
    if as_of.tzinfo is not None:
        as_of = as_of.tz_convert("Asia/Shanghai").tz_localize(None)

    posts = first_observed_post_snapshot(
        load_xueqiu_export(
            posts_path, authorization_path=authorization_path
        )
    )
    bars = _load_bars(bars_path)
    codes = list(args.codes)
    bars = bars[bars["fut_code"].isin(codes) & bars["bar_end"].le(as_of)]
    if bars.empty:
        raise ValueError("No completed market bars are available as of the requested time")
    if not posts["authorized_use"].all():
        raise ValueError("Live feed authorization metadata is not confirmed")
    if not posts["is_content_point_in_time"].all():
        raise ValueError("Live feed contains content without point-in-time snapshots")

    membership = load_index_membership(
        get_engine(args.dsn),
        fut_codes=codes,
        start_date=min(posts["created_at"].min(), bars["trade_date"].min()),
        end_date=as_of,
    )
    mapped_posts, _ = assign_posts_to_futures(
        posts[posts["created_at"].le(as_of)],
        membership,
        fut_codes=codes,
    )

    feature_frames: list[pd.DataFrame] = []
    for fut_code in codes:
        code_bars = bars[bars["fut_code"].eq(fut_code)]
        code_posts = mapped_posts[mapped_posts["fut_code"].eq(fut_code)]
        if code_bars["trade_date"].nunique() < 25 or code_posts.empty:
            continue
        feature_frames.append(
            build_xueqiu_intraday_features(
                code_bars,
                code_posts,
                fut_code=fut_code,
            )
        )
    if not feature_frames:
        raise ValueError("Insufficient warm-up bars or mapped posts for live signals")

    features = augment_cross_product_features(
        pd.concat(feature_frames, ignore_index=True)
    )
    features = apply_xueqiu_signal(features, XueqiuSignalConfig())
    latest = (
        features.sort_values("bar_end")
        .groupby("fut_code", as_index=False)
        .tail(1)
        .sort_values("fut_code")
    )

    market_open = _market_is_open(as_of)
    feed_file_lag = _file_lag_minutes(posts_path, as_of)
    bar_file_lag = _file_lag_minutes(bars_path, as_of)
    rows: list[dict] = []
    global_halts: list[str] = []
    if market_open and feed_file_lag > args.max_feed_lag_minutes:
        global_halts.append("xueqiu_feed_file_stale")
    if market_open and bar_file_lag > args.max_bar_lag_minutes:
        global_halts.append("market_bar_file_stale")

    for row in latest.itertuples(index=False):
        bar_lag = max(
            0.0, float((as_of - pd.Timestamp(row.bar_end)).total_seconds() / 60.0)
        )
        halts = list(global_halts)
        if market_open and bar_lag > args.max_bar_lag_minutes:
            halts.append("latest_completed_bar_stale")
        direction = int(row.xueqiu_signal) if not halts and market_open else 0
        if halts:
            status = "halted"
        elif not market_open:
            status = "market_closed"
        elif direction:
            status = "actionable"
        else:
            status = "healthy_no_signal"
        if halts:
            reason = ",".join(halts)
        elif not market_open:
            reason = "market_closed"
        else:
            reason = row.xueqiu_signal_reason
        rows.append(
            {
                "fut_code": row.fut_code,
                "status": status,
                "as_of": as_of.isoformat(),
                "signal_bar_end": pd.Timestamp(row.bar_end).isoformat(),
                "target_contract": row.target_contract,
                "direction": direction,
                "suggested_notional_position": _position_size(
                    float(row.realized_vol_annualized), direction
                ),
                "confidence": float(row.xueqiu_signal_confidence)
                if direction
                else 0.0,
                "reason": reason,
                "post_count_60m": float(row.post_count_60m),
                "unique_user_count_60m": float(row.unique_user_count_60m),
                "point_in_time_content_coverage_60m": float(
                    row.xq_content_coverage_60m
                ),
                "sentiment_z": float(row.xq_sentiment_z)
                if pd.notna(row.xq_sentiment_z)
                else None,
                "sentiment_velocity_z": float(row.xq_sentiment_velocity_z)
                if pd.notna(row.xq_sentiment_velocity_z)
                else None,
                "bar_lag_minutes": bar_lag,
                "feed_file_lag_minutes": feed_file_lag,
                "bar_file_lag_minutes": bar_file_lag,
            }
        )

    payload = {
        "generated_at": as_of.isoformat(),
        "execution_rule": "signal at completed 15m bar; execute at next bar open",
        "orders_submitted": False,
        "signals": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
