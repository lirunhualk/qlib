from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd


RANK_HISTORY_COLUMNS = ["date", "symbol", "rank", "new_fan_rate", "loyal_fan_rate"]


def _normalize_eastmoney_rank_history(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"时间", "排名", "新晋粉丝", "铁杆粉丝"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{symbol} rank history missing columns: {sorted(missing)}")

    df = raw.rename(
        columns={
            "时间": "date",
            "排名": "rank",
            "新晋粉丝": "new_fan_rate",
            "铁杆粉丝": "loyal_fan_rate",
        }
    )[["date", "rank", "new_fan_rate", "loyal_fan_rate"]].copy()
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["rank", "new_fan_rate", "loyal_fan_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "rank"])
    df = df[df["rank"] > 0]
    return df[RANK_HISTORY_COLUMNS].drop_duplicates(subset=["date", "symbol"]).sort_values(["symbol", "date"])


def fetch_eastmoney_rank_history(symbols: Iterable[str], *, throttle_seconds: float = 0.15) -> pd.DataFrame:
    import akshare as ak

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for symbol in dict.fromkeys(symbols):
        try:
            raw = ak.stock_hot_rank_detail_em(symbol)
            frames.append(_normalize_eastmoney_rank_history(raw, symbol))
        except Exception as exc:  # noqa: BLE001 - keep batch fetch resilient.
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
        time.sleep(throttle_seconds)

    if not frames:
        raise RuntimeError("No Eastmoney rank history was fetched. " + "; ".join(errors[:5]))

    result = pd.concat(frames, ignore_index=True)
    if errors:
        result.attrs["fetch_errors"] = errors
    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_or_fetch_rank_history(
    symbols: Iterable[str],
    *,
    cache_path: Path,
    refresh: bool = False,
    throttle_seconds: float = 0.15,
) -> pd.DataFrame:
    requested = set(dict.fromkeys(symbols))
    cached = pd.DataFrame(columns=RANK_HISTORY_COLUMNS)

    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        missing_cols = set(RANK_HISTORY_COLUMNS).difference(cached.columns)
        if missing_cols:
            raise ValueError(f"Sentiment cache missing columns: {sorted(missing_cols)}")
        cached = cached[RANK_HISTORY_COLUMNS].copy()

    cached_symbols = set(cached["symbol"].dropna().unique()) if not cached.empty else set()
    symbols_to_fetch = sorted(requested if refresh else requested.difference(cached_symbols))

    fetched = pd.DataFrame(columns=RANK_HISTORY_COLUMNS)
    if symbols_to_fetch:
        fetched = fetch_eastmoney_rank_history(symbols_to_fetch, throttle_seconds=throttle_seconds)

    concat_frames: list[pd.DataFrame]
    if refresh:
        concat_frames = [cached[~cached["symbol"].isin(requested)], fetched]
    else:
        concat_frames = [cached, fetched]
    concat_frames = [frame for frame in concat_frames if not frame.empty]
    combined = pd.concat(concat_frames, ignore_index=True) if concat_frames else pd.DataFrame(columns=RANK_HISTORY_COLUMNS)

    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["date", "symbol"], keep="last")
        combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_path, index=False, encoding="utf-8-sig")

    available = combined[combined["symbol"].isin(requested)].copy()
    if available.empty:
        raise RuntimeError("No requested symbols are available in sentiment cache or fetched data.")
    return available.reset_index(drop=True)
