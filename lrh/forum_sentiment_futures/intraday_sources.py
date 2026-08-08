from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .data_loader import MarketDataLoader
from .engine import select_dominant_contract


SINA_MINUTE_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/=/"
    "InnerFuturesNewService.getFewMinLine"
)
INTRADAY_COLUMNS = [
    "ts_code",
    "bar_end",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "oi",
]
EXPECTED_15M_TIMES = {
    "09:45",
    "10:00",
    "10:15",
    "10:30",
    "10:45",
    "11:00",
    "11:15",
    "11:30",
    "13:15",
    "13:30",
    "13:45",
    "14:00",
    "14:15",
    "14:30",
    "14:45",
    "15:00",
}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def build_execution_contract_map(
    loader: MarketDataLoader,
    fut_code: str,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    bundle = loader.load_market_bundle(
        fut_code,
        start_date=pd.Timestamp(start_date).date().isoformat(),
        end_date=pd.Timestamp(end_date).date().isoformat(),
    )
    selections: list[dict] = []
    for trade_date, snapshot in bundle.market.groupby("trade_date", sort=True):
        selections.append(
            {
                "selection_date": pd.Timestamp(trade_date).normalize(),
                "target_contract": select_dominant_contract(snapshot),
            }
        )
    selected = pd.DataFrame(selections).sort_values("selection_date").reset_index(drop=True)
    if len(selected) < 2:
        raise ValueError(f"Not enough daily observations to build {fut_code} contract mapping")
    result = selected.iloc[:-1].copy()
    result["trade_date"] = selected["selection_date"].shift(-1).iloc[:-1].to_numpy()
    result["fut_code"] = fut_code
    return result[["fut_code", "selection_date", "trade_date", "target_contract"]]


def _normalize_sina_payload(payload: object, ts_code: str) -> pd.DataFrame:
    frame = pd.DataFrame(payload)
    if frame.empty:
        raise ValueError(f"Sina returned no intraday bars for {ts_code}")
    if {"d", "o", "h", "l", "c", "v", "p"}.issubset(frame.columns):
        frame = frame.rename(
            columns={
                "d": "bar_end",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "p": "oi",
            }
        )
    elif len(frame.columns) == 7:
        frame.columns = ["bar_end", "open", "high", "low", "close", "volume", "oi"]
    else:
        raise ValueError(f"Unexpected Sina intraday schema for {ts_code}: {list(frame.columns)}")

    frame["bar_end"] = pd.to_datetime(frame["bar_end"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "oi"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["ts_code"] = ts_code
    frame["trade_date"] = frame["bar_end"].dt.normalize()
    frame = frame.dropna(subset=["bar_end", "open", "high", "low", "close"])
    frame = frame.drop_duplicates(subset=["ts_code", "bar_end"], keep="last")
    return frame[INTRADAY_COLUMNS].sort_values("bar_end").reset_index(drop=True)


def fetch_sina_contract_bars(
    ts_code: str,
    *,
    period_minutes: int = 15,
    throttle_seconds: float = 0.10,
    attempts: int = 4,
) -> pd.DataFrame:
    symbol = ts_code.split(".", 1)[0]
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                SINA_MINUTE_URL,
                params={"symbol": symbol, "type": str(period_minutes)},
                headers={"User-Agent": USER_AGENT, "Referer": "https://finance.sina.com.cn/"},
                timeout=30,
            )
            response.raise_for_status()
            if "=(" not in response.text:
                raise ValueError("Unexpected Sina JSONP response")
            payload = json.loads(response.text.split("=(", 1)[1].rsplit(");", 1)[0])
            frame = _normalize_sina_payload(payload, ts_code)
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
            return frame
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch Sina intraday bars for {ts_code}: {last_error}")


def _read_cached_contract(path: Path, ts_code: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["bar_end", "trade_date"])
    missing = set(INTRADAY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Intraday cache {path.name} missing columns: {sorted(missing)}")
    frame = frame[INTRADAY_COLUMNS].copy()
    frame["ts_code"] = ts_code
    return frame


def load_or_fetch_sina_bars(
    contracts: Iterable[str],
    *,
    cache_dir: Path,
    period_minutes: int = 15,
    refresh: bool = False,
    max_workers: int = 4,
    throttle_seconds: float = 0.10,
) -> pd.DataFrame:
    requested = sorted(set(contracts))
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ts_code in requested:
        path = cache_dir / f"{ts_code.replace('.', '_')}_{period_minutes}m.csv"
        if path.exists() and not refresh:
            frames[ts_code] = _read_cached_contract(path, ts_code)
        else:
            missing.append(ts_code)

    errors: list[str] = []
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = {
                pool.submit(
                    fetch_sina_contract_bars,
                    ts_code,
                    period_minutes=period_minutes,
                    throttle_seconds=throttle_seconds,
                ): ts_code
                for ts_code in missing
            }
            total = len(futures)
            for completed, future in enumerate(as_completed(futures), start=1):
                ts_code = futures[future]
                try:
                    frame = future.result()
                    path = cache_dir / f"{ts_code.replace('.', '_')}_{period_minutes}m.csv"
                    frame.to_csv(path, index=False, encoding="utf-8-sig")
                    frames[ts_code] = frame
                except Exception as exc:  # noqa: BLE001 - report all missing contracts together.
                    errors.append(f"{ts_code}: {exc}")
                if completed % 20 == 0 or completed == total:
                    print(f"[intraday] fetched {completed}/{total} contracts")
    if errors:
        raise RuntimeError("Intraday contract fetch was incomplete: " + "; ".join(errors[:10]))
    if set(frames) != set(requested):
        absent = sorted(set(requested).difference(frames))
        raise RuntimeError(f"Intraday cache is incomplete: {absent[:10]}")
    return pd.concat([frames[code] for code in requested], ignore_index=True)


def assemble_dominant_intraday(
    contract_map: pd.DataFrame,
    contract_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = contract_map.copy()
    mapping["trade_date"] = pd.to_datetime(mapping["trade_date"]).dt.normalize()
    bars = contract_bars.copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars["bar_end"] = pd.to_datetime(bars["bar_end"])
    joined = mapping.merge(
        bars,
        left_on=["trade_date", "target_contract"],
        right_on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_many",
    )
    joined["bar_time"] = joined["bar_end"].dt.strftime("%H:%M")

    coverage_rows: list[dict] = []
    complete_dates: list[pd.Timestamp] = []
    for trade_date, frame in joined.groupby("trade_date", sort=True):
        observed_times = set(frame["bar_time"].dropna())
        missing_times = sorted(EXPECTED_15M_TIMES.difference(observed_times))
        extra_times = sorted(observed_times.difference(EXPECTED_15M_TIMES))
        complete = not missing_times
        if complete:
            complete_dates.append(pd.Timestamp(trade_date))
        coverage_rows.append(
            {
                "fut_code": frame["fut_code"].iloc[0],
                "trade_date": trade_date,
                "target_contract": frame["target_contract"].iloc[0],
                "observed_bars": len(observed_times.intersection(EXPECTED_15M_TIMES)),
                "expected_bars": len(EXPECTED_15M_TIMES),
                "complete": complete,
                "missing_times": ",".join(missing_times),
                "extra_times": ",".join(extra_times),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    result = joined[
        joined["trade_date"].isin(complete_dates) & joined["bar_time"].isin(EXPECTED_15M_TIMES)
    ].copy()
    result = result.sort_values(["trade_date", "bar_end"]).drop_duplicates(
        subset=["trade_date", "bar_end"], keep="last"
    )
    result["bar_number"] = result.groupby("trade_date").cumcount()
    return result.reset_index(drop=True), coverage
