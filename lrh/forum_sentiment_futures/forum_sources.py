from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


FORUM_POST_COLUMNS = [
    "board_code",
    "post_id",
    "publish_time",
    "date",
    "user_id",
    "title",
    "post_from",
    "origin_bar_code",
]
PAGE_SIZE = 80
BASE_URL = "https://gbapi.eastmoney.com/webarticlelist/api/Article/Articlelist"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _request_page(
    board: str,
    page: int,
    *,
    timeout_seconds: float = 30.0,
    throttle_seconds: float = 0.05,
    attempts: int = 4,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            params = {
                "code": board,
                "type": 0,
                "p": page,
                "ps": PAGE_SIZE,
                "sorttype": 0,
                "plat": "Web",
                "version": "2022",
                "product": "Guba",
            }
            response = requests.get(
                BASE_URL,
                params=params,
                headers={"User-Agent": USER_AGENT, "Referer": "https://guba.eastmoney.com/"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or "re" not in payload:
                raise ValueError("unexpected article_list payload")
            if throttle_seconds > 0:
                time.sleep(throttle_seconds)
            return payload
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {board} page {page}: {last_error}")


def _page_date_bounds(payload: dict) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    values = pd.to_datetime(
        [row.get("post_publish_time") for row in payload.get("re", [])],
        errors="coerce",
    )
    values = values[~pd.isna(values)]
    if len(values) == 0:
        return None, None
    return pd.Timestamp(values.min()), pd.Timestamp(values.max())


def _discover_last_page(
    board: str,
    start_date: pd.Timestamp,
    *,
    throttle_seconds: float,
) -> tuple[int, dict[int, dict]]:
    pages: dict[int, dict] = {
        1: _request_page(board, 1, throttle_seconds=throttle_seconds)
    }
    total_count = int(pages[1].get("count") or len(pages[1].get("re", [])))
    total_pages = max(1, math.ceil(total_count / PAGE_SIZE))
    low, high, candidate = 1, total_pages, 1

    while low <= high:
        mid = (low + high) // 2
        payload = pages.get(mid)
        if payload is None:
            payload = _request_page(board, mid, throttle_seconds=throttle_seconds)
            pages[mid] = payload
        _, page_max = _page_date_bounds(payload)
        if page_max is not None and page_max.normalize() >= start_date.normalize():
            candidate = mid
            low = mid + 1
        else:
            high = mid - 1

    # Fetch one boundary page as protection against occasional pinned/out-of-order posts.
    return min(total_pages, candidate + 1), pages


def _normalize_page(board: str, payload: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for post in payload.get("re", []):
        publish_time = pd.to_datetime(post.get("post_publish_time"), errors="coerce")
        post_id = post.get("post_id")
        if pd.isna(publish_time) or post_id is None:
            continue
        post_user = post.get("post_user") or {}
        post_guba = post.get("post_guba") or {}
        rows.append(
            {
                "board_code": board,
                "post_id": str(post_id),
                "publish_time": publish_time,
                "date": publish_time.normalize(),
                "user_id": str(post.get("user_id") or post_user.get("user_id") or ""),
                "title": str(post.get("post_title") or ""),
                "post_from": str(post.get("post_from") or ""),
                "origin_bar_code": str(
                    post_guba.get("stockbar_code") or post.get("stockbar_code") or ""
                ),
            }
        )
    return pd.DataFrame(rows, columns=FORUM_POST_COLUMNS)


def fetch_eastmoney_forum_posts(
    boards: Iterable[str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    max_workers: int = 8,
    throttle_seconds: float = 0.05,
) -> pd.DataFrame:
    requested = list(dict.fromkeys(boards))
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError("Forum start_date must not be after end_date")

    cached_pages: dict[tuple[str, int], dict] = {}
    page_jobs: list[tuple[str, int]] = []
    for board in requested:
        last_page, discovered = _discover_last_page(
            board,
            start_ts,
            throttle_seconds=throttle_seconds,
        )
        cached_pages.update({(board, page): payload for page, payload in discovered.items()})
        page_jobs.extend((board, page) for page in range(1, last_page + 1))
        print(f"[forum] {board}: pages 1-{last_page}")

    payloads = dict(cached_pages)
    pending = [job for job in page_jobs if job not in payloads]
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        future_map = {
            pool.submit(
                _request_page,
                board,
                page,
                throttle_seconds=throttle_seconds,
            ): (board, page)
            for board, page in pending
        }
        total = len(future_map)
        for completed, future in enumerate(as_completed(future_map), start=1):
            board, page = future_map[future]
            try:
                payloads[(board, page)] = future.result()
            except Exception as exc:  # noqa: BLE001 - fail after collecting all page errors.
                errors.append(f"{board} page {page}: {exc}")
            if completed % 50 == 0 or completed == total:
                print(f"[forum] fetched {completed}/{total} remaining pages")

    if errors:
        raise RuntimeError("Forum history fetch was incomplete: " + "; ".join(errors[:10]))

    frames = [_normalize_page(board, payloads[(board, page)]) for board, page in page_jobs]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No Eastmoney forum posts were fetched")
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["board_code", "post_id"], keep="last")
    result = result[result["date"].between(start_ts, end_ts)]
    return result.sort_values(["board_code", "publish_time", "post_id"]).reset_index(drop=True)


def load_or_fetch_forum_posts(
    boards: Iterable[str],
    *,
    cache_path: Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    refresh: bool = False,
    max_workers: int = 8,
    throttle_seconds: float = 0.05,
) -> pd.DataFrame:
    requested = set(dict.fromkeys(boards))
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    cached = pd.DataFrame(columns=FORUM_POST_COLUMNS)
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["publish_time", "date"])
        missing = set(FORUM_POST_COLUMNS).difference(cached.columns)
        if missing:
            raise ValueError(f"Forum post cache missing columns: {sorted(missing)}")
        cached = cached[FORUM_POST_COLUMNS].copy()

    covered = not refresh and not cached.empty
    for board in requested:
        board_cache = cached[cached["board_code"] == board]
        if board_cache.empty:
            covered = False
            break
        latest_required = end_ts - pd.Timedelta(days=14)
        if board_cache["date"].min() > start_ts or board_cache["date"].max() < latest_required:
            covered = False
            break

    if not covered:
        fetched = fetch_eastmoney_forum_posts(
            sorted(requested),
            start_date=start_ts,
            end_date=end_ts,
            max_workers=max_workers,
            throttle_seconds=throttle_seconds,
        )
        remaining = cached[~cached["board_code"].isin(requested)]
        frames = [frame for frame in (remaining, fetched) if not frame.empty]
        cached = pd.concat(frames, ignore_index=True)
        cached = cached.drop_duplicates(subset=["board_code", "post_id"], keep="last")
        cached = cached.sort_values(["board_code", "publish_time", "post_id"]).reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache_path, index=False, encoding="utf-8-sig")

    available = cached[cached["board_code"].isin(requested)].copy()
    available = available[available["date"].between(start_ts, end_ts)]
    if available.empty:
        raise RuntimeError("No requested forum posts are available in the requested date range")
    return available.reset_index(drop=True)
