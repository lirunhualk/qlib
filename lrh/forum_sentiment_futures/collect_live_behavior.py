from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrh.forum_sentiment_futures.config import FUTURE_SPECS
from lrh.forum_sentiment_futures.forum_sources import _normalize_page, _request_page


def collect_once(output_path: Path, pages: int = 2) -> dict:
    frames = []
    for spec in FUTURE_SPECS.values():
        for page in range(1, pages + 1):
            payload = _request_page(spec.forum_board_code, page, throttle_seconds=0.25)
            frame = _normalize_page(spec.forum_board_code, payload)
            frame["observed_at"] = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
            frames.append(frame)
    fresh = pd.concat(frames, ignore_index=True).drop_duplicates(["board_code", "post_id"])
    old = pd.read_csv(output_path, dtype={"post_id": str, "user_id": str}, parse_dates=["publish_time", "date", "observed_at"]) if output_path.exists() else fresh.iloc[:0]
    result = pd.concat([old, fresh], ignore_index=True).drop_duplicates(["board_code", "post_id"], keep="first")
    result = result.sort_values(["board_code", "publish_time", "post_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    result.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(output_path)
    return {
        "stored_rows": len(result), "new_rows": len(result) - len(old),
        "latest_publish_time": result.publish_time.max().isoformat() if len(result) else "",
        "polled_at": fresh.observed_at.max().isoformat(), "output_file": str(output_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect the first-observed versions of Eastmoney index-forum posts.")
    parser.add_argument("--output-file", default=str(Path(__file__).parent / "data/eastmoney_live/posts.csv"))
    parser.add_argument("--pages", type=int, default=2, choices=range(1, 6))
    parser.add_argument("--cycles", type=int, default=1, help="1 for one-shot; 0 for continuous collection")
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 30 or args.cycles < 0:
        raise ValueError("poll-seconds must be at least 30 and cycles nonnegative")
    completed = 0
    while args.cycles == 0 or completed < args.cycles:
        result = collect_once(Path(args.output_file), args.pages)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        completed += 1
        if args.cycles == 0 or completed < args.cycles:
            time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
