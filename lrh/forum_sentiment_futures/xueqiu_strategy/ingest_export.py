from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lrh.forum_sentiment_futures.xueqiu_strategy.ingest import (
    append_canonical_csv,
    load_xueqiu_export,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize an authorized Xueqiu export into the local append-only store."
    )
    parser.add_argument("input_file")
    parser.add_argument("--authorization-metadata", required=True)
    parser.add_argument("--observed-at", default=None)
    parser.add_argument(
        "--output-file",
        default=str(
            Path(__file__).resolve().parent.parent
            / "data"
            / "xueqiu"
            / "authorized_posts.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    posts = load_xueqiu_export(
        args.input_file,
        authorization_path=args.authorization_metadata,
        default_observed_at=args.observed_at,
    )
    combined = append_canonical_csv(posts, args.output_file)
    result = {
        "input_rows": len(posts),
        "stored_rows": len(combined),
        "first_created_at": combined["created_at"].min().isoformat()
        if len(combined)
        else None,
        "last_created_at": combined["created_at"].max().isoformat()
        if len(combined)
        else None,
        "output_file": str(Path(args.output_file).resolve()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
