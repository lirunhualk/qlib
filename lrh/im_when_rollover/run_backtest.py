from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrh.im_when_rollover.config import (  # noqa: E402
    DEFAULT_COST_BPS_PER_SIDE,
    DEFAULT_LIQUID_TOP_N,
    DEFAULT_OUTPUT_DIRNAME,
    FUTURE_SPECS,
    build_strategy_specs,
)
from lrh.im_when_rollover.data_loader import MarketDataLoader  # noqa: E402
from lrh.im_when_rollover.db import get_engine  # noqa: E402
from lrh.im_when_rollover.engine import BacktestConfig, run_strategy_backtest  # noqa: E402
from lrh.im_when_rollover.reports import build_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest CFFEX index futures rollover carry strategies.")
    parser.add_argument(
        "--codes",
        nargs="+",
        default=list(FUTURE_SPECS.keys()),
        choices=sorted(FUTURE_SPECS.keys()),
        help="Futures underlyings to backtest.",
    )
    parser.add_argument("--start-date", default=None, help="Inclusive start date, e.g. 2015-01-01.")
    parser.add_argument("--end-date", default=None, help="Inclusive end date, e.g. 2026-04-10.")
    parser.add_argument("--dsn", default=None, help="Override MySQL DSN.")
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    parser.add_argument("--liquid-top-n", type=int, default=DEFAULT_LIQUID_TOP_N)
    parser.add_argument("--output-dir", default=None, help="Directory for CSV outputs.")
    parser.add_argument("--skip-daily", action="store_true", help="Do not persist daily return CSVs.")
    return parser.parse_args()


def resolve_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        return Path(raw_output_dir).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path(__file__).resolve().parent / DEFAULT_OUTPUT_DIRNAME / run_id).resolve()


def main() -> int:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = get_engine(args.dsn)
    loader = MarketDataLoader(engine)
    strategies = build_strategy_specs(liquid_top_n=args.liquid_top_n)
    config = BacktestConfig(cost_bps_per_side=args.cost_bps_per_side)

    all_summary: list[pd.DataFrame] = []
    all_annual: list[pd.DataFrame] = []
    all_rolls: list[pd.DataFrame] = []

    for fut_code in args.codes:
        bundle = loader.load_market_bundle(fut_code, start_date=args.start_date, end_date=args.end_date)
        code_dir = output_dir / fut_code
        code_dir.mkdir(parents=True, exist_ok=True)

        daily_frames: list[pd.DataFrame] = []
        for strategy in strategies:
            daily_frames.append(run_strategy_backtest(bundle.market, strategy, config))

        daily = pd.concat(daily_frames, ignore_index=True)
        reports = build_reports(daily, fut_code, bundle.spot_code)

        reports.summary.to_csv(code_dir / "summary.csv", index=False, encoding="utf-8-sig")
        reports.annual_returns.to_csv(code_dir / "annual_returns.csv", index=False, encoding="utf-8-sig")
        reports.roll_events.to_csv(code_dir / "roll_events.csv", index=False, encoding="utf-8-sig")
        bundle.market.to_csv(code_dir / "market_snapshot.csv", index=False, encoding="utf-8-sig")
        if not args.skip_daily:
            reports.daily_returns.to_csv(code_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")

        all_summary.append(reports.summary)
        all_annual.append(reports.annual_returns)
        all_rolls.append(reports.roll_events)

        top3 = reports.summary.nlargest(3, "annualized_excess_return")[
            ["strategy_name", "annualized_excess_return", "annualized_return", "roll_count"]
        ]
        print(f"\n[{fut_code}] top strategies by annualized excess return")
        print(top3.to_string(index=False))

    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_csv(output_dir / "summary_all.csv", index=False, encoding="utf-8-sig")
        pd.concat(all_annual, ignore_index=True).to_csv(
            output_dir / "annual_returns_all.csv", index=False, encoding="utf-8-sig"
        )
        pd.concat(all_rolls, ignore_index=True).to_csv(output_dir / "roll_events_all.csv", index=False, encoding="utf-8-sig")

    print(f"\nOutputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
