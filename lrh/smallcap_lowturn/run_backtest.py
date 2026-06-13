from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrh.smallcap_lowturn.config import DEFAULT_CONFIG
from lrh.smallcap_lowturn.data_loader import latest_stock_trade_date, load_market_data
from lrh.smallcap_lowturn.db import get_engine
from lrh.smallcap_lowturn.engine import run_backtest
from lrh.smallcap_lowturn.reports import write_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the smallcap_lowturn A-share daily strategy backtest.")
    parser.add_argument("--start-date", type=int, default=DEFAULT_CONFIG.start_date)
    parser.add_argument("--test-start-date", type=int, default=DEFAULT_CONFIG.test_start_date)
    parser.add_argument("--end-date", type=int, default=None)
    parser.add_argument("--benchmark-code", default=DEFAULT_CONFIG.benchmark_code)
    parser.add_argument("--holdings-count", type=int, default=DEFAULT_CONFIG.holdings_count)
    parser.add_argument("--rebalance-interval", type=int, default=DEFAULT_CONFIG.rebalance_interval)
    parser.add_argument("--min-amt20", type=float, default=DEFAULT_CONFIG.min_amt20)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_CONFIG.cost_bps)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "report"),
        help="Directory for summary, daily returns, holdings, and report artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = get_engine()
    end_date = args.end_date or latest_stock_trade_date(engine)
    config = replace(
        DEFAULT_CONFIG,
        start_date=args.start_date,
        test_start_date=args.test_start_date,
        end_date=end_date,
        benchmark_code=args.benchmark_code,
        holdings_count=args.holdings_count,
        rebalance_interval=args.rebalance_interval,
        min_amt20=args.min_amt20,
        cost_bps=args.cost_bps,
    )
    market = load_market_data(
        engine,
        start_date=config.start_date,
        test_start_date=config.test_start_date,
        end_date=end_date,
        benchmark_code=config.benchmark_code,
    )
    result = run_backtest(market.stock, market.benchmark, config)
    write_reports(result, args.output_dir)

    summary = result.summary
    print("smallcap_lowturn backtest complete")
    print(f"range: {summary['start_date']} - {summary['end_date']} ({summary['days']} days)")
    print(f"annualized_return: {summary['annualized_return']:.4%}")
    print(f"max_drawdown: {summary['max_drawdown']:.4%}")
    print(f"sharpe: {summary['sharpe']:.4f}")
    print(f"benchmark_annualized_return: {summary['benchmark_annualized_return']:.4%}")
    print(f"output_dir: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

