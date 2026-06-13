from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrh.factor_ic_study.config import DEFAULT_CONFIG
from lrh.factor_ic_study.data_loader import latest_stock_trade_date, load_raw_tables
from lrh.factor_ic_study.db import get_engine
from lrh.factor_ic_study.evaluation import evaluate
from lrh.factor_ic_study.factors import build_factor_panel
from lrh.factor_ic_study.reports import write_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily A-share factor IC and strategy-return study.")
    parser.add_argument("--start-date", type=int, default=DEFAULT_CONFIG.start_date)
    parser.add_argument("--test-start-date", type=int, default=DEFAULT_CONFIG.test_start_date)
    parser.add_argument("--end-date", type=int, default=None)
    parser.add_argument("--holdings-count", type=int, default=DEFAULT_CONFIG.holdings_count)
    parser.add_argument("--rebalance-interval", type=int, default=DEFAULT_CONFIG.rebalance_interval)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_CONFIG.cost_bps)
    parser.add_argument("--feature-set", choices=["core", "extended"], default=DEFAULT_CONFIG.feature_set)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "report"))
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    engine = get_engine()
    end_date = args.end_date or latest_stock_trade_date(engine)
    config = replace(
        DEFAULT_CONFIG,
        start_date=args.start_date,
        test_start_date=args.test_start_date,
        end_date=end_date,
        holdings_count=args.holdings_count,
        rebalance_interval=args.rebalance_interval,
        cost_bps=args.cost_bps,
        feature_set=args.feature_set,
    )
    raw_tables = load_raw_tables(engine, config.start_date, end_date, feature_set=config.feature_set)
    build = build_factor_panel(raw_tables, config)
    result = evaluate(build.panel, build.factor_columns, config)
    metadata = {
        **asdict(config),
        "end_date": end_date,
        "row_count": int(len(build.panel)),
        "factor_count": int(len(build.factor_columns)),
    }
    write_reports(result, build.factor_sources, args.output_dir, metadata)
    print("factor IC study complete")
    print(f"factor_count: {len(build.factor_columns)}")
    print(f"output_dir: {Path(args.output_dir).resolve()}")
    print("top correlations:")
    print(result.correlation_summary.to_string(index=False))
    print("top factors by abs IC:")
    print(
        result.factor_metrics.sort_values("abs_mean_ic", ascending=False)
        .head(10)[["factor", "mean_ic", "abs_mean_ic", "ic_ir", "chosen_ann", "spread_ann"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
