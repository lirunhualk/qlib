from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrh.forum_sentiment_futures.config import (  # noqa: E402
    DEFAULT_CACHE_DIRNAME,
    DEFAULT_COST_BPS_PER_SIDE,
    DEFAULT_LOOKBACK_YEARS,
    DEFAULT_MIN_SENTIMENT_PERIODS,
    DEFAULT_OUTPUT_DIRNAME,
    DEFAULT_SENTIMENT_WINDOW,
    FUTURE_SPECS,
    build_strategy_specs,
)
from lrh.forum_sentiment_futures.data_loader import MarketDataLoader  # noqa: E402
from lrh.forum_sentiment_futures.db import get_engine  # noqa: E402
from lrh.forum_sentiment_futures.engine import BacktestConfig, run_signal_backtest  # noqa: E402
from lrh.forum_sentiment_futures.features import (  # noqa: E402
    build_basket_sentiment,
    build_forum_sentiment,
    build_signal_features,
)
from lrh.forum_sentiment_futures.forum_sources import load_or_fetch_forum_posts  # noqa: E402
from lrh.forum_sentiment_futures.reports import build_reports, write_dashboard, write_qlib_report  # noqa: E402
from lrh.forum_sentiment_futures.sentiment_sources import load_or_fetch_rank_history  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest forum-sentiment signals on CFFEX index futures.")
    parser.add_argument(
        "--codes",
        nargs="+",
        default=list(FUTURE_SPECS.keys()),
        choices=sorted(FUTURE_SPECS.keys()),
        help="Futures roots to backtest.",
    )
    parser.add_argument("--start-date", default=None, help="Inclusive start date, e.g. 2025-07-07.")
    parser.add_argument("--end-date", default=None, help="Inclusive end date, e.g. 2026-07-07.")
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=DEFAULT_LOOKBACK_YEARS,
        help="Rolling years used when --start-date is omitted.",
    )
    parser.add_argument("--dsn", default=None, help="Override SQLAlchemy DSN. Prefer ASHARE_DB_URL.")
    parser.add_argument(
        "--sentiment-source",
        choices=["forum-posts", "rank-history"],
        default="forum-posts",
        help="Forum posts provide multi-year history; rank history is limited to about one year.",
    )
    parser.add_argument("--refresh-sentiment", action="store_true", help="Refetch the selected sentiment source.")
    parser.add_argument("--sentiment-cache", default=None, help="CSV cache path for Eastmoney rank history.")
    parser.add_argument("--forum-post-cache", default=None, help="CSV cache path for Eastmoney forum posts.")
    parser.add_argument("--forum-workers", type=int, default=8, help="Concurrent forum page requests.")
    parser.add_argument("--output-dir", default=None, help="Directory for reports and CSV outputs.")
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    parser.add_argument("--sentiment-window", type=int, default=DEFAULT_SENTIMENT_WINDOW)
    parser.add_argument("--min-sentiment-periods", type=int, default=DEFAULT_MIN_SENTIMENT_PERIODS)
    parser.add_argument("--throttle-seconds", type=float, default=0.15)
    return parser.parse_args()


def _resolve_output_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).resolve()
    return (Path(__file__).resolve().parent / DEFAULT_OUTPUT_DIRNAME).resolve()


def _resolve_cache_path(raw: str | None) -> Path:
    if raw:
        return Path(raw).resolve()
    return (Path(__file__).resolve().parent / DEFAULT_CACHE_DIRNAME / "eastmoney_rank_history.csv").resolve()


def _resolve_forum_cache_path(raw: str | None) -> Path:
    if raw:
        return Path(raw).resolve()
    return (Path(__file__).resolve().parent / DEFAULT_CACHE_DIRNAME / "eastmoney_forum_posts.csv").resolve()


def _required_symbols(codes: list[str]) -> list[str]:
    symbols: list[str] = []
    for code in codes:
        symbols.extend(FUTURE_SPECS[code].sentiment_symbols)
    return sorted(set(symbols))


def _required_boards(codes: list[str]) -> list[str]:
    return sorted({FUTURE_SPECS[code].forum_board_code for code in codes})


def _resolve_date_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    end_date = pd.Timestamp(args.end_date).normalize() if args.end_date else pd.Timestamp.today().normalize()
    if args.start_date:
        start_date = pd.Timestamp(args.start_date).normalize()
    else:
        start_date = end_date - pd.DateOffset(years=args.lookback_years)
    if start_date > end_date:
        raise ValueError("start-date must not be after end-date")
    warmup_days = max(120, args.sentiment_window * 3)
    warmup_start = start_date - pd.Timedelta(days=warmup_days)
    return start_date, end_date, warmup_start


def _trim_and_rebase(
    daily: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    cost_bps_per_side: float,
) -> pd.DataFrame:
    result = daily[daily["trade_date"].between(start_date, end_date)].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("No backtest observations remain after applying the requested date range")
    if result["strategy"].iloc[0] == "always_long_continuous":
        contract_changed = result["target_contract"].ne(result["target_contract"].shift(1))
        result["turnover"] = 0.0
        result.loc[contract_changed, "turnover"] = 2.0
        result.loc[0, "turnover"] = 1.0
        result["transaction_cost"] = result["turnover"] * (cost_bps_per_side / 10000.0)
        result["strategy_return_net"] = result["strategy_return_gross"] - result["transaction_cost"]
        result["excess_return_net"] = result["strategy_return_net"] - result["benchmark_return"]
    result["nav"] = (1.0 + result["strategy_return_net"]).cumprod()
    result["benchmark_nav"] = (1.0 + result["benchmark_return"]).cumprod()
    result["relative_nav"] = result["nav"] / result["benchmark_nav"]
    return result


def _write_annual_comparison(output_dir: Path, annual_returns: pd.DataFrame) -> pd.DataFrame:
    strategy = annual_returns[annual_returns["strategy"] == "dual_forum_sentiment"].copy()
    strategy = strategy[
        [
            "fut_code",
            "year",
            "start_date",
            "end_date",
            "observations",
            "strategy_annualized_return",
            "strategy_return_net",
            "benchmark_return",
            "active_days",
            "transaction_cost",
        ]
    ].rename(
        columns={
            "strategy_annualized_return": "sentiment_strategy_annualized_return",
            "strategy_return_net": "sentiment_strategy_period_return",
            "benchmark_return": "spot_buy_hold_period_return",
            "active_days": "sentiment_active_days",
            "transaction_cost": "sentiment_transaction_cost",
        }
    )
    spot_annualized = annual_returns[
        annual_returns["strategy"] == "dual_forum_sentiment"
    ][["fut_code", "year", "benchmark_annualized_return"]].rename(
        columns={"benchmark_annualized_return": "spot_buy_hold_annualized_return"}
    )
    holding = annual_returns[annual_returns["strategy"] == "always_long_continuous"][
        ["fut_code", "year", "strategy_return_net", "strategy_annualized_return"]
    ].rename(
        columns={
            "strategy_return_net": "futures_buy_hold_period_return",
            "strategy_annualized_return": "futures_buy_hold_annualized_return",
        }
    )
    comparison = strategy.merge(holding, on=["fut_code", "year"], validate="one_to_one")
    comparison = comparison.merge(spot_annualized, on=["fut_code", "year"], validate="one_to_one")
    relative_period = (
        (1.0 + comparison["sentiment_strategy_period_return"])
        / (1.0 + comparison["futures_buy_hold_period_return"])
        - 1.0
    )
    comparison["relative_to_futures_period_return"] = relative_period
    comparison["relative_to_futures_annualized_return"] = (
        (1.0 + relative_period) ** (252.0 / comparison["observations"]) - 1.0
    )
    code_order = {code: idx for idx, code in enumerate(FUTURE_SPECS)}
    comparison["_code_order"] = comparison["fut_code"].map(code_order)
    comparison = comparison.sort_values(["_code_order", "year"]).drop(columns="_code_order")
    comparison.to_csv(output_dir / "annual_comparison.csv", index=False, encoding="utf-8-sig")
    return comparison


def main() -> int:
    args = parse_args()
    start_date, end_date, warmup_start = _resolve_date_window(args)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path: Path
    rank_history = pd.DataFrame()
    forum_posts = pd.DataFrame()
    if args.sentiment_source == "forum-posts":
        cache_path = _resolve_forum_cache_path(args.forum_post_cache)
        forum_posts = load_or_fetch_forum_posts(
            _required_boards(args.codes),
            cache_path=cache_path,
            start_date=warmup_start,
            end_date=end_date,
            refresh=args.refresh_sentiment,
            max_workers=args.forum_workers,
            throttle_seconds=args.throttle_seconds,
        )
    else:
        cache_path = _resolve_cache_path(args.sentiment_cache)
        rank_history = load_or_fetch_rank_history(
            _required_symbols(args.codes),
            cache_path=cache_path,
            refresh=args.refresh_sentiment,
            throttle_seconds=args.throttle_seconds,
        )

    engine = get_engine(args.dsn)
    loader = MarketDataLoader(engine)
    strategies = build_strategy_specs()
    config = BacktestConfig(cost_bps_per_side=args.cost_bps_per_side, signal_lag_days=1)

    all_daily: list[pd.DataFrame] = []
    all_signals: list[pd.DataFrame] = []
    all_sentiment: list[pd.DataFrame] = []

    for fut_code in args.codes:
        spec = FUTURE_SPECS[fut_code]
        print(f"\n[{fut_code}] loading market and building sentiment features")
        bundle = loader.load_market_bundle(
            fut_code,
            start_date=warmup_start,
            end_date=end_date,
        )
        if args.sentiment_source == "forum-posts":
            sentiment = build_forum_sentiment(
                forum_posts,
                fut_code=fut_code,
                board_code=spec.forum_board_code,
                window=args.sentiment_window,
                min_periods=args.min_sentiment_periods,
            )
        else:
            sentiment = build_basket_sentiment(
                rank_history,
                fut_code=fut_code,
                symbols=spec.sentiment_symbols,
                window=args.sentiment_window,
                min_periods=args.min_sentiment_periods,
            )
        signals = build_signal_features(
            sentiment,
            bundle.spot,
            fut_code=fut_code,
            market_window=args.sentiment_window,
            min_periods=args.min_sentiment_periods,
        )
        all_sentiment.append(sentiment[sentiment["date"].between(start_date, end_date)])
        all_signals.append(signals[signals["trade_date"].between(start_date, end_date)])

        for strategy in strategies:
            daily = run_signal_backtest(bundle.market, signals, strategy, config)
            all_daily.append(
                _trim_and_rebase(
                    daily,
                    start_date,
                    end_date,
                    cost_bps_per_side=args.cost_bps_per_side,
                )
            )

    daily_all = pd.concat(all_daily, ignore_index=True)
    signals_all = pd.concat(all_signals, ignore_index=True)
    sentiment_all = pd.concat(all_sentiment, ignore_index=True)
    reports = build_reports(daily_all)

    reports.summary.to_csv(output_dir / "summary_all.csv", index=False, encoding="utf-8-sig")
    reports.annual_returns.to_csv(output_dir / "annual_returns_all.csv", index=False, encoding="utf-8-sig")
    reports.daily_returns.to_csv(output_dir / "daily_returns_all.csv", index=False, encoding="utf-8-sig")
    signals_all.to_csv(output_dir / "signal_features_all.csv", index=False, encoding="utf-8-sig")
    sentiment_all.to_csv(output_dir / "basket_sentiment_all.csv", index=False, encoding="utf-8-sig")
    annual_comparison = _write_annual_comparison(output_dir, reports.annual_returns)

    for (fut_code, strategy), frame in reports.daily_returns.groupby(["fut_code", "strategy"], sort=False):
        strategy_dir = output_dir / fut_code / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(strategy_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")
        summary = reports.summary[(reports.summary["fut_code"] == fut_code) & (reports.summary["strategy"] == strategy)]
        summary.to_csv(strategy_dir / "summary.csv", index=False, encoding="utf-8-sig")
        write_qlib_report(strategy_dir, frame)

    write_dashboard(output_dir, reports)

    print("\nTop rows by annualized return:")
    print(
        reports.summary[
            [
                "fut_code",
                "strategy_name",
                "annualized_return",
                "max_drawdown",
                "active_days",
                "benchmark_annualized_return",
            ]
        ]
        .sort_values(["fut_code", "annualized_return"], ascending=[True, False])
        .to_string(index=False)
    )
    print(f"\nReports written to: {output_dir}")
    print(f"Sentiment source: {args.sentiment_source}")
    print(f"Sentiment cache: {cache_path}")
    print("\nDual forum sentiment vs continuous futures buy-and-hold (annualized):")
    print(
        annual_comparison[
            [
                "fut_code",
                "year",
                "sentiment_strategy_annualized_return",
                "futures_buy_hold_annualized_return",
                "relative_to_futures_annualized_return",
                "spot_buy_hold_annualized_return",
                "sentiment_active_days",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
