from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrh.forum_sentiment_futures.config import FUTURE_SPECS  # noqa: E402
from lrh.forum_sentiment_futures.data_loader import MarketDataLoader  # noqa: E402
from lrh.forum_sentiment_futures.db import get_engine  # noqa: E402
from lrh.forum_sentiment_futures.forum_sources import load_or_fetch_forum_posts  # noqa: E402
from lrh.forum_sentiment_futures.hybrid_core import (  # noqa: E402
    TrendCoreConfig,
    combine_core_and_intraday,
    run_trend_core,
)
from lrh.forum_sentiment_futures.intraday_engine import (  # noqa: E402
    IntradayStrategyConfig,
    performance_metrics,
    run_intraday_backtest,
)
from lrh.forum_sentiment_futures.intraday_features import build_intraday_features  # noqa: E402
from lrh.forum_sentiment_futures.intraday_reports import (  # noqa: E402
    build_intraday_annual,
    build_intraday_summary,
    write_intraday_dashboard,
    write_intraday_qlib_reports,
    write_results_markdown,
)
from lrh.forum_sentiment_futures.intraday_sources import (  # noqa: E402
    assemble_dominant_intraday,
    build_execution_contract_map,
    load_or_fetch_sina_bars,
)


OVERLAY_MIN_TRAIN_TRADES = 20
OVERLAY_MIN_TRAIN_SHARPE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest 15-minute forum sentiment signals with explicit volatility controls."
    )
    parser.add_argument("--codes", nargs="+", default=list(FUTURE_SPECS), choices=sorted(FUTURE_SPECS))
    parser.add_argument("--start-date", default="2021-07-10")
    parser.add_argument("--end-date", default="2026-07-10")
    parser.add_argument("--train-end-date", default="2024-12-31")
    parser.add_argument("--dsn", default=None, help="Override SQLAlchemy DSN; prefer ASHARE_DB_URL.")
    parser.add_argument("--period-minutes", type=int, default=15, choices=[15])
    parser.add_argument("--refresh-intraday", action="store_true")
    parser.add_argument("--refresh-forum", action="store_true")
    parser.add_argument("--intraday-workers", type=int, default=4)
    parser.add_argument("--forum-workers", type=int, default=6)
    parser.add_argument("--throttle-seconds", type=float, default=0.10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-daily", default=None)
    return parser.parse_args()


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_output_dir(raw: str | None) -> Path:
    return Path(raw).resolve() if raw else (_project_dir() / "intraday_report").resolve()


def _resolve_baseline_path(raw: str | None) -> Path:
    return (
        Path(raw).resolve()
        if raw
        else (_project_dir() / "report" / "daily_returns_all.csv").resolve()
    )


def _candidate_configs() -> list[IntradayStrategyConfig]:
    base = IntradayStrategyConfig(
        market_panic_threshold=0.35,
        long_score_threshold=0.65,
        short_score_threshold=0.65,
        target_volatility=0.06,
        max_position_size=0.50,
        min_position_size=0.10,
        commission_bps_per_side=1.0,
        slippage_bps_per_side=1.0,
        allow_long=False,
        allow_short=True,
        signal_start_slot=4,
        signal_end_slot=7,
        require_new_relevant_token=True,
        max_trades_per_day=1,
    )
    configs: list[IntradayStrategyConfig] = []
    for min_activity, max_activity in [(-0.5, 1.0), (0.0, 0.8)]:
        for excitement_threshold in [0.8, 1.2, 1.6]:
            for holding_bars in [1, 2]:
                for stop_loss, take_profit in [(0.004, 0.006), (0.006, 0.009)]:
                    configs.append(
                        replace(
                            base,
                            forum_excitement_threshold=excitement_threshold,
                            max_holding_bars=holding_bars,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            min_activity_z=min_activity,
                            max_activity_z=max_activity,
                        )
                    )
    return configs


def _complete_strategy_calendar(
    daily: pd.DataFrame,
    contract_map: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    calendar = contract_map[
        contract_map["trade_date"].between(start_date, end_date)
    ][["fut_code", "trade_date"]].drop_duplicates()
    if daily.empty:
        result = calendar.copy()
        result["data_complete"] = False
    else:
        result = calendar.merge(
            daily, on=["fut_code", "trade_date"], how="left", validate="one_to_one"
        )
        result["data_complete"] = result["nav"].notna()
    defaults = {
        "strategy": "intraday_forum_risk_controlled",
        "strategy_name": "Intraday forum sentiment, risk controlled",
        "strategy_return_gross": 0.0,
        "transaction_cost": 0.0,
        "strategy_return_net": 0.0,
        "benchmark_return": 0.0,
        "excess_return_net": 0.0,
        "turnover": 0.0,
        "trade_count": 0,
        "long_trades": 0,
        "short_trades": 0,
        "average_position_size": 0.0,
    }
    for column, value in defaults.items():
        if column not in result:
            result[column] = value
        else:
            result[column] = result[column].fillna(value)
    result["nav"] = (1.0 + result["strategy_return_net"]).cumprod()
    return result.sort_values("trade_date").reset_index(drop=True)


def _assess_overlay_training(
    daily: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
) -> dict:
    train = daily[daily["trade_date"].between(start_date, train_end)].copy()
    metrics = performance_metrics(train)
    eligible = (
        metrics["trade_count"] >= OVERLAY_MIN_TRAIN_TRADES
        and metrics["annualized_return"] > 0.0
        and metrics["sharpe"] >= OVERLAY_MIN_TRAIN_SHARPE
    )
    return {
        "eligible": eligible,
        "minimum_training_trades": OVERLAY_MIN_TRAIN_TRADES,
        "minimum_training_sharpe": OVERLAY_MIN_TRAIN_SHARPE,
        **metrics,
    }


def _disable_intraday_overlay(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    for column in [
        "strategy_return_gross",
        "transaction_cost",
        "strategy_return_net",
        "excess_return_net",
        "turnover",
        "trade_count",
        "long_trades",
        "short_trades",
        "average_position_size",
    ]:
        result[column] = 0
    result["nav"] = 1.0
    return result


def _search_parameters(
    features_by_code: dict[str, pd.DataFrame],
    mappings_by_code: dict[str, pd.DataFrame],
    core_by_code: dict[str, pd.DataFrame],
    *,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[IntradayStrategyConfig, pd.DataFrame]:
    rows: list[dict] = []
    candidates = _candidate_configs()
    for candidate_index, config in enumerate(candidates, start=1):
        portfolio_frames: list[pd.DataFrame] = []
        total_trades = 0
        candidate_trades = 0
        eligible_codes: list[str] = []
        for fut_code, features in features_by_code.items():
            train_features = features[features["trade_date"].between(start_date, train_end)]
            daily, _ = run_intraday_backtest(train_features, config)
            completed = _complete_strategy_calendar(
                daily,
                mappings_by_code[fut_code],
                start_date=start_date,
                end_date=train_end,
            )
            assessment = _assess_overlay_training(
                completed,
                start_date=start_date,
                train_end=train_end,
            )
            candidate_trades += int(assessment["trade_count"])
            if assessment["eligible"]:
                eligible_codes.append(fut_code)
                total_trades += int(assessment["trade_count"])
            else:
                completed = _disable_intraday_overlay(completed)
            core = core_by_code[fut_code][
                ["trade_date", "core_return_net"]
            ].copy()
            comparison = completed[["trade_date", "strategy_return_net"]].merge(
                core,
                on="trade_date",
                how="inner",
                validate="one_to_one",
            )
            comparison["hybrid_return_net"] = (
                comparison["core_return_net"] + comparison["strategy_return_net"]
            )
            portfolio_frames.append(comparison.assign(fut_code=fut_code))
        combined = pd.concat(portfolio_frames, ignore_index=True)
        portfolio = (
            combined.groupby("trade_date", as_index=False)[
                ["strategy_return_net", "core_return_net", "hybrid_return_net"]
            ].mean()
        )
        overlay_metrics = performance_metrics(portfolio)
        core_metrics = performance_metrics(
            portfolio.rename(columns={"strategy_return_net": "overlay_return"}).rename(
                columns={"core_return_net": "strategy_return_net"}
            )
        )
        hybrid_metrics = performance_metrics(
            portfolio.rename(columns={"strategy_return_net": "overlay_return"}).rename(
                columns={"hybrid_return_net": "strategy_return_net"}
            )
        )
        incremental_return = (
            hybrid_metrics["annualized_return"] - core_metrics["annualized_return"]
        )
        yearly_positive = 0
        for year in range(start_date.year + 1, train_end.year + 1):
            year_frame = portfolio[portfolio["trade_date"].dt.year.eq(year)]
            if len(year_frame) < 100:
                continue
            year_metrics = performance_metrics(
                year_frame.rename(columns={"strategy_return_net": "overlay_return"}).rename(
                    columns={"hybrid_return_net": "strategy_return_net"}
                )
            )
            yearly_positive += int(year_metrics["annualized_return"] > 0.0)
        feasible = (
            total_trades >= OVERLAY_MIN_TRAIN_TRADES
            and hybrid_metrics["annualized_volatility"] <= 0.10
            and hybrid_metrics["max_drawdown"] >= -0.12
            and incremental_return > 0.0
        )
        score = (
            incremental_return
            + 0.03 * overlay_metrics["sharpe"]
            - 0.25
            * max(
                0.0,
                hybrid_metrics["annualized_volatility"]
                - core_metrics["annualized_volatility"],
            )
            + 0.002 * yearly_positive
        )
        rows.append(
            {
                "candidate": candidate_index,
                "feasible": feasible,
                "score": score,
                "total_trades": total_trades,
                "candidate_trades": candidate_trades,
                "eligible_codes": ",".join(eligible_codes),
                "incremental_annualized_return": incremental_return,
                "positive_training_years": yearly_positive,
                "overlay_annualized_return": overlay_metrics["annualized_return"],
                "overlay_annualized_volatility": overlay_metrics["annualized_volatility"],
                "overlay_max_drawdown": overlay_metrics["max_drawdown"],
                "overlay_sharpe": overlay_metrics["sharpe"],
                **hybrid_metrics,
                **config.to_dict(),
            }
        )
        print(
            f"[search] {candidate_index}/{len(candidates)} "
            f"hybrid={hybrid_metrics['annualized_return']:.4f} "
            f"increment={incremental_return:.4f} "
            f"vol={hybrid_metrics['annualized_volatility']:.4f} "
            f"mdd={hybrid_metrics['max_drawdown']:.4f} trades={total_trades}"
        )
    search = pd.DataFrame(rows)
    eligible = search[search["feasible"]].copy()
    pool = (
        eligible
        if not eligible.empty
        else search[search["total_trades"] >= OVERLAY_MIN_TRAIN_TRADES].copy()
    )
    if pool.empty:
        raise RuntimeError("No intraday parameter candidate produced enough training trades")
    winner = pool.sort_values(
        ["score", "incremental_annualized_return", "overlay_sharpe"], ascending=False
    ).iloc[0]
    selected = candidates[int(winner["candidate"]) - 1]
    return selected, search.sort_values("score", ascending=False).reset_index(drop=True)


def _load_baselines(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Daily baseline report not found: {path}. Run run_backtest.py before the intraday study."
        )
    source = pd.read_csv(path, parse_dates=["trade_date"])
    original = source[source["strategy"] == "dual_forum_sentiment"][
        ["fut_code", "trade_date", "strategy_return_net"]
    ].rename(columns={"strategy_return_net": "original_daily_return"})
    holding = source[source["strategy"] == "always_long_continuous"][
        [
            "fut_code",
            "trade_date",
            "target_contract",
            "strategy_return_gross",
            "transaction_cost",
            "turnover",
            "strategy_return_net",
        ]
    ].rename(
        columns={
            "target_contract": "futures_target_contract",
            "strategy_return_gross": "futures_gross_return",
            "transaction_cost": "futures_hold_transaction_cost",
            "turnover": "futures_hold_turnover",
            "strategy_return_net": "futures_buy_hold_return",
        }
    )
    return original.merge(holding, on=["fut_code", "trade_date"], validate="one_to_one")


def _build_core_by_code(
    baselines: pd.DataFrame,
    config: TrendCoreConfig,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for fut_code, frame in baselines.groupby("fut_code", sort=False):
        core_input = frame[
            [
                "fut_code",
                "trade_date",
                "futures_target_contract",
                "futures_gross_return",
            ]
        ].rename(
            columns={
                "futures_target_contract": "target_contract",
                "futures_gross_return": "strategy_return_gross",
            }
        )
        result[fut_code] = run_trend_core(core_input, config)
    return result


def _merge_baselines(
    daily: pd.DataFrame,
    baselines: pd.DataFrame,
    core_by_code: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = daily.merge(baselines, on=["fut_code", "trade_date"], how="left", validate="one_to_one")
    core_columns = [
        "fut_code",
        "trade_date",
        "core_signal_price",
        "core_positive_horizons",
        "core_realized_volatility",
        "core_trend_positive",
        "core_rebalance",
        "core_desired_position",
        "core_position",
        "core_contract_roll",
        "core_turnover",
        "core_transaction_cost",
        "core_return_gross",
        "core_return_net",
        "core_trade_count",
        "core_nav",
    ]
    first_core = next(iter(core_by_code.values()))
    core_columns.extend(
        column
        for column in first_core.columns
        if column.startswith("core_ma_") or column.startswith("core_momentum_")
    )
    core = pd.concat(
        [frame[core_columns] for frame in core_by_code.values()], ignore_index=True
    )
    result = result.merge(core, on=["fut_code", "trade_date"], how="left", validate="one_to_one")
    missing = result[
        ["original_daily_return", "futures_buy_hold_return", "core_return_net"]
    ].isna().any(axis=1)
    if missing.any():
        examples = result.loc[missing, ["fut_code", "trade_date"]].head().to_dict("records")
        raise ValueError(f"Daily baseline or trend-core coverage is incomplete: {examples}")
    result = combine_core_and_intraday(result)
    result["benchmark_return"] = result["futures_buy_hold_return"]
    result["excess_return_net"] = result["hybrid_return_net"] - result["benchmark_return"]
    result["nav"] = result.groupby("fut_code")["strategy_return_net"].transform(
        lambda values: (1.0 + values).cumprod()
    )
    result["original_nav"] = result.groupby("fut_code")["original_daily_return"].transform(
        lambda values: (1.0 + values).cumprod()
    )
    result["benchmark_nav"] = result.groupby("fut_code")["futures_buy_hold_return"].transform(
        lambda values: (1.0 + values).cumprod()
    )
    return result


def main() -> int:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    train_end = pd.Timestamp(args.train_end_date).normalize()
    warmup_start = start_date - pd.Timedelta(days=180)
    if not (start_date <= train_end < end_date):
        raise ValueError("train-end-date must fall inside the requested backtest range")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines = _load_baselines(_resolve_baseline_path(args.baseline_daily))
    baselines = baselines[
        baselines["fut_code"].isin(args.codes)
        & baselines["trade_date"].between(start_date, end_date)
    ].copy()
    missing_baseline_codes = set(args.codes).difference(baselines["fut_code"].unique())
    if missing_baseline_codes:
        raise ValueError(f"Missing daily baselines for: {sorted(missing_baseline_codes)}")
    core_config = TrendCoreConfig()
    core_by_code = _build_core_by_code(baselines, core_config)

    data_dir = _project_dir() / "data"
    forum_posts = load_or_fetch_forum_posts(
        [FUTURE_SPECS[code].forum_board_code for code in args.codes],
        cache_path=data_dir / "eastmoney_forum_posts.csv",
        start_date=warmup_start,
        end_date=end_date,
        refresh=args.refresh_forum,
        max_workers=args.forum_workers,
        throttle_seconds=args.throttle_seconds,
    )

    loader = MarketDataLoader(get_engine(args.dsn))
    mappings_by_code: dict[str, pd.DataFrame] = {}
    for fut_code in args.codes:
        print(f"[{fut_code}] building previous-day dominant-contract mapping")
        mappings_by_code[fut_code] = build_execution_contract_map(
            loader,
            fut_code,
            start_date=warmup_start,
            end_date=end_date,
        )
    all_mappings = pd.concat(mappings_by_code.values(), ignore_index=True)
    contract_bars = load_or_fetch_sina_bars(
        all_mappings["target_contract"].unique(),
        cache_dir=data_dir / f"sina_{args.period_minutes}m",
        period_minutes=args.period_minutes,
        refresh=args.refresh_intraday,
        max_workers=args.intraday_workers,
        throttle_seconds=args.throttle_seconds,
    )

    features_by_code: dict[str, pd.DataFrame] = {}
    coverage_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    for fut_code in args.codes:
        dominant_bars, coverage = assemble_dominant_intraday(
            mappings_by_code[fut_code], contract_bars
        )
        coverage_frames.append(coverage)
        print(
            f"[{fut_code}] complete 15m days: {int(coverage['complete'].sum())}/"
            f"{len(coverage)}"
        )
        features = build_intraday_features(
            dominant_bars,
            forum_posts,
            board_code=FUTURE_SPECS[fut_code].forum_board_code,
        )
        features_by_code[fut_code] = features
        feature_frames.append(features)

    selected_config, parameter_search = _search_parameters(
        features_by_code,
        mappings_by_code,
        core_by_code,
        start_date=start_date,
        train_end=train_end,
    )
    print("[search] selected", json.dumps(selected_config.to_dict(), ensure_ascii=False))

    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    candidate_trade_frames: list[pd.DataFrame] = []
    eligibility_rows: list[dict] = []
    for fut_code in args.codes:
        daily, trades = run_intraday_backtest(features_by_code[fut_code], selected_config)
        daily = _complete_strategy_calendar(
            daily,
            mappings_by_code[fut_code],
            start_date=start_date,
            end_date=end_date,
        )
        assessment = _assess_overlay_training(
            daily,
            start_date=start_date,
            train_end=train_end,
        )
        eligibility_rows.append({"fut_code": fut_code, **assessment})
        daily["overlay_eligible"] = bool(assessment["eligible"])
        if not trades.empty:
            trades = trades[trades["trade_date"].between(start_date, end_date)].copy()
            trades["overlay_eligible"] = bool(assessment["eligible"])
            candidate_trade_frames.append(trades)
        if not assessment["eligible"]:
            daily = _disable_intraday_overlay(daily)
            daily["overlay_eligible"] = False
        daily_frames.append(daily)
        if assessment["eligible"] and not trades.empty:
            trade_frames.append(trades)
    daily_all = pd.concat(daily_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    candidate_trades_all = (
        pd.concat(candidate_trade_frames, ignore_index=True)
        if candidate_trade_frames
        else pd.DataFrame()
    )
    overlay_eligibility = pd.DataFrame(eligibility_rows)
    daily_all = _merge_baselines(daily_all, baselines, core_by_code)

    coverage_all = pd.concat(coverage_frames, ignore_index=True)
    coverage_all = coverage_all[coverage_all["trade_date"].between(start_date, end_date)]
    features_all = pd.concat(feature_frames, ignore_index=True)
    features_all = features_all[features_all["trade_date"].between(start_date, end_date)]
    summary = build_intraday_summary(daily_all, train_end)
    annual = build_intraday_annual(daily_all)

    daily_all.to_csv(output_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")
    trades_all.to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")
    candidate_trades_all.to_csv(
        output_dir / "candidate_trades.csv", index=False, encoding="utf-8-sig"
    )
    overlay_eligibility.to_csv(
        output_dir / "overlay_eligibility.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(output_dir / "annual_returns.csv", index=False, encoding="utf-8-sig")
    annual_comparison_columns = [
        "fut_code",
        "year",
        "start_date",
        "end_date",
        "observations",
        "hybrid_core_intraday_annualized_return",
        "hybrid_core_intraday_annualized_volatility",
        "hybrid_core_intraday_max_drawdown",
        "original_daily_sentiment_annualized_return",
        "continuous_futures_hold_annualized_return",
    ]
    annual[annual_comparison_columns].to_csv(
        output_dir / "annual_comparison.csv", index=False, encoding="utf-8-sig"
    )
    coverage_all.to_csv(output_dir / "data_coverage.csv", index=False, encoding="utf-8-sig")
    features_all.to_csv(output_dir / "signal_features.csv", index=False, encoding="utf-8-sig")
    parameter_search.to_csv(output_dir / "parameter_search.csv", index=False, encoding="utf-8-sig")
    (output_dir / "selected_params.json").write_text(
        json.dumps(
            {
                "intraday_overlay_config": selected_config.to_dict(),
                "trend_core_config": core_config.to_dict(),
                "train_start": start_date.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": (train_end + pd.Timedelta(days=1)).date().isoformat(),
                "test_end": end_date.date().isoformat(),
                "market_data_source": "Sina historical contract 15-minute bars",
                "forum_data_source": "Eastmoney index-forum post timestamps and titles",
                "hybrid_definition": "volatility-targeted trend core plus intraday forum overlay",
                "overlay_eligibility_rule": {
                    "minimum_training_trades": OVERLAY_MIN_TRAIN_TRADES,
                    "minimum_training_sharpe": OVERLAY_MIN_TRAIN_SHARPE,
                    "require_positive_training_return": True,
                },
                "overlay_eligible_codes": overlay_eligibility.loc[
                    overlay_eligibility["eligible"], "fut_code"
                ].tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_intraday_qlib_reports(output_dir, daily_all)
    write_intraday_dashboard(output_dir, daily_all, summary, annual)
    write_results_markdown(
        output_dir,
        summary,
        annual,
        coverage_all,
        overlay_eligibility,
    )

    columns = [
        "fut_code",
        "period",
        "strategy",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "sharpe",
        "trade_count",
    ]
    print("\nPerformance summary:")
    print(summary[columns].to_string(index=False))
    print(f"\nReports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
