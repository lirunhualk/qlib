from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrh.forum_sentiment_futures.behavior_signals import (  # noqa: E402
    BehaviorSignalConfig,
    apply_behavior_signal,
    augment_cross_board_features,
    build_behavior_candidates,
)
from lrh.forum_sentiment_futures.intraday_engine import (  # noqa: E402
    performance_metrics,
    run_intraday_backtest,
)


MIN_CODE_TRAIN_TRADES = 8
MIN_CODE_VALIDATION_TRADES = 2
MIN_CODE_TRAIN_SHARPE = 0.25
MIN_PORTFOLIO_TRAIN_TRADES = 20
MIN_PORTFOLIO_VALIDATION_TRADES = 5
DEVELOPMENT_END = pd.Timestamp("2023-12-31")
VALIDATION_START = pd.Timestamp("2024-01-01")


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test updated forum-behavior insights as a risk-controlled incremental "
            "overlay on the existing intraday hybrid strategy."
        )
    )
    parser.add_argument("--start-date", default="2021-07-24")
    parser.add_argument("--end-date", default="2026-07-24")
    parser.add_argument("--train-end-date", default="2024-12-31")
    parser.add_argument(
        "--base-daily",
        default=str(_project_dir() / "insight_update_report" / "legacy" / "daily_returns.csv"),
    )
    parser.add_argument(
        "--feature-path",
        default=str(_project_dir() / "insight_update_report" / "legacy" / "signal_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_project_dir() / "insight_update_report" / "behavior"),
    )
    parser.add_argument(
        "--reuse-search",
        action="store_true",
        help="Reuse candidate_search.csv in output-dir and rerun only the selected candidate.",
    )
    return parser.parse_args()


def _metric_frame(
    frame: pd.DataFrame,
    return_column: str,
    trade_count_column: str | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "strategy_return_net": pd.to_numeric(
                frame[return_column], errors="coerce"
            ).fillna(0.0)
        }
    )
    result["trade_count"] = (
        pd.to_numeric(frame[trade_count_column], errors="coerce").fillna(0).astype(int)
        if trade_count_column
        else 0
    )
    return result


def _metrics(
    frame: pd.DataFrame,
    return_column: str,
    trade_count_column: str | None = None,
) -> dict:
    return performance_metrics(_metric_frame(frame, return_column, trade_count_column))


def _complete_behavior_calendar(
    daily: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    calendar = base[["fut_code", "trade_date"]].drop_duplicates()
    columns = [
        "fut_code",
        "trade_date",
        "strategy_return_gross",
        "transaction_cost",
        "strategy_return_net",
        "turnover",
        "trade_count",
        "long_trades",
        "short_trades",
        "average_position_size",
    ]
    if daily.empty:
        result = calendar.copy()
    else:
        result = calendar.merge(
            daily[columns],
            on=["fut_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )
    defaults = {
        "strategy_return_gross": 0.0,
        "transaction_cost": 0.0,
        "strategy_return_net": 0.0,
        "turnover": 0.0,
        "trade_count": 0,
        "long_trades": 0,
        "short_trades": 0,
        "average_position_size": 0.0,
    }
    for column, value in defaults.items():
        if column not in result:
            result[column] = value
        result[column] = result[column].fillna(value)
    return result.sort_values("trade_date").reset_index(drop=True)


def _disable_behavior(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    for column in [
        "strategy_return_gross",
        "transaction_cost",
        "strategy_return_net",
        "turnover",
        "trade_count",
        "long_trades",
        "short_trades",
        "average_position_size",
    ]:
        result[column] = 0
    return result


def _period(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return frame[frame["trade_date"].between(start, end)].copy()


def _assess_code(
    daily: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
) -> dict:
    train = _period(daily, start_date, train_end)
    development = _period(daily, start_date, min(DEVELOPMENT_END, train_end))
    validation = _period(daily, VALIDATION_START, train_end)
    train_metrics = _metrics(train, "strategy_return_net", "trade_count")
    development_metrics = _metrics(
        development, "strategy_return_net", "trade_count"
    )
    validation_metrics = _metrics(validation, "strategy_return_net", "trade_count")
    eligible = (
        train_metrics["trade_count"] >= MIN_CODE_TRAIN_TRADES
        and validation_metrics["trade_count"] >= MIN_CODE_VALIDATION_TRADES
        and train_metrics["annualized_return"] > 0.0
        and validation_metrics["total_return"] > 0.0
        and train_metrics["sharpe"] >= MIN_CODE_TRAIN_SHARPE
    )
    return {
        "eligible": eligible,
        "train_trades": train_metrics["trade_count"],
        "train_annualized_return": train_metrics["annualized_return"],
        "train_sharpe": train_metrics["sharpe"],
        "development_trades": development_metrics["trade_count"],
        "development_total_return": development_metrics["total_return"],
        "validation_trades": validation_metrics["trade_count"],
        "validation_total_return": validation_metrics["total_return"],
        "validation_sharpe": validation_metrics["sharpe"],
    }


def _combine_base_and_behavior(
    base: pd.DataFrame,
    behavior: pd.DataFrame,
) -> pd.DataFrame:
    behavior_columns = {
        "strategy_return_gross": "behavior_return_gross",
        "transaction_cost": "behavior_transaction_cost",
        "strategy_return_net": "behavior_return_net",
        "turnover": "behavior_turnover",
        "trade_count": "behavior_trade_count",
        "long_trades": "behavior_long_trades",
        "short_trades": "behavior_short_trades",
        "average_position_size": "behavior_average_position_size",
    }
    renamed = behavior.rename(columns=behavior_columns)
    result = base.merge(
        renamed[
            ["fut_code", "trade_date", *behavior_columns.values()]
        ],
        on=["fut_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    for column in behavior_columns.values():
        result[column] = result[column].fillna(0.0)
    result["updated_hybrid_return_gross"] = (
        result["hybrid_return_gross"] + result["behavior_return_gross"]
    )
    result["updated_hybrid_transaction_cost"] = (
        result["hybrid_transaction_cost"] + result["behavior_transaction_cost"]
    )
    result["updated_hybrid_return_net"] = (
        result["hybrid_return_net"] + result["behavior_return_net"]
    )
    result["updated_hybrid_turnover"] = (
        result["hybrid_turnover"] + result["behavior_turnover"]
    )
    result["updated_hybrid_trade_count"] = (
        result["hybrid_trade_count"].fillna(0).astype(int)
        + result["behavior_trade_count"].fillna(0).astype(int)
    )
    result["updated_hybrid_nav"] = result.groupby("fut_code")[
        "updated_hybrid_return_net"
    ].transform(lambda values: (1.0 + values).cumprod())
    return result


def _portfolio(daily: pd.DataFrame) -> pd.DataFrame:
    return_columns = [
        "updated_hybrid_return_net",
        "hybrid_return_net",
        "core_return_net",
        "behavior_return_net",
        "original_daily_return",
        "futures_buy_hold_return",
    ]
    trade_columns = [
        "updated_hybrid_trade_count",
        "hybrid_trade_count",
        "core_trade_count",
        "behavior_trade_count",
    ]
    returns = daily.groupby("trade_date", as_index=False)[return_columns].mean()
    trades = daily.groupby("trade_date", as_index=False)[trade_columns].sum()
    return returns.merge(trades, on="trade_date", validate="one_to_one")


def _segment_comparison(
    portfolio: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    frame = _period(portfolio, start, end)
    base = _metrics(frame, "hybrid_return_net", "hybrid_trade_count")
    updated = _metrics(
        frame,
        "updated_hybrid_return_net",
        "updated_hybrid_trade_count",
    )
    behavior = _metrics(frame, "behavior_return_net", "behavior_trade_count")
    return {
        "observations": updated["observations"],
        "base_annualized_return": base["annualized_return"],
        "base_annualized_volatility": base["annualized_volatility"],
        "base_max_drawdown": base["max_drawdown"],
        "updated_annualized_return": updated["annualized_return"],
        "updated_annualized_volatility": updated["annualized_volatility"],
        "updated_max_drawdown": updated["max_drawdown"],
        "updated_sharpe": updated["sharpe"],
        "incremental_annualized_return": (
            updated["annualized_return"] - base["annualized_return"]
        ),
        "volatility_delta": (
            updated["annualized_volatility"] - base["annualized_volatility"]
        ),
        "max_drawdown_delta": updated["max_drawdown"] - base["max_drawdown"],
        "behavior_annualized_return": behavior["annualized_return"],
        "behavior_total_return": behavior["total_return"],
        "behavior_sharpe": behavior["sharpe"],
        "behavior_trade_count": behavior["trade_count"],
    }


def _prefix(values: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _run_behavior_candidate(
    features_by_code: dict[str, pd.DataFrame],
    base_by_code: dict[str, pd.DataFrame],
    config: BehaviorSignalConfig,
    *,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames: list[pd.DataFrame] = []
    eligibility_rows: list[dict] = []
    for fut_code, features in features_by_code.items():
        signaled = apply_behavior_signal(features, config)
        daily, _ = run_intraday_backtest(
            signaled,
            config.execution_config(),
            signal_column="behavior_signal",
        )
        completed = _complete_behavior_calendar(daily, base_by_code[fut_code])
        assessment = _assess_code(
            completed,
            start_date=start_date,
            train_end=train_end,
        )
        eligibility_rows.append({"fut_code": fut_code, **assessment})
        completed["behavior_eligible"] = bool(assessment["eligible"])
        if not assessment["eligible"]:
            completed = _disable_behavior(completed)
            completed["behavior_eligible"] = False
        daily_frames.append(
            _combine_base_and_behavior(base_by_code[fut_code], completed)
        )
    return pd.concat(daily_frames, ignore_index=True), pd.DataFrame(eligibility_rows)


def _search_candidates(
    features_by_code: dict[str, pd.DataFrame],
    base_by_code: dict[str, pd.DataFrame],
    *,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[BehaviorSignalConfig, pd.DataFrame, bool]:
    candidates = build_behavior_candidates()
    rows: list[dict] = []
    for candidate_id, config in enumerate(candidates, start=1):
        combined, eligibility = _run_behavior_candidate(
            features_by_code,
            base_by_code,
            config,
            start_date=start_date,
            train_end=train_end,
        )
        portfolio = _portfolio(combined)
        development = _segment_comparison(
            portfolio,
            start=start_date,
            end=min(DEVELOPMENT_END, train_end),
        )
        validation = _segment_comparison(
            portfolio,
            start=VALIDATION_START,
            end=train_end,
        )
        train = _segment_comparison(
            portfolio,
            start=start_date,
            end=train_end,
        )

        positive_years = 0
        for year in (2022, 2023, 2024):
            if year > train_end.year:
                continue
            year_metrics = _segment_comparison(
                portfolio,
                start=pd.Timestamp(f"{year}-01-01"),
                end=min(pd.Timestamp(f"{year}-12-31"), train_end),
            )
            positive_years += int(
                year_metrics["incremental_annualized_return"] > 0.0
            )

        feasible = (
            eligibility["eligible"].any()
            and train["behavior_trade_count"] >= MIN_PORTFOLIO_TRAIN_TRADES
            and validation["behavior_trade_count"] >= MIN_PORTFOLIO_VALIDATION_TRADES
            and development["incremental_annualized_return"] > 0.0
            and validation["incremental_annualized_return"] > 0.0
            and development["behavior_total_return"] > 0.0
            and validation["behavior_total_return"] > 0.0
            and train["volatility_delta"] <= 0.003
            and validation["volatility_delta"] <= 0.005
            and train["max_drawdown_delta"] >= -0.01
            and positive_years >= 2
        )
        conservative_sharpe = min(
            development["behavior_sharpe"],
            validation["behavior_sharpe"],
        )
        score = (
            2.0 * validation["incremental_annualized_return"]
            + development["incremental_annualized_return"]
            + train["incremental_annualized_return"]
            + 0.005 * conservative_sharpe
            - max(0.0, train["volatility_delta"])
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "feasible": feasible,
                "score": score,
                "eligible_codes": ",".join(
                    eligibility.loc[eligibility["eligible"], "fut_code"]
                ),
                "positive_training_years": positive_years,
                **config.to_dict(),
                **_prefix(development, "development"),
                **_prefix(validation, "validation"),
                **_prefix(train, "train"),
            }
        )
        print(
            f"[behavior-search] {candidate_id}/{len(candidates)} "
            f"{config.family} eligible="
            f"{rows[-1]['eligible_codes'] or '-'} "
            f"dev={development['incremental_annualized_return']:.4f} "
            f"val={validation['incremental_annualized_return']:.4f} "
            f"trades={train['behavior_trade_count']:.0f}"
        )

    search = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(
        drop=True
    )
    feasible = search[search["feasible"]]
    deployed = not feasible.empty
    winner = feasible.iloc[0] if deployed else search.iloc[0]
    selected = candidates[int(winner["candidate_id"]) - 1]
    return selected, search, deployed


def _run_selected(
    features_by_code: dict[str, pd.DataFrame],
    base_by_code: dict[str, pd.DataFrame],
    config: BehaviorSignalConfig,
    *,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
    deployed: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    candidate_trade_frames: list[pd.DataFrame] = []
    eligibility_rows: list[dict] = []
    signal_frames: list[pd.DataFrame] = []

    for fut_code, features in features_by_code.items():
        signaled = apply_behavior_signal(features, config)
        selected_signals = signaled[signaled["behavior_signal"].ne(0)].copy()
        if not selected_signals.empty:
            signal_frames.append(selected_signals)
        daily, trades = run_intraday_backtest(
            signaled,
            config.execution_config(),
            signal_column="behavior_signal",
        )
        completed = _complete_behavior_calendar(daily, base_by_code[fut_code])
        assessment = _assess_code(
            completed,
            start_date=start_date,
            train_end=train_end,
        )
        eligible = deployed and bool(assessment["eligible"])
        eligibility_rows.append(
            {"fut_code": fut_code, "deployed": eligible, **assessment}
        )
        if not trades.empty:
            trades = trades.copy()
            trades["behavior_eligible"] = eligible
            candidate_trade_frames.append(trades)
            if eligible:
                trade_frames.append(trades)
        if not eligible:
            completed = _disable_behavior(completed)
        completed["behavior_eligible"] = eligible
        daily_frames.append(
            _combine_base_and_behavior(base_by_code[fut_code], completed)
        )

    return (
        pd.concat(daily_frames, ignore_index=True),
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(),
        (
            pd.concat(candidate_trade_frames, ignore_index=True)
            if candidate_trade_frames
            else pd.DataFrame()
        ),
        pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame(),
        pd.DataFrame(eligibility_rows),
    )


def _comparison_row(
    frame: pd.DataFrame,
    *,
    fut_code: str,
    period: str,
) -> dict:
    updated = _metrics(
        frame, "updated_hybrid_return_net", "updated_hybrid_trade_count"
    )
    legacy = _metrics(frame, "hybrid_return_net", "hybrid_trade_count")
    core = _metrics(frame, "core_return_net", "core_trade_count")
    behavior = _metrics(frame, "behavior_return_net", "behavior_trade_count")
    holding = _metrics(frame, "futures_buy_hold_return")
    original = _metrics(frame, "original_daily_return")
    return {
        "fut_code": fut_code,
        "period": period,
        "start_date": frame["trade_date"].min().date().isoformat(),
        "end_date": frame["trade_date"].max().date().isoformat(),
        "observations": len(frame),
        "updated_annualized_return": updated["annualized_return"],
        "legacy_annualized_return": legacy["annualized_return"],
        "annualized_return_delta": (
            updated["annualized_return"] - legacy["annualized_return"]
        ),
        "updated_annualized_volatility": updated["annualized_volatility"],
        "legacy_annualized_volatility": legacy["annualized_volatility"],
        "annualized_volatility_delta": (
            updated["annualized_volatility"] - legacy["annualized_volatility"]
        ),
        "updated_max_drawdown": updated["max_drawdown"],
        "legacy_max_drawdown": legacy["max_drawdown"],
        "max_drawdown_delta": updated["max_drawdown"] - legacy["max_drawdown"],
        "updated_sharpe": updated["sharpe"],
        "legacy_sharpe": legacy["sharpe"],
        "behavior_annualized_return": behavior["annualized_return"],
        "behavior_trade_count": behavior["trade_count"],
        "trend_core_annualized_return": core["annualized_return"],
        "original_daily_annualized_return": original["annualized_return"],
        "continuous_hold_annualized_return": holding["annualized_return"],
        "continuous_hold_annualized_volatility": holding["annualized_volatility"],
        "continuous_hold_max_drawdown": holding["max_drawdown"],
    }


def _build_summary(
    daily: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    period_bounds = {
        "full": (start_date, end_date),
        "development": (start_date, min(DEVELOPMENT_END, train_end)),
        "validation": (VALIDATION_START, train_end),
        "train": (start_date, train_end),
        "test": (train_end + pd.Timedelta(days=1), end_date),
    }
    rows: list[dict] = []
    for fut_code, code_frame in daily.groupby("fut_code", sort=False):
        for period_name, (period_start, period_end) in period_bounds.items():
            frame = _period(code_frame, period_start, period_end)
            if not frame.empty:
                rows.append(
                    _comparison_row(
                        frame,
                        fut_code=fut_code,
                        period=period_name,
                    )
                )
    portfolio = _portfolio(daily)
    for period_name, (period_start, period_end) in period_bounds.items():
        frame = _period(portfolio, period_start, period_end)
        if not frame.empty:
            rows.append(
                _comparison_row(
                    frame,
                    fut_code="PORTFOLIO",
                    period=period_name,
                )
            )
    return pd.DataFrame(rows)


def _build_annual(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    inputs = [(code, frame) for code, frame in daily.groupby("fut_code", sort=False)]
    inputs.append(("PORTFOLIO", _portfolio(daily)))
    for fut_code, source in inputs:
        source = source.copy()
        source["year"] = source["trade_date"].dt.year
        for year, frame in source.groupby("year", sort=True):
            row = _comparison_row(
                frame,
                fut_code=fut_code,
                period=str(int(year)),
            )
            row["year"] = int(year)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_cost_sensitivity(
    daily: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    train_end: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    period_bounds = {
        "full": (start_date, end_date),
        "train": (start_date, train_end),
        "test": (train_end + pd.Timedelta(days=1), end_date),
    }
    rows: list[dict] = []
    for cost_multiplier in (0.0, 0.5, 1.0, 1.5):
        scenario = daily.copy()
        scenario["scenario_return"] = (
            scenario["hybrid_return_net"]
            + scenario["behavior_return_gross"]
            - cost_multiplier * scenario["behavior_transaction_cost"]
        )
        portfolio = scenario.groupby("trade_date", as_index=False)[
            ["scenario_return", "hybrid_return_net"]
        ].mean()
        for period_name, (period_start, period_end) in period_bounds.items():
            frame = _period(portfolio, period_start, period_end)
            candidate = _metrics(frame, "scenario_return")
            legacy = _metrics(frame, "hybrid_return_net")
            rows.append(
                {
                    "period": period_name,
                    "behavior_cost_multiplier": cost_multiplier,
                    "candidate_annualized_return": candidate["annualized_return"],
                    "legacy_annualized_return": legacy["annualized_return"],
                    "annualized_return_delta": (
                        candidate["annualized_return"]
                        - legacy["annualized_return"]
                    ),
                    "candidate_annualized_volatility": candidate[
                        "annualized_volatility"
                    ],
                    "candidate_max_drawdown": candidate["max_drawdown"],
                }
            )
    return pd.DataFrame(rows)


def _production_recommendation(summary: pd.DataFrame) -> tuple[bool, dict]:
    row = summary[
        summary["fut_code"].eq("PORTFOLIO") & summary["period"].eq("test")
    ].iloc[0]
    checks = {
        "positive_test_return_delta": bool(row["annualized_return_delta"] > 0.0),
        "test_volatility_delta_at_most_10bp": bool(
            row["annualized_volatility_delta"] <= 0.001
        ),
        "test_drawdown_not_worse_by_more_than_50bp": bool(
            row["max_drawdown_delta"] >= -0.005
        ),
    }
    return all(checks.values()), checks


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)) and not np.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_results(
    output_dir: Path,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    eligibility: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    config: BehaviorSignalConfig,
    *,
    training_selected: bool,
    production_recommended: bool,
    candidate_count: int,
) -> None:
    full_rows: list[list[str]] = []
    for row in summary[
        summary["period"].eq("full") & summary["fut_code"].ne("PORTFOLIO")
    ].itertuples(index=False):
        full_rows.append(
            [
                row.fut_code,
                _percent(row.updated_annualized_return),
                _percent(row.legacy_annualized_return),
                _percent(row.annualized_return_delta),
                _percent(row.updated_annualized_volatility),
                _percent(row.updated_max_drawdown),
                _percent(row.continuous_hold_annualized_return),
            ]
        )
    test_rows: list[list[str]] = []
    for row in summary[
        summary["period"].eq("test")
        & summary["fut_code"].isin(["IF", "IC", "IH", "IM", "PORTFOLIO"])
    ].itertuples(index=False):
        test_rows.append(
            [
                row.fut_code,
                _percent(row.updated_annualized_return),
                _percent(row.legacy_annualized_return),
                _percent(row.annualized_return_delta),
                _percent(row.annualized_volatility_delta),
                _percent(row.max_drawdown_delta),
                str(int(row.behavior_trade_count)),
            ]
        )
    annual_rows: list[list[str]] = []
    for row in annual[annual["fut_code"].ne("PORTFOLIO")].sort_values(
        ["fut_code", "year"]
    ).itertuples(index=False):
        annual_rows.append(
            [
                row.fut_code,
                str(row.year),
                _percent(row.updated_annualized_return),
                _percent(row.legacy_annualized_return),
                _percent(row.continuous_hold_annualized_return),
                _percent(row.updated_annualized_volatility),
                _percent(row.updated_max_drawdown),
            ]
        )
    eligible_codes = eligibility.loc[eligibility["deployed"], "fut_code"].tolist()
    training_text = (
        f"通过训练门槛并进入样本外测试，品种为 {', '.join(eligible_codes)}"
        if training_selected and eligible_codes
        else "未进入样本外测试；没有候选同时通过开发段和验证段门槛"
    )
    production_text = (
        "建议进入下一阶段仿真"
        if production_recommended
        else "不建议替换原策略；样本外验收未通过"
    )
    test_cost_rows: list[list[str]] = []
    for row in cost_sensitivity[cost_sensitivity["period"].eq("test")].itertuples(
        index=False
    ):
        test_cost_rows.append(
            [
                f"{row.behavior_cost_multiplier:.1f}x",
                _percent(row.candidate_annualized_return),
                _percent(row.annualized_return_delta),
                _percent(row.candidate_annualized_volatility),
                _percent(row.candidate_max_drawdown),
            ]
        )
    report = f"""# 论坛行为新洞见回测

本报告把原帖更新中可由现有历史数据验证的行为假设，作为增量盘中覆盖层叠加到原有
风控混合策略。参数选择仅使用 2024-12-31 及以前的数据，其中 2024 年单独作为
验证段；2025 年以后不参与选择。

选中候选：`{config.family}`。训练筛选状态：{training_text}。
生产结论：**{production_text}**。

## 全区间

{_markdown_table(
    ["品种", "更新年化", "原策略年化", "增量", "更新波动", "更新最大回撤", "期货持有年化"],
    full_rows,
)}

## 样本外

{_markdown_table(
    ["品种", "更新年化", "原策略年化", "年化增量", "波动变化", "回撤变化", "新信号笔数"],
    test_rows,
)}

## 成本敏感性

仅改变新增行为覆盖层的成本，原策略成本保持不变。`0.0x` 是新增覆盖层零成本，
`1.0x` 是单边 1bp 手续费加 1bp 滑点。

{_markdown_table(
    ["新增成本", "样本外候选年化", "相对原策略", "候选波动", "候选最大回撤"],
    test_cost_rows,
)}

## 逐年

不足一年的 2021、2026 仍按 252 个交易日折算年化。

{_markdown_table(
    ["品种", "年份", "更新年化", "原策略年化", "持有年化", "更新波动", "更新最大回撤"],
    annual_rows,
)}

## 选择规则

- 候选共 {candidate_count} 个，覆盖持续关注、情绪一阶速度、反应性/预期性文本、
  符号强度、反讽代理、作者集中度、多板块同步、波动分位和 14:45 行为。
- 单品种训练期至少 8 笔、2024 验证期至少 2 笔、训练 Sharpe 不低于 0.25。
- 组合层面要求开发段和验证段均有正增益，训练期至少 20 笔、验证期至少 5 笔，
  且波动和最大回撤不得显著恶化。
- 信号在 15 分钟 K 线收盘确认，下一根 K 线开盘成交；单边手续费 1bp、滑点
  1bp。新覆盖层目标波动 5%，单品种仓位上限 0.35。
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")


def _write_qlib_reports(
    output_dir: Path,
    daily: pd.DataFrame,
    *,
    production_recommended: bool,
) -> None:
    for fut_code, frame in daily.groupby("fut_code", sort=False):
        qlib_dir = output_dir / fut_code / "qlib_report"
        qlib_dir.mkdir(parents=True, exist_ok=True)
        candidate = frame.sort_values("trade_date").set_index("trade_date")[
            [
                "updated_hybrid_return_gross",
                "futures_buy_hold_return",
                "updated_hybrid_transaction_cost",
                "updated_hybrid_turnover",
            ]
        ].rename(
            columns={
                "updated_hybrid_return_gross": "return",
                "futures_buy_hold_return": "bench",
                "updated_hybrid_transaction_cost": "cost",
                "updated_hybrid_turnover": "turnover",
            }
        )
        candidate.index.name = "date"
        candidate.to_csv(
            qlib_dir / "candidate_qlib_report.csv", encoding="utf-8-sig"
        )
        if production_recommended:
            production = candidate
        else:
            production = frame.sort_values("trade_date").set_index("trade_date")[
                [
                    "hybrid_return_gross",
                    "futures_buy_hold_return",
                    "hybrid_transaction_cost",
                    "hybrid_turnover",
                ]
            ].rename(
                columns={
                    "hybrid_return_gross": "return",
                    "futures_buy_hold_return": "bench",
                    "hybrid_transaction_cost": "cost",
                    "hybrid_turnover": "turnover",
                }
            )
            production.index.name = "date"
        production.to_csv(qlib_dir / "qlib_report.csv", encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    train_end = pd.Timestamp(args.train_end_date).normalize()
    if not start_date <= DEVELOPMENT_END < train_end < end_date:
        raise ValueError(
            "Date range must contain development through 2023, validation in 2024, "
            "and an out-of-sample period after train-end-date"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.base_daily, parse_dates=["trade_date"])
    features = pd.read_csv(args.feature_path, parse_dates=["trade_date", "bar_end"])
    base = base[base["trade_date"].between(start_date, end_date)].copy()
    features = features[features["trade_date"].between(start_date, end_date)].copy()
    if base.empty or features.empty:
        raise ValueError("Base daily returns and signal features must cover the requested range")
    if base.duplicated(["fut_code", "trade_date"]).any():
        raise ValueError("Base daily returns contain duplicate code/date rows")

    features = augment_cross_board_features(features)
    codes = sorted(base["fut_code"].unique())
    missing_feature_codes = set(codes).difference(features["fut_code"].unique())
    if missing_feature_codes:
        raise ValueError(f"Signal features missing codes: {sorted(missing_feature_codes)}")
    base_by_code = {
        code: frame.sort_values("trade_date").reset_index(drop=True)
        for code, frame in base.groupby("fut_code", sort=False)
    }
    train_features_by_code = {
        code: frame[frame["trade_date"].le(train_end)]
        .sort_values(["trade_date", "bar_end"])
        .reset_index(drop=True)
        for code, frame in features.groupby("fut_code", sort=False)
    }
    train_base_by_code = {
        code: frame[frame["trade_date"].le(train_end)].reset_index(drop=True)
        for code, frame in base_by_code.items()
    }

    candidates = build_behavior_candidates()
    search_path = output_dir / "candidate_search.csv"
    if args.reuse_search:
        if not search_path.exists():
            raise FileNotFoundError(
                f"Cannot reuse search because the file does not exist: {search_path}"
            )
        search = pd.read_csv(search_path)
        feasible = search[search["feasible"].astype(str).str.lower().eq("true")]
        deployed = not feasible.empty
        winner = feasible.iloc[0] if deployed else search.iloc[0]
        selected = candidates[int(winner["candidate_id"]) - 1]
    else:
        selected, search, deployed = _search_candidates(
            train_features_by_code,
            train_base_by_code,
            start_date=start_date,
            train_end=train_end,
        )
    print(
        "[behavior-search] selected",
        json.dumps(_json_safe(selected.to_dict()), ensure_ascii=False),
        f"deployed={deployed}",
    )

    features_by_code = {
        code: frame.sort_values(["trade_date", "bar_end"]).reset_index(drop=True)
        for code, frame in features.groupby("fut_code", sort=False)
    }
    daily, trades, candidate_trades, selected_signals, eligibility = _run_selected(
        features_by_code,
        base_by_code,
        selected,
        start_date=start_date,
        train_end=train_end,
        deployed=deployed,
    )
    daily = daily.sort_values(["fut_code", "trade_date"]).reset_index(drop=True)
    finite_returns = np.isfinite(
        daily[
            [
                "updated_hybrid_return_net",
                "hybrid_return_net",
                "behavior_return_net",
            ]
        ].to_numpy(dtype=float)
    ).all()
    if not finite_returns:
        raise ValueError("Backtest produced non-finite daily returns")
    if (daily["updated_hybrid_transaction_cost"] < 0).any():
        raise ValueError("Backtest produced negative transaction costs")

    summary = _build_summary(
        daily,
        start_date=start_date,
        train_end=train_end,
        end_date=end_date,
    )
    annual = _build_annual(daily)
    cost_sensitivity = _build_cost_sensitivity(
        daily,
        start_date=start_date,
        train_end=train_end,
        end_date=end_date,
    )
    production_recommended, recommendation_checks = _production_recommendation(
        summary
    )
    daily["production_return_gross"] = np.where(
        production_recommended,
        daily["updated_hybrid_return_gross"],
        daily["hybrid_return_gross"],
    )
    daily["production_transaction_cost"] = np.where(
        production_recommended,
        daily["updated_hybrid_transaction_cost"],
        daily["hybrid_transaction_cost"],
    )
    daily["production_return_net"] = np.where(
        production_recommended,
        daily["updated_hybrid_return_net"],
        daily["hybrid_return_net"],
    )
    daily["production_turnover"] = np.where(
        production_recommended,
        daily["updated_hybrid_turnover"],
        daily["hybrid_turnover"],
    )
    daily["production_nav"] = daily.groupby("fut_code")[
        "production_return_net"
    ].transform(lambda values: (1.0 + values).cumprod())
    daily.to_csv(output_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")
    candidate_trades.to_csv(
        output_dir / "candidate_trades.csv", index=False, encoding="utf-8-sig"
    )
    selected_signals.to_csv(
        output_dir / "selected_signal_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    eligibility.to_csv(
        output_dir / "behavior_eligibility.csv", index=False, encoding="utf-8-sig"
    )
    search.to_csv(
        output_dir / "candidate_search.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        output_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig"
    )
    annual.to_csv(
        output_dir / "annual_comparison.csv", index=False, encoding="utf-8-sig"
    )
    cost_sensitivity.to_csv(
        output_dir / "cost_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    selection = {
        "training_selected_for_oos_test": deployed,
        "production_recommended": production_recommended,
        "production_recommendation_checks": recommendation_checks,
        "selected_behavior_config": selected.to_dict(),
        "execution_config": selected.execution_config().to_dict(),
        "selection_periods": {
            "development_end": DEVELOPMENT_END.date().isoformat(),
            "validation_start": VALIDATION_START.date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "test_start": (train_end + pd.Timedelta(days=1)).date().isoformat(),
        },
        "eligibility_thresholds": {
            "minimum_code_train_trades": MIN_CODE_TRAIN_TRADES,
            "minimum_code_validation_trades": MIN_CODE_VALIDATION_TRADES,
            "minimum_code_train_sharpe": MIN_CODE_TRAIN_SHARPE,
            "minimum_portfolio_train_trades": MIN_PORTFOLIO_TRAIN_TRADES,
            "minimum_portfolio_validation_trades": MIN_PORTFOLIO_VALIDATION_TRADES,
        },
    }
    (output_dir / "selected_params.json").write_text(
        json.dumps(_json_safe(selection), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_results(
        output_dir,
        summary,
        annual,
        eligibility,
        cost_sensitivity,
        selected,
        training_selected=deployed,
        production_recommended=production_recommended,
        candidate_count=len(candidates),
    )
    _write_qlib_reports(
        output_dir,
        daily,
        production_recommended=production_recommended,
    )

    print("\nUpdated strategy comparison:")
    print(
        summary[
            summary["period"].isin(["full", "test"])
        ][
            [
                "fut_code",
                "period",
                "updated_annualized_return",
                "legacy_annualized_return",
                "annualized_return_delta",
                "updated_annualized_volatility",
                "updated_max_drawdown",
                "behavior_trade_count",
            ]
        ].to_string(index=False)
    )
    print(f"\nReports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
