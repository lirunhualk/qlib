from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lrh.forum_sentiment_futures.config import FUTURE_SPECS
from lrh.forum_sentiment_futures.data_loader import MarketDataLoader
from lrh.forum_sentiment_futures.db import get_engine
from lrh.forum_sentiment_futures.intraday_engine import (
    IntradayStrategyConfig,
    run_intraday_backtest,
)
from lrh.forum_sentiment_futures.intraday_sources import (
    assemble_dominant_intraday,
    build_execution_contract_map,
    load_or_fetch_sina_bars,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.audit import (
    audit_backtest_data,
    write_audit_report,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.features import (
    augment_cross_product_features,
    build_xueqiu_intraday_features,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.ingest import (
    first_observed_post_snapshot,
    load_xueqiu_export,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.membership import (
    assign_posts_to_futures,
    load_index_membership,
)
from lrh.forum_sentiment_futures.xueqiu_strategy.signals import (
    XueqiuSignalConfig,
    apply_xueqiu_signal,
)


def _strategy_dir() -> Path:
    return Path(__file__).resolve().parent


def _project_dir() -> Path:
    return _strategy_dir().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest authorized point-in-time Xueqiu intraday sentiment."
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=list(FUTURE_SPECS),
        choices=sorted(FUTURE_SPECS),
    )
    parser.add_argument("--start-date", default="2021-07-26")
    parser.add_argument("--end-date", default="2026-07-24")
    parser.add_argument("--train-end-date", default="2024-12-31")
    parser.add_argument(
        "--posts-file",
        default=str(_project_dir() / "data" / "xueqiu" / "authorized_posts.csv"),
    )
    parser.add_argument(
        "--authorization-metadata",
        default=str(_project_dir() / "data" / "xueqiu" / "authorization.json"),
    )
    parser.add_argument(
        "--baseline-daily",
        default=str(
            _project_dir()
            / "insight_update_report"
            / "legacy"
            / "daily_returns.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(_project_dir() / "xueqiu_report")
    )
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--refresh-intraday", action="store_true")
    parser.add_argument("--intraday-workers", type=int, default=4)
    parser.add_argument("--throttle-seconds", type=float, default=0.10)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _load_baseline(path: Path, codes: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Baseline daily report not found: {path}")
    frame = pd.read_csv(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame[
        frame["fut_code"].isin(codes)
        & frame["trade_date"].between(start, end)
    ].copy()
    required = {
        "fut_code",
        "trade_date",
        "hybrid_return_net",
        "core_return_gross",
        "core_return_net",
        "core_transaction_cost",
        "core_turnover",
        "core_trade_count",
        "futures_buy_hold_return",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Baseline report missing columns: {sorted(missing)}")
    return frame


def _write_blocked(
    output_dir: Path,
    *,
    message: str,
    posts_file: Path,
    authorization_file: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked_by_data_audit",
        "message": message,
        "posts_file": str(posts_file),
        "authorization_metadata": str(authorization_file),
    }
    (output_dir / "run_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "RESULTS.md").write_text(
        f"""# 雪球盘中情绪策略

状态：**未生成收益**

原因：{message}

系统没有用东方财富数据、网络档案热门帖或当前点赞数替代雪球历史数据。
补齐授权的点时导出与授权元数据后，重新运行同一命令即可生成四品种回测。
""",
        encoding="utf-8",
    )


def _complete_overlay_calendar(
    daily: pd.DataFrame,
    baseline: pd.DataFrame,
    fut_code: str,
) -> pd.DataFrame:
    calendar = baseline[baseline["fut_code"].eq(fut_code)][
        ["fut_code", "trade_date"]
    ].drop_duplicates()
    if daily.empty:
        merged = calendar.copy()
    else:
        merged = calendar.merge(
            daily,
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
    for column, default in defaults.items():
        if column not in merged:
            merged[column] = default
        else:
            merged[column] = merged[column].fillna(default)
    return merged


def _performance(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return {
            "observations": 0,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "total_return": np.nan,
        }
    nav = (1.0 + values).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    annualized_return = float(
        (1.0 + total_return) ** (252.0 / len(values)) - 1.0
    )
    volatility = float(values.std(ddof=0) * np.sqrt(252.0))
    sharpe = (
        float(values.mean() / values.std(ddof=0) * np.sqrt(252.0))
        if values.std(ddof=0) > 0
        else 0.0
    )
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1.0
    return {
        "observations": len(values),
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "total_return": total_return,
    }


RETURN_COLUMNS = {
    "xueqiu_hybrid": "xueqiu_hybrid_return_net",
    "original_hybrid": "original_hybrid_return_net",
    "continuous_futures_hold": "futures_buy_hold_return",
}


def merge_xueqiu_overlay(baseline: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    keys = ["fut_code", "trade_date"]
    mapping = {
        "strategy_return_gross": "xueqiu_overlay_return_gross",
        "transaction_cost": "xueqiu_overlay_transaction_cost",
        "strategy_return_net": "xueqiu_overlay_return_net",
        "turnover": "xueqiu_overlay_turnover",
        "trade_count": "xueqiu_overlay_trade_count",
    }
    selected = overlay[keys + list(mapping)].rename(columns=mapping)
    daily = baseline.merge(selected, on=keys, how="left", validate="one_to_one")
    if daily[list(mapping.values())].isna().any().any():
        raise ValueError("Xueqiu overlay calendar is incomplete")
    daily["xueqiu_hybrid_return_gross"] = (
        daily["core_return_gross"] + daily["xueqiu_overlay_return_gross"]
    )
    daily["xueqiu_hybrid_transaction_cost"] = (
        daily["core_transaction_cost"] + daily["xueqiu_overlay_transaction_cost"]
    )
    daily["xueqiu_hybrid_return_net"] = (
        daily["xueqiu_hybrid_return_gross"] - daily["xueqiu_hybrid_transaction_cost"]
    )
    daily["xueqiu_hybrid_turnover"] = (
        daily["core_turnover"] + daily["xueqiu_overlay_turnover"]
    )
    daily["original_hybrid_return_net"] = daily["hybrid_return_net"]
    return daily.sort_values(keys).reset_index(drop=True)


def _build_comparison(
    daily: pd.DataFrame, train_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    annual_rows: list[dict] = []
    for fut_code, code_frame in daily.groupby("fut_code", sort=True):
        periods = {
            "full": code_frame,
            "train": code_frame[code_frame["trade_date"].le(train_end)],
            "out_of_sample": code_frame[code_frame["trade_date"].gt(train_end)],
        }
        for period, period_frame in periods.items():
            for strategy, column in RETURN_COLUMNS.items():
                summary_rows.append(
                    {
                        "fut_code": fut_code,
                        "period": period,
                        "strategy": strategy,
                        **_performance(period_frame[column]),
                    }
                )
        for year, year_frame in code_frame.groupby(
            code_frame["trade_date"].dt.year, sort=True
        ):
            for strategy, column in RETURN_COLUMNS.items():
                annual_rows.append(
                    {
                        "fut_code": fut_code,
                        "year": int(year),
                        "strategy": strategy,
                        **_performance(year_frame[column]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(annual_rows)


def _write_qlib_reports(output_dir: Path, daily: pd.DataFrame) -> None:
    for fut_code, frame in daily.groupby("fut_code", sort=True):
        qlib_dir = output_dir / fut_code / "qlib_report"
        qlib_dir.mkdir(parents=True, exist_ok=True)
        report = frame.sort_values("trade_date").set_index("trade_date")[
            [
                "xueqiu_hybrid_return_net",
                "futures_buy_hold_return",
                "xueqiu_hybrid_transaction_cost",
                "xueqiu_hybrid_turnover",
            ]
        ].rename(
            columns={
                "xueqiu_hybrid_return_net": "return",
                "futures_buy_hold_return": "bench",
                "xueqiu_hybrid_transaction_cost": "cost",
                "xueqiu_hybrid_turnover": "turnover",
            }
        )
        report.index.name = "date"
        report.to_csv(qlib_dir / "qlib_report.csv", encoding="utf-8-sig")
        (qlib_dir / "README.md").write_text(
            "Qlib-compatible daily report for the trend core plus the "
            "authorized Xueqiu intraday overlay. Orders execute at the next "
            "15-minute bar open; cost includes commission and slippage.\n",
            encoding="utf-8",
        )


def _percent(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.2%}"


def _write_results(
    output_dir: Path,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    full = summary[summary["period"].eq("full")]
    rows: list[str] = []
    for fut_code in sorted(daily["fut_code"].unique()):
        pivot = full[full["fut_code"].eq(fut_code)].set_index("strategy")
        xq = pivot.loc["xueqiu_hybrid"]
        original = pivot.loc["original_hybrid"]
        hold = pivot.loc["continuous_futures_hold"]
        rows.append(
            "| "
            + " | ".join(
                [
                    fut_code,
                    _percent(xq["annualized_return"]),
                    _percent(xq["annualized_volatility"]),
                    _percent(xq["max_drawdown"]),
                    _percent(original["annualized_return"]),
                    _percent(hold["annualized_return"]),
                ]
            )
            + " |"
        )
    report = """# 雪球盘中情绪策略回测

数据审计已通过。雪球信号只使用发帖时点可见的内容、帖子到达速度和作者广度，
不使用事后累计点赞/回复/浏览量。信号在 15 分钟 K 线收盘确认，下一根 K 线开盘成交。

| 品种 | 雪球混合年化 | 年化波动 | 最大回撤 | 原混合年化 | 期货持有年化 |
| --- | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(rows) + """

逐年结果见 `annual_comparison.csv`，训练期/样本外结果见
`comparison_summary.csv`，成交与成本见 `trades.csv` 和
`daily_returns.csv`。
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    summary.to_csv(
        output_dir / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    annual.to_csv(
        output_dir / "annual_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> int:
    args = parse_args()
    codes = list(args.codes)
    start = pd.Timestamp(args.start_date).normalize()
    end = pd.Timestamp(args.end_date).normalize()
    train_end = pd.Timestamp(args.train_end_date).normalize()
    if not start <= train_end < end:
        raise ValueError("train-end-date must fall inside the backtest range")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    posts_path = Path(args.posts_file)
    authorization_path = Path(args.authorization_metadata)
    baseline = _load_baseline(Path(args.baseline_daily), codes, start, end)
    calendar = baseline[["fut_code", "trade_date"]].drop_duplicates()

    if not posts_path.exists() or not authorization_path.exists():
        audit = audit_backtest_data(
            pd.DataFrame(),
            calendar,
            fut_codes=codes,
            start_date=start,
            end_date=end,
        )
        write_audit_report(audit, output_dir)
        missing = []
        if not posts_path.exists():
            missing.append(f"帖子文件不存在：{posts_path}")
        if not authorization_path.exists():
            missing.append(f"授权元数据不存在：{authorization_path}")
        message = "；".join(missing)
        _write_blocked(
            output_dir,
            message=message,
            posts_file=posts_path,
            authorization_file=authorization_path,
        )
        print(message)
        return 2

    posts = first_observed_post_snapshot(
        load_xueqiu_export(
            posts_path, authorization_path=authorization_path
        )
    )
    engine = get_engine(args.dsn)
    membership = load_index_membership(
        engine,
        fut_codes=codes,
        start_date=start,
        end_date=end,
    )
    mapped_posts, membership_coverage = assign_posts_to_futures(
        posts, membership, fut_codes=codes
    )
    membership_coverage.to_csv(
        output_dir / "membership_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit = audit_backtest_data(
        mapped_posts,
        calendar,
        fut_codes=codes,
        start_date=start,
        end_date=end,
    )
    write_audit_report(audit, output_dir)
    if not audit.eligible or args.audit_only:
        message = (
            "仅执行数据审计"
            if args.audit_only
            else "；".join(audit.reasons)
        )
        _write_blocked(
            output_dir,
            message=message,
            posts_file=posts_path,
            authorization_file=authorization_path,
        )
        print(message)
        return 0 if args.audit_only else 2

    warmup_start = start - pd.Timedelta(days=180)
    loader = MarketDataLoader(engine)
    mappings: dict[str, pd.DataFrame] = {}
    for fut_code in codes:
        mappings[fut_code] = build_execution_contract_map(
            loader,
            fut_code,
            start_date=warmup_start,
            end_date=end,
        )
    all_mappings = pd.concat(mappings.values(), ignore_index=True)
    contract_bars = load_or_fetch_sina_bars(
        all_mappings["target_contract"].unique(),
        cache_dir=_project_dir() / "data" / "sina_15m",
        period_minutes=15,
        refresh=args.refresh_intraday,
        max_workers=args.intraday_workers,
        throttle_seconds=args.throttle_seconds,
    )

    feature_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    for fut_code in codes:
        dominant_bars, coverage = assemble_dominant_intraday(
            mappings[fut_code], contract_bars
        )
        coverage_frames.append(coverage)
        feature_frames.append(
            build_xueqiu_intraday_features(
                dominant_bars,
                mapped_posts,
                fut_code=fut_code,
            )
        )
    all_features = augment_cross_product_features(
        pd.concat(feature_frames, ignore_index=True)
    )
    signal_config = XueqiuSignalConfig()
    all_features = apply_xueqiu_signal(all_features, signal_config)
    engine_config = IntradayStrategyConfig(
        max_holding_bars=2,
        stop_loss=0.004,
        take_profit=0.006,
        target_volatility=0.06,
        max_position_size=0.35,
        min_position_size=0.10,
        max_trades_per_day=1,
        cooldown_bars=2,
        daily_loss_limit=0.006,
        commission_bps_per_side=1.0,
        slippage_bps_per_side=1.0,
    )

    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for fut_code in codes:
        code_features = all_features[all_features["fut_code"].eq(fut_code)]
        overlay_daily, trades = run_intraday_backtest(
            code_features,
            engine_config,
            signal_column="xueqiu_signal",
        )
        daily_frames.append(
            _complete_overlay_calendar(overlay_daily, baseline, fut_code)
        )
        if not trades.empty:
            trade_frames.append(trades)
    overlay = pd.concat(daily_frames, ignore_index=True)
    trades_all = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame()
    )
    daily = merge_xueqiu_overlay(baseline, overlay)
    if not trades_all.empty:
        trades_all = trades_all[trades_all["trade_date"].between(start, end)].copy()

    summary, annual = _build_comparison(daily, train_end)
    daily.to_csv(
        output_dir / "daily_returns.csv", index=False, encoding="utf-8-sig"
    )
    trades_all.to_csv(
        output_dir / "trades.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(coverage_frames, ignore_index=True).to_csv(
        output_dir / "intraday_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostic_columns = [
        "fut_code",
        "trade_date",
        "bar_end",
        "target_contract",
        "post_count_60m",
        "unique_user_count_60m",
        "xq_content_coverage_60m",
        "xq_sentiment_z",
        "xq_sentiment_velocity_z",
        "activity_z",
        "market_panic_z",
        "xueqiu_signal",
        "xueqiu_signal_confidence",
        "xueqiu_signal_reason",
    ]
    all_features[diagnostic_columns].to_csv(
        output_dir / "signal_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "selected_params.json").write_text(
        json.dumps(
            {
                "signal": signal_config.to_dict(),
                "execution": engine_config.to_dict(),
                "parameter_selection": "fixed_before_backtest",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_results(output_dir, summary, annual, daily)
    _write_qlib_reports(output_dir, daily)
    (output_dir / "run_status.json").write_text(
        json.dumps(
            {"status": "completed", "data_audit_passed": True},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Backtest completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
