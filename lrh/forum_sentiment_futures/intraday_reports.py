from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

from .intraday_engine import performance_metrics


RETURN_SPECS = {
    "hybrid_core_intraday": ("hybrid_return_net", "hybrid_trade_count"),
    "trend_core": ("core_return_net", "core_trade_count"),
    "intraday_overlay": ("intraday_overlay_return_net", "trade_count"),
    "original_daily_sentiment": ("original_daily_return", None),
    "continuous_futures_hold": ("futures_buy_hold_return", None),
}


def _metric_frame(
    frame: pd.DataFrame,
    return_column: str,
    trade_count_column: str | None,
) -> pd.DataFrame:
    result = frame.copy()
    result["strategy_return_net"] = pd.to_numeric(result[return_column], errors="coerce").fillna(0.0)
    result["trade_count"] = (
        pd.to_numeric(result[trade_count_column], errors="coerce").fillna(0).astype(int)
        if trade_count_column is not None
        else 0
    )
    return result


def build_intraday_summary(daily: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict] = []
    periods = {
        "full": pd.Series(True, index=daily.index),
        "train": daily["trade_date"].le(train_end),
        "test": daily["trade_date"].gt(train_end),
    }
    for fut_code, code_frame in daily.groupby("fut_code", sort=False):
        for period_name, mask in periods.items():
            period_frame = code_frame.loc[mask.reindex(code_frame.index, fill_value=False)].copy()
            if period_frame.empty:
                continue
            for strategy_name, (return_column, trade_count_column) in RETURN_SPECS.items():
                metrics = performance_metrics(
                    _metric_frame(period_frame, return_column, trade_count_column)
                )
                rows.append(
                    {
                        "fut_code": fut_code,
                        "period": period_name,
                        "strategy": strategy_name,
                        "start_date": period_frame["trade_date"].min().date().isoformat(),
                        "end_date": period_frame["trade_date"].max().date().isoformat(),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def build_intraday_annual(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    source = daily.copy()
    source["year"] = source["trade_date"].dt.year
    for (fut_code, year), frame in source.groupby(["fut_code", "year"], sort=False):
        row: dict = {
            "fut_code": fut_code,
            "year": int(year),
            "start_date": frame["trade_date"].min().date().isoformat(),
            "end_date": frame["trade_date"].max().date().isoformat(),
            "observations": len(frame),
            "hybrid_trade_count": int(frame["hybrid_trade_count"].sum()),
            "intraday_trade_count": int(frame["trade_count"].sum()),
            "intraday_active_days": int(frame["trade_count"].gt(0).sum()),
            "hybrid_transaction_cost": float(frame["hybrid_transaction_cost"].sum()),
            "core_transaction_cost": float(frame["core_transaction_cost"].sum()),
            "intraday_transaction_cost": float(
                frame["intraday_overlay_transaction_cost"].sum()
            ),
        }
        for strategy_name, (return_column, trade_count_column) in RETURN_SPECS.items():
            metrics = performance_metrics(
                _metric_frame(frame, return_column, trade_count_column)
            )
            prefix = strategy_name
            row[f"{prefix}_period_return"] = metrics["total_return"]
            row[f"{prefix}_annualized_return"] = metrics["annualized_return"]
            row[f"{prefix}_annualized_volatility"] = metrics["annualized_volatility"]
            row[f"{prefix}_max_drawdown"] = metrics["max_drawdown"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_intraday_qlib_reports(output_dir: Path, daily: pd.DataFrame) -> None:
    for fut_code, frame in daily.groupby("fut_code", sort=False):
        qlib_dir = output_dir / fut_code / "qlib_report"
        qlib_dir.mkdir(parents=True, exist_ok=True)
        report = frame.sort_values("trade_date").set_index("trade_date")[
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
        report.index.name = "date"
        report.to_csv(qlib_dir / "qlib_report.csv", encoding="utf-8-sig")
        (qlib_dir / "README.md").write_text(
            "Qlib-compatible daily report for the volatility-targeted trend core plus "
            "15-minute forum-sentiment overlay. return is gross hybrid return, bench is net "
            "continuous-futures holding return, cost includes core commission plus intraday "
            "commission/slippage, and turnover is total notional turnover.\n",
            encoding="utf-8",
        )


def write_intraday_dashboard(
    output_dir: Path,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
) -> None:
    figures: list[go.Figure] = []
    for fut_code, frame in daily.groupby("fut_code", sort=False):
        frame = frame.sort_values("trade_date").copy()
        fig = go.Figure()
        for return_column, label, color in [
            ("hybrid_return_net", "Hybrid: trend core + 15m forum", "#0f766e"),
            ("core_return_net", "Vol-target trend core", "#2563eb"),
            ("intraday_overlay_return_net", "15m forum overlay", "#9333ea"),
            ("original_daily_return", "Original daily sentiment", "#b45309"),
            ("futures_buy_hold_return", "Continuous futures hold", "#334155"),
        ]:
            nav = (1.0 + frame[return_column].fillna(0.0)).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=frame["trade_date"],
                    y=nav,
                    mode="lines",
                    name=label,
                    line={"color": color},
                )
            )
        fig.update_layout(
            title=f"{fut_code}: risk-controlled hybrid vs baselines",
            height=430,
            margin={"l": 48, "r": 24, "t": 56, "b": 44},
            legend={"orientation": "h", "y": 1.08},
        )
        figures.append(fig)

    figure_html = "\n".join(
        pio.to_html(fig, include_plotlyjs="cdn" if index == 0 else False, full_html=False)
        for index, fig in enumerate(figures)
    )
    summary_html = summary.to_html(index=False, float_format=lambda value: f"{value:.6f}")
    annual_columns = [
        "fut_code",
        "year",
        "hybrid_core_intraday_annualized_return",
        "hybrid_core_intraday_annualized_volatility",
        "hybrid_core_intraday_max_drawdown",
        "trend_core_annualized_return",
        "intraday_overlay_annualized_return",
        "original_daily_sentiment_annualized_return",
        "continuous_futures_hold_annualized_return",
        "intraday_trade_count",
    ]
    annual_html = annual[annual_columns].to_html(
        index=False, float_format=lambda value: f"{value:.6f}"
    )
    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Risk-Controlled Hybrid Futures Backtest</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
        h1, h2 {{ letter-spacing: 0; }}
        table {{ border-collapse: collapse; width: 100%; background: white; }}
        th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; font-size: 12px; text-align: right; }}
        th:first-child, td:first-child, td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
        .table-wrap {{ overflow-x: auto; border: 1px solid #d0d7de; background: white; margin-bottom: 28px; }}
      </style>
    </head>
    <body>
      <h1>Risk-Controlled Hybrid Futures Backtest</h1>
      <h2>Summary</h2>
      <div class="table-wrap">{summary_html}</div>
      <h2>Annual comparison</h2>
      <div class="table-wrap">{annual_html}</div>
      {figure_html}
    </body>
    </html>
    """
    (output_dir / "dashboard.html").write_text(html, encoding="utf-8")


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_results_markdown(
    output_dir: Path,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    coverage: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> None:
    full = summary[
        summary["period"].eq("full")
        & summary["strategy"].isin(
            [
                "hybrid_core_intraday",
                "original_daily_sentiment",
                "continuous_futures_hold",
            ]
        )
    ].pivot(index="fut_code", columns="strategy")
    full_rows: list[list[str]] = []
    for fut_code in full.index:
        full_rows.append(
            [
                fut_code,
                _percent(full.loc[fut_code, ("annualized_return", "hybrid_core_intraday")]),
                _percent(
                    full.loc[fut_code, ("annualized_volatility", "hybrid_core_intraday")]
                ),
                _percent(full.loc[fut_code, ("max_drawdown", "hybrid_core_intraday")]),
                _percent(
                    full.loc[fut_code, ("annualized_return", "original_daily_sentiment")]
                ),
                _percent(
                    full.loc[fut_code, ("annualized_return", "continuous_futures_hold")]
                ),
            ]
        )

    annual_rows: list[list[str]] = []
    for row in annual.sort_values(["fut_code", "year"]).itertuples(index=False):
        annual_rows.append(
            [
                row.fut_code,
                str(row.year),
                _percent(row.hybrid_core_intraday_annualized_return),
                _percent(row.original_daily_sentiment_annualized_return),
                _percent(row.continuous_futures_hold_annualized_return),
                _percent(row.hybrid_core_intraday_annualized_volatility),
                _percent(row.hybrid_core_intraday_max_drawdown),
            ]
        )

    coverage_rows: list[list[str]] = []
    for fut_code, frame in coverage.groupby("fut_code", sort=True):
        complete = int(frame["complete"].sum())
        coverage_rows.append(
            [fut_code, str(complete), str(len(frame)), _percent(complete / len(frame))]
        )
    eligible_codes = eligibility.loc[eligibility["eligible"], "fut_code"].tolist()
    eligible_text = ", ".join(eligible_codes) if eligible_codes else "无"

    report = f"""# 风控混合策略回测结果

回测区间为 2021-07-12 至 2026-07-10；IM 自 2022-07-25 起。训练期截至
2024-12-31，2025-01-02 之后为样本外。收益均已扣除交易成本。

## 全区间

{_markdown_table(
    ["品种", "混合年化", "混合波动", "混合最大回撤", "原日频年化", "期货持有年化"],
    full_rows,
)}

## 逐年比较

2021 和 2026 是不足一年的区间，表中按 252 个交易日折算年化。

{_markdown_table(
    ["品种", "年份", "混合年化", "原日频年化", "期货持有年化", "混合波动", "混合最大回撤"],
    annual_rows,
)}

## 数据覆盖

{_markdown_table(["品种", "完整15分钟交易日", "目标交易日", "覆盖率"], coverage_rows)}

盘中增强仅在训练样本达到至少 20 笔、训练 Sharpe 不低于 0.5 且净收益为正时部署。
本次达到门槛的品种：{eligible_text}。未部署品种的候选交易仍保留在
`candidate_trades.csv`，便于审计，正式成交记录在 `trades.csv`。

## 口径

- 趋势核心：20/60/120 日均线分别配合 5/10/20 日动量，至少两个周期为正；20 日实现波动率定仓，仓位上限 0.5，每 5 个交易日调仓。
- 盘中增强：论坛帖子映射到其发布后第一根 15 分钟 K 线，收盘确认信号，下一根 K 线开盘成交；只在 10:45-11:30 检查新出现的亢奋词。
- 成本：核心单边 1bp；盘中单边 1bp 手续费加 1bp 滑点；换月按平旧开新计双边换手。
- 期货持有：主力连续期货始终做多，包含换月成本，不是现货指数收益。
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")
