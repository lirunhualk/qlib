from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .engine import BacktestResult


def _date_str(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")


def _write_dashboard(result: BacktestResult, output_dir: Path) -> None:
    daily = result.daily.copy()
    annual = result.annual_returns.copy()
    daily["date_str"] = _date_str(daily["date"])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=("Equity", "Drawdown", "Calendar-year return"),
    )
    fig.add_trace(go.Scatter(x=daily["date_str"], y=daily["equity"], name="strategy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["date_str"], y=daily["bench_equity"], name=result.config.benchmark_code), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily["date_str"], y=daily["drawdown"], name="strategy drawdown"), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily["date_str"], y=daily["bench_drawdown"], name="benchmark drawdown"), row=2, col=1)
    fig.add_trace(go.Bar(x=annual["year"], y=annual["strategy_return"], name="strategy annual"), row=3, col=1)
    fig.add_trace(go.Bar(x=annual["year"], y=annual["benchmark_return"], name="benchmark annual"), row=3, col=1)
    fig.update_layout(
        title="smallcap_lowturn backtest",
        template="plotly_white",
        height=980,
        barmode="group",
        yaxis_tickformat=".2f",
        yaxis2_tickformat=".0%",
        yaxis3_tickformat=".0%",
    )
    fig.write_html(output_dir / "dashboard.html", include_plotlyjs="cdn")


def _write_qlib_report(result: BacktestResult, output_dir: Path) -> None:
    report_dir = output_dir / "qlib_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    qlib_report = result.daily[["date", "return", "bench", "cost", "turnover"]].copy()
    qlib_report["date"] = _date_str(qlib_report["date"])
    qlib_report.to_csv(report_dir / "qlib_report.csv", index=False)
    (report_dir / "README.md").write_text(
        "\n".join(
            [
                "# Qlib-Compatible Report",
                "",
                "Columns are daily `return`, `bench`, `cost`, and `turnover`.",
                "The HTML dashboard in the parent directory is rendered with Plotly as a CLI-safe fallback.",
            ]
        ),
        encoding="utf-8",
    )


def write_reports(result: BacktestResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    result.daily.to_csv(output / "daily_returns.csv", index=False)
    result.annual_returns.to_csv(output / "annual_returns.csv", index=False)
    pd.DataFrame([result.summary]).to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if not result.holdings.empty:
        result.holdings.to_csv(output / "holdings.csv", index=False)
    _write_qlib_report(result, output)
    _write_dashboard(result, output)

