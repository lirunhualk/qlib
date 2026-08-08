from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

from .compat import qlib_style_risk_analysis


@dataclass(frozen=True)
class ReportBundle:
    summary: pd.DataFrame
    annual_returns: pd.DataFrame
    daily_returns: pd.DataFrame


def _year_fraction(dates: pd.Series) -> float:
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())
    years = (end - start).days / 365.25
    return max(years, 1 / 365.25)


def _annualize_period_return(period_return: float, observations: int) -> float:
    if observations <= 0 or period_return <= -1.0:
        return float("nan")
    return float((1.0 + period_return) ** (252.0 / observations) - 1.0)


def build_reports(daily_results: pd.DataFrame) -> ReportBundle:
    daily = daily_results.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily["year"] = daily["trade_date"].dt.year

    summary_rows: list[dict] = []
    annual_rows: list[dict] = []
    for (fut_code, strategy), frame in daily.groupby(["fut_code", "strategy"], sort=False):
        frame = frame.sort_values("trade_date").reset_index(drop=True)
        strat_ra = qlib_style_risk_analysis(frame["strategy_return_net"], mode="product")
        bench_ra = qlib_style_risk_analysis(frame["benchmark_return"], mode="product")
        excess_ra = qlib_style_risk_analysis(frame["excess_return_net"], mode="sum")
        years = _year_fraction(frame["trade_date"])
        active_days = int(frame["position"].ne(0).sum())
        long_days = int(frame["position"].gt(0).sum())
        short_days = int(frame["position"].lt(0).sum())
        win_rate = frame.loc[frame["position"].ne(0), "strategy_return_net"].gt(0).mean()

        summary_rows.append(
            {
                "fut_code": fut_code,
                "strategy": strategy,
                "strategy_name": frame["strategy_name"].iloc[0],
                "start_date": frame["trade_date"].min().date().isoformat(),
                "end_date": frame["trade_date"].max().date().isoformat(),
                "observations": len(frame),
                "active_days": active_days,
                "long_days": long_days,
                "short_days": short_days,
                "active_day_ratio": active_days / len(frame) if len(frame) else 0.0,
                "annual_turnover": frame["turnover"].sum() / years,
                "win_rate_active_days": float(win_rate) if pd.notna(win_rate) else 0.0,
                "total_return": frame["nav"].iloc[-1] - 1.0,
                "annualized_return": float(strat_ra.loc["annualized_return", "risk"]),
                "max_drawdown": float(strat_ra.loc["max_drawdown", "risk"]),
                "information_ratio": float(strat_ra.loc["information_ratio", "risk"]),
                "benchmark_total_return": frame["benchmark_nav"].iloc[-1] - 1.0,
                "benchmark_annualized_return": float(bench_ra.loc["annualized_return", "risk"]),
                "relative_total_return": frame["relative_nav"].iloc[-1] - 1.0,
                "annualized_excess_return": float(excess_ra.loc["annualized_return", "risk"]),
                "excess_max_drawdown": float(excess_ra.loc["max_drawdown", "risk"]),
                "total_transaction_cost": float(frame["transaction_cost"].sum()),
            }
        )

        for year, year_frame in frame.groupby("year"):
            observations = len(year_frame)
            strategy_period_return = (1.0 + year_frame["strategy_return_net"]).prod() - 1.0
            benchmark_period_return = (1.0 + year_frame["benchmark_return"]).prod() - 1.0
            relative_period_return = (
                (1.0 + strategy_period_return) / (1.0 + benchmark_period_return) - 1.0
            )
            annual_rows.append(
                {
                    "fut_code": fut_code,
                    "strategy": strategy,
                    "strategy_name": frame["strategy_name"].iloc[0],
                    "year": int(year),
                    "start_date": year_frame["trade_date"].min().date().isoformat(),
                    "end_date": year_frame["trade_date"].max().date().isoformat(),
                    "observations": observations,
                    "strategy_return_net": strategy_period_return,
                    "benchmark_return": benchmark_period_return,
                    "strategy_annualized_return": _annualize_period_return(
                        strategy_period_return, observations
                    ),
                    "benchmark_annualized_return": _annualize_period_return(
                        benchmark_period_return, observations
                    ),
                    "relative_period_return": relative_period_return,
                    "relative_annualized_return": _annualize_period_return(
                        relative_period_return, observations
                    ),
                    "excess_return_net": year_frame["excess_return_net"].sum(),
                    "active_days": int(year_frame["position"].ne(0).sum()),
                    "transaction_cost": float(year_frame["transaction_cost"].sum()),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["fut_code", "annualized_return"], ascending=[True, False]
    )
    annual_returns = pd.DataFrame(annual_rows).sort_values(["fut_code", "strategy", "year"])
    return ReportBundle(summary=summary, annual_returns=annual_returns, daily_returns=daily)


def build_qlib_report_frame(daily_frame: pd.DataFrame) -> pd.DataFrame:
    report = (
        daily_frame.sort_values("trade_date")
        .set_index("trade_date")[["strategy_return_gross", "benchmark_return", "transaction_cost", "turnover"]]
        .rename(
            columns={
                "strategy_return_gross": "return",
                "benchmark_return": "bench",
                "transaction_cost": "cost",
            }
        )
    )
    report.index.name = "date"
    return report


def write_qlib_report(strategy_dir: Path, daily_frame: pd.DataFrame) -> None:
    qlib_dir = strategy_dir / "qlib_report"
    qlib_dir.mkdir(parents=True, exist_ok=True)
    report = build_qlib_report_frame(daily_frame)
    report.to_csv(qlib_dir / "qlib_report.csv", encoding="utf-8-sig")
    (qlib_dir / "README.md").write_text(
        "Daily qlib-compatible report. return is gross intraday futures return; "
        "bench is spot-index close-to-close buy-and-hold return; cost is explicit transaction cost.\n",
        encoding="utf-8",
    )


def write_dashboard(output_dir: Path, reports: ReportBundle) -> None:
    summary_html = reports.summary.to_html(index=False, float_format=lambda x: f"{x:.6f}")

    figures: list[go.Figure] = []
    for fut_code, frame in reports.daily_returns.groupby("fut_code", sort=False):
        fig = go.Figure()
        for strategy, strat_frame in frame.groupby("strategy", sort=False):
            strat_frame = strat_frame.sort_values("trade_date")
            fig.add_trace(
                go.Scatter(
                    x=strat_frame["trade_date"],
                    y=strat_frame["nav"],
                    mode="lines",
                    name=strat_frame["strategy_name"].iloc[0],
                )
            )
        bench = frame.sort_values("trade_date").drop_duplicates("trade_date")
        fig.add_trace(
            go.Scatter(
                x=bench["trade_date"],
                    y=bench["benchmark_nav"],
                    mode="lines",
                    name="Spot index buy-and-hold",
                line=dict(dash="dot", color="#555555"),
            )
        )
        fig.update_layout(title=f"{fut_code} NAV", height=440, margin=dict(l=48, r=24, t=56, b=44))
        figures.append(fig)

    figures_html = "\n".join(
        pio.to_html(fig, include_plotlyjs="cdn" if idx == 0 else False, full_html=False)
        for idx, fig in enumerate(figures)
    )
    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Forum Sentiment Futures Backtest</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
        table {{ border-collapse: collapse; width: 100%; background: white; }}
        th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; font-size: 12px; text-align: right; }}
        th:first-child, td:first-child, td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
        .table-wrap {{ overflow-x: auto; border: 1px solid #d0d7de; background: white; }}
      </style>
    </head>
    <body>
      <h1>Forum Sentiment Futures Backtest</h1>
      <div class="table-wrap">{summary_html}</div>
      {figures_html}
    </body>
    </html>
    """
    (output_dir / "dashboard.html").write_text(html, encoding="utf-8")
