from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

from qlib.contrib.report.graph import BaseGraph, ScatterGraph, SubplotsGraph

from .compat import qlib_style_risk_analysis


def build_qlib_report_normal(daily_frame: pd.DataFrame) -> pd.DataFrame:
    report_df = (
        daily_frame.sort_values("trade_date")
        .set_index("trade_date")[["strategy_return_gross", "transaction_cost", "benchmark_return", "turnover"]]
        .rename(
            columns={
                "strategy_return_gross": "return",
                "transaction_cost": "cost",
                "benchmark_return": "bench",
            }
        )
    )
    report_df.index.name = "date"
    return report_df


def build_qlib_analysis_df(report_normal_df: pd.DataFrame) -> pd.DataFrame:
    analysis = {
        "excess_return_without_cost": qlib_style_risk_analysis(
            report_normal_df["return"] - report_normal_df["bench"], N=252, mode="sum"
        ),
        "excess_return_with_cost": qlib_style_risk_analysis(
            report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"],
            N=252,
            mode="sum",
        ),
    }
    return pd.concat(analysis)


def _calculate_maximum(df: pd.DataFrame, is_ex: bool = False):
    if is_ex:
        end_date = df["cum_ex_return_wo_cost_mdd"].idxmin()
        start_date = df.loc[df.index <= end_date]["cum_ex_return_wo_cost"].idxmax()
    else:
        end_date = df["return_wo_mdd"].idxmin()
        start_date = df.loc[df.index <= end_date]["cum_return_wo_cost"].idxmax()
    return start_date, end_date


def _calculate_mdd(series: pd.Series) -> pd.Series:
    return series - series.cummax()


def _calculate_report_data(df: pd.DataFrame) -> pd.DataFrame:
    index_names = df.index.names
    df.index = df.index.strftime("%Y-%m-%d")

    report_df = pd.DataFrame()
    report_df["cum_bench"] = df["bench"].cumsum()
    report_df["cum_return_wo_cost"] = df["return"].cumsum()
    report_df["cum_return_w_cost"] = (df["return"] - df["cost"]).cumsum()
    report_df["return_wo_mdd"] = _calculate_mdd(report_df["cum_return_wo_cost"])
    report_df["return_w_cost_mdd"] = _calculate_mdd((df["return"] - df["cost"]).cumsum())
    report_df["cum_ex_return_wo_cost"] = (df["return"] - df["bench"]).cumsum()
    report_df["cum_ex_return_w_cost"] = (df["return"] - df["bench"] - df["cost"]).cumsum()
    report_df["cum_ex_return_wo_cost_mdd"] = _calculate_mdd(report_df["cum_ex_return_wo_cost"])
    report_df["cum_ex_return_w_cost_mdd"] = _calculate_mdd(report_df["cum_ex_return_w_cost"])
    report_df["turnover"] = df["turnover"]
    report_df.sort_index(ascending=True, inplace=True)
    report_df.index.names = index_names
    return report_df


def qlib_report_graph(report_df: pd.DataFrame) -> Iterable[go.Figure]:
    native = _calculate_report_data(report_df.copy())
    max_start_date, max_end_date = _calculate_maximum(native)
    ex_max_start_date, ex_max_end_date = _calculate_maximum(native, True)

    index_name = native.index.name
    temp_df = native.reset_index()
    temp_df.loc[-1] = 0
    temp_df = temp_df.shift(1)
    temp_df.loc[0, index_name] = "T0"
    temp_df.set_index(index_name, inplace=True)
    temp_df.iloc[0] = 0
    native = temp_df

    default_kind_map = dict(kind="ScatterGraph", kwargs={"mode": "lines+markers"})
    fill_args = {"fill": "tozeroy", "mode": "lines+markers"}
    column_row_col = [
        ("cum_bench", dict(row=1, col=1)),
        ("cum_return_wo_cost", dict(row=1, col=1)),
        ("cum_return_w_cost", dict(row=1, col=1)),
        ("return_wo_mdd", dict(row=2, col=1, graph_kwargs=fill_args)),
        ("return_w_cost_mdd", dict(row=3, col=1, graph_kwargs=fill_args)),
        ("cum_ex_return_wo_cost", dict(row=4, col=1)),
        ("cum_ex_return_w_cost", dict(row=4, col=1)),
        ("turnover", dict(row=5, col=1)),
        ("cum_ex_return_w_cost_mdd", dict(row=6, col=1, graph_kwargs=fill_args)),
        ("cum_ex_return_wo_cost_mdd", dict(row=7, col=1, graph_kwargs=fill_args)),
    ]

    subplot_layout = {}
    for idx in range(1, 8):
        subplot_layout.update({f"yaxis{idx}": dict(zeroline=True, showline=True, showticklabels=True)})
        subplot_layout.update({f"xaxis{idx}": dict(showline=idx == 7, type="category", tickangle=45)})

    layout = dict(
        height=1200,
        title=" ",
        shapes=[
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": max_start_date,
                "y0": 0.55,
                "x1": max_end_date,
                "y1": 1,
                "fillcolor": "#d3d3d3",
                "opacity": 0.3,
                "line": {"width": 0},
            },
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": ex_max_start_date,
                "y0": 0,
                "x1": ex_max_end_date,
                "y1": 0.55,
                "fillcolor": "#d3d3d3",
                "opacity": 0.3,
                "line": {"width": 0},
            },
        ],
    )

    figure = SubplotsGraph(
        df=native,
        layout=layout,
        sub_graph_data=column_row_col,
        subplots_kwargs=dict(
            shared_xaxes=True,
            vertical_spacing=0.01,
            rows=7,
            cols=1,
            row_width=[1, 1, 1, 3, 1, 1, 3],
            print_grid=False,
        ),
        kind_map=default_kind_map,
        sub_graph_layout=subplot_layout,
    ).figure
    return (figure,)


def _get_all_risk_analysis(risk_df: pd.DataFrame) -> pd.DataFrame:
    risk_df = risk_df.unstack()
    risk_df.columns = risk_df.columns.droplevel(0)
    return risk_df.drop("mean", axis=1)


def _get_monthly_risk_analysis_with_report(report_normal_df: pd.DataFrame) -> pd.DataFrame:
    grouped = report_normal_df.groupby([report_normal_df.index.year, report_normal_df.index.month], group_keys=False)
    monthly_df = pd.DataFrame()
    for gp_m in sorted(set(grouped.size().index)):
        month_report = grouped.get_group(gp_m)
        if len(month_report) < 3:
            continue
        month_days = pd.Timestamp(year=gp_m[0], month=gp_m[1], day=1).days_in_month
        temp_df = pd.concat(
            {
                "excess_return_without_cost": qlib_style_risk_analysis(
                    month_report["return"] - month_report["bench"], N=252, mode="sum"
                ),
                "excess_return_with_cost": qlib_style_risk_analysis(
                    month_report["return"] - month_report["bench"] - month_report["cost"],
                    N=252,
                    mode="sum",
                ),
            }
        )
        temp_df["date"] = pd.Timestamp(year=gp_m[0], month=gp_m[1], day=month_days)
        monthly_df = pd.concat([monthly_df, temp_df], sort=False)
    return monthly_df


def _get_monthly_analysis_with_feature(monthly_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    monthly_gp = monthly_df.reset_index().groupby("level_1", group_keys=False)
    feature_df = monthly_gp.get_group(feature).set_index(["level_0", "level_1"])
    result = feature_df.pivot_table(index="date", values=["risk"], columns=feature_df.index)
    result.columns = map(lambda item: "_".join(item[-1]), result.columns)
    result.index = result.index.strftime("%Y-%m")
    return result


def qlib_risk_analysis_graph(analysis_df: pd.DataFrame, report_normal_df: pd.DataFrame) -> Iterable[go.Figure]:
    figures: list[go.Figure] = []
    if analysis_df is not None and not analysis_df.empty:
        figures.append(
            SubplotsGraph(
                _get_all_risk_analysis(analysis_df),
                kind_map=dict(kind="BarGraph", kwargs={}),
                subplots_kwargs={"rows": 1, "cols": 4},
            ).figure
        )

    monthly_df = _get_monthly_risk_analysis_with_report(report_normal_df)
    for feature in ["annualized_return", "max_drawdown", "information_ratio", "std"]:
        temp_df = _get_monthly_analysis_with_feature(monthly_df, feature)
        figures.append(
            ScatterGraph(
                temp_df,
                layout=dict(title=feature, xaxis=dict(type="category", tickangle=45)),
                graph_kwargs={"mode": "lines+markers"},
            ).figure
        )
    return figures


def save_strategy_report_html(
    report_path: Path,
    *,
    title: str,
    summary_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    report_normal_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
) -> None:
    figures = list(qlib_report_graph(report_normal_df)) + list(qlib_risk_analysis_graph(analysis_df, report_normal_df))
    figures_html = "\n".join(
        pio.to_html(fig, include_plotlyjs="cdn" if idx == 0 else False, full_html=False)
        for idx, fig in enumerate(figures)
    )

    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>{title}</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        table {{ border-collapse: collapse; margin-bottom: 24px; }}
        th, td {{ border: 1px solid #d0d0d0; padding: 6px 10px; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
        h1, h2 {{ margin-top: 28px; }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      <h2>Summary</h2>
      {summary_df.to_html(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x), border=0)}
      <h2>Annual Returns</h2>
      {annual_df.to_html(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x), border=0)}
      <h2>Qlib Native Graphs</h2>
      {figures_html}
    </body>
    </html>
    """
    report_path.write_text(html, encoding="utf-8")
