from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

from qlib.contrib.report.graph import BaseGraph, ScatterGraph, SubplotsGraph

from .compat import qlib_style_risk_analysis

TRACE_NAME_MAP = {
    "cum_bench": "cum bench",
    "cum_return_wo_cost": "cum return no cost",
    "cum_return_w_cost": "cum return with cost",
    "return_wo_mdd": "drawdown no cost",
    "return_w_cost_mdd": "drawdown with cost",
    "cum_ex_return_wo_cost": "cum excess no cost",
    "cum_ex_return_w_cost": "cum excess with cost",
    "cum_ex_return_wo_cost_mdd": "excess drawdown no cost",
    "cum_ex_return_w_cost_mdd": "excess drawdown with cost",
    "turnover": "turnover",
    "std": "std",
    "annualized return": "annualized return",
    "information ratio": "information ratio",
    "max drawdown": "max drawdown",
}

REPORT_PAGE_STYLE = """
        body { font-family: Arial, sans-serif; margin: 0; background: #f3f5f8; color: #111827; }
        .report-shell { max-width: 1360px; margin: 0 auto; padding: 16px 20px 32px; }
        table { border-collapse: collapse; margin-bottom: 0; width: 100%; min-width: 980px; background: #ffffff; }
        th, td { border: 1px solid #d0d7de; padding: 5px 8px; font-size: 13px; line-height: 1.35; text-align: right; white-space: nowrap; }
        th:first-child, td:first-child { text-align: left; }
        table.dataframe th:nth-child(3),
        table.dataframe td:nth-child(3),
        table.dataframe th:nth-child(4),
        table.dataframe td:nth-child(4) { white-space: normal; min-width: 110px; max-width: 160px; }
        thead th { white-space: normal; vertical-align: bottom; }
        h1 { margin: 0 0 16px; font-size: 26px; line-height: 1.2; }
        h2 { margin: 24px 0 12px; font-size: 18px; line-height: 1.3; }
        .table-scroll { overflow-x: auto; margin-bottom: 18px; border: 1px solid #dbe2ea; border-radius: 10px; background: #ffffff; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04); }
        .table-scroll table { border: 0; }
        .table-scroll::-webkit-scrollbar { height: 10px; }
        .table-scroll::-webkit-scrollbar-thumb { background: #c6cfda; border-radius: 999px; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }
        .summary-card { border: 1px solid #dbe2ea; border-radius: 10px; background: #ffffff; padding: 10px 12px; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04); }
        .summary-label { font-size: 12px; line-height: 1.35; color: #475569; margin-bottom: 4px; word-break: break-word; }
        .summary-value { font-size: 16px; line-height: 1.35; font-weight: 600; word-break: break-word; }
        .plot-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; margin-top: 14px; }
        .plot-grid > div,
        .plot-card { min-width: 0; overflow: visible; }
        .plotly-graph-div,
        .js-plotly-plot,
        .plot-container,
        .svg-container { max-width: 100% !important; }
        .plotly-graph-div { border: 1px solid #dbe2ea; border-radius: 10px; background: #ffffff; box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04); }
        @media (max-width: 900px) {
          .report-shell { padding: 12px; }
          h1 { font-size: 22px; }
          table { min-width: 860px; }
          th, td { font-size: 12px; padding: 4px 6px; }
        }
"""

REPORT_PAGE_SCRIPT = """
      <script>
        function simplifyTraceName(name) {
          if (!name) {
            return name;
          }
          var renamed = String(name);
          var exactMap = {
            "cum_bench": "cum bench",
            "cum_return_wo_cost": "cum return no cost",
            "cum_return_w_cost": "cum return with cost",
            "return_wo_mdd": "drawdown no cost",
            "return_w_cost_mdd": "drawdown with cost",
            "cum_ex_return_wo_cost": "cum excess no cost",
            "cum_ex_return_w_cost": "cum excess with cost",
            "cum_ex_return_wo_cost_mdd": "excess drawdown no cost",
            "cum_ex_return_w_cost_mdd": "excess drawdown with cost",
            "turnover": "turnover"
          };
          if (exactMap[renamed]) {
            return exactMap[renamed];
          }
          renamed = renamed.replaceAll("excess_return_with_cost_", "with cost ");
          renamed = renamed.replaceAll("excess_return_without_cost_", "without cost ");
          renamed = renamed.replaceAll("_annualized_return", "");
          renamed = renamed.replaceAll("_max_drawdown", "");
          renamed = renamed.replaceAll("_information_ratio", "");
          renamed = renamed.replaceAll("_std", "");
          renamed = renamed.replaceAll("_", " ");
          return renamed.replace(/\\s+/g, " ").trim();
        }

        window.addEventListener("load", function () {
          document.querySelectorAll("table.dataframe").forEach(function (table) {
            if (table.parentElement && table.parentElement.classList.contains("table-scroll")) {
              return;
            }
            var wrapper = document.createElement("div");
            wrapper.className = "table-scroll";
            table.parentNode.insertBefore(wrapper, table);
            wrapper.appendChild(table);
          });

          var tables = Array.from(document.querySelectorAll("table.dataframe"));
          var summaryTable = tables[0];
          if (summaryTable) {
            var headerCells = Array.from(summaryTable.querySelectorAll("thead th"));
            var summaryRow = summaryTable.querySelector("tbody tr");
            if (summaryRow && summaryRow.children.length === headerCells.length) {
              var summaryGrid = document.createElement("div");
              summaryGrid.className = "summary-grid";
              Array.from(summaryRow.children).forEach(function (cell, index) {
                var card = document.createElement("div");
                card.className = "summary-card";

                var label = document.createElement("div");
                label.className = "summary-label";
                label.textContent = headerCells[index].textContent;

                var value = document.createElement("div");
                value.className = "summary-value";
                value.textContent = cell.textContent;

                card.appendChild(label);
                card.appendChild(value);
                summaryGrid.appendChild(card);
              });

              var summaryContainer = summaryTable.parentElement;
              summaryContainer.insertAdjacentElement("afterend", summaryGrid);
              summaryContainer.style.display = "none";
            }
          }

          var graphDivs = Array.from(document.querySelectorAll(".plotly-graph-div"));
          if (!graphDivs.length || !window.Plotly) {
            return;
          }

          graphDivs.forEach(function (div) {
            if (div.parentElement) {
              div.parentElement.classList.add("plot-card");
            }
            var traceIndexes = [];
            var traceNames = [];
            (div.data || []).forEach(function (trace, index) {
              traceIndexes.push(index);
              traceNames.push(simplifyTraceName(trace.name || ""));
            });
            if (traceIndexes.length) {
              window.Plotly.restyle(div, { name: traceNames }, traceIndexes);
            }
          });

          var firstGraphContainer = graphDivs[0].parentElement;
          var secondaryContainers = graphDivs.slice(1).map(function (div) {
            return div.parentElement;
          });

          if (secondaryContainers.length) {
            var grid = document.createElement("div");
            grid.className = "plot-grid";
            firstGraphContainer.insertAdjacentElement("afterend", grid);
            secondaryContainers.forEach(function (container) {
              grid.appendChild(container);
            });
          }

          graphDivs.forEach(function (div, index) {
            var isPrimary = index === 0;
            var isRiskBar = index === 1;
            var layout = { font: { size: 12 } };

            if (isPrimary) {
              layout.height = 760;
              layout.margin = { l: 56, r: 160, t: 16, b: 56, pad: 4 };
              layout.legend = {
                x: 1.02,
                xanchor: "left",
                y: 1,
                yanchor: "top",
                bgcolor: "rgba(255,255,255,0.78)"
              };
            } else if (isRiskBar) {
              layout.height = 420;
              layout.margin = { l: 48, r: 32, t: 56, b: 44, pad: 4 };
              layout.showlegend = false;
            } else {
              layout.height = 320;
              layout.margin = { l: 56, r: 64, t: 54, b: 64, pad: 4 };
              layout.legend = {
                orientation: "h",
                x: 0,
                xanchor: "left",
                y: 1.16,
                yanchor: "bottom",
                bgcolor: "rgba(255,255,255,0.72)"
              };
            }

            window.Plotly.relayout(div, layout).then(function () {
              window.Plotly.Plots.resize(div);
            });
          });
        });
      </script>
"""


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
    risk_df = risk_df.drop("mean", axis=1)
    risk_df.index = ["with cost" if idx == "excess_return_with_cost" else "without cost" for idx in risk_df.index]
    return risk_df.rename(columns=lambda col: _short_trace_name(col))


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


def _short_trace_name(name: str | None) -> str | None:
    if not name:
        return name
    if name in TRACE_NAME_MAP:
        return TRACE_NAME_MAP[name]

    short_name = str(name)
    replacements = [
        ("excess_return_with_cost_", "with cost "),
        ("excess_return_without_cost_", "without cost "),
        ("_annualized_return", ""),
        ("_max_drawdown", ""),
        ("_information_ratio", ""),
        ("_std", ""),
        ("_", " "),
    ]
    for old, new in replacements:
        short_name = short_name.replace(old, new)
    return " ".join(short_name.split())


def _style_figure(fig: go.Figure, kind: str) -> go.Figure:
    for trace in fig.data:
        if getattr(trace, "name", None):
            trace.name = _short_trace_name(trace.name)

    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)

    if kind == "main":
        fig.update_layout(
            autosize=True,
            height=760,
            font=dict(size=12),
            margin=dict(l=56, r=160, t=16, b=56, pad=4),
            legend=dict(x=1.02, xanchor="left", y=1, yanchor="top", bgcolor="rgba(255,255,255,0.78)"),
        )
    elif kind == "risk_bar":
        fig.update_layout(
            autosize=True,
            height=420,
            font=dict(size=12),
            margin=dict(l=48, r=32, t=56, b=44, pad=4),
            showlegend=False,
        )
    else:
        fig.update_layout(
            autosize=True,
            height=320,
            font=dict(size=12),
            margin=dict(l=56, r=64, t=54, b=64, pad=4),
            legend=dict(
                orientation="h",
                x=0,
                xanchor="left",
                y=1.16,
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.72)",
            ),
        )
    fig.update_layout(title=dict(x=0.5, xanchor="center"))
    return fig


def qlib_risk_analysis_graph(analysis_df: pd.DataFrame, report_normal_df: pd.DataFrame) -> Iterable[go.Figure]:
    figures: list[go.Figure] = []
    if analysis_df is not None and not analysis_df.empty:
        risk_summary = _get_all_risk_analysis(analysis_df)
        figures.append(
            SubplotsGraph(
                risk_summary,
                kind_map=dict(kind="BarGraph", kwargs={}),
                subplots_kwargs={
                    "rows": 2,
                    "cols": 2,
                    "print_grid": False,
                    "subplot_titles": risk_summary.columns.tolist(),
                },
            ).figure
        )

    monthly_df = _get_monthly_risk_analysis_with_report(report_normal_df)
    for feature in ["annualized_return", "max_drawdown", "information_ratio", "std"]:
        temp_df = _get_monthly_analysis_with_feature(monthly_df, feature)
        temp_df = temp_df.rename(columns=lambda col: _short_trace_name(col))
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
    if figures:
        figures[0] = _style_figure(figures[0], "main")
        for idx in range(1, len(figures)):
            figures[idx] = _style_figure(figures[idx], "risk_bar" if idx == 1 else "risk_line")

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
{REPORT_PAGE_STYLE}
      </style>
    </head>
    <body>
      <div class="report-shell">
      <h1>{title}</h1>
      <h2>Summary</h2>
      {summary_df.to_html(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x), border=0)}
      <h2>Annual Returns</h2>
      {annual_df.to_html(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x), border=0)}
      <h2>Qlib Native Graphs</h2>
      {figures_html}
      </div>
      {REPORT_PAGE_SCRIPT}
    </body>
    </html>
    """
    report_path.write_text(html, encoding="utf-8")
