from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrh.im_when_rollover.config import (  # noqa: E402
    DEFAULT_COST_BPS_PER_SIDE,
    DEFAULT_LIQUID_TOP_N,
    DEFAULT_REPORT_DIRNAME,
    FUTURE_SPECS,
    build_strategy_specs,
)
from lrh.im_when_rollover.data_loader import MarketDataLoader  # noqa: E402
from lrh.im_when_rollover.db import get_engine  # noqa: E402
from lrh.im_when_rollover.engine import BacktestConfig, run_strategy_backtest  # noqa: E402
from lrh.im_when_rollover.qlib_native_report import (  # noqa: E402
    build_qlib_analysis_df,
    build_qlib_report_normal,
    save_strategy_report_html,
)
from lrh.im_when_rollover.reports import build_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest CFFEX index futures rollover carry strategies.")
    parser.add_argument(
        "--codes",
        nargs="+",
        default=list(FUTURE_SPECS.keys()),
        choices=sorted(FUTURE_SPECS.keys()),
        help="Futures underlyings to backtest.",
    )
    parser.add_argument("--start-date", default=None, help="Inclusive start date, e.g. 2015-01-01.")
    parser.add_argument("--end-date", default=None, help="Inclusive end date, e.g. 2026-04-10.")
    parser.add_argument("--dsn", default=None, help="Override MySQL DSN.")
    parser.add_argument("--cost-bps-per-side", type=float, default=DEFAULT_COST_BPS_PER_SIDE)
    parser.add_argument("--liquid-top-n", type=int, default=DEFAULT_LIQUID_TOP_N)
    parser.add_argument("--output-dir", default=None, help="Directory for reports and CSV outputs.")
    parser.add_argument("--skip-daily", action="store_true", help="Do not persist daily return CSVs.")
    return parser.parse_args()


def resolve_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        return Path(raw_output_dir).resolve()
    return (Path(__file__).resolve().parent / DEFAULT_REPORT_DIRNAME).resolve()


def _write_code_index(code_dir: Path, fut_code: str, summary_df: pd.DataFrame) -> None:
    link_rows = []
    for row in summary_df.sort_values("annualized_excess_return", ascending=False).itertuples():
        link_rows.append(
            "<tr>"
            f"<td><a href=\"{row.strategy}/report.html\">{row.strategy_name}</a></td>"
            f"<td>{row.annualized_excess_return:.6f}</td>"
            f"<td>{row.annualized_return:.6f}</td>"
            f"<td>{row.roll_count}</td>"
            "</tr>"
        )
    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>{fut_code} Report Index</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        table {{ border-collapse: collapse; }}
        th, td {{ border: 1px solid #d0d0d0; padding: 6px 10px; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
      </style>
    </head>
    <body>
      <h1>{fut_code} Strategy Reports</h1>
      <table>
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Annualized Excess Return</th>
            <th>Annualized Return</th>
            <th>Roll Count</th>
          </tr>
        </thead>
        <tbody>
          {"".join(link_rows)}
        </tbody>
      </table>
    </body>
    </html>
    """
    (code_dir / "index.html").write_text(html, encoding="utf-8")


def _write_root_index(root_dir: Path, summary_all: pd.DataFrame) -> None:
    rows = []
    for row in summary_all.sort_values(["fut_code", "annualized_excess_return"], ascending=[True, False]).itertuples():
        rows.append(
            "<tr>"
            f"<td>{row.fut_code}</td>"
            f"<td><a href=\"{row.fut_code}/{row.strategy}/report.html\">{row.strategy_name}</a></td>"
            f"<td>{row.annualized_excess_return:.6f}</td>"
            f"<td>{row.annualized_return:.6f}</td>"
            f"<td>{row.roll_count}</td>"
            "</tr>"
        )
    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Index Futures Rollover Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        table {{ border-collapse: collapse; }}
        th, td {{ border: 1px solid #d0d0d0; padding: 6px 10px; text-align: right; }}
        th:first-child, td:first-child, td:nth-child(2) {{ text-align: left; }}
      </style>
    </head>
    <body>
      <h1>Index Futures Rollover Report</h1>
      <p>Signal timing: T-1 close snapshot. Execution timing: T open.</p>
      <table>
        <thead>
          <tr>
            <th>Future</th>
            <th>Strategy</th>
            <th>Annualized Excess Return</th>
            <th>Annualized Return</th>
            <th>Roll Count</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>
    </body>
    </html>
    """
    (root_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = get_engine(args.dsn)
    loader = MarketDataLoader(engine)
    strategies = build_strategy_specs(liquid_top_n=args.liquid_top_n)
    config = BacktestConfig(cost_bps_per_side=args.cost_bps_per_side, signal_lag_days=1, execution_price="open")

    all_summary: list[pd.DataFrame] = []
    all_annual: list[pd.DataFrame] = []
    all_rolls: list[pd.DataFrame] = []

    for fut_code in args.codes:
        bundle = loader.load_market_bundle(fut_code, start_date=args.start_date, end_date=args.end_date)
        code_dir = output_dir / fut_code
        code_dir.mkdir(parents=True, exist_ok=True)

        daily_frames: list[pd.DataFrame] = []
        for strategy in strategies:
            daily_frames.append(run_strategy_backtest(bundle.market, strategy, config))

        daily = pd.concat(daily_frames, ignore_index=True)
        reports = build_reports(daily, fut_code, bundle.spot_code)

        reports.summary.to_csv(code_dir / "summary.csv", index=False, encoding="utf-8-sig")
        reports.annual_returns.to_csv(code_dir / "annual_returns.csv", index=False, encoding="utf-8-sig")
        reports.roll_events.to_csv(code_dir / "roll_events.csv", index=False, encoding="utf-8-sig")
        bundle.market.to_csv(code_dir / "market_snapshot.csv", index=False, encoding="utf-8-sig")
        if not args.skip_daily:
            reports.daily_returns.to_csv(code_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")

        for strategy in strategies:
            strategy_dir = code_dir / strategy.name
            strategy_dir.mkdir(parents=True, exist_ok=True)
            strategy_daily = daily[daily["strategy"] == strategy.name].copy()
            strategy_summary = reports.summary[reports.summary["strategy"] == strategy.name].copy()
            strategy_annual = reports.annual_returns[reports.annual_returns["strategy"] == strategy.name].copy()
            report_normal_df = build_qlib_report_normal(strategy_daily)
            analysis_df = build_qlib_analysis_df(report_normal_df)

            strategy_summary.to_csv(strategy_dir / "summary.csv", index=False, encoding="utf-8-sig")
            strategy_annual.to_csv(strategy_dir / "annual_returns.csv", index=False, encoding="utf-8-sig")
            report_normal_df.to_csv(strategy_dir / "report_normal.csv", encoding="utf-8-sig")
            analysis_df.to_csv(strategy_dir / "analysis_df.csv", encoding="utf-8-sig")
            if not args.skip_daily:
                strategy_daily.to_csv(strategy_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")

            save_strategy_report_html(
                strategy_dir / "report.html",
                title=f"{fut_code} - {strategy.display_name}",
                summary_df=strategy_summary,
                annual_df=strategy_annual,
                report_normal_df=report_normal_df,
                analysis_df=analysis_df,
            )

        _write_code_index(code_dir, fut_code, reports.summary)

        all_summary.append(reports.summary)
        all_annual.append(reports.annual_returns)
        all_rolls.append(reports.roll_events)

        top3 = reports.summary.nlargest(3, "annualized_excess_return")[
            ["strategy_name", "annualized_excess_return", "annualized_return", "roll_count"]
        ]
        print(f"\n[{fut_code}] top strategies by annualized excess return")
        print(top3.to_string(index=False))

    if all_summary:
        summary_all = pd.concat(all_summary, ignore_index=True)
        annual_all = pd.concat(all_annual, ignore_index=True)
        rolls_all = pd.concat(all_rolls, ignore_index=True)

        summary_all.to_csv(output_dir / "summary_all.csv", index=False, encoding="utf-8-sig")
        annual_all.to_csv(output_dir / "annual_returns_all.csv", index=False, encoding="utf-8-sig")
        rolls_all.to_csv(output_dir / "roll_events_all.csv", index=False, encoding="utf-8-sig")
        _write_root_index(output_dir, summary_all)

    print(f"\nReports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
