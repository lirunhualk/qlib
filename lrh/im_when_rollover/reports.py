from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .compat import qlib_style_risk_analysis


@dataclass(frozen=True)
class ReportBundle:
    summary: pd.DataFrame
    annual_returns: pd.DataFrame
    roll_events: pd.DataFrame
    daily_returns: pd.DataFrame


def _year_fraction(index: pd.Series) -> float:
    start = pd.Timestamp(index.min())
    end = pd.Timestamp(index.max())
    years = (end - start).days / 365.25
    return max(years, 1 / 365.25)


def build_reports(daily_results: pd.DataFrame, fut_code: str, spot_code: str) -> ReportBundle:
    daily = daily_results.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    daily["year"] = daily["trade_date"].dt.year

    summary_rows: list[dict] = []
    annual_rows: list[dict] = []

    for strategy, frame in daily.groupby("strategy", sort=False):
        frame = frame.sort_values("trade_date").reset_index(drop=True)
        strat_ra = qlib_style_risk_analysis(frame["strategy_return"], N=252, mode="product")
        bench_ra = qlib_style_risk_analysis(frame["benchmark_return"], N=252, mode="product")
        excess_ra = qlib_style_risk_analysis(frame["excess_return"], N=252, mode="product")

        years = _year_fraction(frame["trade_date"])
        total_return = frame["nav"].iloc[-1] - 1.0
        benchmark_total_return = frame["benchmark_nav"].iloc[-1] - 1.0
        relative_total_return = frame["relative_nav"].iloc[-1] - 1.0
        roll_count = int(frame["rolled"].sum())

        summary_rows.append(
            {
                "fut_code": fut_code,
                "spot_code": spot_code,
                "strategy": strategy,
                "strategy_name": frame["strategy_name"].iloc[0],
                "start_date": frame["trade_date"].min().date().isoformat(),
                "end_date": frame["trade_date"].max().date().isoformat(),
                "observations": len(frame),
                "roll_count": roll_count,
                "rolls_per_year": roll_count / years,
                "annual_turnover": frame["turnover"].sum() / years,
                "total_return": total_return,
                "annualized_return": float(strat_ra.loc["annualized_return", "risk"]),
                "max_drawdown": float(strat_ra.loc["max_drawdown", "risk"]),
                "information_ratio": float(strat_ra.loc["information_ratio", "risk"]),
                "benchmark_total_return": benchmark_total_return,
                "benchmark_annualized_return": float(bench_ra.loc["annualized_return", "risk"]),
                "relative_total_return": relative_total_return,
                "annualized_excess_return": float(excess_ra.loc["annualized_return", "risk"]),
                "excess_max_drawdown": float(excess_ra.loc["max_drawdown", "risk"]),
                "excess_information_ratio": float(excess_ra.loc["information_ratio", "risk"]),
            }
        )

        for year, year_frame in frame.groupby("year"):
            annual_rows.append(
                {
                    "fut_code": fut_code,
                    "spot_code": spot_code,
                    "strategy": strategy,
                    "strategy_name": frame["strategy_name"].iloc[0],
                    "year": int(year),
                    "strategy_return": (1.0 + year_frame["strategy_return"]).prod() - 1.0,
                    "benchmark_return": (1.0 + year_frame["benchmark_return"]).prod() - 1.0,
                    "excess_return": (1.0 + year_frame["excess_return"]).prod() - 1.0,
                    "roll_count": int(year_frame["rolled"].sum()),
                }
            )

    roll_events = daily[daily["rolled"]].copy()
    roll_events = roll_events[
        [
            "trade_date",
            "strategy",
            "strategy_name",
            "contract_before_trade",
            "target_contract",
            "holding_contract",
            "old_contract_trade_days_to_expiry",
            "old_contract_vwap",
            "new_contract_vwap",
            "new_contract_close",
            "holding_annualized_carry",
        ]
    ]
    roll_events.insert(0, "fut_code", fut_code)
    roll_events.insert(1, "spot_code", spot_code)

    return ReportBundle(
        summary=pd.DataFrame(summary_rows).sort_values(["fut_code", "annualized_excess_return"], ascending=[True, False]),
        annual_returns=pd.DataFrame(annual_rows).sort_values(["fut_code", "strategy", "year"]),
        roll_events=roll_events.sort_values(["trade_date", "strategy"]),
        daily_returns=daily.sort_values(["strategy", "trade_date"]),
    )
