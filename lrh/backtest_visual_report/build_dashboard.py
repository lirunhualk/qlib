from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / "smallcap_lowturn" / "report"
DEFAULT_OUTPUT = ROOT / "index.html"


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int_date(value: str) -> str:
    text = str(value).strip()
    if "-" in text:
        return text[:10]
    return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")


def fmt_pct(value: float, digits: int = 2, signed: bool = True) -> str:
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value * 100:.{digits}f}%"


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def compound(values: list[float]) -> float:
    total = 1.0
    for value in values:
        total *= 1.0 + value
    return total - 1.0


def longest_streak(values: list[float], positive: bool) -> int:
    best = 0
    current = 0
    for value in values:
        ok = value > 0 if positive else value < 0
        if ok:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values)


def percentile_rank(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    return sum(1 for item in values if item <= value) / len(values)


def normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    avg = mean(values)
    sigma = pstdev(values)
    if sigma == 0:
        return [0.0 for _ in values]
    return [(item - avg) / sigma for item in values]


def moving_average(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = []
    acc: list[float] = []
    for value in values:
        acc.append(value)
        if len(acc) > window:
            acc.pop(0)
        result.append(mean(acc) if len(acc) == window else None)
    return result


def rolling_ir(excess_returns: list[float], window: int = 126) -> list[float | None]:
    result: list[float | None] = []
    acc: list[float] = []
    for value in excess_returns:
        acc.append(value)
        if len(acc) > window:
            acc.pop(0)
        if len(acc) < window:
            result.append(None)
            continue
        sigma = safe_stdev(acc)
        result.append(mean(acc) / sigma * math.sqrt(252) if sigma else None)
    return result


def read_summary(source: Path) -> dict[str, Any]:
    with (source / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_daily(source: Path) -> list[dict[str, Any]]:
    path = source / "daily_returns.csv"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "date": parse_int_date(row["date"]),
                    "return": parse_float(row.get("return")),
                    "gross_return": parse_float(row.get("gross_return")),
                    "bench": parse_float(row.get("bench")),
                    "cost": parse_float(row.get("cost")),
                    "turnover": parse_float(row.get("turnover")),
                    "holdings": parse_float(row.get("holdings")),
                    "cash_weight": parse_float(row.get("cash_weight")),
                    "equity": parse_float(row.get("equity"), 1.0),
                    "drawdown": parse_float(row.get("drawdown")),
                    "bench_equity": parse_float(row.get("bench_equity"), 1.0),
                    "bench_drawdown": parse_float(row.get("bench_drawdown")),
                    "blocked_buys": parse_float(row.get("blocked_buys")),
                    "blocked_sells": parse_float(row.get("blocked_sells")),
                }
            )
    return rows


def read_annual(source: Path) -> list[dict[str, Any]]:
    path = source / "annual_returns.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "year": str(row["year"]),
                    "strategy": parse_float(row.get("strategy_return")),
                    "benchmark": parse_float(row.get("benchmark_return")),
                }
            )
    return rows


def read_holdings(source: Path) -> list[dict[str, Any]]:
    path = source / "holdings.csv"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "rebalance_date": parse_int_date(row["rebalance_date"]),
                    "signal_date": parse_int_date(row["signal_date"]),
                    "ts_code": row.get("ts_code", ""),
                    "name": row.get("name", ""),
                    "target_weight": parse_float(row.get("target_weight")),
                    "score": parse_float(row.get("score")),
                    "total_mv": parse_float(row.get("total_mv")),
                    "circ_mv": parse_float(row.get("circ_mv")),
                    "amt20": parse_float(row.get("amt20")),
                    "turn20": parse_float(row.get("turn20")),
                    "vol60": parse_float(row.get("vol60")),
                    "ret20": parse_float(row.get("ret20")),
                    "ret60": parse_float(row.get("ret60")),
                }
            )
    return rows


def monthly_returns(daily: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in daily:
        dt = datetime.strptime(row["date"], "%Y-%m-%d")
        grouped[(str(dt.year), dt.month)].append(row["return"])
    years = sorted({year for year, _month in grouped})
    heatmap: list[list[Any]] = []
    for y_index, year in enumerate(years):
        for month in range(1, 13):
            value = compound(grouped.get((year, month), [])) if (year, month) in grouped else None
            heatmap.append([month - 1, y_index, None if value is None else round(value * 100, 2)])
    return years, heatmap


def annual_ir_rows(daily: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in daily:
        if row["bench"] == 0.0 and row["date"] == daily[0]["date"]:
            continue
        grouped[row["date"][:4]].append(row["return"] - row["bench"])
    rows: list[dict[str, Any]] = []
    for year in sorted(grouped):
        values = grouped[year]
        sigma = safe_stdev(values)
        rows.append(
            {
                "year": year,
                "ir": mean(values) / sigma * math.sqrt(252) if sigma else 0.0,
                "mean_excess": mean(values) * 252 if values else 0.0,
                "sample": len(values),
            }
        )
    return rows


def calc_snapshot_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["rebalance_date"]].append(row)

    dates = sorted(by_date)
    concentration = []
    exposure_raw = {
        "小市值": [],
        "低换手": [],
        "低波动": [],
        "短动量": [],
        "流动性": [],
        "综合得分": [],
    }
    top_holdings: list[dict[str, Any]] = []

    for date in dates:
        group = by_date[date]
        hhi = sum(item["target_weight"] ** 2 for item in group)
        effective = 1.0 / hhi if hhi else 0.0
        weights = [item["target_weight"] for item in group]
        total_weight = sum(weights) or 1.0

        def wavg(key: str, transform=lambda value: value) -> float:
            return sum(item["target_weight"] * transform(max(item[key], 1e-12)) for item in group) / total_weight

        concentration.append(
            {
                "date": date,
                "holdings": len(group),
                "effective_holdings": effective,
                "hhi": hhi,
                "avg_score": wavg("score", lambda value: value),
            }
        )
        exposure_raw["小市值"].append(-wavg("total_mv", math.log))
        exposure_raw["低换手"].append(-wavg("turn20", lambda value: value))
        exposure_raw["低波动"].append(-wavg("vol60", lambda value: value))
        exposure_raw["短动量"].append(wavg("ret20", lambda value: value))
        exposure_raw["流动性"].append(wavg("amt20", math.log))
        exposure_raw["综合得分"].append(wavg("score", lambda value: value))

    if dates:
        latest = by_date[dates[-1]]
        top_holdings = sorted(latest, key=lambda item: item["score"], reverse=True)[:30]

    exposure = {
        key: [round(value, 3) for value in normalize_series(values)]
        for key, values in exposure_raw.items()
    }
    latest_exposure = {
        key: values[-1] if values else 0.0
        for key, values in exposure.items()
    }

    latest_group = by_date[dates[-1]] if dates else []
    bucket_charts = build_bucket_charts(latest_group)

    return {
        "dates": dates,
        "concentration": concentration,
        "exposure": exposure,
        "latestExposure": latest_exposure,
        "topHoldings": top_holdings,
        "bucketCharts": bucket_charts,
    }


def build_bucket_charts(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not rows:
        return {}

    def bucket_by_quantile(key: str, labels: tuple[str, str, str], reverse: bool = False) -> list[dict[str, Any]]:
        values = sorted(item[key] for item in rows)
        groups = {label: {"weight": 0.0, "count": 0, "avg": 0.0} for label in labels}
        for item in rows:
            rank = percentile_rank(values, item[key])
            if rank <= 1 / 3:
                label = labels[2] if reverse else labels[0]
            elif rank <= 2 / 3:
                label = labels[1]
            else:
                label = labels[0] if reverse else labels[2]
            groups[label]["weight"] += item["target_weight"]
            groups[label]["count"] += 1
            groups[label]["avg"] += item[key]
        result = []
        for label in labels:
            count = groups[label]["count"] or 1
            result.append(
                {
                    "name": label,
                    "weight": groups[label]["weight"],
                    "count": groups[label]["count"],
                    "avg": groups[label]["avg"] / count,
                }
            )
        return result

    return {
        "市值分组": bucket_by_quantile("total_mv", ("小市值", "中市值", "大市值"), reverse=True),
        "换手分组": bucket_by_quantile("turn20", ("低换手", "中换手", "高换手")),
        "波动分组": bucket_by_quantile("vol60", ("低波动", "中波动", "高波动")),
        "动量分组": bucket_by_quantile("ret20", ("弱动量", "中动量", "强动量")),
    }


def build_metrics(summary: dict[str, Any], daily: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [row["return"] for row in daily]
    bench = [row["bench"] for row in daily]
    valid_excess = [r - b for r, b in zip(returns, bench) if not (r == 0.0 and b == 0.0)]
    daily_win = sum(1 for value in returns if value > 0) / len(returns) if returns else 0.0
    excess_win = sum(1 for value in valid_excess if value > 0) / len(valid_excess) if valid_excess else 0.0
    sigma = safe_stdev(valid_excess)
    ir = mean(valid_excess) / sigma * math.sqrt(252) if sigma else 0.0
    t_stat = mean(valid_excess) / (sigma / math.sqrt(len(valid_excess))) if sigma and valid_excess else 0.0
    p_value = math.erfc(abs(t_stat) / math.sqrt(2))

    by_month: dict[str, list[float]] = defaultdict(list)
    for row in daily:
        by_month[row["date"][:7]].append(row["return"])
    month_values = [compound(values) for values in by_month.values()]
    month_win = sum(1 for value in month_values if value > 0) / len(month_values) if month_values else 0.0

    return {
        "strategy": [
            {"label": "累计策略收益", "value": fmt_pct(parse_float(summary.get("cumulative_return")), 2), "tone": "up"},
            {"label": "年化策略收益", "value": fmt_pct(parse_float(summary.get("annualized_return")), 2), "tone": "up"},
            {"label": "策略夏普", "value": fmt_num(parse_float(summary.get("sharpe")), 2), "tone": "neutral"},
            {"label": "最大回撤", "value": fmt_pct(parse_float(summary.get("max_drawdown")), 2), "tone": "down"},
            {"label": "日胜率", "value": fmt_pct(daily_win, 2, signed=False), "tone": "neutral"},
            {"label": "月胜率", "value": fmt_pct(month_win, 2, signed=False), "tone": "neutral"},
            {"label": "平均换手", "value": fmt_pct(parse_float(summary.get("average_turnover")), 2, signed=False), "tone": "neutral"},
            {"label": "最长连涨", "value": f"{longest_streak(returns, True)} 天", "tone": "neutral"},
            {"label": "最长连跌", "value": f"{longest_streak(returns, False)} 天", "tone": "neutral"},
        ],
        "excess": [
            {
                "label": "累计超额收益",
                "value": fmt_pct(
                    parse_float(summary.get("cumulative_return"))
                    - parse_float(summary.get("benchmark_cumulative_return")),
                    2,
                ),
                "tone": "up",
            },
            {
                "label": "年化超额收益",
                "value": fmt_pct(
                    parse_float(summary.get("annualized_return"))
                    - parse_float(summary.get("benchmark_annualized_return")),
                    2,
                ),
                "tone": "up",
            },
            {"label": "信息比率 IR", "value": fmt_num(ir, 2), "tone": "neutral"},
            {"label": "IR t 统计量", "value": fmt_num(t_stat, 2), "tone": "neutral"},
            {"label": "p 值", "value": f"{p_value:.4f}", "tone": "neutral"},
            {"label": "超额日胜率", "value": fmt_pct(excess_win, 2, signed=False), "tone": "neutral"},
        ],
        "benchmark": [
            {
                "label": "基准累计收益",
                "value": fmt_pct(parse_float(summary.get("benchmark_cumulative_return")), 2),
                "tone": "neutral",
            },
            {
                "label": "基准年化收益",
                "value": fmt_pct(parse_float(summary.get("benchmark_annualized_return")), 2),
                "tone": "neutral",
            },
            {
                "label": "基准最大回撤",
                "value": fmt_pct(parse_float(summary.get("benchmark_max_drawdown")), 2),
                "tone": "down",
            },
            {
                "label": "平均持仓数",
                "value": fmt_num(parse_float(summary.get("average_holdings")), 1),
                "tone": "neutral",
            },
            {
                "label": "总交易成本",
                "value": fmt_pct(parse_float(summary.get("total_cost")), 2, signed=False),
                "tone": "neutral",
            },
        ],
        "ir": {
            "value": ir,
            "tStat": t_stat,
            "pValue": p_value,
            "sample": len(valid_excess),
            "meanDailyExcess": mean(valid_excess) if valid_excess else 0.0,
            "dailyVol": sigma,
        },
    }


def build_data(source: Path) -> dict[str, Any]:
    summary = read_summary(source)
    daily = read_daily(source)
    annual = read_annual(source)
    holdings = read_holdings(source)

    dates = [row["date"] for row in daily]
    returns = [row["return"] for row in daily]
    bench = [row["bench"] for row in daily]
    excess = [row["return"] - row["bench"] for row in daily]
    rolling_ir_values = rolling_ir(excess)
    turnover_ma = moving_average([row["turnover"] for row in daily], 20)
    years, heatmap = monthly_returns(daily)
    annual_ir = annual_ir_rows(daily)
    snapshot = calc_snapshot_stats(holdings)

    excess_cumulative = [row["equity"] - row["bench_equity"] for row in daily]

    factor_names = ["小市值", "低换手", "低波动", "短动量", "流动性"]
    factor_weights = [0.36, 0.23, 0.16, 0.10, 0.15]
    factor_contribution = []
    for name, weight in zip(factor_names, factor_weights):
        latest = snapshot["latestExposure"].get(name, 0.0)
        factor_contribution.append(
            {
                "name": name,
                "value": (summary["cumulative_return"] - summary["benchmark_cumulative_return"]) * weight * (1 + latest * 0.08),
                "exposure": latest,
            }
        )

    brinson_proxy = {
        "dates": dates,
        "allocation": [value * 0.34 for value in excess_cumulative],
        "selection": [value * 0.55 for value in excess_cumulative],
        "interaction": [value * 0.11 for value in excess_cumulative],
        "excess": excess_cumulative,
        "summary": [
            {"label": "配置效应（代理）", "value": fmt_pct(excess_cumulative[-1] * 0.34 if excess_cumulative else 0.0), "tone": "up"},
            {"label": "选股效应（代理）", "value": fmt_pct(excess_cumulative[-1] * 0.55 if excess_cumulative else 0.0), "tone": "up"},
            {"label": "交互效应（代理）", "value": fmt_pct(excess_cumulative[-1] * 0.11 if excess_cumulative else 0.0), "tone": "neutral"},
            {"label": "超额收益", "value": fmt_pct(excess_cumulative[-1] if excess_cumulative else 0.0), "tone": "up"},
        ],
    }

    meta = {
        "title": "Alpha 指增归因分析",
        "strategy": summary.get("strategy", "smallcap_lowturn"),
        "benchmark": summary.get("benchmark_code", "000852.SH"),
        "dateRange": f"{parse_int_date(str(summary.get('start_date')))} - {parse_int_date(str(summary.get('end_date')))}",
        "days": int(summary.get("days", len(daily))),
        "sourceDir": str(source),
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "收益、回撤、换手和稳定性指标来自 smallcap_lowturn/report 的真实回测结果。",
            "Brinson/Barra 页当前使用持仓特征和策略权重生成代理视图；真实行业归因和 Barra 风险归因需要行业分类、因子暴露、因子收益和个股收益明细。",
        ],
    }

    return {
        "meta": meta,
        "summary": summary,
        "metrics": build_metrics(summary, daily),
        "overview": {
            "dates": dates,
            "strategyReturn": [(row["equity"] - 1.0) * 100 for row in daily],
            "benchmarkReturn": [(row["bench_equity"] - 1.0) * 100 for row in daily],
            "excessReturn": [value * 100 for value in excess_cumulative],
            "drawdown": [row["drawdown"] * 100 for row in daily],
            "benchmarkDrawdown": [row["bench_drawdown"] * 100 for row in daily],
            "turnover": [row["turnover"] * 100 for row in daily],
            "turnoverMa": [None if value is None else value * 100 for value in turnover_ma],
            "cashWeight": [row["cash_weight"] * 100 for row in daily],
            "rollingIr": rolling_ir_values,
            "blockedBuys": [row["blocked_buys"] for row in daily],
            "blockedSells": [row["blocked_sells"] for row in daily],
        },
        "annual": annual,
        "annualIr": annual_ir,
        "monthly": {"years": years, "values": heatmap},
        "holdings": snapshot,
        "brinson": brinson_proxy,
        "barra": {
            "factorContribution": factor_contribution,
            "factorNames": factor_names,
            "factorWeights": factor_weights,
        },
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Alpha 指增归因分析</title>
  <link href="https://cdn.jsdelivr.net/npm/remixicon@4.6.0/fonts/remixicon.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1017;
      --panel: #131922;
      --panel-2: #161d27;
      --stroke: #273140;
      --stroke-soft: #202938;
      --text: #e6edf6;
      --muted: #8a98ab;
      --muted-2: #66758a;
      --blue: #2f89ff;
      --red: #ff454a;
      --green: #1db56d;
      --orange: #d59b22;
      --purple: #9b69ff;
      --cyan: #46c7e8;
      --shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
    }

    * { box-sizing: border-box; }
    html { min-width: 1120px; }
    body {
      margin: 0;
      font-family: Inter, "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }

    button, select {
      font: inherit;
      color: inherit;
    }

    .shell {
      display: grid;
      grid-template-columns: 208px minmax(920px, 1fr);
      min-height: 100vh;
    }

    .topbar {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 20;
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: #111720;
      border-bottom: 1px solid var(--stroke);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .brand-mark {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      color: var(--blue);
      font-size: 18px;
    }

    .brand-title {
      font-size: 17px;
      font-weight: 800;
      white-space: nowrap;
    }

    .strategy-control {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }

    .strategy-control select {
      height: 32px;
      border: 1px solid var(--stroke);
      border-radius: 6px;
      background: #0c121a;
      padding: 0 30px 0 10px;
      outline: none;
      font-weight: 700;
    }

    .top-meta {
      display: flex;
      align-items: center;
      gap: 16px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .upload-button {
      border: 1px solid rgba(47, 137, 255, 0.75);
      background: rgba(47, 137, 255, 0.12);
      color: var(--blue);
      height: 32px;
      border-radius: 6px;
      padding: 0 12px;
      font-weight: 700;
      cursor: pointer;
    }

    .sidebar {
      position: fixed;
      top: 52px;
      bottom: 0;
      left: 0;
      width: 208px;
      padding: 34px 10px 16px;
      background: #121821;
      border-right: 1px solid var(--stroke);
      overflow-y: auto;
    }

    .nav-title {
      margin: 0 0 12px;
      padding-left: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }

    .nav-button {
      width: 100%;
      height: 42px;
      display: flex;
      align-items: center;
      gap: 10px;
      border: 0;
      border-radius: 7px;
      background: transparent;
      color: var(--muted);
      padding: 0 12px;
      cursor: pointer;
      text-align: left;
      font-size: 14px;
      font-weight: 700;
    }

    .nav-button i {
      width: 18px;
      color: #74839a;
      font-size: 17px;
    }

    .nav-button.active {
      color: var(--blue);
      background: #193153;
    }

    .nav-button.active i { color: var(--blue); }

    main {
      grid-column: 2;
      padding: 82px 28px 36px;
    }

    .page {
      display: none;
    }

    .page.active {
      display: block;
    }

    .page-header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 24px;
    }

    h1 {
      margin: 0;
      font-size: 26px;
      line-height: 1.1;
      font-weight: 850;
    }

    .subhead {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .mode-tabs {
      display: inline-flex;
      border: 1px solid var(--stroke);
      border-radius: 7px;
      padding: 3px;
      background: #0c121a;
    }

    .mode-tabs button {
      height: 28px;
      min-width: 54px;
      border: 0;
      border-radius: 5px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 12px;
      font-weight: 800;
    }

    .mode-tabs button.active {
      color: var(--blue);
      background: #132642;
    }

    .context-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      align-items: center;
      min-height: 52px;
      margin-bottom: 18px;
      padding: 0 18px;
      border: 1px solid var(--stroke);
      border-radius: 9px;
      background: var(--panel);
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    .context-strip strong { color: var(--text); }
    .context-pill {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      height: 28px;
      padding: 0 9px;
      border: 1px solid var(--stroke);
      border-radius: 5px;
      background: #0d131c;
      color: var(--text);
    }

    .section-label {
      margin: 18px 0 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 850;
      text-transform: uppercase;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }

    .metric-card {
      min-height: 86px;
      padding: 16px 18px;
      border: 1px solid var(--stroke);
      border-radius: 8px;
      background: var(--panel);
    }

    .metric-label {
      display: block;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }

    .metric-value {
      font-size: 23px;
      line-height: 1;
      font-weight: 850;
      color: var(--text);
    }

    .metric-value.up { color: var(--red); }
    .metric-value.down { color: var(--green); }

    .panel-grid {
      display: grid;
      gap: 16px;
    }

    .two-col {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }

    .wide-left {
      grid-template-columns: minmax(0, 2fr) minmax(320px, 0.85fr);
    }

    .panel {
      border: 1px solid var(--stroke);
      border-radius: 10px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      min-height: 48px;
      padding: 16px 18px 0;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }

    .panel-title {
      margin: 0;
      color: #9eacc0;
      font-size: 13px;
      font-weight: 850;
      letter-spacing: 0;
    }

    .panel-note {
      margin: 6px 0 0;
      color: var(--muted-2);
      font-size: 12px;
      line-height: 1.55;
    }

    .icon-button {
      width: 34px;
      height: 28px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--stroke);
      border-radius: 6px;
      background: #0d131c;
      color: var(--muted);
      cursor: pointer;
    }

    .chart {
      width: 100%;
      height: 430px;
    }

    .chart.tall { height: 560px; }
    .chart.short { height: 300px; }

    .info-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      padding: 16px 18px 18px;
    }

    .info-block {
      min-height: 92px;
      padding: 14px;
      border: 1px solid var(--stroke-soft);
      border-radius: 8px;
      background: #101720;
    }

    .info-block h3 {
      margin: 0 0 8px;
      font-size: 13px;
    }

    .info-block p {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.65;
    }

    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    .data-table th,
    .data-table td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--stroke-soft);
      text-align: right;
      white-space: nowrap;
    }

    .data-table th:first-child,
    .data-table td:first-child,
    .data-table th:nth-child(2),
    .data-table td:nth-child(2) {
      text-align: left;
    }

    .data-table th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 850;
    }

    .data-table td {
      color: var(--text);
      font-weight: 650;
    }

    .scroll-table {
      max-height: 440px;
      overflow: auto;
      padding: 6px 10px 12px;
    }

    .notice {
      margin-bottom: 16px;
      padding: 14px 16px;
      border: 1px solid rgba(47, 137, 255, 0.35);
      border-radius: 8px;
      background: rgba(47, 137, 255, 0.08);
      color: #a9c8ef;
      font-size: 13px;
      line-height: 1.6;
    }

    .suggestions {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }

    @media (max-width: 1280px) {
      html { min-width: 980px; }
      .shell { grid-template-columns: 186px minmax(790px, 1fr); }
      .sidebar { width: 186px; }
      main { padding-left: 22px; padding-right: 22px; }
      .metric-grid { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
      .two-col, .wide-left { grid-template-columns: 1fr; }
      .suggestions { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <span class="brand-mark"><i class="ri-bar-chart-box-line"></i></span>
      <div class="brand-title">Alpha 指增归因分析</div>
      <div class="strategy-control">
        <span>策略</span>
        <select aria-label="策略">
          <option>smallcap_lowturn</option>
        </select>
      </div>
    </div>
    <div class="top-meta">
      <span>中证1000（000852.SH） · 策略归因报告</span>
      <span><i class="ri-user-3-line"></i> qianyang</span>
      <button class="upload-button" type="button"><i class="ri-upload-2-line"></i> 上传策略</button>
    </div>
  </header>

  <div class="shell">
    <aside class="sidebar">
      <p class="nav-title">分析模块</p>
      <button class="nav-button active" type="button" data-page="overview"><i class="ri-dashboard-3-line"></i><span>总览仪表盘</span></button>
      <button class="nav-button" type="button" data-page="brinson"><i class="ri-bar-chart-grouped-line"></i><span>Brinson归因</span></button>
      <button class="nav-button" type="button" data-page="barra"><i class="ri-radar-line"></i><span>Barra归因</span></button>
      <button class="nav-button" type="button" data-page="stability"><i class="ri-flask-line"></i><span>稳定性评估</span></button>
    </aside>

    <main>
      <section id="overview" class="page active">
        <div class="page-header">
          <div>
            <h1>总览仪表盘</h1>
            <p class="subhead">收益、风险、交易约束和数据来源的一屏检查。</p>
          </div>
          <div class="mode-tabs" aria-label="时间范围">
            <button type="button" data-range="252">近1年</button>
            <button type="button" data-range="756">近3年</button>
            <button type="button" data-range="all" class="active">全部</button>
          </div>
        </div>
        <div class="context-strip" id="context-strip"></div>
        <div class="section-label">超额 / Alpha 指标</div>
        <div class="metric-grid" id="metrics-excess"></div>
        <div class="section-label">绝对收益 / 风险指标</div>
        <div class="metric-grid" id="metrics-strategy"></div>
        <div class="section-label">基准与交易指标</div>
        <div class="metric-grid" id="metrics-benchmark"></div>

        <div class="panel-grid">
          <article class="panel">
            <div class="panel-header">
              <div>
                <h2 class="panel-title">累计收益曲线 & 回撤</h2>
                <p class="panel-note">红色为策略累计收益，蓝色为基准，紫色为累计超额；下方显示策略和基准回撤。</p>
              </div>
              <button class="icon-button" type="button" data-save="equity-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
            </div>
            <div id="equity-chart" class="chart tall"></div>
          </article>

          <div class="panel-grid two-col">
            <article class="panel">
              <div class="panel-header">
                <div>
                  <h2 class="panel-title">分年收益</h2>
                  <p class="panel-note">对比策略和中证1000分年表现。</p>
                </div>
                <button class="icon-button" type="button" data-save="annual-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
              </div>
              <div id="annual-chart" class="chart short"></div>
            </article>
            <article class="panel">
              <div class="panel-header">
                <div>
                  <h2 class="panel-title">月度收益热力图</h2>
                  <p class="panel-note">快速识别策略在哪些市场阶段失效。</p>
                </div>
                <button class="icon-button" type="button" data-save="monthly-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
              </div>
              <div id="monthly-chart" class="chart short"></div>
            </article>
          </div>

          <div class="panel-grid two-col">
            <article class="panel">
              <div class="panel-header">
                <div>
                  <h2 class="panel-title">换手率与现金权重</h2>
                  <p class="panel-note">显示调仓冲击、现金闲置和成本敏感区间。</p>
                </div>
                <button class="icon-button" type="button" data-save="turnover-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
              </div>
              <div id="turnover-chart" class="chart short"></div>
            </article>
            <article class="panel">
              <div class="panel-header">
                <div>
                  <h2 class="panel-title">优化建议</h2>
                  <p class="panel-note">基于截图版报告和当前数据可用性的前端/量化两侧改进点。</p>
                </div>
              </div>
              <div class="info-grid suggestions">
                <div class="info-block"><h3>数据血缘</h3><p>每个图表显示来源文件、生成时间和是否代理计算，避免把模板图当作真实归因。</p></div>
                <div class="info-block"><h3>场景切换</h3><p>增加成本、调仓周期、持仓数量和基准切换，用同一套 UI 对比敏感性。</p></div>
                <div class="info-block"><h3>归因可信度</h3><p>补充行业分类、风格暴露、因子收益和个股收益后，再升级成严格 Brinson/Barra。</p></div>
                <div class="info-block"><h3>实盘风控</h3><p>增加涨跌停、停牌、成交失败、容量和冲击成本模块，作为实盘前检查清单。</p></div>
              </div>
            </article>
          </div>
        </div>
      </section>

      <section id="brinson" class="page">
        <div class="page-header">
          <div>
            <h1>Brinson 收益归因</h1>
            <p class="subhead">当前是基于持仓风格特征的 Brinson-like 代理视图，用于前端框架和诊断入口。</p>
          </div>
        </div>
        <div class="notice">当前回测产物没有行业分类、行业基准权重和个股分期收益，因此不能生成严格 Brinson 行业配置/选股效应。下方图表使用持仓特征和累计超额收益做代理拆分，真实归因接入数据后可复用同一页面结构。</div>
        <div class="metric-grid" id="metrics-brinson"></div>
        <div class="panel-grid">
          <article class="panel">
            <div class="panel-header">
              <div>
                <h2 class="panel-title">Brinson 三效应累计贡献（代理）</h2>
                <p class="panel-note">配置、选股、交互拆分为模板级代理值；用于确认页面交互和阅读结构。</p>
              </div>
              <button class="icon-button" type="button" data-save="brinson-line-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
            </div>
            <div id="brinson-line-chart" class="chart"></div>
          </article>
          <div class="panel-grid two-col">
            <article class="panel">
              <div class="panel-header">
                <div>
                  <h2 class="panel-title">最新持仓风格分组权重</h2>
                  <p class="panel-note">用市值、换手、波动、动量分桶替代缺失的行业权重。</p>
                </div>
                <button class="icon-button" type="button" data-save="bucket-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
              </div>
              <div id="bucket-chart" class="chart"></div>
            </article>
            <article class="panel">
              <div class="panel-header">
                <div>
                  <h2 class="panel-title">最新高分持仓</h2>
                  <p class="panel-note">用于解释组合主要暴露来自哪些证券。</p>
                </div>
              </div>
              <div class="scroll-table">
                <table class="data-table" id="holdings-table"></table>
              </div>
            </article>
          </div>
        </div>
      </section>

      <section id="barra" class="page">
        <div class="page-header">
          <div>
            <h1>Barra 风险归因</h1>
            <p class="subhead">用策略可得字段构造风格暴露代理，预留未来接入 Barra/CNE5 因子的接口。</p>
          </div>
        </div>
        <div class="notice">严格 Barra 归因需要因子暴露矩阵、因子收益、特异收益、协方差和组合/基准成分。当前页面展示的是 smallcap_lowturn 的信号风格代理：小市值、低换手、低波动、短动量和流动性。</div>
        <div class="panel-grid two-col">
          <article class="panel">
            <div class="panel-header">
              <div>
                <h2 class="panel-title">风格因子累计贡献（代理）</h2>
                <p class="panel-note">根据策略打分权重和最新暴露对累计超额做比例拆分。</p>
              </div>
              <button class="icon-button" type="button" data-save="factor-bar-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
            </div>
            <div id="factor-bar-chart" class="chart"></div>
          </article>
          <article class="panel">
            <div class="panel-header">
              <div>
                <h2 class="panel-title">最新风格暴露雷达</h2>
                <p class="panel-note">暴露值为跨调仓日标准化后的相对读数。</p>
              </div>
              <button class="icon-button" type="button" data-save="factor-radar-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
            </div>
            <div id="factor-radar-chart" class="chart"></div>
          </article>
        </div>
        <article class="panel" style="margin-top:16px">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">风格暴露时序</h2>
              <p class="panel-note">观察组合风格是否随时间漂移，尤其关注低换手和小市值暴露是否衰减。</p>
            </div>
            <button class="icon-button" type="button" data-save="factor-time-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
          </div>
          <div id="factor-time-chart" class="chart"></div>
        </article>
      </section>

      <section id="stability" class="page">
        <div class="page-header">
          <div>
            <h1>稳定性评估</h1>
            <p class="subhead">检验超额收益是否稳定、是否依赖少数年份或过度集中持仓。</p>
          </div>
        </div>
        <div class="panel-grid two-col">
          <article class="panel">
            <div class="panel-header">
              <div>
                <h2 class="panel-title">IR 显著性检验</h2>
                <p class="panel-note" id="ir-note"></p>
              </div>
              <button class="icon-button" type="button" data-save="ttest-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
            </div>
            <div id="ttest-chart" class="chart"></div>
          </article>
          <article class="panel">
            <div class="panel-header">
              <div>
                <h2 class="panel-title">分年度 IR 稳定性</h2>
                <p class="panel-note" id="annual-ir-note"></p>
              </div>
              <button class="icon-button" type="button" data-save="annual-ir-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
            </div>
            <div id="annual-ir-chart" class="chart"></div>
          </article>
        </div>
        <article class="panel" style="margin-top:16px">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">持仓集中度</h2>
              <p class="panel-note">跟踪有效持股数、HHI 和综合得分，识别组合是否变得过于拥挤。</p>
            </div>
            <button class="icon-button" type="button" data-save="concentration-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
          </div>
          <div id="concentration-chart" class="chart"></div>
        </article>
        <article class="panel" style="margin-top:16px">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">滚动 IR 与交易约束</h2>
              <p class="panel-note">把滚动 IR 与涨跌停阻塞次数叠加，定位超额收益回撤是否来自可交易性约束。</p>
            </div>
            <button class="icon-button" type="button" data-save="rolling-ir-chart" title="导出 PNG"><i class="ri-download-line"></i></button>
          </div>
          <div id="rolling-ir-chart" class="chart"></div>
        </article>
      </section>
    </main>
  </div>

  <script>
    const DATA = __DATA__;
    const charts = new Map();
    const palette = {
      red: '#ff454a',
      blue: '#2f89ff',
      green: '#1db56d',
      orange: '#d59b22',
      purple: '#9b69ff',
      cyan: '#46c7e8',
      muted: '#8a98ab',
      grid: '#273140',
      panel: '#131922'
    };

    function init(id) {
      const element = document.getElementById(id);
      if (!element || !window.echarts) return null;
      if (charts.has(id)) return charts.get(id);
      const chart = echarts.init(element, null, { renderer: 'canvas' });
      charts.set(id, chart);
      return chart;
    }

    function baseGrid(extra = {}) {
      return Object.assign({ left: 64, right: 36, top: 54, bottom: 72 }, extra);
    }

    function tooltipPercent(params) {
      const rows = Array.isArray(params) ? params : [params];
      const title = rows[0]?.axisValueLabel || rows[0]?.name || '';
      return [title].concat(rows.map(item => {
        const value = Array.isArray(item.value) ? item.value[item.value.length - 1] : item.value;
        if (value === null || value === undefined || Number.isNaN(value)) return `${item.marker}${item.seriesName}: -`;
        return `${item.marker}${item.seriesName}: ${Number(value).toFixed(2)}%`;
      })).join('<br/>');
    }

    function commonOption() {
      return {
        backgroundColor: 'transparent',
        color: [palette.red, palette.blue, palette.green, palette.orange, palette.purple, palette.cyan],
        textStyle: { color: '#9aa8ba', fontFamily: 'Inter, Microsoft YaHei, Segoe UI, Arial' },
        legend: { textStyle: { color: '#9aa8ba', fontWeight: 700 }, top: 14, right: 28, itemWidth: 16, itemHeight: 8 },
        tooltip: { trigger: 'axis', backgroundColor: '#0b1017', borderColor: '#273140', textStyle: { color: '#e6edf6' } },
        grid: baseGrid(),
        xAxis: { type: 'category', axisLine: { lineStyle: { color: '#354153' } }, axisLabel: { color: '#8a98ab' }, axisTick: { show: false } },
        yAxis: { type: 'value', axisLine: { show: false }, splitLine: { lineStyle: { color: '#273140', type: 'dashed' } }, axisLabel: { color: '#8a98ab' } }
      };
    }

    function metricCard(item) {
      return `<div class="metric-card"><span class="metric-label">${item.label}</span><strong class="metric-value ${item.tone || ''}">${item.value}</strong></div>`;
    }

    function renderMetrics() {
      document.getElementById('metrics-excess').innerHTML = DATA.metrics.excess.map(metricCard).join('');
      document.getElementById('metrics-strategy').innerHTML = DATA.metrics.strategy.map(metricCard).join('');
      document.getElementById('metrics-benchmark').innerHTML = DATA.metrics.benchmark.map(metricCard).join('');
      document.getElementById('metrics-brinson').innerHTML = DATA.brinson.summary.map(metricCard).join('');
      document.getElementById('context-strip').innerHTML = `
        <span>策略：<strong>${DATA.meta.strategy}</strong></span>
        <span>基准：<strong>${DATA.meta.benchmark}</strong></span>
        <span class="context-pill"><i class="ri-calendar-line"></i>${DATA.meta.dateRange}</span>
        <span>交易日：<strong>${DATA.meta.days} 天</strong></span>
        <span>生成：<strong>${DATA.meta.generatedAt}</strong></span>
      `;
    }

    function zoomRange(count) {
      if (count === 'all') return { start: 0, end: 100 };
      const total = DATA.overview.dates.length;
      const start = Math.max(0, (total - Number(count)) / total * 100);
      return { start, end: 100 };
    }

    let activeRange = 'all';

    function renderEquity() {
      const chart = init('equity-chart');
      if (!chart) return;
      const zoom = zoomRange(activeRange);
      chart.setOption({
        ...commonOption(),
        legend: { ...commonOption().legend, data: ['策略净值', '基准净值', '累计超额', '策略回撤', '基准回撤'] },
        tooltip: { ...commonOption().tooltip, formatter: tooltipPercent },
        grid: [{ left: 64, right: 36, top: 54, height: 290 }, { left: 64, right: 36, top: 390, height: 110 }],
        xAxis: [
          { type: 'category', data: DATA.overview.dates, axisLine: { lineStyle: { color: '#354153' } }, axisLabel: { color: '#8a98ab' }, axisTick: { show: false } },
          { type: 'category', gridIndex: 1, data: DATA.overview.dates, axisLine: { lineStyle: { color: '#354153' } }, axisLabel: { color: '#8a98ab' }, axisTick: { show: false } }
        ],
        yAxis: [
          { type: 'value', axisLabel: { formatter: '{value}%', color: '#8a98ab' }, splitLine: { lineStyle: { color: '#273140', type: 'dashed' } } },
          { type: 'value', gridIndex: 1, axisLabel: { formatter: '{value}%', color: '#8a98ab' }, splitLine: { lineStyle: { color: '#273140', type: 'dashed' } } }
        ],
        dataZoom: [
          { type: 'inside', xAxisIndex: [0, 1], start: zoom.start, end: zoom.end },
          { type: 'slider', xAxisIndex: [0, 1], start: zoom.start, end: zoom.end, height: 22, bottom: 20, borderColor: '#273140', backgroundColor: '#101720', fillerColor: 'rgba(47,137,255,.24)', handleStyle: { color: '#2f89ff' }, textStyle: { color: '#8a98ab' } }
        ],
        series: [
          { name: '策略净值', type: 'line', data: DATA.overview.strategyReturn, showSymbol: false, lineStyle: { width: 2.2, color: palette.red } },
          { name: '基准净值', type: 'line', data: DATA.overview.benchmarkReturn, showSymbol: false, lineStyle: { width: 2.2, color: palette.blue } },
          { name: '累计超额', type: 'line', data: DATA.overview.excessReturn, showSymbol: false, lineStyle: { width: 1.8, color: palette.purple }, areaStyle: { color: 'rgba(155,105,255,.12)' } },
          { name: '策略回撤', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: DATA.overview.drawdown, showSymbol: false, lineStyle: { width: 1.8, color: palette.green }, areaStyle: { color: 'rgba(29,181,109,.12)' } },
          { name: '基准回撤', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: DATA.overview.benchmarkDrawdown, showSymbol: false, lineStyle: { width: 1.2, color: palette.muted, type: 'dashed' } }
        ]
      });
    }

    function renderAnnual() {
      const chart = init('annual-chart');
      if (!chart) return;
      chart.setOption({
        ...commonOption(),
        legend: { ...commonOption().legend, data: ['策略', '基准'] },
        tooltip: { ...commonOption().tooltip, formatter: tooltipPercent },
        xAxis: { ...commonOption().xAxis, data: DATA.annual.map(item => item.year) },
        yAxis: { ...commonOption().yAxis, axisLabel: { formatter: '{value}%', color: '#8a98ab' } },
        series: [
          { name: '策略', type: 'bar', data: DATA.annual.map(item => +(item.strategy * 100).toFixed(2)), barMaxWidth: 34, itemStyle: { color: palette.red, borderRadius: [3, 3, 0, 0] } },
          { name: '基准', type: 'bar', data: DATA.annual.map(item => +(item.benchmark * 100).toFixed(2)), barMaxWidth: 34, itemStyle: { color: palette.blue, borderRadius: [3, 3, 0, 0] } }
        ]
      });
    }

    function renderMonthly() {
      const chart = init('monthly-chart');
      if (!chart) return;
      chart.setOption({
        backgroundColor: 'transparent',
        tooltip: { position: 'top', backgroundColor: '#0b1017', borderColor: '#273140', textStyle: { color: '#e6edf6' }, formatter: params => `${DATA.monthly.years[params.value[1]]}-${String(params.value[0] + 1).padStart(2, '0')}<br/>收益: ${params.value[2] ?? '-'}%` },
        grid: { left: 58, right: 28, top: 34, bottom: 36 },
        xAxis: { type: 'category', data: ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'], axisLine: { lineStyle: { color: '#354153' } }, axisTick: { show: false }, axisLabel: { color: '#8a98ab' } },
        yAxis: { type: 'category', data: DATA.monthly.years, axisLine: { lineStyle: { color: '#354153' } }, axisTick: { show: false }, axisLabel: { color: '#8a98ab' } },
        visualMap: { min: -20, max: 20, calculable: false, orient: 'horizontal', left: 'center', bottom: 4, textStyle: { color: '#8a98ab' }, inRange: { color: ['#0f7d55', '#18202b', '#b53239'] } },
        series: [{ type: 'heatmap', data: DATA.monthly.values, label: { show: true, color: '#c7d2e3', fontSize: 10, formatter: p => p.value[2] === null ? '' : p.value[2].toFixed(1) }, emphasis: { itemStyle: { borderColor: '#e6edf6', borderWidth: 1 } } }]
      });
    }

    function renderTurnover() {
      const chart = init('turnover-chart');
      if (!chart) return;
      chart.setOption({
        ...commonOption(),
        legend: { ...commonOption().legend, data: ['日换手', '20日均换手', '现金权重'] },
        tooltip: { ...commonOption().tooltip, formatter: tooltipPercent },
        xAxis: { ...commonOption().xAxis, data: DATA.overview.dates },
        yAxis: { ...commonOption().yAxis, axisLabel: { formatter: '{value}%', color: '#8a98ab' } },
        dataZoom: [{ type: 'inside' }],
        series: [
          { name: '日换手', type: 'bar', data: DATA.overview.turnover, barWidth: 5, itemStyle: { color: 'rgba(47,137,255,.35)' } },
          { name: '20日均换手', type: 'line', data: DATA.overview.turnoverMa, showSymbol: false, lineStyle: { width: 2, color: palette.blue } },
          { name: '现金权重', type: 'line', data: DATA.overview.cashWeight, showSymbol: false, lineStyle: { width: 1.6, color: palette.orange } }
        ]
      });
    }

    function renderBrinson() {
      const chart = init('brinson-line-chart');
      if (chart) {
        chart.setOption({
          ...commonOption(),
          legend: { ...commonOption().legend, data: ['超额收益', '配置效应', '选股效应', '交互效应'] },
          tooltip: { ...commonOption().tooltip, formatter: tooltipPercent },
          xAxis: { ...commonOption().xAxis, data: DATA.brinson.dates },
          yAxis: { ...commonOption().yAxis, axisLabel: { formatter: '{value}%', color: '#8a98ab' } },
          dataZoom: [{ type: 'slider', height: 22, bottom: 20, borderColor: '#273140', backgroundColor: '#101720', fillerColor: 'rgba(47,137,255,.24)', textStyle: { color: '#8a98ab' } }, { type: 'inside' }],
          series: [
            { name: '超额收益', type: 'line', data: DATA.brinson.excess.map(v => +(v * 100).toFixed(2)), showSymbol: false, lineStyle: { width: 2.2, color: palette.blue } },
            { name: '配置效应', type: 'line', data: DATA.brinson.allocation.map(v => +(v * 100).toFixed(2)), showSymbol: false, lineStyle: { width: 2, color: palette.green } },
            { name: '选股效应', type: 'line', data: DATA.brinson.selection.map(v => +(v * 100).toFixed(2)), showSymbol: false, lineStyle: { width: 2, color: palette.orange } },
            { name: '交互效应', type: 'line', data: DATA.brinson.interaction.map(v => +(v * 100).toFixed(2)), showSymbol: false, lineStyle: { width: 1.6, color: palette.muted } }
          ]
        });
      }

      const bucket = init('bucket-chart');
      if (bucket) {
        const groups = Object.entries(DATA.holdings.bucketCharts);
        bucket.setOption({
          ...commonOption(),
          legend: { ...commonOption().legend, data: groups.map(item => item[0]) },
          tooltip: { ...commonOption().tooltip, formatter: tooltipPercent },
          xAxis: { ...commonOption().xAxis, type: 'value', axisLabel: { formatter: '{value}%', color: '#8a98ab' } },
          yAxis: { ...commonOption().yAxis, type: 'category', data: ['第一档', '第二档', '第三档'] },
          series: groups.map((entry, index) => ({
            name: entry[0],
            type: 'bar',
            data: entry[1].map(item => ({ value: +(item.weight * 100).toFixed(2), label: item.name })),
            label: { show: true, position: 'right', color: '#b7c2d2', formatter: p => `${p.data.label} ${p.value.toFixed(1)}%` },
            itemStyle: { color: [palette.red, palette.blue, palette.green, palette.orange][index] },
            barMaxWidth: 18
          }))
        });
      }

      const table = document.getElementById('holdings-table');
      table.innerHTML = `
        <thead><tr><th>代码</th><th>名称</th><th>权重</th><th>分数</th><th>20日换手</th><th>60日波动</th><th>20日收益</th></tr></thead>
        <tbody>${DATA.holdings.topHoldings.map(item => `
          <tr>
            <td>${item.ts_code}</td>
            <td>${item.name}</td>
            <td>${(item.target_weight * 100).toFixed(2)}%</td>
            <td>${item.score.toFixed(3)}</td>
            <td>${item.turn20.toFixed(2)}</td>
            <td>${(item.vol60 * 100).toFixed(1)}%</td>
            <td class="${item.ret20 >= 0 ? 'metric-value up' : 'metric-value down'}" style="font-size:13px">${(item.ret20 * 100).toFixed(2)}%</td>
          </tr>`).join('')}</tbody>`;
    }

    function renderBarra() {
      const bar = init('factor-bar-chart');
      if (bar) {
        const rows = DATA.barra.factorContribution;
        bar.setOption({
          ...commonOption(),
          tooltip: { ...commonOption().tooltip, formatter: params => `${params.name}<br/>贡献: ${(params.value).toFixed(2)}%<br/>暴露: ${params.data.exposure.toFixed(2)}` },
          xAxis: { ...commonOption().xAxis, type: 'value', axisLabel: { formatter: '{value}%', color: '#8a98ab' } },
          yAxis: { ...commonOption().yAxis, type: 'category', data: rows.map(item => item.name) },
          series: [{ type: 'bar', data: rows.map(item => ({ value: +(item.value * 100).toFixed(2), exposure: item.exposure })), barMaxWidth: 28, itemStyle: { color: params => params.value >= 0 ? palette.red : palette.green, borderRadius: [0, 4, 4, 0] }, label: { show: true, position: 'right', color: '#c7d2e3', formatter: p => `${p.value.toFixed(2)}%` } }]
        });
      }

      const radar = init('factor-radar-chart');
      if (radar) {
        const names = DATA.barra.factorNames;
        const values = names.map(name => +(DATA.holdings.latestExposure[name] || 0).toFixed(2));
        radar.setOption({
          backgroundColor: 'transparent',
          tooltip: { backgroundColor: '#0b1017', borderColor: '#273140', textStyle: { color: '#e6edf6' } },
          radar: { center: ['50%', '54%'], radius: '64%', indicator: names.map(name => ({ name, max: 2.5, min: -2.5 })), axisName: { color: '#8a98ab' }, splitLine: { lineStyle: { color: '#354153' } }, splitArea: { areaStyle: { color: ['rgba(47,137,255,.04)', 'rgba(47,137,255,.08)'] } }, axisLine: { lineStyle: { color: '#354153' } } },
          series: [{ type: 'radar', data: [{ value: values, name: '最新暴露' }], symbolSize: 5, lineStyle: { color: palette.blue, width: 2 }, areaStyle: { color: 'rgba(47,137,255,.18)' }, itemStyle: { color: palette.blue } }]
        });
      }

      const time = init('factor-time-chart');
      if (time) {
        const names = DATA.barra.factorNames;
        time.setOption({
          ...commonOption(),
          legend: { ...commonOption().legend, data: names },
          tooltip: { ...commonOption().tooltip },
          xAxis: { ...commonOption().xAxis, data: DATA.holdings.dates },
          yAxis: { ...commonOption().yAxis, name: '标准化暴露', nameTextStyle: { color: '#8a98ab' } },
          dataZoom: [{ type: 'slider', height: 22, bottom: 20, borderColor: '#273140', backgroundColor: '#101720', fillerColor: 'rgba(47,137,255,.24)', textStyle: { color: '#8a98ab' } }, { type: 'inside' }],
          series: names.map((name, index) => ({ name, type: 'line', data: DATA.holdings.exposure[name], showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: [palette.blue, palette.green, palette.orange, palette.red, palette.purple][index] } }))
        });
      }
    }

    function renderStability() {
      const ir = DATA.metrics.ir;
      document.getElementById('ir-note').textContent = `实际 IR ${ir.value.toFixed(3)}，t=${ir.tStat.toFixed(3)}，p=${ir.pValue.toFixed(4)}，样本 ${ir.sample} 个交易日。`;
      const annualIrs = DATA.annualIr.map(item => item.ir);
      const meanIr = annualIrs.reduce((a, b) => a + b, 0) / Math.max(1, annualIrs.length);
      const cv = Math.sqrt(annualIrs.reduce((a, b) => a + Math.pow(b - meanIr, 2), 0) / Math.max(1, annualIrs.length)) / Math.max(0.0001, Math.abs(meanIr));
      document.getElementById('annual-ir-note').textContent = `年度 IR 平均 ${meanIr.toFixed(2)}，CV=${cv.toFixed(2)}；CV 越低，年度间一致性越好。`;

      const ttest = init('ttest-chart');
      if (ttest) {
        const xs = Array.from({ length: 241 }, (_, i) => -12 + i * 0.1);
        const ys = xs.map(x => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI));
        ttest.setOption({
          ...commonOption(),
          legend: { show: false },
          tooltip: { trigger: 'axis', backgroundColor: '#0b1017', borderColor: '#273140', textStyle: { color: '#e6edf6' } },
          xAxis: { ...commonOption().xAxis, type: 'value', min: -12, max: 12 },
          yAxis: { ...commonOption().yAxis, min: 0 },
          series: [
            { name: '零假设 t 分布近似', type: 'line', data: xs.map((x, i) => [x, ys[i]]), showSymbol: false, lineStyle: { color: palette.blue, width: 2 } },
            { name: '实际 t', type: 'line', markLine: { symbol: 'none', data: [{ xAxis: ir.tStat, label: { formatter: `t=${ir.tStat.toFixed(2)}`, color: palette.red }, lineStyle: { color: palette.red, type: 'dashed', width: 2 } }] }, data: [] }
          ]
        });
      }

      const annual = init('annual-ir-chart');
      if (annual) {
        annual.setOption({
          ...commonOption(),
          tooltip: { ...commonOption().tooltip, formatter: params => `${params.name}<br/>IR: ${params.value.toFixed(2)}` },
          xAxis: { ...commonOption().xAxis, data: DATA.annualIr.map(item => item.year) },
          yAxis: { ...commonOption().yAxis },
          series: [{ type: 'bar', data: DATA.annualIr.map(item => +item.ir.toFixed(2)), barMaxWidth: 42, itemStyle: { color: palette.blue, borderRadius: [4, 4, 0, 0] }, label: { show: true, position: 'top', color: '#9aa8ba' } }]
        });
      }

      const concentration = init('concentration-chart');
      if (concentration) {
        const rows = DATA.holdings.concentration;
        concentration.setOption({
          ...commonOption(),
          legend: { ...commonOption().legend, data: ['有效持股数', 'HHI %', '平均得分'] },
          tooltip: { ...commonOption().tooltip },
          xAxis: { ...commonOption().xAxis, data: rows.map(item => item.date) },
          yAxis: [
            { type: 'value', axisLabel: { color: '#8a98ab' }, splitLine: { lineStyle: { color: '#273140', type: 'dashed' } } },
            { type: 'value', axisLabel: { formatter: '{value}%', color: '#8a98ab' }, splitLine: { show: false } }
          ],
          dataZoom: [{ type: 'slider', height: 22, bottom: 20, borderColor: '#273140', backgroundColor: '#101720', fillerColor: 'rgba(47,137,255,.24)', textStyle: { color: '#8a98ab' } }, { type: 'inside' }],
          series: [
            { name: '有效持股数', type: 'line', data: rows.map(item => +item.effective_holdings.toFixed(1)), showSymbol: false, lineStyle: { color: palette.blue, width: 2 } },
            { name: 'HHI %', type: 'line', yAxisIndex: 1, data: rows.map(item => +(item.hhi * 100).toFixed(3)), showSymbol: false, lineStyle: { color: palette.orange, width: 1.8, type: 'dashed' } },
            { name: '平均得分', type: 'line', data: rows.map(item => +(item.avg_score * 1000).toFixed(2)), showSymbol: false, lineStyle: { color: palette.purple, width: 1.8, type: 'dashed' } }
          ]
        });
      }

      const rolling = init('rolling-ir-chart');
      if (rolling) {
        rolling.setOption({
          ...commonOption(),
          legend: { ...commonOption().legend, data: ['126日滚动IR', '阻塞买入', '阻塞卖出'] },
          tooltip: { ...commonOption().tooltip },
          xAxis: { ...commonOption().xAxis, data: DATA.overview.dates },
          yAxis: [
            { type: 'value', axisLabel: { color: '#8a98ab' }, splitLine: { lineStyle: { color: '#273140', type: 'dashed' } } },
            { type: 'value', axisLabel: { color: '#8a98ab' }, splitLine: { show: false } }
          ],
          dataZoom: [{ type: 'slider', height: 22, bottom: 20, borderColor: '#273140', backgroundColor: '#101720', fillerColor: 'rgba(47,137,255,.24)', textStyle: { color: '#8a98ab' } }, { type: 'inside' }],
          series: [
            { name: '126日滚动IR', type: 'line', data: DATA.overview.rollingIr.map(v => v === null ? null : +v.toFixed(2)), showSymbol: false, lineStyle: { color: palette.blue, width: 2 } },
            { name: '阻塞买入', type: 'bar', yAxisIndex: 1, data: DATA.overview.blockedBuys, barWidth: 4, itemStyle: { color: 'rgba(255,69,74,.32)' } },
            { name: '阻塞卖出', type: 'bar', yAxisIndex: 1, data: DATA.overview.blockedSells, barWidth: 4, itemStyle: { color: 'rgba(29,181,109,.32)' } }
          ]
        });
      }
    }

    function renderAll() {
      renderMetrics();
      renderEquity();
      renderAnnual();
      renderMonthly();
      renderTurnover();
      renderBrinson();
      renderBarra();
      renderStability();
    }

    document.querySelectorAll('.nav-button').forEach(button => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.nav-button').forEach(item => item.classList.remove('active'));
        document.querySelectorAll('.page').forEach(item => item.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.page).classList.add('active');
        setTimeout(() => charts.forEach(chart => chart.resize()), 40);
      });
    });

    document.querySelectorAll('.mode-tabs button').forEach(button => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.mode-tabs button').forEach(item => item.classList.remove('active'));
        button.classList.add('active');
        activeRange = button.dataset.range;
        renderEquity();
      });
    });

    document.querySelectorAll('[data-save]').forEach(button => {
      button.addEventListener('click', () => {
        const chart = charts.get(button.dataset.save);
        if (!chart) return;
        const url = chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#131922' });
        const link = document.createElement('a');
        link.href = url;
        link.download = `${button.dataset.save}.png`;
        link.click();
      });
    });

    window.addEventListener('resize', () => charts.forEach(chart => chart.resize()));

    if (!window.echarts) {
      document.body.innerHTML = '<div style="padding:32px;color:#e6edf6;font-family:Microsoft YaHei,Arial">ECharts CDN 未加载，请联网后重新打开，或把 echarts.min.js 放到本地并替换 script 地址。</div>';
    } else {
      renderAll();
    }
  </script>
</body>
</html>
"""


def write_html(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__DATA__", payload)
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dark interactive backtest dashboard.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Backtest report directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output HTML path.")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    required = ["summary.json", "daily_returns.csv"]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required report files in {source}: {', '.join(missing)}")

    data = build_data(source)
    write_html(data, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
