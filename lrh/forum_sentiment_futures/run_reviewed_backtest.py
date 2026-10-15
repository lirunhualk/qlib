from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lrh.forum_sentiment_futures.config import FUTURE_SPECS
from lrh.forum_sentiment_futures.data_loader import MarketDataLoader
from lrh.forum_sentiment_futures.db import get_engine
from lrh.forum_sentiment_futures.hybrid_core import TrendCoreConfig, run_trend_core
from lrh.forum_sentiment_futures.intraday_engine import (
    IntradayStrategyConfig, performance_metrics, run_intraday_backtest,
)
from lrh.forum_sentiment_futures.intraday_features import build_intraday_features
from lrh.forum_sentiment_futures.intraday_sources import (
    assemble_dominant_intraday, build_execution_contract_map, load_or_fetch_sina_bars,
)


@dataclass(frozen=True)
class ActivityConfig:
    activity_z: float = 0.5
    speed_z: float = 0.5
    both_sides: bool = True
    minimum_day_move: float = 0.004
    min_posts: int = 5
    min_authors: int = 3


def activity_signal(features: pd.DataFrame, config: ActivityConfig, *, price_only: bool = False) -> pd.Series:
    common = features["bar_slot"].between(2, 12) & features["realized_vol_annualized"].between(0.01, 0.35)
    if not price_only:
        common &= (
            features["post_count_60m"].ge(config.min_posts)
            & features["unique_user_count_60m"].ge(config.min_authors)
            & features["activity_z"].ge(config.activity_z)
            & features["activity_speed_z"].ge(config.speed_z)
            & features["post_count"].gt(0)
        )
    long = common & features["day_return"].le(-config.minimum_day_move) & features["bar_return"].ge(0)
    short = common & features["day_return"].ge(config.minimum_day_move) & features["bar_return"].le(0)
    signal = pd.Series(0, index=features.index, dtype="int8")
    signal.loc[long] = 1
    if config.both_sides:
        signal.loc[short] = -1
    return signal


def executable_daily(market: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    source = market.set_index(["trade_date", "ts_code"], verify_integrity=True)
    rows = []
    previous_contract = None
    for row in mapping.sort_values("trade_date").itertuples(index=False):
        contract = row.target_contract
        today = source.loc[(row.trade_date, contract)]
        old = source.loc[(row.trade_date, previous_contract or contract)]
        previous_close = float(old["prev_close"])
        if not np.isfinite(previous_close) or previous_close <= 0:
            previous_contract = contract
            continue
        gap = float(old["open"] / previous_close - 1.0)
        intraday = float(today["close"] / today["open"] - 1.0)
        turnover = 1.0 if previous_contract is None else (2.0 if contract != previous_contract else 0.0)
        gross = (1.0 + gap) * (1.0 + intraday) - 1.0
        rows.append({
            "fut_code": row.fut_code, "trade_date": row.trade_date,
            "target_contract": contract, "overnight_return": gap,
            "intraday_return": intraday, "strategy_return_gross": gross,
            "hold_turnover": turnover, "hold_cost": turnover / 10000.0,
            "hold_return": gross - turnover / 10000.0,
        })
        previous_contract = contract
    return pd.DataFrame(rows)


def evaluate(returns: pd.Series, trade_counts: pd.Series | None = None) -> dict:
    if returns.isna().any():
        raise ValueError("Missing returns cannot be interpreted as flat days")
    return performance_metrics(pd.DataFrame({"strategy_return_net": returns, "trade_count": 0 if trade_counts is None else trade_counts}))


def complete_daily(daily: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    keys = ["fut_code", "trade_date"]
    columns = ["strategy_return_gross", "strategy_return_net", "transaction_cost", "turnover", "trade_count"]
    if daily.empty:
        result = calendar[keys].copy()
        for column in columns:
            result[column] = 0.0
    else:
        result = calendar[keys].merge(daily[keys + columns], on=keys, how="left", validate="one_to_one")
        result[columns] = result[columns].fillna(0.0)
    return result


def trade_counts(frame: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "production":
        return frame["production_trade_count"]
    if strategy == "core":
        return frame["core_trade_count"]
    if strategy == "hold":
        return frame["hold_turnover"].gt(0).astype(int)
    column = f'{strategy.removesuffix("_overlay")}_trade_count'
    result = frame[column]
    if not strategy.endswith("_overlay"):
        result = result + frame["core_trade_count"]
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(output: Path, daily: pd.DataFrame, summary: pd.DataFrame, annual: pd.DataFrame, manifest: dict) -> None:
    full = summary[summary["period"].eq("full")]
    headline = full[full["strategy"].isin(["production", "behavior", "hold"])].copy()
    headline = headline[["fut_code", "strategy", "annualized_return", "annualized_volatility", "max_drawdown", "total_return", "trade_count"]]
    for column in ("annualized_return", "annualized_volatility", "max_drawdown", "total_return"):
        headline[column] = headline[column].map(lambda value: f"{value:.2%}")
    annual_wide = annual.pivot(index=["fut_code", "year", "observations"], columns="strategy", values="annualized_return")
    annual_wide = annual_wide[["production", "legacy", "behavior", "core", "price_only", "hold"]].reset_index()
    formatted_annual = annual_wide.copy()
    for column in ["production", "legacy", "behavior", "core", "price_only", "hold"]:
        formatted_annual[column] = formatted_annual[column].map(lambda value: f"{value:.2%}")
    frozen = manifest["production_overlay_by_code"]
    portfolio = summary[(summary.fut_code == "equal_weight_4") & (summary.period == "full")].set_index("strategy")
    headline_conclusion = (
        f'四品种等权生产组合年化 {portfolio.loc["production", "annualized_return"]:.2%}、'
        f'年化波动 {portfolio.loc["production", "annualized_volatility"]:.2%}、'
        f'最大回撤 {portfolio.loc["production", "max_drawdown"]:.2%}；'
        f'强制启用纯行为覆盖层的组合年化 {portfolio.loc["behavior", "annualized_return"]:.2%}。'
        '行为信号未通过训练部署门槛，本次没有证据支持收益提升，生产口径只保留趋势核心。'
        if all(value["selected"] == "none" for value in frozen.values())
        else '覆盖层部署与否由训练期门槛决定，样本外结果见 summary.csv。'
    )
    out_of_sample = summary[(summary.period == "out_of_sample") & summary.strategy.isin(["production", "behavior", "behavior_overlay", "price_only", "hold"])].copy()
    out_of_sample = out_of_sample[["fut_code", "strategy", "annualized_return", "annualized_volatility", "max_drawdown", "trade_count"]]
    for column in ("annualized_return", "annualized_volatility", "max_drawdown"):
        out_of_sample[column] = out_of_sample[column].map(lambda value: f"{value:.2%}")
    report = f'''# 财经论坛行为策略审查与实际回测

{headline_conclusion}

数据：东方财富四个指数股吧真实帖子、本机 MySQL 期货日线和新浪分合约 15 分钟 K 线。
请求区间：{manifest["start_date"]} 至 {manifest["end_date"]}；IM 从上市后首个可执行交易日开始。
本次实际使用 {manifest["post_rows"]:,} 条帖子。雪球没有可用历史流，本次已按用户要求更换数据源。

## 全期结果

{headline.to_markdown(index=False)}

## 样本外

{out_of_sample.to_markdown(index=False)}

## 每年年化

{formatted_annual.to_markdown(index=False)}

2021、2026 和 IM 的 2022 年是不完整年度，按 252 个交易日折算年化；年度实际收益同时保存于 annual_comparison.csv。

## 选择与归因

候选训练区间截止 2024-12-31，2025-01-01 以后只用于评估；2026-07-27 至最新日期是本轮新增数据区间。
纯行为版本共 8 个候选，采用发帖量、独立作者广度和活动变化速度。纯价格版本保留同样的价格和执行条件。
生产覆盖层选择：{json.dumps(frozen, ensure_ascii=False)}。选择只依赖训练期净收益、Sharpe 和交易笔数，未用样本外收益选参数。
生产策略包含趋势核心，因此混合收益不能归因于论坛；summary.csv 同时列出独立覆盖层和纯价格消融。

## 修复内容

- 仓位在信号 K 线收盘时锁定；下一根 K 线数据不参与定仓。
- 60 分钟作者数取作者 ID 的并集，防止重复作者被误算为情绪广度。
- 趋势核心隔夜使用旧仓，开盘后使用新仓；换月隔夜损益来自实际旧合约。
- 跳空穿越止损按开盘成交，日内损益和成本逐笔对账；回撤含初始资金。
- 雪球模块的覆盖层收益使用独立列名，修复基准同名列覆盖问题。

## 成本、执行与数据限制

行为与情绪信号使用发布时间加 60 秒的假设采集延迟，在 15 分钟收盘确认后于下一根 K 线开盘成交。
帖子只映射至其后 15 分钟内的行情 K 线，跨夜、午间休市以及缺失交易日的帖子不挤入下一根 K 线。
60 分钟窗口指同日最近 4 根交易 K 线；午休不计入交易时间。
覆盖层单边手续费 1bp、滑点 1bp，成本敏感性另测 0/2/4/8bp 总单边成本；核心与持有单边 1bp。
目标波动是定仓参数而非保证，核心仓位上限 0.5、覆盖层上限 0.35、每日最多一笔覆盖交易。
名义仓位允许小数，未按整数合约、保证金、逐笔盘口、涨跌停排队和容量建模。
历史帖子为当前可见存续帖，删除帖和漏收不可恢复；文本版本可能被编辑，纯行为版本不使用正文。
可得的作者与发布时间仍是回溯数据，60 秒采集延迟是研究假设，并非历史实时采集证据。
盘中缺任一标准 K 线时覆盖层不交易，核心按日线继续；所有缺口列在 data_coverage.csv。

复现命令、参数、源文件 SHA256 和环境时间保存于 manifest.json；日报保存于各策略/品种 qlib_report。
'''
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    figures = []
    for code, frame in daily.groupby("fut_code", sort=True):
        fig = go.Figure()
        for strategy, label in [("production", "生产组合"), ("core", "趋势核心"), ("behavior", "纯行为候选组合"), ("hold", "一直持有")]:
            fig.add_trace(go.Scatter(x=frame.trade_date, y=(1 + frame[f"{strategy}_return"]).cumprod(), name=label))
        fig.update_layout(title=f"{code} 扣成本净值", template="plotly_white", height=380)
        figures.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if not figures else False))
        dd = go.Figure()
        for strategy, label in [("production", "生产组合"), ("hold", "一直持有")]:
            nav = (1 + frame[f"{strategy}_return"]).cumprod()
            dd.add_trace(go.Scatter(x=frame.trade_date, y=nav / nav.cummax().clip(lower=1) - 1, name=label))
        dd.update_layout(title=f"{code} 回撤", yaxis_tickformat=".0%", template="plotly_white", height=280)
        figures.append(pio.to_html(dd, full_html=False, include_plotlyjs=False))
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>论坛行为实际回测</title><style>body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;color:#1f2937}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #ddd}.table{overflow-x:auto}section{max-width:1400px;margin:auto}</style><section><h1>论坛行为实际回测</h1><p>' + headline_conclusion + '</p><p>' + f'{manifest["start_date"]} 至 {manifest["end_date"]} | 东方财富 | 扣除成本 | 训练截止 2024-12-31' + '</p><div class="table">' + headline.to_html(index=False) + '</div><h2>样本外</h2><div class="table">' + out_of_sample.to_html(index=False) + '</div><h2>逐年年化</h2><div class="table">' + formatted_annual.to_html(index=False) + '</div>' + ''.join(figures) + '<p>详细方法、源文件哈希、样本外和成本敏感性见同目录 REPORT.md、manifest.json 和 CSV 文件。图表加载需要访问 Plotly CDN。</p></section></html>'
    (output / "dashboard.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2021-10-11")
    parser.add_argument("--end-date", default="2026-10-09")
    parser.add_argument("--posts-file", default=str(Path(__file__).parent / "data/eastmoney_forum_posts_20261011.csv"))
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "reviewed_report_20261011"))
    parser.add_argument("--refresh-intraday", action="store_true")
    args = parser.parse_args()
    start, end, train_end = pd.Timestamp(args.start_date), pd.Timestamp(args.end_date), pd.Timestamp("2024-12-31")
    warmup = start - pd.Timedelta(days=220)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    posts_path = Path(args.posts_file)
    posts = pd.read_csv(posts_path, parse_dates=["publish_time", "date"], dtype={"post_id": str, "user_id": str})
    if posts.duplicated(["board_code", "post_id"]).any():
        raise ValueError("Duplicate forum post keys")
    posts["publish_time"] += pd.Timedelta(seconds=60)
    # Activity variants are independent of potentially revised historical text.
    behavior_posts = posts.copy()
    behavior_posts["title"] = ""
    loader = MarketDataLoader(get_engine())
    mappings, bundles, core_by_code = {}, {}, {}
    for code in FUTURE_SPECS:
        mappings[code] = build_execution_contract_map(loader, code, start_date=warmup, end_date=end)
        bundles[code] = loader.load_market_bundle(code, start_date=warmup.date().isoformat(), end_date=end.date().isoformat())
        hold = executable_daily(bundles[code].market, mappings[code])
        core_by_code[code] = run_trend_core(hold, TrendCoreConfig())
    all_mapping = pd.concat(mappings.values(), ignore_index=True)
    # Network progress is captured; only declared JSON fields cross the CLI boundary.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        bars = load_or_fetch_sina_bars(all_mapping.target_contract.unique(), cache_dir=Path(__file__).parent / "data/sina_15m", refresh=args.refresh_intraday)
    feature_by_code, coverage_frames = {}, []
    for code, mapping in mappings.items():
        dominant, coverage = assemble_dominant_intraday(mapping, bars)
        coverage_frames.append(coverage)
        activity = build_intraday_features(dominant, behavior_posts, board_code=FUTURE_SPECS[code].forum_board_code)
        text_features = build_intraday_features(dominant, posts, board_code=FUTURE_SPECS[code].forum_board_code)
        feature_by_code[code] = (activity, text_features)
    coverage_all = pd.concat(coverage_frames, ignore_index=True)
    coverage_all[coverage_all.trade_date.between(start, end)].to_csv(output / "data_coverage.csv", index=False, encoding="utf-8-sig")

    candidates = [ActivityConfig(level, speed, both) for level in (0.5, 1.25) for speed in (0.5, 1.25) for both in (False, True)]
    execution = IntradayStrategyConfig(max_holding_bars=2, stop_loss=0.004, take_profit=0.006, target_volatility=0.06, max_position_size=0.35, min_position_size=0.10, max_trades_per_day=1, cooldown_bars=2, daily_loss_limit=0.006)
    legacy_params = json.loads((Path(__file__).parent / "insight_update_report/legacy/selected_params.json").read_text(encoding="utf-8"))
    legacy_dict = legacy_params["intraday_overlay_config"]
    legacy_values = {key: value for key, value in legacy_dict.items() if key in IntradayStrategyConfig.__dataclass_fields__}
    if not legacy_values:
        raise ValueError("Saved original intraday parameters cannot be found")
    legacy_config = IntradayStrategyConfig(**legacy_values)

    daily_frames, trade_frames, search_rows, sensitivity_rows, choices = [], [], [], [], {}
    for code, (activity, text_features) in feature_by_code.items():
        core = core_by_code[code]
        calendar = core[core.trade_date.between(start, end)].copy()
        results = []
        for candidate_id, candidate in enumerate(candidates):
            selected = activity.copy()
            selected["review_signal"] = activity_signal(selected, candidate)
            day, trades = run_intraday_backtest(selected, execution, signal_column="review_signal")
            day = complete_daily(day, calendar)
            train = day[day.trade_date.le(train_end)]
            metrics = performance_metrics(train)
            search_rows.append({"fut_code": code, "candidate_id": candidate_id, **asdict(candidate), **metrics})
            results.append((metrics, candidate_id, day, trades))
        winner = max(results, key=lambda item: (item[0]["sharpe"], item[0]["annualized_return"]))
        metrics, candidate_id, behavior, behavior_trades = winner
        legacy_day, legacy_trades = run_intraday_backtest(text_features, legacy_config)
        legacy_day = complete_daily(legacy_day, calendar)
        legacy_metrics = performance_metrics(legacy_day[legacy_day.trade_date.le(train_end)])
        behavior_ok = metrics["trade_count"] >= 20 and metrics["sharpe"] >= 0.5 and metrics["annualized_return"] > 0
        legacy_ok = legacy_metrics["trade_count"] >= 20 and legacy_metrics["sharpe"] >= 0.5 and legacy_metrics["annualized_return"] > 0
        choice = "behavior" if behavior_ok and (not legacy_ok or metrics["sharpe"] > legacy_metrics["sharpe"]) else ("legacy" if legacy_ok else "none")
        choices[code] = {"selected": choice, "behavior_candidate_id": candidate_id, "behavior_eligible": bool(behavior_ok), "legacy_eligible": bool(legacy_ok)}
        selected = activity.copy()
        selected["review_signal"] = activity_signal(selected, candidates[candidate_id], price_only=True)
        price_day, price_trades = run_intraday_backtest(selected, execution, signal_column="review_signal")
        price_day = complete_daily(price_day, calendar)
        frame = calendar.copy()
        variants = {"behavior": behavior, "legacy": legacy_day, "price_only": price_day}
        for name, day in variants.items():
            day = day.rename(columns={column: f"{name}_{column}" for column in ("strategy_return_gross", "strategy_return_net", "transaction_cost", "turnover", "trade_count")})
            frame = frame.merge(day, on=["fut_code", "trade_date"], validate="one_to_one")
            frame[f"{name}_return"] = frame["core_return_net"] + frame[f"{name}_strategy_return_net"]
            frame[f"{name}_overlay_return"] = frame[f"{name}_strategy_return_net"]
        frame["production_return"] = frame["core_return_net"] + (frame[f"{choice}_strategy_return_net"] if choice != "none" else 0.0)
        frame["production_gross"] = frame["core_return_gross"] + (frame[f"{choice}_strategy_return_gross"] if choice != "none" else 0.0)
        frame["production_cost"] = frame["core_transaction_cost"] + (frame[f"{choice}_transaction_cost"] if choice != "none" else 0.0)
        frame["production_turnover"] = frame["core_turnover"] + (frame[f"{choice}_turnover"] if choice != "none" else 0.0)
        frame["core_return"] = frame["core_return_net"]
        frame["production_trade_count"] = frame["core_trade_count"] + (frame[f"{choice}_trade_count"] if choice != "none" else 0)
        daily_frames.append(frame)
        for name, trades in [("behavior", behavior_trades), ("legacy", legacy_trades), ("price_only", price_trades)]:
            if not trades.empty:
                trades = trades[trades.trade_date.between(start, end)].copy()
                trades["variant"] = name
                trades["deployed"] = name == choice
                trade_frames.append(trades)
        for total_bps in (0.0, 2.0, 4.0, 8.0):
            for name in ("behavior", "legacy", "price_only"):
                returns = frame["core_return_net"] + frame[f"{name}_strategy_return_gross"] - frame[f"{name}_turnover"] * total_bps / 10000
                for period, mask in [("full", pd.Series(True, index=frame.index)), ("out_of_sample", frame.trade_date.gt(train_end))]:
                    sensitivity_rows.append({"fut_code": code, "variant": name, "period": period, "total_bps_per_side": total_bps, **evaluate(returns[mask])})
        delayed = activity.copy()
        delayed["review_signal"] = activity_signal(delayed, candidates[candidate_id]).groupby(delayed.trade_date).shift(1).fillna(0)
        delayed_day, _ = run_intraday_backtest(delayed, execution, signal_column="review_signal")
        delayed_day = complete_daily(delayed_day, calendar)
        frame["delayed_behavior_return"] = frame["core_return_net"] + delayed_day["strategy_return_net"].to_numpy()
        frame["delayed_behavior_trade_count"] = delayed_day["trade_count"].to_numpy()

    daily = pd.concat(daily_frames, ignore_index=True).sort_values(["fut_code", "trade_date"])
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary_rows, annual_rows = [], []
    strategies = ["production", "legacy", "behavior", "core", "price_only", "hold", "behavior_overlay", "legacy_overlay", "price_only_overlay", "delayed_behavior"]
    for code, frame in daily.groupby("fut_code"):
        periods = {"full": frame, "train": frame[frame.trade_date.le(train_end)], "out_of_sample": frame[frame.trade_date.gt(train_end)], "new_since_july": frame[frame.trade_date.ge("2026-07-27")]}
        for period, subset in periods.items():
            for strategy in strategies:
                summary_rows.append({"fut_code": code, "period": period, "strategy": strategy, **evaluate(subset[f"{strategy}_return"], trade_counts(subset, strategy)), "start_date": str(subset.trade_date.min().date()), "end_date": str(subset.trade_date.max().date())})
        for year, subset in frame.groupby(frame.trade_date.dt.year):
            for strategy in strategies:
                annual_rows.append({"fut_code": code, "year": year, "strategy": strategy, **evaluate(subset[f"{strategy}_return"], trade_counts(subset, strategy))})
        for strategy, gross, cost, turnover in [("production", "production_gross", "production_cost", "production_turnover"), ("core", "core_return_gross", "core_transaction_cost", "core_turnover")]:
            destination = output / strategy / code / "qlib_report"
            destination.mkdir(parents=True, exist_ok=True)
            frame.set_index("trade_date")[[gross, "hold_return", cost, turnover]].rename(columns={gross: "return", "hold_return": "bench", cost: "cost", turnover: "turnover"}).rename_axis("date").to_csv(destination / "qlib_report.csv", encoding="utf-8-sig")
    common = daily.groupby("trade_date").filter(lambda frame: frame.fut_code.nunique() == 4)
    portfolio = common.groupby("trade_date")[[f"{name}_return" for name in strategies]].mean()
    for period, subset in [("full", portfolio), ("out_of_sample", portfolio[portfolio.index > train_end]), ("new_since_july", portfolio[portfolio.index >= "2026-07-27"])]:
        for strategy in strategies:
            portfolio_count = int(trade_counts(common[common.trade_date.between(subset.index.min(), subset.index.max())], strategy).sum())
            summary_rows.append({"fut_code": "equal_weight_4", "period": period, "strategy": strategy, **evaluate(subset[f"{strategy}_return"]), "trade_count": portfolio_count, "start_date": str(subset.index.min().date()), "end_date": str(subset.index.max().date())})
    summary, annual = pd.DataFrame(summary_rows), pd.DataFrame(annual_rows)
    daily.to_csv(output / "daily_returns.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(search_rows).to_csv(output / "training_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity_rows).to_csv(output / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(output / "annual_comparison.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(output / "portfolio_daily.csv", encoding="utf-8-sig")
    manifest = {
        "start_date": args.start_date, "end_date": args.end_date, "train_end": str(train_end.date()),
        "post_rows": len(posts), "post_file": str(posts_path.resolve()), "post_sha256": _sha(posts_path),
        "data_source": "eastmoney_index_forums_mysql_cffex_sina_15m", "execution": execution.to_dict(),
        "legacy_parameters": legacy_config.to_dict(), "core": TrendCoreConfig().to_dict(),
        "production_overlay_by_code": choices, "selection_uses_out_of_sample": False,
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "code_sha256": {path.name: _sha(path) for path in [Path(__file__), Path(__file__).parent / "intraday_engine.py", Path(__file__).parent / "intraday_features.py", Path(__file__).parent / "hybrid_core.py"]},
        "used_qlib": False, "renderer": "plotly", "collection_delay_seconds": 60,
        "bar_cache_sha256": {str(path.name): _sha(path) for path in sorted((Path(__file__).parent / "data/sina_15m").glob("*.csv")) if path.name.split("_")[0] + ".CFX" in set(all_mapping.target_contract)},
        "command": "python lrh/forum_sentiment_futures/run_reviewed_backtest.py --start-date " + args.start_date + " --end-date " + args.end_date,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(output, daily, summary, annual, manifest)
    print(json.dumps({"report_path": str(output.resolve()), "post_rows": len(posts), "daily_rows": len(daily), "trade_rows": len(trades), "completed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
