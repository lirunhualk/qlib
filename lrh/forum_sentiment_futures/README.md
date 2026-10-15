# 财经论坛情绪股指期货策略

最新审查与实际回测位于 `reviewed_report_20261011`，区间更新至 2026-10-09，
并已修复成交时点、收益归因和作者广度问题。下方 2026-07 的结果属于旧区间留档；
当前结论以 `reviewed_report_20261011/REPORT.md` 为准。

本目录参考集思录文章中“用财经论坛行为数据生成股指期货信号”的研究思路，构建了
`IF/IC/IH/IM` 四品种的可复现回测。文章没有公开完整原始数据和精确公式，因此这里是
独立实现，不声称复刻博主实盘。

当前包含三套研究：

- 原始日频策略：论坛日频情绪在 T 日收盘后形成信号，T+1 日内交易。
- 风控混合策略：多周期趋势核心控制风险，15 分钟论坛情绪只做经过训练门槛验证的盘中增强。
- 原帖更新研究：增加情绪一阶速度、反应性/预期性文本、多板块同步、波动分位和
  14:45 行为候选，并用 2025 年后的样本外区间决定是否替换原策略。

## 数据

- 论坛：东方财富沪深300、中证500、上证50、中证1000指数股吧，共 81,216 条历史帖子；使用帖子 ID、作者、标题和精确发布时间，当前缓存到 2026-07-24。
- 日线：本机 `aShare` MySQL 的 `ts_fut_daily`、`ts_fut_basic`、`ts_index_daily`。
- 15 分钟行情：逐日用前一交易日信息选择主力合约，再拼接新浪历史合约 15 分钟 K 线，缓存于 `data/sina_15m`。
- Tushare 的 [`ft_mins`](https://tushare.pro/document/2?doc_id=313) 支持期货分钟数据，但当前账号没有该接口权限，所以没有把它作为本次回测数据源。
- 含 180 天预热期的完整 15 分钟交易日覆盖：IF `1325/1330`、IC `1326/1330`、IH `1323/1330`、IM `966/970`。正式回测区间各品种覆盖率均高于 99.5%；缺任意标准 K 线的交易日不生成盘中信号。

浏览量、点赞量和回复量会随时间累积，历史页面看到的是当前值，存在未来数据泄漏，因而不进入信号。

## 混合策略

### 趋势核心

- 20/60/120 日均线分别与 5/10/20 日动量配对，至少两个周期同时为正才持有多仓。
- 指标整体滞后一交易日；当天收益不会进入当天仓位。
- 20 日实现波动率定仓，目标风险 10%，仓位上限 0.5，仓位按 0.1 取整。
- 每 5 个交易日调仓；合约换月按平旧、开新计算双边换手。
- 核心成本为单边 1bp。

### 盘中增强

- 帖子映射到发布时间之后第一根 15 分钟 K 线，信号在该 K 线收盘确认，下一根 K 线开盘成交。
- 特征使用最近 60 分钟帖子数、独立用户数、金融恐慌/亢奋词及表情数量；标准化只使用同一盘中时段此前 40 个交易日的数据。
- 10:45-11:30 出现新的亢奋词、市场未恐慌且短期动量不强时做空；一天最多一笔，持有一根 K 线。
- 仓位上限 0.5；单边手续费 1bp、滑点 1bp。
- 参数只用 2024-12-31 以前的数据选择。单品种训练期至少 20 笔、Sharpe 不低于 0.5 且净收益为正才部署。最终只有 IH 达标；其他品种候选交易仅留作审计。

表情标签仍保留为诊断变量，但不直接触发交易。实验中扩大词典或反复使用同一个 60 分钟情绪窗口会显著增加换手，净收益被成本吞噬。

## 回测结果

区间为 2021-07-26 至 2026-07-24，IM 从上市后的 2022-07-25 开始。2025-01-02
以后是完全样本外区间。

| 品种 | 混合年化 | 年化波动 | 最大回撤 | 原日频年化 | 期货持有年化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IF | 2.19% | 5.57% | -7.73% | -0.84% | 0.79% |
| IC | 4.13% | 5.93% | -7.20% | -1.56% | 8.04% |
| IH | 0.40% | 5.09% | -6.77% | -0.04% | -1.29% |
| IM | 3.03% | 5.75% | -7.19% | 0.58% | 10.26% |

混合策略四品种均提高了原日频策略的全期年化收益，并把最大回撤从约
`12.7%-16.3%` 降至约 `7.1%-7.7%`。它没有在 IC、IM 的单边牛市中超过始终持有，
但持有基准的年化波动为 `18.3%-26.0%`、最大回撤为 `34.4%-38.6%`。

完整逐年收益、波动和回撤见 `insight_update_report/legacy/RESULTS.md` 与
`insight_update_report/legacy/annual_comparison.csv`。2021、2026 是不完整年度，
报告按 252 个交易日折算年化。

## 原帖更新验证

新研究实现了原帖更新中可由当前数据检验的 106 个候选，包括日内情绪一阶速度、
短文本反应性、预期性文本、标点/正负号、多板块关注同步、作者集中度、因果波动
分位和 14:45 行为。训练冠军在全区间把等权组合年化从 `2.36%` 提高到 `2.44%`，
波动基本不变；但 2025 年后的样本外年化从 `5.41%` 降到 `5.28%`。新增覆盖层即使
按零成本计算，样本外增量仍为负，因此生产口径不替换原策略。

洞见映射、不可验证的数据边界和详细结论见
`insight_update_report/UPDATED_INSIGHTS.md`，候选结果见
`insight_update_report/behavior/RESULTS.md`。

## 运行

在仓库根目录执行日频基线：

```powershell
python .\lrh\forum_sentiment_futures\run_backtest.py `
  --codes IF IC IH IM `
  --start-date 2021-07-24 `
  --end-date 2026-07-24 `
  --refresh-sentiment `
  --output-dir .\lrh\forum_sentiment_futures\insight_update_report\baseline
```

执行 15 分钟混合回测：

```powershell
python .\lrh\forum_sentiment_futures\run_intraday_backtest.py `
  --codes IF IC IH IM `
  --start-date 2021-07-24 `
  --end-date 2026-07-24 `
  --train-end-date 2024-12-31 `
  --refresh-intraday `
  --baseline-daily .\lrh\forum_sentiment_futures\insight_update_report\baseline\daily_returns_all.csv `
  --output-dir .\lrh\forum_sentiment_futures\insight_update_report\legacy
```

验证原帖更新中的行为候选：

```powershell
python .\lrh\forum_sentiment_futures\run_behavior_update_backtest.py `
  --start-date 2021-07-24 `
  --end-date 2026-07-24 `
  --train-end-date 2024-12-31
```

原策略最新基准位于 `insight_update_report/legacy`，新增行为研究位于
`insight_update_report/behavior`：

- `RESULTS.md`：全区间与逐年可读报告。
- `annual_comparison.csv`、`comparison_summary.csv`：四品种、原策略与持有基准比较。
- `daily_returns.csv`、`trades.csv`：候选日收益和盘中成交。
- `candidate_search.csv`、`behavior_eligibility.csv`：训练筛选和品种门槛审计。
- `cost_sensitivity.csv`、`selected_params.json`：成本敏感性和生产验收结论。
- `<fut_code>/qlib_report/qlib_report.csv`：生产推荐口径；未通过验收时自动回退原策略。

## 雪球授权数据策略

`xueqiu_strategy` 是独立的数据接入和盘中信号实现。雪球现行服务协议禁止网页爬虫和
规避反爬，因此该实现只接受取得授权的供应商导出或实时数据流，不包含 WAF/Cookie
绕过代码。它支持 DataSeed 中文 Excel 字段、通用 CSV/JSONL/Parquet、历史指数成分
映射、点时正文审计、实时延迟熔断和四品种 Qlib 兼容回测。

当前本机没有授权雪球历史文件，所以 `xueqiu_report` 只生成数据阻断报告，不生成或
伪造收益。数据要求、导入命令和实时信号用法见
`xueqiu_strategy/README.md`。

## 局限

历史论坛页面不包含已删除帖子，标题词典也不等同于完整语义模型；新浪分钟接口不是
带服务承诺的机构级数据源。回测没有模拟盘口深度、涨跌停排队、保证金变化和资金容量。
盘中增强样本仍少，尤其不能仅凭全期正收益判断可实盘。上线前应接入实时论坛流和正式
分钟行情，并做仿真盘、延迟监控和逐笔滑点校准。

## 2026-10-11 审查与真实回测

本次按用户要求允许更换行为数据源，使用新抓取的东方财富指数股吧帖子和更新至
2026-10-09 的期货行情。审查修复、训练与样本外分离、纯价格消融及成本敏感性见
`REVIEW.md` 和 `reviewed_report_20261011/REPORT.md`；可打开的图表位于同目录
`dashboard.html`。

```powershell
python .\lrh\forum_sentiment_futures\run_reviewed_backtest.py `
  --start-date 2021-10-11 --end-date 2026-10-09
```

新的实时采集器保存首次观测时间和首次可见文本，不覆盖早期版本：

```powershell
python .\lrh\forum_sentiment_futures\collect_live_behavior.py --cycles 1
python .\lrh\forum_sentiment_futures\collect_live_behavior.py --cycles 0 --poll-seconds 60
```

这两条命令分别是一次采集和持续采集，输出到 `data/eastmoney_live/posts.csv`。
采集已实际验证；策略是否部署仍以回测门槛为准。
