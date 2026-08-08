# 雪球盘中情绪股指期货策略

本子目录实现雪球数据的授权接入、点时审计、15 分钟信号、下一根 K 线成交和
`IF/IC/IH/IM` 回测。它不会绕过雪球的 Cookie、WAF 或登录保护，也不包含网页爬虫。

雪球现行《服务协议》明确禁止使用爬虫、抓取工具和规避非正常浏览识别机制：

- https://xueqiu.com/about/terms

可接入的历史数据包括取得使用授权的供应商导出或自有合规数据流。例如，公开产品页
显示 DataSeed `D1704` 覆盖 2011-07 至 2025-09；FactSet Marketplace 也列有
Xueqiu Analytics。产品存在不等于本机已购买或取得授权。

- https://www.cndataseed.com/data-detail/3760?database_id=850
- https://www.factset.com/marketplace/catalog/product/xueqiu-analytics-for-china-retail-investor-sentiment

## 关键约束

- 默认只使用发帖时间、帖子到达速度、作者广度和发帖时点文本。
- 点赞、回复、转发、浏览量除非带历史快照时间，否则不进入任何信号。
- 供应商的当前帖子正文可能包含事后编辑；只有明确声明为发帖时点快照才进入情绪词。
- 信号在 15 分钟 K 线收盘确认，下一根 K 线开盘成交。
- 默认目标年化风险 `6%`，盘中覆盖仓位上限 `0.35`，一天最多一笔，单边手续费和
  滑点各 `1bp`。
- 参数在看到雪球回测结果前固定，不使用全样本调参。

## 数据格式

`ingest_export.py` 支持 CSV、TSV、Excel、Parquet、JSON 和 JSONL。它能直接识别
DataSeed 样表中的中文字段：

`股票代码`、`ID`、`发帖者ID`、`帖子发表者呢称`、`帖子内容`、`发帖时间`、
`点赞量`、`转发量`、`浏览量`、`评论量`。

通用导出至少需要：

| 字段 | 含义 |
| --- | --- |
| `post_id` | 稳定的帖子 ID |
| `user_id` | 作者 ID |
| `created_at` | 北京时间精确发布时间 |
| `symbol_codes` | `SH600519`、`SH000300` 等，可用逗号或竖线分隔 |
| `text` | 发帖时点正文 |
| `content_observed_at` | 正文快照时间 |
| `observed_at` | 数据流首次观测时间，实时信号必需 |

先根据实际合同复制并修改 `authorization.example.json`。仅当供应商确实保证相应
条件时，才可把布尔项设为 `true`。

## 导入

```powershell
python .\lrh\forum_sentiment_futures\xueqiu_strategy\ingest_export.py `
  D:\authorized-data\xueqiu.xlsx `
  --authorization-metadata D:\authorized-data\authorization.json
```

帖子中的上市公司代码按本机 `ts_index_weight` 历史快照映射到四个指数；直接提及
`SH000300/SH000905/SH000016/SH000852` 的帖子直接映射。

## 五年回测

```powershell
python .\lrh\forum_sentiment_futures\xueqiu_strategy\run_backtest.py `
  --start-date 2021-07-26 `
  --end-date 2026-07-24 `
  --train-end-date 2024-12-31
```

程序首先生成 `xueqiu_report/DATA_AUDIT.md`。五年覆盖率、帖子密度、授权声明、
点时正文或成分时效任一不合格时，进程以代码 `2` 结束，并且不输出收益指标。

审计通过后输出：

- `daily_returns.csv`：雪球覆盖层、趋势核心、原混合策略和持有基准。
- `annual_comparison.csv`：四品种逐年年化收益、波动和回撤。
- `comparison_summary.csv`：全期、训练期和 2025 年后样本外比较。
- `trades.csv`：下一根 K 线成交、双边成本和退出原因。
- `<品种>/qlib_report/qlib_report.csv`：Qlib 兼容日报。

## 实时信号

实时入口读取授权帖子增量文件和正式行情源落地的 15 分钟 K 线文件，不负责下单：

```powershell
python .\lrh\forum_sentiment_futures\xueqiu_strategy\run_live_signal.py `
  --bars-file D:\authorized-data\cffex_15m_live.parquet
```

输出含 `direction`、建议名义仓位、置信度、原因和数据延迟。开盘期间帖子文件或行情
文件超时会返回 `halted`；非交易时段返回 `market_closed`。`orders_submitted` 永远为
`false`。
