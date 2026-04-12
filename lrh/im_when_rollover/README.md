# 股指期货贴水移仓研究

该目录提供一个可直接运行的日频回测脚手架，用本机 `aShare` MySQL 库中的 `ts_fut_*`、`ts_index_daily` 表回测中金所股指期货长期持有多头、通过不同换月策略吃贴水的效果。

## 覆盖品种

- `IF` 对应 `000300.SH`
- `IC` 对应 `000905.SH`
- `IH` 对应 `000016.SH`
- `IM` 对应 `000852.SH`

`IM` 的本地库起始日期是 `2022-07-22`，其余品种覆盖更早历史。

## 策略集合

- 主力成交量
- 主力持仓量
- 每日重选最大年化贴水
- 最大年化贴水（仅在流动性前 N 个合约中选择，默认 `N=2`）
- 近月提前 `5/3/1` 个交易日换月
- 次近月提前 `5/3/1` 个交易日换月
- 最大年化贴水持有至到期

## 关键假设

- 收益口径：
  使用合约收盘价计算持有收益；换仓日假设以当日 `VWAP` 成交。
- `VWAP` 近似：
  `amount * 10000 / (vol * multiplier)`；若分母缺失则退化为收盘价。
- 交易成本：
  默认单边 `1bp`，换仓日按双边扣减。
- 年化贴水：
  使用 `(现货 - 期货) / 现货 * 365 / 距离到期日自然日`。
- 年化指标：
  使用与 `qlib.contrib.evaluate.risk_analysis(..., mode="product")` 一致的复利口径，本地兼容实现按 `252` 交易日年化。

## 运行方式

在仓库根目录执行：

```powershell
python .\lrh\im_when_rollover\run_backtest.py --codes IF IC IH IM
```

只跑 `IM`：

```powershell
python .\lrh\im_when_rollover\run_backtest.py --codes IM
```

指定时间区间和输出目录：

```powershell
python .\lrh\im_when_rollover\run_backtest.py `
  --codes IM `
  --start-date 2022-07-22 `
  --end-date 2026-04-10 `
  --output-dir .\lrh\im_when_rollover\outputs\im_sample
```

## 输出文件

每个品种会生成：

- `summary.csv`: 策略汇总，含年化收益、年化超额收益、回撤、换仓次数等
- `annual_returns.csv`: 分年收益
- `roll_events.csv`: 换仓事件明细
- `market_snapshot.csv`: 合约横截面快照
- `daily_returns.csv`: 日收益明细（可用 `--skip-daily` 跳过）

根输出目录还会额外生成跨品种汇总文件：

- `summary_all.csv`
- `annual_returns_all.csv`
- `roll_events_all.csv`
