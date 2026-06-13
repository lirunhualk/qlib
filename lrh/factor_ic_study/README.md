# factor_ic_study

日频 A 股多因子 IC 与单因子策略收益关系研究。

## 口径

- 信号日：使用当日收盘后可见的横截面因子。
- 收益标签：为了匹配可交易性，因子在 `T` 日预测 `T+1` 开盘到 `T+2` 开盘收益。
- IC：每日横截面 Spearman rank IC。
- 股票池：非 ST、非退市名、非北交所、上市满一年、价格不低于 3 元、20 日均成交额不低于约 2000 万元。
- 单因子策略：每个因子分别按高因子值和低因子值各选 200 只，20 日调仓，扣 10 bps 成本。
- 关系检验：
  - `mean_ic` vs `spread_ann`：有符号 IC 与高因子组减低因子组年化收益。
  - `abs_mean_ic` vs `chosen_ann`：按 IC 符号选择高/低方向后的多头年化收益。

## 运行

```powershell
python .\lrh\factor_ic_study\run_factor_ic_study.py --end-date 20260608
```

默认 `--feature-set core` 覆盖价格、估值/市值/换手、技术指标、资金流和交易约束字段。需要加载两融、基金持仓、财务、质押、股东户数、回购等更重的辅助表时使用：

```powershell
python .\lrh\factor_ic_study\run_factor_ic_study.py --end-date 20260608 --feature-set extended
```

输出默认写入：

```text
D:\github\qlib\lrh\factor_ic_study\report
```

## 当前 core 结果

已运行：

```powershell
python .\lrh\factor_ic_study\run_factor_ic_study.py --end-date 20260608 --feature-set core
```

覆盖 `101` 个核心日频因子，来源包括价格/市值/估值/换手、派生流动性、派生波动、Tushare 技术指标和资金流。

主要相关性结论：

| 关系 | Pearson | p-value | 结论 |
| --- | ---: | ---: | --- |
| `mean_ic` vs `spread_ann` | 0.764 | 1.41e-20 | 显著正相关 |
| `mean_ic` vs `high_ann` | 0.811 | 8.14e-25 | 显著正相关 |
| `abs_mean_ic` vs `chosen_ann` | 0.159 | 0.112 | 不显著 |
| `abs_ic_ir` vs `chosen_ann` | 0.146 | 0.145 | 不显著 |
| `abs_mean_ic` vs `abs_spread_ann` | 0.453 | 1.92e-6 | 显著正相关 |

解释：

- IC 与高因子组减低因子组的收益差有显著关系，这符合 IC 的定义：横截面排序预测能力越强，多空排序组合越容易赚钱。
- 但 `abs(IC)` 与“按 IC 方向只做多”的单因子策略收益不显著。A 股 long-only 收益明显受到小市值、流动性、市场 beta、回撤暴露和成本的共同影响，不能只看 IC 大小。
- 当前样本里 IC 较强的因子主要集中在低成交额波动、低换手、低波动和流动性代理上；按收益排序，小市值和流动性因子更靠前。

代表性结果：

| 因子 | mean IC | 方向 | 对应多头年化 | 高减低年化 |
| --- | ---: | --- | ---: | ---: |
| `amt_std_5d` | -0.0535 | 低值 | 28.43% | -32.17% |
| `amt_mean_5d` | -0.0499 | 低值 | 28.56% | -31.35% |
| `turnover_rate_f` | -0.0504 | 低值 | 5.21% | -25.76% |
| `buy_sm_amount_ratio` | 0.0410 | 高值 | 16.30% | 23.44% |
| `total_mv` | -0.0141 | 低值 | 38.37% | -31.17% |

完整文件：

- `report/factor_metrics.csv`
- `report/daily_ic.csv`
- `report/correlation_summary.csv`
- `report/ic_vs_strategy_scatter.png`
- `report/top_abs_ic.png`
