# backtest_visual_report

深色交互式量化回测可视化页面，参考截图中的 Alpha 指增归因报告样式实现。

默认读取：

```text
D:\github\qlib\lrh\smallcap_lowturn\report
```

生成：

```text
D:\github\qlib\lrh\backtest_visual_report\index.html
```

## 生成命令

在仓库根目录执行：

```powershell
python .\lrh\backtest_visual_report\build_dashboard.py
```

指定其他回测目录：

```powershell
python .\lrh\backtest_visual_report\build_dashboard.py `
  --source .\lrh\smallcap_lowturn\report `
  --output .\lrh\backtest_visual_report\index.html
```

## 页面模块

- `总览仪表盘`: 累计收益、基准、超额、回撤、分年收益、月度热力图、换手和优化建议。
- `Brinson归因`: 当前基于持仓特征和累计超额生成 Brinson-like 代理视图。
- `Barra归因`: 当前基于小市值、低换手、低波动、短动量、流动性生成风格暴露代理视图。
- `稳定性评估`: IR 显著性、分年度 IR、持仓集中度、滚动 IR 和交易约束。

## 数据限制

`smallcap_lowturn/report` 目前没有行业分类、行业基准权重、个股分期收益、Barra 因子暴露和因子收益。因此页面里的 Brinson/Barra 是可视化框架和诊断入口，不是严格归因结果。

要升级成真实归因，建议新增：

- 每日/调仓日持仓行业分类和组合行业权重。
- 对应基准行业权重。
- 个股收益或行业收益，用于配置、选股和交互效应拆分。
- Barra/CNE 风格因子暴露、因子收益、特异收益和协方差。
