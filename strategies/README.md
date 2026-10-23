# 策略

当前只保留一条冻结的商品策略：**按池宽1倍加慢趋势空头**（`market_clock_slow_short`）。完整实现、多头和空头信号、规格核对都在 [`market_clock_slow_short/`](market_clock_slow_short/)；唯一策略由 [`catalog.py`](catalog.py) 登记。

## 规则

### 多头

收盘时，五因子市场分数 `ts_low`、`ts_high`、`dfp_top3`、`ts_low_pm`、`ts_high_pm` 大于 0，且该品种滞后一个交易日的主力对次主力年化展期收益严格大于 0，目标权重为 +1。

### 空头

- 市场趋势：池内等权收盘到收盘日收益累计 60、120、250 个交易日后取符号，再求平均；均值小于 0 时为熊市。历史不足时不算熊市。
- 品种趋势：该品种 60、120、250 日收益符号的平均。
- 熊市日，只有品种趋势小于 0 且滞后一日展期收益小于 0 时，目标权重才为 −1。
- 多头和空头的 carry 条件互斥，同一品种不会在同日同时进入两条腿。

两个规则使用同一常量 `TREND_WINDOWS = (60, 120, 250)`。它与冻结 JSON 的 `short_trend_windows`、`bear_trend_windows` 同时核对；不能只改代码窗口而不更新冻结规格。冻结策略不能原地改参数，规则变化须新建策略并由用户冻结。

### 执行、预算与风控

- 两条腿均按当年池宽分 1 倍名义敞口；每个入选品种约为权益 / 当年池宽。
- 收盘信号在下一交易日第一根可交易开盘执行：有夜盘时是夜盘开盘，否则是日盘开盘。
- 权重不变则继续持有，只交易目标手数与现有手数的差额。
- 峰值回撤达到 10% 时清仓并停 10 个交易日；展期收益池内某品种某年缺失超过 5% 会报错。

## 研究状态

慢趋势策略于 2026-10-22 冻结。它不是已证明可部署的优势；研究期表现不代表样本外证据。2022–2024 已经被看过，2025 尚未打开，任何 2022+ 测试都须由用户在当轮明确要求。**不读 2026。**

2026-10-23 移除三窗口趋势版本（20/60/120）。该版本曾冻结并做过研究比较，但已不再是当前策略或可选项；历史比较数字保留在 [`docs/guide.md`](../docs/guide.md)，不得据此把它当作第二本在跑。

## 常用命令

```bash
python -m backtest.engine --research
python -m backtest.engine --start 2016 --end 2021
python -m backtest.engine --research
python -m backtest.engine --research --by-year
python -m research.market_clock_slow_short_factors
```

冻结 JSON 为 [`market_clock_slow_short.json`](market_clock_slow_short/market_clock_slow_short.json)。只在策略尚未冻结时填写 `status: frozen`、`frozen_at` 并令 `chosen == parameters`；冻结后不修改。策略规则与研究局限见 [`docs/guide.md`](../docs/guide.md)。
