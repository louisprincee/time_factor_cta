# 策略

当前有两条冻结策略和一条研究期候选，名单在 [`catalog.py`](catalog.py)。每条的 JSON 规格和代码在同目录，导入时互相核对，不一致就报错。

| 策略 | ID | 目录 | 冻结 |
|---|---|---|---|
| 按池宽1倍 | market_clock_pool | [`market_clock_carry/`](market_clock_carry/) | 2026-10-21 |
| 按池宽1倍加慢趋势空头 | market_clock_slow_short | [`market_clock_slow_short/`](market_clock_slow_short/) | 2026-10-22 |
| 按池宽1.5倍加慢趋势空头（研究期候选） | market_clock_slow_short_15 | [`market_clock_slow_short_15/`](market_clock_slow_short_15/) | 未冻结 |

## 按池宽1倍

五个因子 `ts_low`、`ts_high`、`dfp_top3`、`ts_low_pm`、`ts_high_pm` 乘事前方向，各品种按自己过去 250 日（不含当天）做 z，截断 ±3，五个取平均，再在当年池内取平均，得到当天的市场分数。

收盘同时满足两件才做多：当天分数大于 0；该品种主力对次主力的年化展期收益，用前一个交易日的数，严格大于 0。近月高于远月时这个数为正。每个入选品种拿 1 倍 × 权益 / 当年池宽。下一交易日第一根开盘成交：有夜盘时是夜盘开盘，没有夜盘时是日盘开盘。

权重不变就留着，只成交手数差额。峰值回撤 10% 清仓，停 10 个交易日。展期收益池内某品种某年缺失超过 5% 就报错。

## 按池宽1倍加慢趋势空头

多头腿和按池宽1倍完全一样，代码直接复用。另加空头腿：

- 熊市：池内等权收盘到收盘日收益累计 60、120、250 日后的符号平均小于 0。不足 250 天不算熊市。
- 熊市日，品种自身 60、120、250 日收益符号的平均小于 0，并且滞后一日展期收益小于 0，权重 −1，也按当年池宽分 1 倍。
- 一个品种同一天只会在一条腿上：多头要展期 > 0，空头要展期 < 0。

成交、留仓、熔断和费用都和按池宽1倍一样。

## 按池宽1.5倍加慢趋势空头（研究期候选）

复用冻结策略的信号和风控，只把多头、空头每条合格持仓的名义预算都提高到 1.5 倍。候选尚未冻结，只能跑 2016–2021 研究期；预先登记的比较门槛和报告项见 [`market_clock_slow_short_15.json`](market_clock_slow_short_15/market_clock_slow_short_15.json)。

策略动机、研究期结果和信号层检验见 [`docs/guide.md`](../docs/guide.md)。

```bash
python -m backtest.engine --research                 # 2016–2021，不写名字运行所有已登记策略
python -m backtest.engine 按池宽1倍 --research       # 只跑这一条
python -m backtest.engine 按池宽1.5倍加慢趋势空头 --research  # 1.5倍研究期候选
python -m backtest.engine --research --by-year
python -m research.market_clock_carry_factors        # 信号层检验
```

## 冻结

JSON 里 `status: frozen`、写了 `frozen_at`、`chosen` 等于 `parameters` 才算冻结。`--validation`、`--year`、`--oos` 先记台账再读；没冻结的策略在写台账之前就拒绝。冻结的 JSON 不改，规则变了就是新 ID 的新策略。不读 2026。

## 已删除

旧的入选平分1倍、长多按池宽1.5倍、入选平分1.5倍、五因子正展期熊市过滤、对偶空头和熊市应对比较脚本在 2026-10-22 删除；当前慢趋势空头 1.5 倍候选是新登记的独立策略。更早的只做多全市场时钟、四条尾部策略和展期 carry 探索也已删除。2022 年及以后的台账和结果已在 2026-10-22 清空，见 [`README.md`](../README.md) 的数据边界一节。
