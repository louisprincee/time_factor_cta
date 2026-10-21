# 策略

当前策略是 [`market_clock_carry`](market_clock_carry/)，四条共用一套信号，2026-10-21 一起冻结，四份 JSON 规格在同目录，导入时跟代码核对。`catalog.py` 是空的。

## 当日分数且展期为正

五个因子 `ts_low`、`ts_high`、`dfp_top3`、`ts_low_pm`、`ts_high_pm` 乘事前方向，各品种按自己过去 250 日（不含当天）做 z，截断 ±3，五个取平均，再在当年池内取平均，得到当天的市场分数。

收盘同时满足两件才入选：当天分数大于 0；该品种主力对次主力的年化展期收益，用前一个交易日的数，严格大于 0。近月高于远月时这个数为正。下一交易日第一根开盘成交：有夜盘时是夜盘开盘，没有夜盘时是日盘开盘。

| 策略 | ID | 分仓 | 敞口上限 |
|---|---|---|---|
| 按池宽1倍 | market_clock_pool | 每个入选品种 1 倍 × 权益 / 当年池宽 | 1 |
| 入选平分1倍 | market_clock_carry | 入选品种平分，入选数一变全部重新平分 | 1 |
| 按池宽1.5倍 | market_clock_pool_15 | 每个入选品种 1.5 倍 × 权益 / 当年池宽 | 1.5 |
| 入选平分1.5倍 | market_clock_carry_15 | 入选品种平分，入选数一变全部重新平分 | 1.5 |

权重不变就留着，只成交手数差额。峰值回撤 10% 清仓，停 10 个交易日。展期收益池内某品种某年缺失超过 5% 就报错。

研究期数字、信号层检验和为什么这么做，写在 [`docs/当日分数且展期为正.md`](../docs/当日分数且展期为正.md)。

```bash
python -m backtest.engine --research                 # 2016–2021，不写名字四条都跑
python -m backtest.engine 按池宽1倍 --research       # 只跑这一条
python -m backtest.engine --research --by-year
python -m research.market_clock_carry_factors        # 信号层检验
```

## 冻结

四条都是 `status: frozen`、`frozen_at: 2026-10-21`，`chosen` 等于 `parameters`，冻结记录也写进了台账（`kind: decision`）。`--validation`、`--year`、`--oos` 可以打开 2022 及以后，先记台账再读；以后新加的策略没冻结时，这些命令会在写台账之前拒绝。不读 2026。

## 已删除

只做多全市场时钟、冻结过的四条尾部策略、展期 carry 探索策略都已删除，记录在 [`docs/全市场时钟做法与表现.md`](../docs/全市场时钟做法与表现.md) 和 git 历史里。
