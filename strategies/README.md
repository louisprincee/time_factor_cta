# 策略

仓库登记三条冻结策略：两条商品策略和一条股指策略。冻结表示规则已固定，不代表样本外验证或实盘优势；三条规则本次复跑使用的都是已经看过的研究期数据，**任何 2022 年及以后数据均未打开**。

| 市场 | 策略 | ID | 冻结日 | 规则差异 |
|---|---|---|---|---|
| 商品 | 按池宽 1 倍加三窗口趋势空头 | `market_clock_slow_short_20_60_120` | 2026-10-23 | 原五因子市场分数 + 正 carry 多头；趋势、负 carry 空头 |
| 商品 | 商品原策略（仅将分数换成收盘位置） | `commodity_close_location_reversal_strict_swap` | 2026-10-27 | 只把多头市场分数替换为池内 CLV < 0.5；正 carry 筛选和空头腿不变 |
| 股指 | 股指原策略（仅将分数换成收盘位置） | `index_close_location_reversal_strict_swap` | 2026-10-27 | 只把 IF/IH/IC 的多头市场分数替换为等权 CLV < 0.5；正 carry 筛选和空头腿不变 |

商品策略由 [`catalog.py`](catalog.py) 登记，股指策略由 [`index_catalog.py`](index_catalog.py) 独立登记；商品和股指正式账本都由 `backtest.engine` 运行并写入 `runs/strategies/` 下各自的市场目录。冻结 JSON 位于各自策略目录，规则、`chosen` 和代码常量会在运行时核对；冻结后不得修改。新规则必须使用新 ID。

## 共同规则

- 趋势窗口为 `TREND_WINDOWS = (20, 60, 120)`。
- 多头要求当日收盘信号成立，且**滞后一交易日**的年化展期收益严格大于 0；目标权重 +1。
- 空头只在市场趋势小于 0 时启用，且要求品种趋势小于 0、滞后一日展期收益小于 0；目标权重 −1。空头覆盖多头。
- 按池宽分配 1 倍名义敞口，下一交易日第一根可交易开盘执行；目标权重不变则不调整仓位。
- 峰值回撤达到 10% 时清仓并停 10 个交易日。
- 商品 2011–2021、股指 IF/IH/IC 共同上市期 2015-04-16–2021 是本次冻结策略比较的研究日历。首个有效信号可能因数据覆盖和趋势预热而更晚。

原商品收盘位置策略 `commodity_close_location_reversal`（不对多头应用 carry 过滤）和股指简单候选 `index_close_location_reversal`（只做等权长仓）仍作为研究代码中的探索规则，不在冻结策略注册表中。它们基于已看过的研究结果提出或检验，不能当作确认性结果，也不得用于打开 OOS。

## 本次研究结果

三条冻结策略的正式账本由 `backtest.engine` 运行；股指单品种诊断也由 engine 运行。`python -m research.commodity_close_location_reversal --three-book-comparison --overwrite` 只读取这些账本，生成商品组件反事实分析、IF/IH/IC 单品种和共同组合品种贡献图。研究归因结果目录为 `runs/research/three_strategy_clv_comparison/`。

- 商品原策略：2011-01-14 至 2021-12-30 评分，1 亿元，历史手续费和每边 1 跳；年化净收益 5.64%、Sharpe 1.03、最大回撤 −6.48%。
- 商品严格替换：同样的研究期和成本；年化净收益 6.21%、Sharpe 1.12、最大回撤 −5.99%。
- 股指严格替换：首个信号日 2015-04-20，至 2021-12-30；年化净收益 11.94%、Sharpe 0.88、最大回撤 −26.35%。1 亿元、2015 年每边 0.2 点，之后每边 1 个前一年估算 tick，加历史手续费。
- 商品多头的“信号 + carry”是联合条件，不能拆成唯一可加的因果贡献。单独运行的分数组合、CLV 组合、carry 组合及空头组合是反事实账本，不可相加。股指单品种结果也不是独立样本外验证。

策略细节和经济机制证据见 [`docs/guide.md`](../docs/guide.md)。常用命令：

```bash
python -m backtest.engine --research
python -m backtest.engine market_clock_slow_short_20_60_120 --research
python -m backtest.engine commodity_close_location_reversal_strict_swap --research
python -m backtest.engine index_close_location_reversal_strict_swap --research --index-individuals
python -m research.commodity_close_location_reversal --three-book-comparison --overwrite
python -m research.market_clock_slow_short_factors
```

不要因策略已冻结就自行启动 OOS；只有用户明确指定窗口后才能运行，并须遵守 [`AGENTS.md`](../AGENTS.md) 中的数据分区与台账规则。
