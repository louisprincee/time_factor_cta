# 回测

入口是 `python -m backtest.engine`，区间必须显式指定：

```bash
python -m backtest.engine --research                 # 2011–2021 一本账
python -m backtest.engine --start 2016 --end 2021     # 冻结时的研究期
python -m backtest.engine --research --by-year       # 每年一本新账
python -m backtest.engine --year 2019
python -m backtest.engine --start 2018 --end 2020
python -m backtest.engine --validation               # 2022，策略冻结后
python -m backtest.engine --year 2024                # 单年，冻结后
python -m backtest.engine --oos [--by-year]          # 2023–2025，冻结后
```

当前 `strategies/catalog.py` 登记两条冻结商品策略；不指定策略名时，商品入口会运行两条。股指策略由 `strategies/index_catalog.py` 独立登记，但同样由 `backtest.engine` 调用共享整数手账本运行。股指正式组合只开放 2015–2021 研究期，不读 2022+：

```bash
python -m backtest.engine market_clock_slow_short_20_60_120 --research
python -m backtest.engine commodity_close_location_reversal_strict_swap --research
python -m backtest.engine index_close_location_reversal_strict_swap --research
python -m backtest.engine index_close_location_reversal_strict_swap --research --index-individuals
```

股指组合结果位于 `runs/strategies/index_close_location_reversal_strict_swap/index/research/`。添加 `--index-individuals` 会由同一 engine 额外执行 IF、IH、IC 三本单品种诊断账本，分别写入 `.../index/individual/<品种>/research/`；这些是事后诊断，不是独立冻结策略。研究归因脚本只读取这些账本并生成汇总和图，不再模拟策略账本。

2022 及以后：点到的策略里有任何一条没冻结，整批直接拒绝，不写台账也不读行情。冻结了就先写台账再读（2022 记 `year_test`，2023 起走 `final_evaluation`，都带冻结日期、`chosen` 和代码与规格文件的哈希）。台账里已经打开过的窗口只打印旧结果；结果文件丢了，要先追加一条 `kind: void` 并写原因才能重开。不读 2026。

2026-10-26 用户决定：2022–2025 是当前商品原策略的样本外，原策略从未在这几年运行过；2026-10-27 冻结的替换策略也没有 OOS 结果。用户明确说开始看样本外之前一律不打开任何 2022+ 窗口。台账（`data/commodity/oos/ledger.jsonl`）目前是空的，打开时先写台账再读。

## 账本

`simulate_target_lots`：整数手、按目标仓位交易。策略给出每个品种在执行日开盘的目标权重（−3 到 3，负数做空；当前策略用 −1、0、1）。

- 名义金额 = |权重| × 敞口上限 × 期初权益 / 当年池宽，向下取整到整数手，手数与权重同号。
- 权重不变就继续持有，不随价格或权益调手数，只成交差额。
- 盈亏 = 有符号手数 × 合约乘数 × 复权开盘价变动（加法复权，差值除以原始价，不做比值）。合约乘数按所持合约取：RU1207 以前 5 吨、FU1202–FU1812 50 吨、PB1109–PB1308 25 吨（`tfcta.costs.CONTRACT_MULTIPLIER_RANGES`），换月到乘数不同的合约时名义不变、手数向下取整。
- 换月在旧主力前一日收盘和新主力当日开盘上收手续费和滑点，不结转真实合约结算盈亏。
- 无开盘价或最高等于最低（涨跌停代理）当日不成交，次日重试。持仓期间缺复权开盘变动直接报错。
- 整本账一个峰值回撤熔断：10% 清仓、停 `halt_days`（当前策略为 10）个交易日，恢复当日把熔断用的峰值改成当时权益（`daily.csv` 的 `熔断口径回撤`）。`summary.csv` 的 `最大回撤` 从初始资金起算，不受这个重置影响。
- 保证金按名义金额 20% 估计，超过权益就清仓。
- 1 亿元初始权益。年化收益、波动率和 Sharpe 用每年 242 个交易日。`summary.csv` 和 `annual.csv` 同时报告 Calmar 比率，定义为年化收益 / 最大回撤绝对值；年度 Calmar 按该年度的有效交易日数年化。

手续费按历史费率：按金额 ×1.01、按手数 +0.01 元；手续费表最多向后沿用 31 天。米筐的上期所费率从 2013-10-08 才有，2011-01-01 到 2013-10-07 的 AG、AL、AU、CU、RB、RU、ZN 用各自 2013-10-08 的费率往前沿用（用户 2026-10-23 决定，`backfill_shfe_fees`），这是假设；把这段费率改成 0 或翻倍，2011–2013 年净收益变动不到 0.2 个百分点。大商所 2010 年起是真实费率。滑点是上一年估计 tick 每边 1 跳，上一年估不出时最多再沿用一年。超出就是缺成本，成交当天报错。

`load_research(first_year=None)` 读研究期因子缓存和日线，默认从 2016 起；`first_year` 早于 2016 时，之前几年的品种池用落盘的 yearly_stats 按同一门槛重建，重建的 2015 必须等于落盘的，并从五个因子都有值的那天（2011-01-14）起计分；`load_through_year(y)` 在研究期后面接上 2022 至 y 年现算的五个因子（研究期最后 60 天两边必须一致，否则报错），2023 起只能在最终评估块内调用。`ledger_inputs()` 把一个区间的价格、换月、合约代码、费用和滑点对齐好。

## 输出

商品结果在 `runs/strategies/<id>/commodity/<区间>/`，股指结果在 `runs/strategies/<id>/index/<区间>/`。区间是 `research`、单年 `2019` 或 `2016-2021` 这样的起止年。每个账本目录包含 `summary.csv`、`annual.csv`、`daily.csv`、`positions.csv`、`events.csv` 和绩效图。研究期规格文件仍位于对应的 `runs/research/<id>/protocol.json`。

绩效函数 `performance`、`sharpe_ratio`、`max_drawdown`、`newey_west_t`、`deflated_sharpe`、`pbo_cscv` 也在 `backtest/engine.py`。

## 三策略收盘位置替换对照

先用 `backtest.engine` 生成三本冻结策略账本；股指还需加 `--index-individuals` 生成 IF/IH/IC 单品种诊断账本。然后运行 `python -m research.commodity_close_location_reversal --three-book-comparison`，研究规格在 `config/research/three_strategy_clv_comparison.json`。研究脚本只读账本并做归因和绘图，不执行策略回测：

1. 商品冻结原策略，2011–2021；
2. 商品严格替换版：仅将五因子时间分数 `> 0` 换为池内等权收盘位置 `< 0.5`，原有滞后一日 carry > 0 多头门槛、趋势空头、预算、执行和风控不变；
3. IF/IH/IC 严格替换版：对股指迁移策略作相同替换，保留其 carry 多头门槛与趋势空头，日历从三品种共同上市日 2015-04-16 起。实际首个有目标的收益日为 2015-04-20。

三个结果均使用 1 亿元账本、10% 回撤熔断、历史手续费和每边 1 跳滑点。股指 2015 年因 IH/IC 没有上一年估计 tick，按已核实的中金所固定 tick 0.2 点计；其后使用上一年估计 tick。实际结果（研究期、样本内）：

| 策略 | 计分区间 | 年化净收益 | 净 Sharpe | 最大回撤 | 年化成本占权益 |
|---|---|---:|---:|---:|---:|
| 商品原策略 | 2011-01-14–2021-12-30 | 5.64% | 1.03 | −6.48% | 3.13% |
| 商品仅替换为收盘位置 | 2011-01-14–2021-12-30 | 6.21% | 1.12 | −5.99% | 3.16% |
| 股指仅替换为收盘位置 | 2015-04-20–2021-12-30 | 11.94% | 0.88 | −26.35% | 0.93% |

贡献按实际整数手持仓的多头、空头毛盈亏拆分，实际交易成本单列，并校验各项可逐日复算账本净收益；IF、IH、IC 另分别拆分。年度贡献图采用“逐日 PnL / 当日期初权益”累加，不复利，因此与复利后的年度净收益不同。图表和明细位于 `runs/research/three_strategy_clv_comparison/`，包括 `plots/comparison/nav_comparison.png`、`plots/commodity_annual_contributions.png`、`plots/index_annual_symbol_contributions.png` 和 `plots/cumulative_leg_contributions.png`。比较入口与原商品 CLV 研究合并在 `research/commodity_close_location_reversal.py` 中，使用 `--three-book-comparison` 区分运行模式。

这些窗口都与此前研究日期重叠，不是样本外验证。股指按 1 亿元运行时未建模历史中金所开仓限额；2015–2018 年潜在手数可能超过限额，不能把该段账本结果视为可执行容量或实盘表现。研究只读取商品 2011–2021、股指 2015–2021 研究分区，不读取 2022+ 或 2026。
