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

不写策略名时运行 `strategies/catalog.py` 里的两条冻结策略；写了策略名就只跑该策略：`python -m backtest.engine 按池宽1倍加三窗口趋势空头 --research`，或 `--strategy 按池宽1倍加慢趋势空头 --year 2019`。只做商品。

2022 及以后：点到的策略里有任何一条没冻结，整批直接拒绝，不写台账也不读行情。冻结了就先写台账再读（2022 记 `year_test`，2023 起走 `final_evaluation`，都带冻结日期、`chosen` 和代码与规格文件的哈希）。台账里已经打开过的窗口只打印旧结果；结果文件丢了，要先追加一条 `kind: void` 并写原因才能重开。不读 2026。

2026-10-22 按用户决定清空了商品 OOS 台账（`data/commodity/oos/ledger.jsonl`）和 2022 年及以后的结果文件。清空前，2022–2024 已经为当时的策略打开过，2025 没有打开记录。空台账不让任何年份变干净：2022–2024 重跑只算复现。台账空了以后，`--validation`、`--year`、`--oos` 会重新读数据、写新台账，不会只打印旧结果。

## 账本

`simulate_target_lots`：整数手、按目标仓位交易。策略给出每个品种在执行日开盘的目标权重（−3 到 3，负数做空；两条策略都用 −1、0、1）。

- 名义金额 = |权重| × 敞口上限 × 期初权益 / 当年池宽，向下取整到整数手，手数与权重同号。
- 权重不变就继续持有，不随价格或权益调手数，只成交差额。
- 盈亏 = 有符号手数 × 合约乘数 × 复权开盘价变动（加法复权，差值除以原始价，不做比值）。合约乘数按所持合约取：RU1207 以前 5 吨、FU1202–FU1812 50 吨、PB1109–PB1308 25 吨（`tfcta.costs.CONTRACT_MULTIPLIER_RANGES`），换月到乘数不同的合约时名义不变、手数向下取整。
- 换月在旧主力前一日收盘和新主力当日开盘上收手续费和滑点，不结转真实合约结算盈亏。
- 无开盘价或最高等于最低（涨跌停代理）当日不成交，次日重试。持仓期间缺复权开盘变动直接报错。
- 整本账一个峰值回撤熔断：10% 清仓、停 `halt_days`（两条都是 10）个交易日，恢复当日把熔断用的峰值改成当时权益（`daily.csv` 的 `熔断口径回撤`）。`summary.csv` 的 `最大回撤` 从初始资金起算，不受这个重置影响。
- 保证金按名义金额 20% 估计，超过权益就清仓。
- 1 亿元初始权益。年化收益、波动率和 Sharpe 用每年 242 个交易日。

手续费按历史费率：按金额 ×1.01、按手数 +0.01 元；手续费表最多向后沿用 31 天。米筐的上期所费率从 2013-10-08 才有，2011-01-01 到 2013-10-07 的 AG、AL、AU、CU、RB、RU、ZN 用各自 2013-10-08 的费率往前沿用（用户 2026-10-23 决定，`backfill_shfe_fees`），这是假设；把这段费率改成 0 或翻倍，2011–2013 年净收益变动不到 0.2 个百分点。大商所 2010 年起是真实费率。滑点是上一年估计 tick 每边 1 跳，上一年估不出时最多再沿用一年。超出就是缺成本，成交当天报错。

`load_research(first_year=None)` 读研究期因子缓存和日线，默认从 2016 起；`first_year` 早于 2016 时，之前几年的品种池用落盘的 yearly_stats 按同一门槛重建，重建的 2015 必须等于落盘的，并从五个因子都有值的那天（2011-01-14）起计分；`load_through_year(y)` 在研究期后面接上 2022 至 y 年现算的五个因子（研究期最后 60 天两边必须一致，否则报错），2023 起只能在最终评估块内调用。`ledger_inputs()` 把一个区间的价格、换月、合约代码、费用和滑点对齐好。

## 输出

`runs/strategies/<id>/commodity/<区间>/`，区间是 `research`（2011–2021）、`oos`、单年 `2019`，或 `2016-2021` 这样的起止年。每个区间里有 `summary.csv`、`annual.csv`、`daily.csv`、`positions.csv`、`events.csv`，净值图在 `plots/<策略名>/`。研究期每条还写一份 `runs/research/<id>/protocol.json`（当时的规格）。

绩效函数 `performance`、`sharpe_ratio`、`max_drawdown`、`newey_west_t`、`deflated_sharpe`、`pbo_cscv` 也在 `backtest/engine.py`。
