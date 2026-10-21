# 回测

入口是 `python -m backtest.engine`，区间必须显式指定：

```bash
python -m backtest.engine --research                 # 2016–2021 一本账
python -m backtest.engine --research --by-year       # 每年一本新账
python -m backtest.engine --year 2019
python -m backtest.engine --start 2018 --end 2020
python -m backtest.engine --validation               # 2022，策略冻结后
python -m backtest.engine --year 2024                # 单年，冻结后
python -m backtest.engine --oos [--by-year]          # 2023–2025，冻结后
```

不写策略名时四条都跑，写了只跑那一条：`python -m backtest.engine 按池宽1倍 --research`，或 `--strategy 入选平分1.5倍 --year 2019`。只做商品。

2022 及以后：四条里有任何一条没冻结，整批直接拒绝，不写台账也不读行情。冻结了就先写台账再读（2022 记 `year_test`，2023 起走 `final_evaluation`，都带冻结日期、`chosen` 和代码与规格文件的哈希）。台账里已经打开过的窗口只打印旧结果；结果文件丢了，要先追加一条 `kind: void` 并写原因才能重开。不读 2026。

## 账本

`simulate_target_lots`：整数手、按目标仓位交易。策略给出每个品种在执行日开盘的目标权重（−3 到 3，负数做空，当前策略只用 0 和 1）。

- 名义金额 = 权重 × 敞口上限 × 期初权益 / 分母，向下取整到整数手。分母 `pool` 是当年池宽；`selected` 是当天权重不为 0 的池内品种数，入选数一变，所有持仓按新的份数重新平分。
- 权重不变（`selected` 还要入选数不变）就继续持有，不随价格或权益调手数，只成交差额。
- 盈亏 = 有符号手数 × 合约乘数 × 复权开盘价变动（加法复权，差值除以原始价，不做比值）。
- 换月在旧主力前一日收盘和新主力当日开盘上收手续费和滑点，不结转真实合约结算盈亏。
- 无开盘价或最高等于最低（涨跌停代理）当日不成交，次日重试。持仓期间缺复权开盘变动直接报错。
- 整本账一个峰值回撤熔断：当前策略 10% 清仓、停 10 个交易日，恢复当日把熔断用的峰值改成当时权益（`daily.csv` 的 `熔断口径回撤`）。`summary.csv` 的 `最大回撤` 从初始资金起算，不受这个重置影响。
- 保证金按名义金额 20% 估计，超过权益就清仓。
- 1 亿元初始权益。年化收益、波动率和 Sharpe 用每年 242 个交易日。

手续费按历史费率：按金额 ×1.01、按手数 +0.01 元；手续费表最多向后沿用 31 天。滑点是上一年估计 tick 每边 1 跳，上一年估不出时最多再沿用一年。超出就是缺成本，成交当天报错。

`load_research()` 读研究期因子缓存和日线；`load_through_year(y)` 在研究期后面接上 2022 至 y 年现算的五个因子（研究期最后 60 天两边必须一致，否则报错），2023 起只能在最终评估块内调用。`ledger_inputs()` 把一个区间的价格、换月、合约代码、费用和滑点对齐好。

## 输出

`runs/strategies/<id>/commodity/<区间>/`，区间是 `research`、`oos`、单年 `2019`，或 `2018-2020`。每个区间里有 `summary.csv`、`annual.csv`、`daily.csv`、`positions.csv`、`events.csv`，净值图在 `plots/<策略名>/`。研究期每条还写一份 `runs/research/<id>/protocol.json`（当时的规格）。

绩效函数 `performance`、`sharpe_ratio`、`max_drawdown`、`newey_west_t`、`deflated_sharpe`、`pbo_cscv` 也在 `backtest/engine.py`。
