# 回测

2026-10-29清理：未提交的全局回撤预算、状态过滤及锚价缩放等实验扩展已撤回，原账本与冻结策略规则不变。已做研究见 [`docs/index_research_review.md`](../docs/index_research_review.md)。当前2022–2026全部封闭，包括已有结果；下文历史OOS命令不构成重新运行或读取授权。

入口是 `python -m backtest.engine`，区间必须显式指定：

```bash
python -m backtest.engine --research                 # 2011–2021 一本账
python -m backtest.engine --start 2016 --end 2021     # 冻结时的研究期
python -m backtest.engine --research --by-year       # 每年一本新账
python -m backtest.engine --year 2019
python -m backtest.engine --start 2018 --end 2020
python -m backtest.engine index_close_location_bear_short --research  # 熊市空头含波动率缩放，只跑 2015–2021
python -m backtest.engine --market index --validation  # 熊市空头的 2022 已打开，只打印保存的结果
python -m backtest.engine --year 2024                # 单年，冻结后
python -m backtest.engine --oos [--by-year]          # 2023–2025，冻结后
```

当前 `strategies/commodity_catalog.py` 登记两条冻结商品策略；不指定策略名时，商品入口会运行两条。股指策略由 `strategies/index_catalog.py` 独立登记。不点名时股指入口跑熊市空头（含波动率缩放），研究期 2015–2021，`--validation` 只打印已打开的 2022。不读 2023–2025，不读 2026。

```bash
python -m backtest.engine market_clock_slow_short_20_60_120 --research
python -m backtest.engine 商品收盘位置策略 --research
python -m backtest.engine --market index --research
```

账本结果直接位于 `runs/strategies/<策略>/<区间>/`，不再按商品或股指增加目录层。图表直接位于各账本的 `plots/` 下；股指熊市空头的研究期账本另写 `plots/vs_buy_hold.png` 和 `side_days.csv`。

2022 及以后：点到的策略里有任何一条没冻结，整批直接拒绝，不写台账也不读行情。冻结了就先写台账再读（2022 记 `year_test`，2023 起走 `final_evaluation`，都带冻结日期、`chosen` 和代码与规格文件的哈希）。台账里已经打开过的窗口只打印旧结果；结果文件丢了，要先追加一条 `kind: void` 并写原因才能重开。不读 2026。

2026-10-26 用户决定：2022–2025 是当前商品策略的样本外，商品原策略从未在这几年运行过；2026-10-27 冻结的收盘位置策略也没有 OOS 结果。用户明确说开始看样本外之前一律不打开任何 2022+ 窗口。台账（`data/commodity/oos/ledger.jsonl`）目前是空的，打开时先写台账再读。

## 账本

`simulate_target_lots`：整数手、按目标仓位交易。策略给出每个品种在执行日开盘的目标权重（−3 到 3，负数做空；当前策略用 −1、0、1）。

- 名义金额 = |权重| × 敞口上限 × 期初权益 / 当年池宽，向下取整到整数手，手数与权重同号。
- 权重不变就继续持有，不随价格或权益调手数，只成交差额。
- 盈亏 = 有符号手数 × 合约乘数 × 复权开盘价变动（加法复权，差值除以原始价，不做比值）。合约乘数按所持合约取：RU1207 以前 5 吨、FU1202–FU1812 50 吨、PB1109–PB1308 25 吨（`tfcta.costs.CONTRACT_MULTIPLIER_RANGES`），换月到乘数不同的合约时名义不变、手数向下取整。
- 换月在旧主力前一日收盘和新主力当日开盘上收手续费和滑点，不结转真实合约结算盈亏。
- 无开盘价或最高等于最低（涨跌停代理）当日不成交，次日重试。持仓期间缺复权开盘变动直接报错。
- 整本账一个峰值回撤熔断：10% 清仓、停 `halt_days`（当前策略为 10）个交易日，恢复当日把熔断用的峰值改成当时权益（`daily.csv` 的 `熔断口径回撤`）。`summary.csv` 的 `最大回撤` 从初始资金起算，不受这个重置影响。
- 保证金按名义金额 20% 估计，超过权益就清仓。
- 1 亿元初始权益。股指账本下单时再按当天的中金所非套保开仓限额截断，超出的当天不成交，之后补。年化收益、波动率和 Sharpe 用每年 242 个交易日。`summary.csv` 和 `annual.csv` 同时报告 Calmar 比率，定义为年化收益 / 最大回撤绝对值；年度 Calmar 按该年度的有效交易日数年化。
- `simulate_target_lots` 支持可选 `open_slippage={"buy", "sell", "auction_lots"}`：每个品种日先按 `auction_lots` 对所有开盘成交腿共用竞价额度，额度内用一跳成本，超出部分按买卖方向的成本率；换月旧合约仍按前收一跳。`daily.csv` 另记录开盘成交手数、集合竞价成交手数及两类开盘滑点成本，便于核对参与比例和加权跳数。

手续费按历史费率：按金额 ×1.01、按手数 +0.01 元；手续费表最多向后沿用 31 天。米筐的上期所费率从 2013-10-08 才有，2011-01-01 到 2013-10-07 的 AG、AL、AU、CU、RB、RU、ZN 用各自 2013-10-08 的费率往前沿用（用户 2026-10-23 决定，`backfill_shfe_fees`），这是假设；把这段费率改成 0 或翻倍，2011–2013 年净收益变动不到 0.2 个百分点。大商所 2010 年起是真实费率。滑点是上一年估计 tick 每边 1 跳，上一年估不出时最多再沿用一年。超出就是缺成本，成交当天报错。

`load_research(first_year=None)` 读研究期因子缓存和日线，默认从 2016 起；`first_year` 早于 2016 时，之前几年的品种池用落盘的 yearly_stats 按同一门槛重建，重建的 2015 必须等于落盘的，并从五个因子都有值的那天（2011-01-14）起计分；`load_through_year(y)` 在研究期后面接上 2022 至 y 年现算的五个因子（研究期最后 60 天两边必须一致，否则报错），2023 起只能在最终评估块内调用。`ledger_inputs()` 把一个区间的价格、换月、合约代码、费用和滑点对齐好。

## 输出

结果位于 `runs/strategies/<策略>/<区间>/`。区间是 `research`、单年 `2019` 或 `2016-2021` 这样的起止年。每个账本目录包含 `summary.csv`、`annual.csv`、`daily.csv`、`positions.csv`、`events.csv` 和 `plots/performance.png`。研究期规格文件仍位于对应的 `runs/research/<id>/protocol.json`。

绩效函数 `performance`、`sharpe_ratio`、`max_drawdown`、`newey_west_t`、`deflated_sharpe`、`pbo_cscv` 也在 `backtest/engine.py`。

## 商品两本冻结策略的图表报告

先用 `backtest.engine` 跑两本商品冻结账本，再运行 `python -m research.report.strategy_report`（规格 `config/research/strategy_report.json`）。报告按冻结规则重算完整策略并与 engine 账本逐日核对，再在同一账本、同样成本下跑组成部分和买入持有基准：

- `runs/reports/commodity/plots/`：`nav_drawdown.png`（两本商品策略净值与回撤）、`vs_buy_hold.png`（对比池内等权买入持有）、`components.png`（只保留时间信号、展期、多头腿、空头腿时的净值）。
- 目录里另有 `summary.csv`、`annual.csv`、`leg_contributions.csv`（完整账本多头腿、空头腿、成本逐年贡献，不复利，可复算净收益）、`daily_returns.csv`。

组成部分是各自独立的账本，不能相加。结果都是研究期样本内描述。
