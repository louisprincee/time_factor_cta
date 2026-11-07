# 回测

股指当前只注册三本冻结策略：纯 CLV 等权（默认）、CLV 因子倾斜和标准 55/20 突破趋势袖套。`--market index --research` 运行这三本。每本仍写自己相对等权买入持有的总图和逐年图；全部跑完后，引擎再写一张含三策略加基准的 `performance.png`，以及每年一张同样包含全部策略和基准的图，目录是 `runs/index/research/index_strategy_comparison/plots/`。样本外回测尚未运行；用户未明确要求前不得运行。因子倾斜策略的 OOS 因子输入管道已实现，但尚未用真实 OOS 数据验证。此前的动量底仓、均线/MACD/EMA 补仓及非弱收盘突破规则不再注册；历史实现和研究材料保留供核对，不可通过策略 ID 运行。商品不变。规则见[`股指策略说明`](../docs/index_guide.md)。

入口是 `python -m backtest.engine`，区间必须显式指定：

```bash
python -m backtest.engine --research                 # 2011–2021 一本账
python -m backtest.engine --start 2016 --end 2021     # 冻结时的研究期
python -m backtest.engine --research --by-year       # 每年一本新账
python -m backtest.engine --start 2018 --end 2020
python -m backtest.engine --market index --research # 当前三本股指策略，2015–2021，并写总对比图
python -m backtest.engine index_tactical_clv_only --research
python -m backtest.engine --oos
python -m backtest.engine --market index --oos
```

商品仍登记两条冻结策略。股指研究统一使用1000万、l1_30m口径。

```bash
python -m backtest.engine market_clock_slow_short_20_60_120 --research
python -m backtest.engine 商品收盘位置策略 --research
python -m backtest.engine --market index --research
```

股指口径为1000万、l1_30m。开盘各腿按连续交易前30分钟L1卖一/买一成本，旧约换月前收一跳。

账本在 `runs/<市场>/strategies/<策略>/<区间>/`。图表在该目录的 `plots/performance.png`。股指另外逐年生成该策略与等权买入持有的年内累计净值图 `plots/<年份>.png`，并写 `targets.csv`。一次跑多本股指策略时，引擎最后再写 `runs/index/research/index_strategy_comparison/plots/performance.png` 和同年目录下的逐年图，每张都包含全部策略和基准。研究期读 `research/`，样本外读 `oos/`。股指 OOS 尚未运行；因子历史只用于预热，不输出 OOS 预热段的交易目标。

## 账本

`simulate_target_lots`：整数手、按目标仓位交易。策略给出每个品种在执行日开盘的目标权重（−3 到 3，负数做空；当前策略用 −1、0、1）。

- 名义金额 = |权重| × 敞口上限 × 期初权益 / 当年池宽，向下取整到整数手，手数与权重同号。
- 权重不变就继续持有，不随价格或权益调手数，只成交差额。
- 盈亏 = 有符号手数 × 合约乘数 × 复权开盘价变动（加法复权，差值除以原始价，不做比值）。合约乘数按所持合约取：RU1207 以前 5 吨、FU1202–FU1812 50 吨、PB1109–PB1308 25 吨（`tfcta.costs.CONTRACT_MULTIPLIER_RANGES`），换月到乘数不同的合约时名义不变、手数向下取整。
- 换月在旧主力前一日收盘和新主力当日开盘上收手续费和滑点，不结转真实合约结算盈亏。
- 无开盘价或最高等于最低（涨跌停代理）当日不成交，次日重试。持仓期间缺复权开盘变动直接报错。
- 整本账一个峰值回撤熔断：10% 清仓、停 `halt_days`（当前策略为 10）个交易日，恢复当日把熔断用的峰值改成当时权益（`daily.csv` 的 `熔断口径回撤`）。`summary.csv` 的 `最大回撤` 从初始资金起算，不受这个重置影响。
- 保证金按名义金额 20% 估计，超过权益就清仓。
- 商品 1 亿元初始权益；股指研究期固定 1000 万（见上）。年化收益、波动率和 Sharpe 用每年 242 个交易日。`summary.csv` 和 `annual.csv` 同时报告 Calmar 比率，定义为年化收益 / 最大回撤绝对值；年度 Calmar 按该年度的有效交易日数年化。
- `simulate_target_lots` 支持可选 `open_slippage={"buy", "sell", "auction_lots"}`：每个品种日先按 `auction_lots` 对所有开盘成交腿共用竞价额度，额度内用一跳成本，超出部分按买卖方向的成本率；换月旧合约仍按前收一跳。也可改给 `auction_buy_lots` 和 `auction_buy_rate`：只有买入腿先用额度，按给定成本率成交，卖出不进竞价；两种竞价键不能同时给。`daily.csv` 另记录开盘成交手数、集合竞价成交手数及两类开盘滑点成本，便于核对参与比例和加权跳数。

手续费按历史费率：按金额 ×1.01、按手数 +0.01 元；手续费表最多向后沿用 31 天。米筐的上期所费率从 2013-10-08 才有，2011-01-01 到 2013-10-07 的 AG、AL、AU、CU、RB、RU、ZN 用各自 2013-10-08 的费率往前沿用（用户 2026-10-23 决定，`backfill_shfe_fees`），这是假设；把这段费率改成 0 或翻倍，2011–2013 年净收益变动不到 0.2 个百分点。大商所 2010 年起是真实费率。滑点是上一年估计 tick 每边 1 跳，上一年估不出时最多再沿用一年。超出就是缺成本，成交当天报错。

`load_research(first_year=None)` 读研究期因子缓存和日线，默认从 2016 起；`first_year` 早于 2016 时，之前几年的品种池用落盘的 yearly_stats 按同一门槛重建，并从五个因子都有值的那天起计分。`load_through_year(y)` 把研究期接到 2022 至 y 年现算的因子（研究期最后 60 天两边必须一致，否则报错）。`ledger_inputs()` 把一个区间的价格、换月、合约代码、费用和滑点对齐好。

## 输出

结果位于 `runs/<市场>/strategies/<策略>/<区间>/`。区间是 `research`、商品 `oos`、单年 `2019` 或一个自定义年份区间。每个账本目录包含 `summary.csv`、`annual.csv`、`daily.csv`、`positions.csv`、`events.csv` 和 `plots/performance.png`。股指研究账本另有逐年策略/等权买入持有图 `plots/2015.png` 至 `plots/2021.png`。股指的 `summary.csv` 另报同一账本口径下等权买入持有的累计/年化收益、Sharpe、最大回撤，策略减买入持有的年化差、逐日超额年化及 NW t（10 阶）、相对买入持有的 beta 和年化 alpha；`annual.csv` 另报逐年买入持有净收益、超额收益和买入持有年度内最大回撤。商品研究期规格写在同一目录的 `protocol.json`。

绩效函数 `performance`、`sharpe_ratio`、`max_drawdown`、`newey_west_t`、`deflated_sharpe`、`pbo_cscv` 也在 `backtest/engine.py`。

## 商品两本冻结策略的图表报告

先用 `backtest.engine` 跑两本商品冻结账本，再运行 `python -m research.report.strategy_report`（规格 `config/research/strategy_report.json`）。报告按冻结规则重算完整策略并与 engine 账本逐日核对，再在同一账本、同样成本下跑组成部分和买入持有基准：

- `runs/commodity/research/strategy_report/plots/`：`nav_drawdown.png`（两本商品策略净值与回撤）、`vs_buy_hold.png`（对比池内等权买入持有）、`components.png`（只保留时间信号、展期、多头腿、空头腿时的净值）。
- 目录里另有 `summary.csv`、`annual.csv`、`leg_contributions.csv`（完整账本多头腿、空头腿、成本逐年贡献，不复利，可复算净收益）、`daily_returns.csv`。

组成部分是各自独立的账本，不能相加。结果都是研究期样本内描述。
