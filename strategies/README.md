# 策略

正在用的是 [`market_clock_carry`](market_clock_carry/)。五日符号和当日分数的对照账都已删除。参数在同目录的 JSON 里，导入时跟代码核对。

## 当日分数且展期为正

五个因子 `ts_low`、`ts_high`、`dfp_top3`、`ts_low_pm`、`ts_high_pm` 乘事前方向，各品种按自己过去 250 日（不含当天）做 z，五个取平均，再在当年池内取平均，得到当天的市场分数。

下一开盘做多要同时满足两件：当天分数大于 0；该品种主力对次主力的年化展期收益，再用前一个交易日的数，严格大于 0。近月高于远月时这个数为正。否则这个品种权重是 0。名义敞口上限 1 倍。峰值回撤 10% 清仓，停 10 个交易日。

研究期 2016–2021 是同一窗上的又一次查看：年化 7.2%，Sharpe 1.36，净值路径回撤 −5.7%，beta 0.25，年化 alpha +3.7%，残差 t 2.20，年化成本 3.0%。六年每年净收益都为正。

```bash
python -m backtest.engine --research                 # 2016–2021 一本账
python -m backtest.engine --research --by-year
python -m backtest.engine --year 2019
python -m backtest.engine --start 2018 --end 2020
python -m backtest.engine --validation               # 2022，先记台账
python -m backtest.engine --oos                      # 2023–2025 一本账，先记台账
python -m backtest.engine --oos --by-year
python -m backtest.engine --year 2024
```

2022 及以后先记台账。这些年份以前看过，不是干净样本外。展期收益读到 2025 年底为止，2026 的行丢掉。

账本是 1 亿元、名义敞口上限 1 倍、20% 保证金假设。费用和滑点见 [`backtest/README.md`](../backtest/README.md)。`python -m backtest.engine --research`、`--validation` 和 `--oos` 都跑这一条。

## 已删除

四条尾部策略的做法和数字在 [`docs/四策略做法与表现.md`](../docs/四策略做法与表现.md)。只做多全市场时钟和时钟做多五日跌破的代码、结果都已删除；时钟当时的做法和逐年数字在 [`docs/全市场时钟做法与表现.md`](../docs/全市场时钟做法与表现.md)。上面的命令现在只跑当日分数且展期为正。
