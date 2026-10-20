# 策略

每条策略单独一个目录，冻结参数写在同目录的 JSON 里，导入时跟代码核对，避免两边悄悄不一致。

## 冻结的四条（记录）

四条都是纯多头。时间信息都是低点时戳减高点时戳，必须严格高于该品种自己过去 250 个有效交易日、不含当天的 80% 分位。收盘出信号，下一可交易日开盘进，持有 5 个有效交易日。

- [`trend_ema3`](trend_ema3/)：过去 10 日上涨，并且低点减高点的 3 日指数平均处在最高 20%。
- [`trend_high_atr`](trend_high_atr/)：过去 10 日上涨，用未经平滑的尾部，只在 14 日真实波幅高于该品种自己过去 250 日中位数时做。
- [`trend_adx_range`](trend_adx_range/)：过去 10 日上涨，用未经平滑的尾部，只在 14 日 ADX 小于等于 20 时做。
- [`regime_two_leg`](regime_two_leg/)：一本账分两条腿。ADX 小于等于 20 时只看未经平滑的尾部，不要求 10 日上涨。ADX 大于 20 时，要同时有 10 日上涨、同一条尾部和高波动。

四条已在 2026-10-19 一起冻结。2022 已经用来做过取舍，分腿还用过 2022 的状态分解。样本外只跑一次 2023–2025。

## 当前主线

`research_catalog.py` 只注册一条：[`market_clock_long`](market_clock_long/)，2026-10-20 冻结，参数在同目录的 `market_clock_long.json`。它不进冻结四条的名单，样本外只能用 `--year` 按年打开。

- **信号**：研报五因子 `ts_low`、`ts_high`、`dfp_top3`、`ts_low_pm`、`ts_high_pm` 乘事前方向后，按各品种自身过去 250 日做 z，五个取平均，再在当年池内取平均，得到一个市场分数。分数大于 0 当日看多，否则看平，不做空。
- **仓位**：目标权重 = 最近 5 个收盘里看多的占比 × 账户乘数（池子 20 日波动高于 250 日时降仓，250 日累计收益为负时再减半）× 品种相对权重（60 日波动倒数，池内平均为 1，上限 3）。看多占比不变时沿用上次的目标，下一交易日开盘只成交手数差额。满仓约 1.5 倍权益，整本账一个 20% 熔断。

研究期的六个登记变体在 [`config/research/market_clock_long_robustness.json`](../config/research/market_clock_long_robustness.json)，用 `python -m research.market_clock_long` 重跑，结果在 `runs/research/market_clock_long/`：

| 变体 | 净 Sharpe | 年化 | 年化波动 | 最大回撤 | beta | 年化 alpha | 下跌月捕获 | 黑色占毛盈亏 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 等权主规格 | 1.31 | 13.2% | 9.8% | −11.8% | 0.71 | 3.4% | 0.50 | 44% |
| 收盘位置对照 | 1.13 | 11.3% | 9.9% | −13.6% | 0.71 | 1.6% | 0.61 | 48% |
| 波动降仓 | 1.46 | 13.2% | 8.8% | −11.7% | 0.62 | 4.5% | 0.44 | 44% |
| 趋势帽 | 1.29 | 12.1% | 9.2% | −11.8% | 0.65 | 3.1% | 0.47 | 45% |
| 波动倒数权重 | 1.35 | 11.2% | 8.1% | −10.4% | 0.58 | 3.2% | 0.42 | 36% |
| 三者合一（冻结） | 1.43 | 10.2% | 7.0% | −10.4% | 0.48 | 3.6% | 0.35 | 36% |

板块占比门槛原登记 35%，冻结的版本 36.2% 没过；看到结果后放宽到 40% 才冻结，这是事后放宽。三条机制检验（同涨同跌、放量、持仓下降）都不显著。冻结后打开的 2022、2023 见 [`docs/全市场时钟做法与表现.md`](../docs/全市场时钟做法与表现.md)。

```bash
python -m backtest.engine --experimental-research   # 研究期，写到 runs/strategies/market_clock_long/commodity/research/
python -m research.market_clock_long                # 研究期登记变体、稳健性网格、机制检验
python -m backtest.engine --year 2024               # 按顺序打开下一年，写到 commodity/2024/
```

四条从仓库根目录运行：

```bash
python -m backtest.engine --research
python -m backtest.engine --market index --research
python -m backtest.engine --validation
```

`--market index` 把同一四条策略放到股指期货上。商品是默认市场。结果在 `runs/strategies/<策略>/commodity/` 或 `index/` 下，再分 `research/`、`validation/`、`oos/`。做法和现有数字见 [`docs/四策略做法与表现.md`](../docs/四策略做法与表现.md)。

账本是 1 亿元初始权益、名义敞口上限 1.5 倍、20% 保证金假设、20% 峰值回撤熔断。费用和滑点、输出表见 [`backtest/README.md`](../backtest/README.md)。
