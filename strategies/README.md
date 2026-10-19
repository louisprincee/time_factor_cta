# 策略

每条策略单独一个目录，里面是实现和一份参数说明。JSON 写下样本边界和参数，导入时跟代码核对，避免两边悄悄不一致。

四条都是纯多头。时间信息都是低点时戳减高点时戳，必须严格高于该品种自己过去 250 个有效交易日、不含当天的 80% 分位。收盘出信号，下一可交易日开盘进，持有 5 个有效交易日。

- [`trend_ema3`](trend_ema3/)：过去 10 日上涨，并且低点减高点的 3 日指数平均处在最高 20%。
- [`trend_high_atr`](trend_high_atr/)：过去 10 日上涨，用未经平滑的尾部，只在 14 日真实波幅高于该品种自己过去 250 日中位数时做。
- [`trend_adx_range`](trend_adx_range/)：过去 10 日上涨，用未经平滑的尾部，只在 14 日 ADX 小于等于 20 时做。
- [`regime_two_leg`](regime_two_leg/)：一本账分两条腿。ADX 小于等于 20 时只看未经平滑的尾部，不要求 10 日上涨。ADX 大于 20 时，要同时有 10 日上涨、同一条尾部和高波动。

四条已在 2026-10-19 一起冻结。2022 已经用来做过取舍，分腿还用过 2022 的状态分解。样本外只跑一次 2023–2025。

## 探索策略

下面三条单独注册在 `research_catalog.py`，不加入冻结 catalog，也不允许通过 OOS 入口运行。它们只用于商品 2016–2021 的探索性比较，不是对已读样本外的重新验证。

- `cross_sectional_clock`：每日在当年流动性池内按全日时钟分数排序，做多最高 20%、做空最低 20%，两边等数量；若分界有并列则当天不交易。每边至少需要 10 个有效品种。
- `trend_high_atr_carry`：沿用高 ATR 多头规则，并要求前一交易日已知的主力/次主力年化展期收益率大于 0。carry 使用现有一日滞后对齐，不把当天未完成数据用于当天信号。
- `trend_carry_full_clock`：10 日趋势与滞后 carry 同号决定方向，包含夜盘的全交易日时钟必须落在同方向历史尾部；按当日流动性池中位 ATR% / 本品种 ATR% 缩放仓位，缩放范围为 0.5–2 倍。carry 符号不直接解释成升水/贴水，供应方公式尚未验证。旧纯日盘结果文件仍保留作历史记录，但纯日盘策略已从 catalog 和执行入口移除。

从仓库根目录运行独立研究回测：

```bash
python -m backtest.engine --experimental-research
```

结果写入各策略自己的 `runs/strategies/<策略>/commodity/research/`。截面空头和组合方向使用带符号整数手数，交易费用和保证金按绝对手数计，损益按方向计。此入口不会读取 2022 或 2023 年以后数据。

组合参数见 [`config/research/trend_carry_full_clock.json`](../config/research/trend_carry_full_clock.json)。组件策略和上一版日盘组合结果在本版本定义前已查看，因此它是事后探索候选，不能作为确认性结论或干净样本外验证。

从仓库根目录运行：

```bash
python -m backtest.engine --research
python -m backtest.engine --market index --research
python -m backtest.engine --validation
```

`--market index` 把同一四条策略放到股指期货上。商品是默认市场。结果在 `runs/strategies/<策略>/commodity/` 或 `index/` 下，再分 `research/`、`validation/`、`oos/`。做法和现有数字见 [`docs/四策略做法与表现.md`](../docs/四策略做法与表现.md)。

账本是 1 亿元初始权益、名义敞口上限 1.5 倍、20% 保证金假设、20% 峰值回撤熔断。费用和滑点、输出表见 [`backtest/README.md`](../backtest/README.md)。
