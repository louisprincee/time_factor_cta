# 回测

回测入口为 `python -m backtest.engine`，运行区间必须显式指定：

```bash
python -m backtest.engine --research                 # 2016–2021 一本账
python -m backtest.engine --research --by-year       # 每年一本新账
python -m backtest.engine --validation               # 2022
python -m backtest.engine --oos                      # 2023–2025 一本账
python -m backtest.engine --oos --by-year
python -m backtest.engine --year 2019
python -m backtest.engine --start 2018 --end 2020
python -m backtest.engine --start 2022 --end 2024 --by-year
```

样本内和样本外都用这三种：整段一本账、单独一年、从某年到某年。`--by-year` 把区间拆成每年一本新的 1 亿元账。碰到 2022 及以后先记台账。结果文件已经在的区间只打印，不再读行情。

当前只有商品上的五日符号，`--market index` 会拒绝。四条尾部策略的做法在 `docs/四策略做法与表现.md`。

整数手数账本：1 亿元初始权益、名义敞口上限 1.5 倍、20% 保证金假设。冻结的只做多时钟是 20% 峰值回撤、停到下一年。五日符号是 10% 峰值回撤、停 10 个交易日。费率取对应历史手续费缓存，滑点按前一年 tick 估算。年化收益、波动率和 Sharpe 使用每年 242 个交易日。已删除的四条纯多头（持有 5 日、20% 熔断停到下一年）只留在 `docs/四策略做法与表现.md`。

输出再按区间放在 `research/`、`validation/` 和 `oos/`。每个区间里是
`summary.csv`、`annual.csv`、`daily.csv`、`positions.csv`、`events.csv`，净值图在同级的 `plots/`。
商品用当年流动性池里的全部品种。股指研究期用 IF、IH、IC。

当前策略是当日分数且展期为正，名义敞口上限 1 倍。结果写到 `runs/strategies/market_clock_carry/commodity/` 下的 `research`、`2022`、`oos` 或 `起始-结束`。2022 及以后先记台账。这些年份以前看过，不是干净样本外。展期收益只用到 2025-12-31。

`--experimental-research` 跑 `strategies/research_catalog.py`，现在只有五日符号，只读 2016–2021。

策略声明 `EXECUTION = "target"` 时改用 `simulate_target_lots`：策略的 `targets()` 给出每个品种在执行日开盘的目标权重（0 到 3，等权满仓为 1），名义金额 = 权重 × 1.5 × 期初权益 / 池宽，开盘只成交目标与现有手数的差额，权重不变不调手数，整本账一个熔断。两个账本都是纯多头。`ledger_inputs()` 给出一个区间对齐好的价格、换月、合约代码、费用和滑点，研究脚本直接调用账本时用它。

手续费表只向后沿用最多 31 天的旧费率，tick 表在上一年估不出时最多再沿用一年；超出就是缺成本，成交当天直接报错，不会悄悄用旧值。

整数手账本以连续主力的加法复权开盘价变动估算持仓盈亏；换月在旧主力前一日收盘和新
主力当日开盘上估算手续费与滑点，不结转真实合约现金流或结算盈亏。换月、持仓缺价、
自然年切换及样本末未到期持仓有合成边界测试；持仓期间缺失复权开盘变动会显式报错。
持仓日原始开盘价缺失按不可成交处理，退出委托延迟，并以最后有效开盘价记录敞口。
样本末的未平仓持仓不自动强平，期末结果不含假定的清算费用。因此结果仍是研究估算，
不是交易所级成交仿真。

已移除四板块裁剪。账本使用当年流动性池的全部商品。样本外结果记在
`data/commodity/oos/ledger.jsonl`，全品种这次已经跑完，不能再读 2023–2025。
