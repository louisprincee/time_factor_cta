# 回测

回测入口为 `python -m backtest.engine`，运行区间必须显式指定：

```bash
python -m backtest.engine --research                 # 商品，2016–2021
python -m backtest.engine --market index --research  # 股指，2016–2021
python -m backtest.engine --validation               # 2022
python -m backtest.engine --oos                      # 2023–2025，同一市场只打开一次
python -m backtest.engine --experimental-research    # 新探索策略，商品2016–2021
```

`--market commodity` 是商品流动性池，也是默认值。`--market index` 跑 IF、IH、IC。2022 验证只用这三个；IM 只在样本外里按前一年流动性决定是否进池。股指没有夜盘，时点按当日日盘第一根到最后一根归一化。两个市场的结果分开写：`runs/strategies/<策略>/commodity/` 和 `runs/strategies/<策略>/index/`。股指样本外另记 `data/index/oos/ledger.jsonl`。

两种模式都使用整数合约手数账本，一次跑 `strategies/catalog.py` 里的四条纯多头：
趋势加三日平滑尾部、趋势加尾部且高波动、趋势加尾部且 ADX 震荡、趋势与震荡分腿。持有 5 个有效交易日；
1 亿元初始权益、名义敞口上限 1.5 倍、20% 保证金假设、20% 峰值回撤熔断。费率取对应历史手续费缓存，
滑点按前一年 tick 估算。年化收益、波动率和 Sharpe 使用每年 242 个交易日。

输出再按区间放在 `research/`、`validation/` 和 `oos/`。每个区间里是
`summary.csv`、`annual.csv`、`daily.csv`、`positions.csv`、`events.csv`，净值图在同级的 `plots/`。
商品用当年流动性池里的全部品种。股指研究期用 IF、IH、IC。

`--experimental-research` 单独运行 `strategies/research_catalog.py` 注册的三条商品探索策略：时钟截面多空、时钟加正展期收益过滤，以及趋势/carry 同向确认的全交易日时钟组合（含夜盘）。纯日盘策略已移除。实验入口只读取 2016–2021 研究分片，不读取 2022 或 2023 年以后；结果仍使用整数手账本、历史费率、上一年 tick 滑点和相同风险约束。该 catalog 不参与冻结名单和 OOS 审计，组合结果属于事后探索。

手续费表只向后沿用最多 31 天的旧费率，tick 表在上一年估不出时最多再沿用一年；超出就是缺成本，成交当天直接报错，不会悄悄用旧值。

整数手账本以连续主力的加法复权开盘价变动估算持仓盈亏；换月在旧主力前一日收盘和新
主力当日开盘上估算手续费与滑点，不结转真实合约现金流或结算盈亏。换月、持仓缺价、
自然年切换及样本末未到期持仓有合成边界测试；持仓期间缺失复权开盘变动会显式报错。
持仓日原始开盘价缺失按不可成交处理，退出委托延迟，并以最后有效开盘价记录敞口。
样本末的未平仓持仓不自动强平，期末结果不含假定的清算费用。因此结果仍是研究估算，
不是交易所级成交仿真。

已移除四板块裁剪。账本使用当年流动性池的全部商品。样本外结果记在
`data/commodity/oos/ledger.jsonl`，全品种这次已经跑完，不能再读 2023–2025。
