# 当前策略

商品两条冻结策略保持不变。股指唯一注册策略是 [`index_clv_rally_base_long`](index_clv_rally_base_long/index_clv_rally_base_long.json)：等权收盘收益的 63/126/252 日动量多数表决决定半仓多空；另一半只在 pooled CLV<0.4 时加多头，冲高 z>0 只把这一半乘 1.5。未满窗口投多头票，完整窗口收益为零投空头票。没有单边上涨；半仓和三个长度是用户批准的固定设计，不是按收益搜索的。

研究期年化 10.72%、Sharpe 1.09、最大回撤 −10.30%，平均绝对敞口 0.434，6/7 年为正，2021 年 −1.03%。相对 MACD 日收益差 NW t 0.64，相对买入持有 t 0.70，都不显著。规格未冻结，样本外不运行。底仓对照登记在 `config/research/index_base_momentum.json`。

```bash
python -m backtest.engine --market index --research
python -m backtest.engine index_clv_rally_base_long --research
```

账本和逐年图在 `runs/index/strategies/index_clv_rally_base_long/research/`。经济和已否定的交易记录见 [`index_guide.md`](../docs/index_guide.md)，退出注册的旧规则见 [`index_research_review.md`](../docs/index_research_review.md)。
