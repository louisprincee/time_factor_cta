# 当前策略

商品两条冻结策略保持不变。股指唯一注册策略是 [`index_clv_rally_base_long`](index_clv_rally_base_long/index_clv_rally_base_long.json)：21/63/126 日上涨窗口数量决定50%、33.33%、16.67%、0%的多头底仓，剩余单位预算仅在pooled CLV<0.4时使用；冲高z>0只把CLV预算乘1.5。没有空头。缺预热保留多头票，完整零收益不投多票；窗口及预算档位为用户确认，未搜索。

研究期年化16.82%、Sharpe1.44、最大回撤−14.50%，平均实际敞口48.02%，6/7年为正、2021年−0.92%。相对上一多空版收益差NW t2.45，但回撤加深；整体改进判定失败。研究登记在 `config/research/index_base_budget_transfer.json`，未冻结、不运行样本外。

```bash
python -m backtest.engine --market index --research
python -m backtest.engine index_clv_rally_base_long --research
```

账本和逐年图在 `runs/index/strategies/index_clv_rally_base_long/research/`。经济和已否定的交易记录见 [`index_guide.md`](../docs/index_guide.md)，退出注册的旧规则见 [`index_research_review.md`](../docs/index_research_review.md)。
