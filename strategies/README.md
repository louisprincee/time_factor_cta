# 当前策略

商品两条冻结策略保持不变。股指唯一注册策略为已冻结的 [`index_clv_rally_base_long_2x`](index_clv_rally_base_long_2x/index_clv_rally_base_long_2x.json)：21/63/126日上涨窗口数量决定50%、33.33%、16.67%、0%的多头底仓，剩余单位预算仅在pooled CLV<0.4时使用；冲高z>0时只把CLV预算乘2。没有空头。缺预热保留多头票，完整零收益不投多票；窗口及预算档位为用户确认，未搜索。历史1.5倍基线仅保留结果和账本，原策略实现、配置及注册已移除。

2倍版已按用户批准冻结，研究期年化18.95%、Sharpe1.49、最大回撤−14.87%，平均实际敞口50.13%；相对预登记历史1.5倍基线的收益、Sharpe和回撤筛选均通过。最大实际敞口201.49%，最大保证金权益占用40.30%，未发生保证金追缴。冻结不等于验证；OOS尚未运行，须用户明确要求后才运行。研究登记和结果在 [`index_guide.md`](../docs/index_guide.md)。

```bash
python -m backtest.engine --market index --research
python -m backtest.engine index_clv_rally_base_long_2x --research
```

账本和逐年图在 `runs/index/strategies/index_clv_rally_base_long_2x/research/`；历史1.5倍账本仍在 `runs/index/strategies/index_clv_rally_base_long/research/`。经济和已否定的交易记录见 [`index_guide.md`](../docs/index_guide.md)，退出注册的旧规则见 [`index_research_review.md`](../docs/index_research_review.md)。
