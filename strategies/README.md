# 当前冻结策略

商品两条不变。股指默认仍为用户2026-11-02选定的 `index_clv_rally_fixed_long`；另登记单品种信号活跃等权、汇总信号品种等权、20日相对收益排名和20日逆波动率四本冻结策略。原策略及默认路由不变。样本外尚未回测。

| 市场 | 策略 | 冻结日 | 规则 |
|---|---|---|---|
| 商品 | 商品原策略 | 2026-10-23 | 五因子市场分数、正carry多头及趋势负carry空头 |
| 商品 | 商品收盘位置策略 | 2026-10-27 | CLV替代市场分数，其余商品规则不变 |
| 股指 | 弱收盘冲高固定权重多头 | 2026-11-02 | CLV<0.4次日做多，冲高z>0乘1.5，固定IF/IH/IC约29.72/20.79/49.48%，无空头 |
| 股指 | 单品种弱收盘活跃等权多头 | 2026-11-03 | 各自CLV<0.4次日触发，有信号品种等分预算；保留冲高放大和20日波动缩放 |
| 股指 | 汇总弱收盘等权多头 | 2026-11-03 | IF/IH/IC等权CLV均值<0.4次日共同触发，三个品种等权；保留冲高放大和20日波动缩放 |
| 股指 | 股指弱收盘20日收益排名多头 | 2026-11-03 | pooled CLV<0.4触发；按20日相对收益排名配权；保留冲高放大和波动缩放 |
| 股指 | 股指弱收盘20日逆波动率多头 | 2026-11-03 | pooled CLV<0.4触发；按20日逆波动率配权；单品种乘数封顶2并再分配 |

股指规则与逐年收益见 [`index_guide.md`](../docs/index_guide.md)。原策略JSON在 [`index_clv_rally_fixed_long.json`](index_close_location/index_clv_rally_fixed_long.json)，另外两本策略JSON分别见 [`index_clv_individual_equal_long.json`](index_clv_individual_equal/index_clv_individual_equal_long.json) 和 [`index_clv_pooled_equal_long.json`](index_clv_pooled_equal/index_clv_pooled_equal_long.json)。原策略仍为 `--market index` 默认项。

```bash
python -m backtest.engine --research
python -m backtest.engine --market index --research
python -m backtest.engine index_clv_individual_equal_long --research
python -m backtest.engine index_clv_pooled_equal_long --research
python -m backtest.engine --oos
python -m backtest.engine --market index --oos
python -m research.report.strategy_report
```

商品执行规则见 [`guide.md`](../docs/guide.md)。股指研究采用固定1000万元、历史费用及L1 30分钟执行。研究结果为样本内历史结果，不代表未来表现或可部署优势。
