# 当前策略

商品两条冻结策略保持不变。`strategies/index_catalog.py` 是股指策略的唯一注册表；`index_tactical_books.py` 实现纯 CLV 和因子倾斜两本书，不维护第二份注册列表。股指正式注册且默认运行的策略仅保留以下三本，均只做多 IF/IH/IC：

1. **纯 CLV 等权**（默认，`index_tactical_clv_only`）：市场等权 CLV<0.4 时三品种等权持有全部择时预算，否则现金；冲高 z>0 只将 CLV 预算乘 2。
2. **CLV 因子倾斜**（`index_tactical_clv_factor_select`）：弱收盘时三品种都参与，预算按 1+有利票分配。因子格由同一研究样本事后筛选，显著性和独立证据不足。
3. **标准 55/20 突破趋势袖套**（`index_clv_breakout_sleeve`）：在纯 CLV 规则上增加固定 0.5 总预算的多头袖套；收盘高于此前 55 日最高价时入场，收盘低于此前 20 日最低价时退出，活跃品种等权。

研究期指标显示，纯 CLV 风险调整表现最好；标准突破累计收益稍高，但 Sharpe 更低、回撤更深，配对增量不显著。因子倾斜是用户指定保留的唯一因子策略，不能将其同样本的选择结果解释为已验证优势。三本仅有 2015–2021 研究期结果，OOS 尚未运行，不得在用户明确要求前运行。策略选择与证据边界见 [`docs/index_guide.md`](../docs/index_guide.md)。

已退役规则（动量底仓、牛市/MACD/EMA 补半仓、非弱收盘突破）不再属于 `INDEX_BOOKS` 或 OOS 集合，也不能用策略 ID 运行。牛市/MACD/EMA 变体的策略包装和无复现用途的协议已删除，历史规则与结果保留在研究文档中；动量归因逻辑并入历史归因研究脚本。非弱收盘突破的研究 runner 与协议仍保留，仅用于复现历史结果。旧三策略比较协议已删除；当前集合的后续比较须先遵照 [`当前对比预登记`](../config/research/index_selected_strategy_comparison.json)。

```bash
python -m backtest.engine --market index --research
python -m backtest.engine index_tactical_clv_only --research
python -m research.index strategy comparison
```

各策略账本及其相对等权买入持有的总图和逐年图位于 `runs/index/strategies/<策略ID>/research/`。当前三策略加基准的总图和逐年图位于 `runs/index/research/index_strategy_comparison/plots/`。
