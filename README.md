# 商品期货时间因子研究

研究价格停留时间、高低点出现的先后，能否为商品期货提供可交易的信息。2016–2021 是研究期。2022 已经被用过，不能再当作从未看过的验证集。2023–2025 的分片已从本地样本外目录移除。

研究读取接口对 2023–2025 保持硬锁，年份已经结束也不能解锁。最终测试需要另行建立冻结版本校验流程。正确性审查及修复记录见 [RepositoryAudit_20261005.md](docs/RepositoryAudit_20261005.md)。

当前没有冻结策略。早盘规则的作图和比较在：

```bash
python scripts/plot_morning_rule.py
python scripts/select_main_book.py
```

日频因子和对照回测：

```bash
python scripts/build_factors.py
python scripts/research.py --specs config/research_candidates.json
```

首次准备数据用 `scripts/step1_shard_minutes.py` 和 `scripts/step2_universe.py`。完整原始单体可能含更晚的年份，日常研究只读已经隔离的研究分片。

因子在收盘才知道的，下一交易日开盘成交。日内规则用历史开仓费和平今费，每边 1 个最小变动价位。这是连续名义仓位的研究模型，没有整数手、保证金、涨跌停和盘口冲击。

四组预登记早盘候选的完整研究在 [RobustMorningResearch_20261005.md](docs/RobustMorningResearch_20261005.md)。本轮没有候选达到稳健性目标，没有冻结策略；复现入口为 `python scripts/research_robust_morning.py`。

```bash
python -m pytest
```
