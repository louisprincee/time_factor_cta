"""兴业证券高频时间维度因子：商品期货分钟频复现。见 docs/Design.md。

模块结构
--------
config                        全局常量：路径、时间切分、时段、参数网格、因子方向、样本外守卫
data.shard_io                 分钟分片读写，验证期 / 样本外守卫只写在这里
data.bars                     日线聚合、日收益与前瞻收益口径、跨分区拼接分钟、乘法复权价
data.sessions                 日内时序坐标与 NIGHT/AM/PM 时段划分
data.universe                 时点有效品种池（研究期、2022 验证、样本外）
data.sectors                  五大板块分类
data.synth                    合成分钟数据（测试用，复刻真实面板结构）
factors.intraday              分钟级时间因子：滚动阈值、持续期、DFP、时间戳
factors.daily                 日频技术指标与慢变量（动量、偏度、低波、展期、carry）
factors.external              外部基本面因子的构建与读取（跨分区连续）
factors.cache                 时间因子逐品种落盘与读取
factors.library               因子库：事前 z 分数、先验方向、全部因子拼装成 SignalSet
research.analysis.stats       时序 IC、六项绩效、walk-forward 统计
research.analysis.screen      板块异质性筛选
research.backtest.engine      周频调仓、下一开盘成交、扣费、等权组合、换手
research.backtest.costs       tick 表与滑点
research.backtest.strategy    自选因子与板块的等权书、配置指纹、通过门槛
research.workflow.context     运行留痕、JSON 落盘、研究期上下文
research.workflow.history     验证期 / 样本外的分钟拼接与因子重算
research.workflow.ledger      验证期与样本外使用台账
"""
