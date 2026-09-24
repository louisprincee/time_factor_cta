"""兴业证券高频时间维度因子：商品期货分钟频复现。见 docs/Design.md。

模块结构
--------
config                 全局常量：路径、切分、时段、参数网格、因子方向、样本外守卫
data.sessions          日内时序坐标与 NIGHT/AM/PM 时段划分
data.shard_io          分钟分片读写，样本外守卫只写在这里
data.universe          时点有效品种池
data.synth             合成分钟数据（测试用，复刻真实面板结构）
factors.duration       持续期核心（滚动阈值 + 逐日向量化计算）
factors.factors        持续期族、时间戳族、方向符号
factors.factor_cache   日频因子落盘与读取
research.panel         日收益、滚动 MAD、滚动分位信号
research.stats         时序 IC 与六项绩效
research.book          等权回测与分族合成
research.protocol      walk-forward、中心点选参、冻结配置
research.jobs          落盘路径与各步共用的读取
"""
