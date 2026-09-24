"""因子层：持续期、日频因子，以及按 (N, M) 落盘的缓存。

duration      持续期定义与滚动阈值
factors       持续期族、时间戳族、方向符号
factor_cache  日频因子落盘与读取

落盘的是原始因子值。方向符号在 research 入模前乘上，不写进缓存。
"""
from . import duration, factor_cache, factors
