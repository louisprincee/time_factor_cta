"""因子层：持续期、四个日频因子，以及按 (N, M) 落盘的缓存。

落盘的是原始因子值。方向符号在 research 入模前乘上，不写进缓存。
"""
from . import duration, factor_cache, factors
