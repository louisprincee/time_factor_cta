"""事先登记的研究：两份研报的因子筛选、当前策略的区间回测和信号层检验。2022 及以后只走冻结和台账。"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
