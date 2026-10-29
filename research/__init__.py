"""还在用的研究代码：商品账本在 commodity/，图表在 report/。已经做完的机制检验脚本已删除。2022 及以后只走冻结和台账。"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
