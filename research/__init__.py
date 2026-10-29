"""商品账本在 commodity/，图表在 report/。股指机制检验脚本在本目录。2022 及以后只走冻结和台账。"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
