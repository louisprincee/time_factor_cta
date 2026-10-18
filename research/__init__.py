"""两篇研报有效因子的研究。这里不构造策略，也不跑样本外。"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
