"""慢趋势空头的 1.5 倍敞口候选；交易信号复用已冻结的 1 倍策略。"""
from pathlib import Path

from strategies import market_clock_slow_short as S


class Book(S.Book):
    ID = "market_clock_slow_short_15"
    NAME = "按池宽1.5倍加慢趋势空头"
    GROSS = 1.5
    CONFIG_PATH = Path(__file__).with_name("market_clock_slow_short_15.json")
    DETAIL = (
        "信号和风控同按池宽1倍加慢趋势空头；多空腿都按当年池宽分1.5倍名义敞口"
        "/下一交易日第一根开盘成交/权重不变则留仓/回撤10%清仓并停10个交易日"
    )

    @property
    def FILES(self) -> list[Path]:
        return [Path(S.__file__), Path(__file__), self.CONFIG_PATH]


BOOK = Book()
BOOK.load_config()
