"""慢趋势空头的三窗口候选：熊市判断换成池子 20/60/120 日三个等权，去掉 250 日；其余同冻结的 1 倍策略。"""
from pathlib import Path

from strategies import market_clock_slow_short as S


class Book(S.Book):
    ID = "market_clock_slow_short_fast3"
    NAME = "按池宽1倍加三窗口趋势空头"
    BEAR_TREND_WINDOWS = (20, 60, 120)
    CONFIG_PATH = Path(__file__).with_name("market_clock_slow_short_fast3.json")
    DETAIL = (
        "多头同按池宽1倍；池子20/60/120日累计收益符号均值小于0时，"
        "品种自身60/120/250日收益符号均值小于0且滞后一日年化展期收益小于0的做空1份"
        "/两腿都按当年池宽分1倍名义敞口/下一交易日第一根开盘成交/权重不变则留仓/回撤10%清仓并停10个交易日"
    )

    @property
    def FILES(self) -> list[Path]:
        return [Path(S.__file__), Path(__file__), self.CONFIG_PATH]


BOOK = Book()
BOOK.load_config()
