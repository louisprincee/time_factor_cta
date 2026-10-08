"""把复盘里的时序动量、波动抬升和均价突破接到早盘规则上。

过滤只用前一交易日收盘已经知道的值。截面多空不进这条早盘规则：
文章自己把下半年的亏损归到强弱排名突然反转，而这条规则是每个品种用自己的过去。
研究期 2016–2021。不读 2022，也不按 2022 已经看到的亏损来挑选过滤。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tfcta.data import bars as B
from tfcta.data import universe as U
from tfcta.factors import daily
from tfcta.research import stats
import plot_morning_rule as morning
import select_main_book as book

SLOT = "早盘持有到11:30"


def lagged(frame, master, symbols):
    return frame.shift(1).reindex(index=master, columns=symbols)


def agree(side, trend):
    trend_side = np.sign(trend)
    keep = np.isfinite(trend) & (trend_side != 0) & (trend_side == np.sign(side))
    return np.where(keep, side, 0.0)


def stand_down_when_vol_rises(side, rise):
    keep = np.isfinite(rise) & (rise <= 0)
    return np.where(keep, side, 0.0)


def report(name, daily_pnl, index):
    series = pd.Series(daily_pnl, index=index)
    perf = stats.performance(series)
    sharpes = [float(stats.sharpe_ratio(series[series.index.year == year])) for year in range(2016, 2022)]
    returns = [float(stats.performance(series[series.index.year == year])["ann_return"]) for year in range(2016, 2022)]
    print(
        f"{name}  全期夏普 {stats.sharpe_ratio(series):.2f}  年化 {perf['ann_return']*100:.2f}%"
        f"  回撤 {perf['max_drawdown']*100:.2f}%  最差年夏普 {min(sharpes):.2f}",
        flush=True,
    )
    print("  年化 " + "  ".join(f"{2016+i}:{returns[i]*100:.2f}%" for i in range(6)), flush=True)
    print("  夏普 " + "  ".join(f"{2016+i}:{sharpes[i]:.2f}" for i in range(6)), flush=True)


def main():
    universe = {year: names for year, names in U.load_universe().items() if 2016 <= year <= 2021}
    symbols = sorted({name for group in universe.values() for name in group if name in morning.costs.MULTIPLIER})
    needed = {SLOT: book.SLOTS[SLOT]}
    master, signal, gross, cost = book.build("research", needed)
    side = signal[SLOT]
    bars = B.load_daily_bars(symbols)
    close, closew = bars["close"], bars["closew"]
    mom = lagged(daily.tsmom_sign(close, closew, daily.REVIEW_MOM_WINDOW), master, symbols).to_numpy()
    rise = lagged(daily.vol_change_sign(close, closew, daily.REVIEW_VOL_WINDOW), master, symbols).to_numpy()
    breakout = lagged(daily.ma_breakout(closew, daily.REVIEW_VOL_WINDOW), master, symbols).to_numpy()
    books = {
        "原规则，单品种20%": side,
        "和3日时序动量同向才做": agree(side, mom),
        "20日波动在抬升则不做": stand_down_when_vol_rises(side, rise),
        "动量同向，且波动没有抬升": stand_down_when_vol_rises(agree(side, mom), rise),
        "和20日均线突破同向才做": agree(side, breakout),
    }
    for name, filtered in books.items():
        weight = morning.allocate(filtered, 0.20)
        report(name, morning.daily_pnl(weight, gross[SLOT], cost[SLOT]), master)


if __name__ == "__main__":
    main()
