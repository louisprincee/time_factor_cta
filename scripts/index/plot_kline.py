"""Generate adjusted continuous daily candlestick charts for IF, IH, and IC."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tfcta import config as C  # noqa: E402
from tfcta.data import load_shard  # noqa: E402

SYMBOLS = ("IF", "IH", "IC")
CHART_START = pd.Timestamp("2016-01-04")
CHART_END = pd.Timestamp("2021-12-31")
OUT = C.RUNS_DIR / "index" / "gtja_time" / "plots"
UP_COLOR = "#C0392B"
DOWN_COLOR = "#1F7A4D"


def daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars into daily adjusted OHLCV candles."""
    required = ("trading_date", "openw", "highw", "loww", "closew", "volume")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"股指分钟分片缺少 K 线字段: {missing}")
    if frame.empty:
        raise ValueError("股指分钟分片为空")

    data = frame.sort_index(kind="mergesort").copy()
    data["_trading_date"] = pd.to_datetime(data["trading_date"]).dt.normalize()
    bars = data.groupby("_trading_date", sort=True).agg(
        open=("openw", "first"),
        high=("highw", "max"),
        low=("loww", "min"),
        close=("closew", "last"),
        volume=("volume", lambda values: values.sum(min_count=1)),
    )
    bars.index.name = "trading_date"
    return bars


def plot_symbol(bars: pd.DataFrame, symbol: str, output: Path) -> None:
    x = np.arange(len(bars))
    opened = bars.open.to_numpy(dtype="float64")
    high = bars.high.to_numpy(dtype="float64")
    low = bars.low.to_numpy(dtype="float64")
    close = bars.close.to_numpy(dtype="float64")
    volume = bars.volume.to_numpy(dtype="float64")
    colors = np.where(close >= opened, UP_COLOR, DOWN_COLOR)
    candles = [
        Rectangle((i - 0.32, min(opened[i], close[i])),
                  0.64, max(abs(close[i] - opened[i]), 1e-8))
        for i in range(len(bars))
    ]

    fig, (price_axis, volume_axis) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    price_axis.add_collection(LineCollection(
        [((i, low[i]), (i, high[i])) for i in x],
        colors=colors, linewidths=0.6))
    price_axis.add_collection(PatchCollection(
        candles, facecolors=colors, edgecolors=colors, linewidths=0.5))
    price_axis.set_xlim(-1, len(bars))
    price_axis.set_ylim(np.nanmin(low), np.nanmax(high))
    price_axis.set_ylabel("Adjusted price")
    price_axis.grid(axis="y", alpha=0.25)

    volume_axis.bar(x, volume / 1e4, color=colors, width=0.7)
    volume_axis.set_ylabel("Volume (10k)")
    volume_axis.grid(axis="y", alpha=0.25)

    years = sorted(bars.index.year.unique())
    ticks = [int(np.flatnonzero(bars.index.year == year)[0]) for year in years]
    volume_axis.set_xticks(ticks, [str(year) for year in years])
    volume_axis.set_xlabel("Trading year")
    fig.suptitle(
        f"{symbol} adjusted continuous daily candles "
        f"({bars.index.min():%Y-%m-%d} – {bars.index.max():%Y-%m-%d})",
        x=0.08, ha="left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        frame = load_shard(
            symbol, directory=C.INDEX_RESEARCH_DIR,
            columns=["trading_date", "openw", "highw", "loww", "closew", "volume"])
        C.assert_no_holdout_dates(frame.trading_date, what=f"{symbol} 股指 K 线")
        bars = daily_bars(frame)
        bars = bars.loc[CHART_START:CHART_END]
        if bars.empty:
            raise ValueError(f"{symbol} 在 {CHART_START.date()}..{CHART_END.date()} 没有日线")
        output = OUT / f"{symbol}_daily_kline.png"
        plot_symbol(bars, symbol, output)
        print(f"{symbol}: {len(bars)} 日线 -> {output}", flush=True)


if __name__ == "__main__":
    main()
