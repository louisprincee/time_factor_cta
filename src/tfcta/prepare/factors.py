"""只在研究期构建时间因子缓存。"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd
from tfcta import config as C
from tfcta.data import build_universe, load_universe
from tfcta.factors import build_symbol


def research_symbols() -> list[str]:
    """研究期回测会读到的品种：2016–2021 用落盘池，2011–2015 用同一门槛重建。"""
    stored = load_universe()
    symbols = {s for year, members in stored.items() if 2016 <= year <= 2021 for s in members}
    stats = pd.read_csv(C.UNIVERSE_DIR / "yearly_stats.csv")
    early, _ = build_universe(stats, years=list(range(C.EARLY_START.year, C.STUDY_START.year)))
    if early[C.STUDY_START.year - 1] != sorted(stored[C.STUDY_START.year - 1]):
        raise ValueError("重建的 2015 品种池与落盘的不一致")
    symbols.update(
        s for year in range(C.EARLY_START.year, C.STUDY_START.year) for s in early[year]
    )
    return sorted(symbols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols',nargs='+')
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    symbols = args.symbols or research_symbols()
    for i,symbol in enumerate(symbols,1):
        result = build_symbol(symbol,[(C.IC_REFERENCE_LOOKBACK,C.IC_REFERENCE_PCT)],overwrite=args.overwrite)
        print(f'[{i}/{len(symbols)}] {symbol}: {result}',flush=True)


if __name__=='__main__':
    main()
