"""只在研究期构建时间因子缓存。"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tfcta import config as C
from tfcta.data import load_universe
from tfcta.factors import build_symbol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols',nargs='+')
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    universe = load_universe()
    symbols = args.symbols or sorted({s for y,ss in universe.items() if 2016<=y<=2021 for s in ss})
    for i,symbol in enumerate(symbols,1):
        result = build_symbol(symbol,[(C.IC_REFERENCE_LOOKBACK,C.IC_REFERENCE_PCT)],overwrite=args.overwrite)
        print(f'[{i}/{len(symbols)}] {symbol}: {result}',flush=True)


if __name__=='__main__':
    main()
