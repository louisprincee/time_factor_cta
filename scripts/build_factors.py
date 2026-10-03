"""Build research-only time factors and carry control. Never opens 2022/OOS."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tfcta import config as C
from tfcta.data import universe as U
from tfcta.factors import cache, external


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols',nargs='+')
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    universe = U.load_universe()
    symbols = args.symbols or sorted({s for y,ss in universe.items() if 2016<=y<=2021 for s in ss})
    for i,symbol in enumerate(symbols,1):
        result = cache.build_symbol(symbol,[(C.IC_REFERENCE_LOOKBACK,C.IC_REFERENCE_PCT)],overwrite=args.overwrite)
        print(f'[{i}/{len(symbols)}] {symbol}: {result}',flush=True)
    result = external.build_partition('research',symbols)
    print(f'Research carry caches: {len(result)}',flush=True)


if __name__=='__main__':
    main()
