"""合成日内经典策略书。

示例：
    python scripts/combine_intraday_books.py \
        --input runs/intraday_classics_2010_2021 \
        --books SR:atr RU:rbreaker
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta.research.intraday.engine import combine_daily_returns  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--books", nargs="+", required=True,
                        help="书写法 SYMBOL:STRATEGY，例如 SR:atr RU:rbreaker")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.input)
    books = {}
    for spec in args.books:
        symbol, separator, strategy = spec.partition(":")
        if not separator:
            parser.error(f"书格式错误: {spec}")
        path = root / f"{symbol.upper()}_{strategy}_daily.csv"
        if not path.exists():
            parser.error(f"找不到日收益文件: {path}")
        books[spec] = pd.read_csv(path, index_col=0, parse_dates=True)
    result = combine_daily_returns(books)
    output = Path(args.output) if args.output else root / "portfolio_daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(output)
    print(pd.Series(result.metrics).to_string())
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())