"""Build partitioned daily factors from cached RQData external inputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta.factors.external_factor_cache import PARTITIONS, build_partition  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="构造外部商品因子并写入因子库")
    parser.add_argument("--symbols", nargs="+", default=["all"],
                        help="商品代码，默认 all1m 商品池")
    parser.add_argument("--partitions", nargs="+", choices=PARTITIONS,
                        default=list(PARTITIONS), help="默认构建三个隔离分区")
    args = parser.parse_args()
    symbols = None if args.symbols == ["all"] else [value.upper() for value in args.symbols]

    failed = False
    for partition in args.partitions:
        try:
            results = build_partition(partition, symbols)
        except Exception as exc:
            print(f"{partition} 构建失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed = True
            continue
        written = sum(item.get("rows", 0) > 0 for item in results.values())
        print(f"{partition}: {written}/{len(results)} 个品种已写入")
    print(f"因子目录: data/factor_daily/external")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())