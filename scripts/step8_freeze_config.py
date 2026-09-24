"""第 8 步：把研究期的决定写成 frozen_config.yaml。只归档，不跑 2022 及以后。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfcta import config as C                    # noqa: E402
from tfcta.research import freeze, paths         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--note', default='')
    args = ap.parse_args()

    sel_path = paths.selection_path()
    if not sel_path.exists():
        print(f"找不到 {sel_path}\n请先运行 step5_param_scan.py。数据未就绪时不要空写冻结文件。")
        return 2

    selection = paths.load_json(sel_path)
    try:
        text = freeze.render(selection, note=args.note)
    except freeze.FreezeError as e:
        print(e)
        return 1

    C.ensure_dirs()
    dest = paths.frozen_path()
    dest.write_text(text, encoding='utf-8')
    run = paths.run_dir('step8')
    (run / 'frozen_config.yaml').write_text(text, encoding='utf-8')
    print(f"已冻结 {len(selection)} 个因子的参数")
    print(f"写入 {dest}")
    print(f"留痕 {run}")
    print("2022-01-01 及以后本阶段不跑。若要做裁决，只能用这份文件里的参数跑一次。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
