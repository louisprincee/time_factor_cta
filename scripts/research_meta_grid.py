"""早盘元策略的参数面：窗口长度 × 窗口口径 × 切换方式 × 基础单，在 2016–2022 上看是否平滑，并算 DSR 和 PBO。

    python scripts/research_meta_grid.py

网格写在 config/meta_grid.json。输出在 runs/morning_oor/meta_grid/。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tfcta import config as C
from tfcta.research import context, stats
import research_morning_meta as meta
import research_morning_oor as oor

CONFIG = ROOT / "config" / "meta_grid.json"
OUT = oor.OUT / "meta_grid"
PROJECT_TRIALS = 300  # 整个项目在早盘上的粗略尝试次数


def trailing_days_mean(table, side, days):
    """最近 days 个交易日内已平仓单的单笔均值，只用前一交易日及以前；窗口内没有单时为空。"""
    active = side != 0
    net = (side * table.gross_1130 - table.cost_1130).where(active)
    calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
    total = net.groupby(table.trading_date).sum().reindex(calendar, fill_value=0.0)
    count = active.groupby(table.trading_date).sum().reindex(calendar, fill_value=0)
    rolling = total.rolling(days, min_periods=days).sum() / count.rolling(days, min_periods=days).sum().replace(0, np.nan)
    return table.trading_date.map(rolling.shift(1))


def grid(spec):
    for book in spec["books"]:
        yield book, "原样", None, None
        for kind, sizes in spec["windows"].items():
            for size in sizes:
                for mode in spec["modes"]:
                    yield book, mode, kind, size


def main():
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    first, last = spec["evaluation"]["period"]
    table = oor.prepare_through_2022(base, first)
    C.assert_validation_2022_dates(table.trading_date[table.trading_date.dt.year == 2022], "元策略网格 2022")
    books = oor.base_books(table, base)
    calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
    rows, curves = [], {}
    for book, mode, kind, size in grid(spec):
        side = books[book]
        if mode != "原样":
            score = meta.trailing_mean(table, side, size) if kind == "笔数" else trailing_days_mean(table, side, size)
            side = meta.switched(side, score, mode)
        weight = oor.weights(table, side, base["sizing"])
        ret = oor.daily(table, weight, "主口径", calendar)
        stressed = oor.daily(table, weight, "滑点2跳", calendar)
        name = book + "·" + mode + ("" if kind is None else f"·{size}{kind}")
        dev, later = ret[ret.index.year <= 2021], ret[ret.index.year == 2022]
        rows.append({"组合": name, "基础单": book, "方式": mode, "口径": kind or "-", "窗口": size or 0,
                     "2016-2022夏普": stats.sharpe_ratio(ret), "2016-2021夏普": stats.sharpe_ratio(dev),
                     "2022夏普": stats.sharpe_ratio(later), "2跳夏普": stats.sharpe_ratio(stressed),
                     "交易笔数": int((weight != 0).sum()),
                     **{f"{y}夏普": stats.sharpe_ratio(p) for y, p in ret.groupby(ret.index.year)}})
        curves[name] = ret
    result = pd.DataFrame(rows).set_index("组合")
    matrix = pd.DataFrame(curves)
    pbo = stats.pbo_cscv(matrix, spec["evaluation"]["pbo_blocks"])
    best = result["2016-2022夏普"].idxmax()
    trial_sharpes = result["2016-2022夏普"].to_numpy()
    dsr_grid = stats.deflated_sharpe(curves[best], trial_sharpes)
    dsr_project = stats.deflated_sharpe(curves[best], trial_sharpes, PROJECT_TRIALS)
    frozen = "主策略·亏损反手·30笔数"
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "grid.csv", encoding="utf-8-sig")
    matrix.to_csv(out / "daily.csv", encoding="utf-8-sig")
    summary = {"最好的组合": best, "PBO": pbo, "DSR（网格内）": dsr_grid, f"DSR（约 {PROJECT_TRIALS} 次尝试）": dsr_project,
               "冻结组合": {"组合": frozen, **result.loc[frozen, ["2016-2021夏普", "2022夏普", "2016-2022夏普"]].to_dict()}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    top = list(result["2016-2022夏普"].nlargest(5).index) + [frozen, "主策略·原样"]
    print("图表：", context.plot_performance(matrix[list(dict.fromkeys(top))], out / "plots", "元策略参数面 2016–2022（前五与冻结组合）"))
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    shown = ["交易笔数", "2016-2021夏普", "2022夏普", "2016-2022夏普", "2跳夏普"]
    print(result[shown].sort_values("2016-2022夏普", ascending=False).round(2).to_string())
    for book in spec["books"]:
        for mode in spec["modes"]:
            part = result[(result.基础单 == book) & (result.方式 == mode)]
            print(f"\n{book}·{mode}：窗口 → 2016-2021 / 2022 夏普")
            print(part.set_index(["口径", "窗口"])[["2016-2021夏普", "2022夏普"]].round(2).T.to_string())
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda v: round(float(v), 3)))
    print("结果：", out)


if __name__ == "__main__":
    main()
