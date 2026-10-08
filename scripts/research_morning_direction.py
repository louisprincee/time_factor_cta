"""开盘偏离的方向判别：流动性冲击逆向、新信息顺向。只在研究期诊断和评估预先声明的规则。

    python scripts/research_morning_direction.py

规则、特征和选择标准写在 config/morning_direction.json；成本、仓位沿用 config/morning_oor.json。
输出在 runs/morning_oor/direction/。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tfcta import config as C
import research_morning_oor as oor

CONFIG = ROOT / "config" / "morning_direction.json"
OUT = oor.OUT / "direction"
YEARS = range(2016, 2022)


def share_same_sign(sign, groups):
    """同组其他品种中 dev15 同号的比例；其他有效品种少于 2 个记 0.5。"""
    frame = pd.DataFrame({"pos": sign > 0, "neg": sign < 0})
    pos = frame.pos.groupby(groups).transform("sum")
    neg = frame.neg.groupby(groups).transform("sum")
    same = np.where(sign > 0, pos - 1, np.where(sign < 0, neg - 1, np.nan))
    total = pos + neg - (sign != 0)
    return pd.Series(np.where(total >= 2, same / total.where(total > 0), 0.5), index=sign.index)


def add_features(table, eff_window):
    """全部只用 09:16 及以前的信息；同日其他品种的 dev15 在决策时点已知。表须按品种、日期排好。"""
    t = table
    sign = np.sign(t.dev15).fillna(0.0)
    by = t.groupby("symbol")
    t["cont"] = sign * t.gross_1130
    t["log_relvol"] = np.log(t.relvol.where(t.relvol > 0))
    t["peer_share"] = share_same_sign(sign, [t.trading_date, t.sector])
    t["market_share"] = share_same_sign(sign, t.trading_date)
    t["seq_agree"] = sum((np.sign(t[c]) == sign).astype(float) for c in ("ret_night", "gap", "ret_pre"))
    t["order_pre"] = sign * (t.hclock_pre - t.lclock_pre)
    t["clock_own"] = np.where(sign > 0, t.hclock_td, t.lclock_td)
    t["night_aligned"] = sign * t.ret_night.fillna(0.0)
    t["gap_aligned"] = sign * t.gap
    t["vwap_aligned"] = sign * t.vwap_dev
    t["dev_z"] = t.dev15.abs() / by.dev15.transform(lambda s: s.shift(1).rolling(60, min_periods=45).std())
    t["eff_median"] = by.eff_pre.transform(
        lambda s: s.shift(1).rolling(eff_window, min_periods=eff_window * 3 // 4).median())
    return t


def tradable(table, rule):
    """样本：偏离够大、成本够低、有 11:30 行情。是否池内、有无波动估计另判。"""
    return ((table.dev15.abs() > rule["dev_threshold"]) & (table.est_cost < rule["max_est_cost"])
            & table.gross_1130.notna() & table.cost_1130.notna())


def ridge_predictions(table, features, rule, spec):
    """逐年外推：y 年只用 y 年以前的样本拟合；返回预测顺向收益和每年的系数。"""
    x = table[features]
    year = table.trading_date.dt.year
    finite = np.isfinite(x).all(axis=1)
    usable = tradable(table, rule) & finite
    pred, coefs = pd.Series(np.nan, index=table.index), {}
    for y in YEARS:
        train = usable & (year >= spec["first_train_year"]) & (year < y)
        lo, hi = table.cont[train].quantile([spec["clip_quantile"], 1 - spec["clip_quantile"]])
        mu, sd = x[train].mean(), x[train].std().replace(0, 1.0)
        model = Ridge(alpha=spec["alpha"]).fit((x[train] - mu) / sd, table.cont[train].clip(lo, hi))
        test = (year == y) & finite
        if test.any():
            pred[test] = model.predict((x[test] - mu) / sd)
        coefs[y] = pd.Series(model.coef_, index=features)
    return pred, pd.DataFrame(coefs).T


def side_of(table, spec, base, name, pred=None):
    """+1 多、−1 空。follow = 顺着偏离方向，fade = 逆着。"""
    rule, kind = base["rule"], spec["rules"][name]["kind"]
    params = spec["rules"][name]
    main = oor.sides(table, rule, base["candidates"][base["main"]])
    if kind == "reference":
        return main
    sign = np.sign(table.dev15).fillna(0.0)
    ok = tradable(table, rule) & table.member & table.sigma.notna()
    loud = table.relvol >= params.get("follow_relvol", np.inf)
    common = table.peer_share >= params.get("follow_share", np.inf)
    votes = loud.astype(int) + common.astype(int) + (table.eff_pre > table.eff_median).astype(int)
    if kind == "relvol":
        ok &= table.relvol.notna()
        follow, fade = loud, ~loud
    elif kind == "peer":
        follow, fade = common, table.peer_share <= params["fade_share"]
    elif kind == "votes":
        ok &= table.relvol.notna() & table.eff_median.notna()
        follow, fade = votes >= 2, votes == 0
    elif kind == "main_plus_votes":
        ok &= table.relvol.notna() & table.eff_median.notna() & (main == 0)
        return main + sign.where(ok & (votes >= 2), 0.0)
    elif kind == "sector_residual":
        return oor.sides(table, rule, {**base["candidates"][base["main"]], "sector_residual": True})
    elif kind == "sector_follow":
        return oor.sides(table, rule, {**base["candidates"][base["main"]], "sector_follow": True})
    elif kind == "sector_both":
        fade = oor.sides(table, rule, {**base["candidates"][base["main"]], "sector_residual": True})
        follow = oor.sides(table, rule, {**base["candidates"][base["main"]], "sector_follow": True})
        if ((fade != 0) & (follow != 0)).any():
            raise ValueError("残差逆向和板块跟随在同一行同时触发")
        return fade + follow
    elif kind == "ridge":
        follow, fade = pred > table.est_cost, pred < -table.est_cost
    else:
        raise ValueError(f"未知规则类型 {kind}")
    return (sign.where(ok & follow, 0.0) - sign.where(ok & fade & ~follow, 0.0)).fillna(0.0)


def bins_of(values):
    if values.nunique() <= 6:
        return values
    return pd.qcut(values.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])


def diagnostics(table, sample, features):
    """各特征分档的延续比例和顺向收益，及逐年 Q高−Q低 的差。只描述，不改规则。"""
    rows, spreads = [], []
    s = table[sample]
    year = s.trading_date.dt.year
    for f in features:
        valid = s[f].notna()
        b = bins_of(s.loc[valid, f])
        g = s.loc[valid].groupby(b, observed=True)
        part = pd.DataFrame({"笔数": g.size(), "均值": g[f].mean(), "延续比例": g.cont.apply(lambda c: (c > 0).mean()),
                             "顺向收益bp": g.cont.mean() * 1e4})
        rows.append(part.reset_index(names="档").assign(特征=f))
        order = sorted(b.unique())
        top, bottom = order[-1], order[0]
        cont = s.loc[valid, "cont"]
        diff = (cont[b == top].groupby(year[valid][b == top]).mean()
                - cont[b == bottom].groupby(year[valid][b == bottom]).mean()) * 1e4
        total = (cont[b == top].mean() - cont[b == bottom].mean()) * 1e4
        spreads.append({"特征": f, "全期高减低bp": total, **{str(y): diff.get(y, np.nan) for y in YEARS},
                        "同号年数": int((np.sign(diff) == np.sign(total)).sum())})
    return pd.concat(rows, ignore_index=True)[["特征", "档", "笔数", "均值", "延续比例", "顺向收益bp"]], \
        pd.DataFrame(spreads)


def legs(table, side, weight):
    """把成交拆成顺向腿和逆向腿：笔数、扣费后单笔均值和逐年均值。"""
    active = weight != 0
    net = (np.sign(weight) * table.gross_1130 - table.cost_1130)[active]
    kind = np.where(side[active] == np.sign(table.dev15[active]), "顺向", "逆向")
    year = table.trading_date[active].dt.year
    out = {}
    for leg in ("顺向", "逆向"):
        n = net[kind == leg]
        by_year = n.groupby(year[kind == leg]).mean() * 1e4
        out[leg] = {"笔数": len(n), "单笔净bp": n.mean() * 1e4 if len(n) else np.nan,
                    "胜率": (n > 0).mean() if len(n) else np.nan, "为正年数": int((by_year > 0).sum()),
                    **{str(y): by_year.get(y, np.nan) for y in YEARS}}
    return out


def select(rows, leg_rows, main):
    """按配置里的标准逐条判定；返回通过与否和入选者，不回写配置。"""
    passed = {}
    for name, row in rows.items():
        follow, fade = leg_rows[name]["顺向"], leg_rows[name]["逆向"]
        passed[name] = bool(name != main and follow["笔数"] >= 100 and follow["单笔净bp"] > 0
                            and follow["为正年数"] >= 4 and fade["单笔净bp"] > 0
                            and row["夏普95%下限"] > rows[main]["夏普95%下限"] and row["滑点2跳夏普"] > 0)
    winners = [n for n, ok in passed.items() if ok]
    chosen = max(winners, key=lambda n: rows[n]["夏普95%下限"]) if winners else None
    return passed, chosen


def main():
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    rule = base["rule"]
    eff_window = spec["rules"]["信息票数"]["eff_window"]
    ridge_spec = spec["rules"]["岭回归"]
    full = oor.prepare("research", base, years=(ridge_spec["first_train_year"], 2021))
    full = add_features(full, eff_window)
    full = oor.add_sector_deviation(full, spec["rules"]["板块残差逆向"]["min_peers"])
    features = list(spec["features"])
    pred, coefs = ridge_predictions(full, features, rule, ridge_spec)
    full["ridge_pred"] = pred
    table = full[full.trading_date.dt.year >= YEARS[0]].reset_index(drop=True)
    C.assert_no_holdout_dates(table.trading_date, "早盘方向研究")
    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    sample = tradable(table, rule) & table.member & table.sigma.notna()
    bins, spreads = diagnostics(table, sample, features)
    bins.to_csv(out / "diagnostics.csv", index=False, encoding="utf-8-sig")
    spreads.to_csv(out / "spread_by_year.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv(out / "ridge_coef.csv", encoding="utf-8-sig")

    calendar = pd.DatetimeIndex(sorted(table.trading_date.unique()))
    curves, rows, leg_rows, years = {}, {}, {}, {}
    for name in spec["rules"]:
        side = side_of(table, spec, base, name, table.ridge_pred)
        weight = oor.weights(table, side, base["sizing"])
        ret = oor.daily(table, weight, "主口径", calendar)
        stressed = oor.daily(table, weight, "滑点2跳", calendar)
        low, high = oor.bootstrap_sharpe(ret)
        rows[name] = {**oor.summary(ret, table, weight, "主口径"), "夏普95%下限": low, "夏普95%上限": high,
                      "滑点2跳夏普": oor.stats.sharpe_ratio(stressed)}
        leg_rows[name] = legs(table, side, weight)
        curves[name] = ret
        years[name] = oor.yearly(ret)
    passed, chosen = select(rows, leg_rows, "主策略")
    result = pd.DataFrame(rows).T.assign(通过=pd.Series(passed))
    result.to_csv(out / "rules.csv", encoding="utf-8-sig")
    leg_table = pd.concat({n: pd.DataFrame(v).T for n, v in leg_rows.items()}, names=["规则", "腿"])
    leg_table.to_csv(out / "legs.csv", encoding="utf-8-sig")
    yearly = pd.concat(years, names=["规则", "年份"])
    yearly.to_csv(out / "yearly.csv", encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(out / "daily.csv", encoding="utf-8-sig")
    oor.plot(curves, out / "nav.png", "开盘偏离方向判别 2016–2021（含现金日，扣费后）")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"样本 {int(sample.sum())} 笔，延续比例 {(table.cont[sample] > 0).mean():.3f}，"
          f"平均顺向收益 {table.cont[sample].mean() * 1e4:.2f}bp（毛）")
    print(bins.round(4).to_string(index=False))
    print(spreads.round(2).to_string(index=False))
    print(coefs.round(5).to_string())
    print(result.round(4).to_string())
    print(leg_table.round(2).to_string())
    print("入选：", chosen or "无（研究期的事前变量不能区分两类偏离）")
    print("结果：", out)


if __name__ == "__main__":
    main()
