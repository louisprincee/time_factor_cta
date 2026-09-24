"""walk-forward、中心点选参、冻结配置。

训练窗口里没有模型要拟合。绩效只在测试年上计算，再用中心点距离选一组参数，
写进冻结文件。这一层不跑 2022 及以后。
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .. import config as C


def walk_forward_folds(test_years: list[int] | None = None) -> list[dict]:
    years = C.WF_TEST_YEARS_LIST if test_years is None else list(test_years)
    rows = []
    for y in years:
        y = int(y)
        rows.append({
            'test_year': y,
            'train_start': y - C.WF_TRAIN_YEARS,
            'train_end': y - 1,
            'sparse_night': y in C.WF_FOLDS_WITH_SPARSE_NIGHT,
        })
    return rows


def backward_years() -> list[int]:
    """2010-2014 向后时间外。只看符号，不得选参。"""
    return list(range(C.BACKWARD_OOS_START.year, C.BACKWARD_OOS_END.year + 1))


def stitch_test_years(port: pd.Series, years: list[int]) -> pd.Series:
    """把各测试年的组合收益按时间拼成一条。空年份跳过，不插 0。"""
    parts = []
    for y in years:
        sl = port.loc[port.index.year == int(y)].dropna()
        if len(sl):
            parts.append(sl)
    if not parts:
        return pd.Series(dtype='float64', name='port_ret')
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep='first')].sort_index()
    out.name = 'port_ret'
    return out


class NoCandidate(RuntimeError):
    """扫描结果里没有一行能同时算出年化收益和收益风险比。"""


def _zscore(x: pd.Series) -> pd.Series:
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        return pd.Series(0.0, index=x.index)
    return (x - mu) / sd


def center_distance(perf: pd.DataFrame,
                    ret_col: str = 'ann_return',
                    sr_col: str = 'ret_risk') -> pd.DataFrame:
    """给绩效表加上 z_return、z_ret_risk、distance。无效行的距离为 NaN。"""
    out = perf.copy()
    ok = out[ret_col].notna() & out[sr_col].notna()
    out['z_return'] = np.nan
    out['z_ret_risk'] = np.nan
    out['distance'] = np.nan
    if int(ok.sum()) == 0:
        return out
    zr = _zscore(out.loc[ok, ret_col])
    zs = _zscore(out.loc[ok, sr_col])
    out.loc[ok, 'z_return'] = zr.to_numpy()
    out.loc[ok, 'z_ret_risk'] = zs.to_numpy()
    out.loc[ok, 'distance'] = np.sqrt(zr.to_numpy() ** 2 + zs.to_numpy() ** 2)
    return out


def select_center(perf: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """返回 (中心点那一行, 收益风险比 argmax 那一行, 带距离的全表)。"""
    scored = center_distance(perf)
    ok = scored.dropna(subset=['ann_return', 'ret_risk', 'distance'])
    if ok.empty:
        raise NoCandidate('没有同时具有年化收益和收益风险比的参数，无法选参')
    center = ok.loc[ok['distance'].idxmin()]
    argmax = ok.loc[ok['ret_risk'].idxmax()]
    return center, argmax, scored


_HOLDOUT_YEARS = set(range(2022, 2030))


class FreezeError(RuntimeError):
    """选择结果里混进了样本外年份，拒绝冻结。"""


def _bad_year(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, np.integer)) and int(v) in _HOLDOUT_YEARS:
        return True
    if isinstance(v, str) and len(v) >= 4 and v[:4].isdigit() and int(v[:4]) in _HOLDOUT_YEARS:
        return True
    return False


def assert_research_selection(selection: dict) -> None:
    """选参结果里出现 2022 及以后的年份或日期，就拒绝写冻结文件。"""
    def walk(o, path: str) -> None:
        if _bad_year(o):
            raise FreezeError(f"选参结果 {path} 含样本外年份 {o}，本阶段不能冻结")
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
    walk(selection, 'selection')


def _scalar(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'null'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    s = str(v)
    if s == '' or any(c in s for c in ':#{}[]&*!|>\'"%@`,'):
        return json.dumps(s, ensure_ascii=False)
    return s


def to_yaml(obj, indent: int = 0) -> str:
    """够用的 YAML 子集：字典、列表、标量。不引入 PyYAML 依赖。"""
    sp = '  ' * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(to_yaml(v, indent + 1).rstrip('\n'))
            else:
                lines.append(f"{sp}{k}: {_scalar(v)}")
        return '\n'.join(lines) + '\n'
    if isinstance(obj, list):
        lines = []
        for v in obj:
            if isinstance(v, dict):
                lines.append(f"{sp}-")
                lines.append(to_yaml(v, indent + 1).rstrip('\n'))
            elif isinstance(v, list):
                lines.append(f"{sp}-")
                lines.append(to_yaml(v, indent + 1).rstrip('\n'))
            else:
                lines.append(f"{sp}- {_scalar(v)}")
        return '\n'.join(lines) + '\n'
    return f"{sp}{_scalar(obj)}\n"


def build_payload(selection: dict,
                  generated: str | None = None,
                  note: str = '') -> dict:
    assert_research_selection(selection)
    factors = []
    for name in sorted(selection):
        block = selection[name]
        factors.append({
            'name': name,
            'sign': int(C.ALL_SIGNS.get(name, 0)),
            'has_prior': name in C.PRIOR_FACTORS,
            'center': block.get('center', {}),
            'argmax_ret_risk': block.get('argmax_ret_risk', {}),
        })
    return {
        'generated': generated or datetime.now().isoformat(timespec='seconds'),
        'stage': 'research_frozen',
        'note': note or '开发层冻结。2022-01-01 及以后本阶段不跑、不看、不统计。',
        'research_end': str(C.RESEARCH_END),
        'holdout_start': str(C.HOLDOUT_START),
        'study_start': str(C.STUDY_START),
        'universe_file': 'universe/universe_by_year.json',
        'sign_rule': '方向来自论文先验与上一轮小时频方向性结果，在因子上乘符号，使值越大越看多。不允许按本研究的 IC 翻转。',
        'center_rule': '先对年化收益和收益风险比各自 z-score（ddof=0），再取欧氏距离最小的参数。原文未做 z-score，这一步是必要修正。',
        'signal_rule': 'factor_t 与 [t-W, t-1] 的分位数比较，不含当日。次日开盘成交。',
        'return_rule': 'day_ret[t]=(openw[t+1]-openw[t])/open[t]',
        'standardize': {
            'method': 'rolling_mad',
            'window': C.STD_WINDOW,
            'divisor': f'{C.STD_MAD_MULT:g} * MAD',
            'clip': C.STD_CLIP,
        },
        'fee_base': C.FEE_BASE,
        'fee_grid': list(C.FEE_GRID),
        'slippage_rule': ('每次换手按 n_ticks × tick / 当年价位中位数 收取单边比例滑点。'
                          'tick 与价位都逐 (品种, 年) 由 research/costs.py 从分钟 close 估出，'
                          '不查交易所表：取非零绝对一阶差分中出现得足够频繁的最小档（众数会偏大一倍）。'
                          '写成 tick 数而非固定比例，是为了保留品种间实测 13.7 倍的成本差异。'),
        'slippage_ticks': C.SLIPPAGE_TICKS,
        'slippage_tick_grid': list(C.SLIPPAGE_TICK_GRID),
        'slippage_table': 'research/tick_size.csv',
        'oos_criteria': {
            'sr_decay_max': C.OOS_SR_DECAY_MAX,
            'mdd_ratio_max': C.OOS_MDD_RATIO_MAX,
            'sign_must_hold': True,
            'run_once': True,
        },
        'combos': {k: list(v) for k, v in C.COMBO_GROUPS.items()},
        'factors': factors,
    }


def render(selection: dict, generated: str | None = None, note: str = '') -> str:
    payload = build_payload(selection, generated=generated, note=note)
    header = (
        "# 时间维度因子研究期冻结配置\n"
        "# 本文件生成之后，2022-01-01 及以后的数据只允许按这里的参数跑一次。\n"
        "# 若根据那次结果改了本文件里的任何参数，该区间即降级为验证集。\n"
    )
    return header + to_yaml(payload)


def clean_record(row: pd.Series, keys: list[str]) -> dict:
    """把绩效表的一行收成可写入 JSON 的普通字典。"""
    out = {}
    for k in keys:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        if k in ('lookback', 'window', 'n_days'):
            out[k] = int(v)
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


PARAM_KEYS = [
    'lookback', 'pct', 'window', 'q_low', 'q_high',
    'ann_return', 'ann_vol', 'ret_risk', 'calmar', 'win_rate', 'max_drawdown',
    'n_days', 'distance',
]
