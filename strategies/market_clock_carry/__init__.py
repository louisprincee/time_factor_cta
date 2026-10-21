"""收盘定权重，下一交易日第一根开盘成交。

有夜盘时这一根是夜盘开盘，没有夜盘时是日盘开盘。夜盘算下一个交易日。
四条共用一套信号，只差分仓方式和名义敞口上限，规格在同目录的四份 JSON。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tfcta import config as C

FACTORS = ("ts_low", "ts_high", "dfp_top3", "ts_low_pm", "ts_high_pm")
Z_WINDOW = 250
Z_MIN_PERIODS = 125
Z_CLIP = 3.0
MIN_FACTORS = 3
MIN_SYMBOLS = 10
ID = "market_clock_carry"
DRAWDOWN_LIMIT = 0.10
HALT_DAYS = 10
CARRY_MAX_MISSING = 0.05      # 池内某品种某年展期收益缺失超过 5% 就报错，不悄悄当成不入选
ROLL_YIELD_ROOT = C.COMMODITY_DATA / "external_rqdata"


def _load_config(path, strategy_id, gross, gross_split) -> dict:
    """规格必须事先写明。探索期不能带冻结标记；冻结后 chosen 必须等于 parameters。"""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    parameters = config.get("parameters") or {}
    sample = config.get("sample", "")
    status = config.get("status")
    if status == "exploratory_predeclared":
        state_ok = (
            not config.get("frozen_at") and not config.get("chosen")
            and "不读取2022及2023年以后" in sample
        )
    elif status == "frozen":
        state_ok = bool(config.get("frozen_at")) and config.get("chosen") == parameters
    else:
        state_ok = False
    if (
        not state_ok
        or config.get("id") != strategy_id
        or "不读 2026" not in sample
        or tuple(parameters.get("factors", ())) != FACTORS
        or parameters.get("roll_yield") != "annualized_yield"
        or parameters.get("roll_yield_lag_days") != 1
        or parameters.get("gross_exposure_limit") != gross
        or parameters.get("gross_split") != gross_split
        or parameters.get("rebalance_on_count_change") is not (gross_split == "selected")
        or parameters.get("drawdown_breaker") != DRAWDOWN_LIMIT
        or parameters.get("halt_days") != HALT_DAYS
    ):
        raise ValueError(f"{strategy_id} 的登记与实现不一致")
    return config


class Book:
    """同一套信号。只差分仓方式和名义敞口上限。"""

    def __init__(self, strategy_id, name, gross, gross_split, filename):
        self.ID = strategy_id
        self.NAME = name
        self.GROSS = gross
        self.GROSS_SPLIT = gross_split
        self.CONFIG_PATH = Path(__file__).with_name(filename)
        sizing = "入选品种平分，入选数一变全部重新平分" if gross_split == "selected" else "按当年池宽分"
        self.DETAIL = (
            "五因子收盘分数大于0且滞后一日年化展期收益大于0的品种"
            f"{sizing}{gross:g}倍名义敞口/下一交易日第一根开盘成交/权重不变则留仓/回撤10%清仓并停10个交易日"
        )

    def load_config(self) -> dict:
        return _load_config(self.CONFIG_PATH, self.ID, self.GROSS, self.GROSS_SPLIT)

    def frozen(self) -> dict:
        """冻结了才能打开 2022 及以后。没冻结直接拒绝，不写台账。"""
        config = self.load_config()
        if config.get("status") != "frozen":
            raise C.HoldoutViolation(
                f"{self.NAME}（{self.ID}）还没冻结，不能打开 2022 及以后。"
                "先在它的 JSON 里写 status=frozen、frozen_at，并让 chosen 等于 parameters。"
            )
        return config


BOOKS = (
    Book("market_clock_pool", "按池宽1倍", 1.0, "pool", "market_clock_pool.json"),
    Book(ID, "入选平分1倍", 1.0, "selected", "market_clock_carry.json"),
    Book("market_clock_pool_15", "按池宽1.5倍", 1.5, "pool", "market_clock_pool_15.json"),
    Book("market_clock_carry_15", "入选平分1.5倍", 1.5, "selected", "market_clock_carry_15.json"),
)


def book_named(name: str) -> Book:
    """用策略 ID 或中文名取一条。"""
    key = str(name).strip()
    matches = [book for book in BOOKS if key in (book.ID, book.NAME)]
    if len(matches) != 1:
        known = "、".join(f"{book.NAME}（{book.ID}）" for book in BOOKS)
        raise ValueError(f"没有这个策略 {name}。可选：{known}")
    return matches[0]


def load_config() -> dict:
    return book_named(ID).load_config()


def pool_mask(universe: dict, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    pool = pd.DataFrame(False, index=index, columns=columns)
    for year, members in universe.items():
        pool.loc[index.year == int(year), [s for s in members if s in pool.columns]] = True
    return pool


def _own_history_z(frame: pd.DataFrame) -> pd.DataFrame:
    """每列只用自己过去 250 个有值观测（不含当天）的均值和标准差。"""
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for symbol in frame.columns:
        values = frame[symbol].dropna()
        prior = values.shift(1).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS)
        std = prior.std()
        z = (values - prior.mean()) / std.where(std > 0)
        out[symbol] = z.clip(-Z_CLIP, Z_CLIP).reindex(frame.index)
    return out


def market_score(signed: dict, universe: dict, index: pd.DatetimeIndex) -> pd.Series:
    """收盘时已知的全市场分数。输入已乘事前方向，越大越看多。"""
    missing = [name for name in FACTORS if name not in signed]
    if missing:
        raise KeyError(f"当日分数缺少因子 {missing}")
    columns = sorted({s for members in universe.values() for s in members})
    frames = {}
    for name in FACTORS:
        frame = signed[name].reindex(columns=columns)
        covered = frame.dropna(how="all").index
        if covered.empty or covered.max() < index.max() or covered.min() > index.min():
            raise ValueError(f"{name} 没有覆盖回测区间 {index.min().date()}..{index.max().date()}")
        frames[name] = _own_history_z(frame).reindex(index)
    stacked = np.stack([frames[name].to_numpy(dtype="float64") for name in FACTORS])
    count = np.isfinite(stacked).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 全是 NaN 的格子下面会被 count 屏蔽
        mean = np.nanmean(stacked, axis=0)
    per_symbol = pd.DataFrame(np.where(count >= MIN_FACTORS, mean, np.nan), index=index, columns=columns)
    per_symbol = per_symbol.where(pool_mask(universe, index, columns))
    enough = per_symbol.notna().sum(axis=1) >= MIN_SYMBOLS
    return per_symbol.mean(axis=1).where(enough)


def _roll_yield_dirs(end: pd.Timestamp) -> list[Path]:
    """研究期在根目录，2022、2023–2025 在各自的分区里。2023 起只在最终评估块内读。"""
    folders = [ROLL_YIELD_ROOT / "roll_yield"]
    if end >= pd.Timestamp(C.HOLDOUT_START):
        folders.append(ROLL_YIELD_ROOT / "validation_2022" / "roll_yield")
    if end >= pd.Timestamp(C.STRICT_OOS_START):
        C.assert_oos_research_locked(end.date())
        folders.append(ROLL_YIELD_ROOT / "holdout_locked" / "roll_yield")
    return folders


def roll_yield(index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """主力对次主力的年化展期收益，未滞后。只保留不晚于区间末日的行。"""
    end = pd.Timestamp(index.max())
    if end > pd.Timestamp(C.DEFAULT_OOS_END):
        raise C.HoldoutViolation("展期收益不读 2026")
    frames = {}
    for symbol in columns:
        parts = []
        for folder in _roll_yield_dirs(end):
            path = folder / f"{symbol}_main_sub.pkl"
            if not path.is_file():
                continue
            frame = pd.read_pickle(path)
            dates = pd.DatetimeIndex(frame.index.get_level_values(-1))
            keep = dates <= end
            parts.append(pd.Series(frame["annualized_yield"].to_numpy()[keep], index=dates[keep]))
        if parts:
            merged = pd.concat(parts)
            frames[symbol] = merged[~merged.index.duplicated(keep="last")].sort_index()
    return pd.DataFrame(frames, columns=list(columns)).reindex(index)


def require_carry_coverage(carry: pd.DataFrame, pool: pd.DataFrame) -> None:
    """池内每个品种每年的展期收益缺失不能超过 CARRY_MAX_MISSING，否则报出是哪些品种哪些年。"""
    pool = pool.reindex(index=carry.index, columns=carry.columns, fill_value=False)
    missing = carry.isna() & pool
    share = missing.groupby(carry.index.year).sum() / pool.groupby(carry.index.year).sum().replace(0, np.nan)
    bad = share.stack()
    bad = bad[bad > CARRY_MAX_MISSING]
    if len(bad):
        listed = "、".join(f"{year} {symbol} 缺 {value:.0%}" for (year, symbol), value in bad.items())
        raise ValueError(f"展期收益覆盖不足，先补下载再跑：{listed}")


def decision_weights(score: pd.Series, carry: pd.DataFrame) -> pd.DataFrame:
    """收盘决策。分数和已经滞后一天的展期收益都严格大于 0 才做多。"""
    allowed = carry.reindex(index=score.index).gt(0)
    side = score.gt(0).astype("float64")
    return pd.DataFrame(1.0, index=score.index, columns=carry.columns).mul(side, axis=0).where(allowed, 0.0)


def targets(signed: dict, universe: dict, index: pd.DatetimeIndex, columns,
            carry: pd.DataFrame | None = None) -> pd.DataFrame:
    """收盘决策后移到下一交易日第一根开盘。给了 carry 就用它（测试用），否则读展期收益并核对覆盖。"""
    columns = pd.Index(columns)
    pool = pool_mask(universe, index, columns)
    if carry is None:
        carry = roll_yield(index, columns)
        require_carry_coverage(carry, pool)
    score = market_score(signed, universe, index).reindex(index)
    lagged = carry.reindex(index=index, columns=columns).shift(1)
    return decision_weights(score, lagged).shift(1).fillna(0.0).where(pool, 0.0)


for _book in BOOKS:
    _book.load_config()
