"""路径、时间分区和数据约定。"""
from __future__ import annotations

import datetime as _dt
import os as _os
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
# src/tfcta/config.py -> src/tfcta -> src -> time_factor_cta
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 分钟频原始单体 pickle（扩展名是 .txt，内容是 pickle），在 data/data_min/ 下。
# 可用环境变量 TFCTA_MINUTE_DIR 指向别处；默认数据目录受 /data/ 忽略规则保护。
# 原始数据跟着项目走，不依赖同级的 backtest_cta_pack 是否存在。
MINUTE_RAW_DIR = Path(_os.environ.get("TFCTA_MINUTE_DIR", PROJECT_ROOT / "data" / "data_min"))
MINUTE_MONOLITH = MINUTE_RAW_DIR / "future_all1mdata_20100101-20251231.txt"

# 数据根目录。设置环境变量 TFCTA_DATA_ROOT 可整体重定向——用于在合成数据上演练
# 整条管道而不污染真实分片。生产运行不要设置它。
DATA_ROOT = Path(_os.environ.get("TFCTA_DATA_ROOT", PROJECT_ROOT / "data"))

# step1 产出
SHARD_ROOT = DATA_ROOT / "minute_shards"
RESEARCH_DIR = SHARD_ROOT / "research"                     # <= 2021-12-31，可自由使用
HOLDOUT_DIR = SHARD_ROOT / "holdout_locked"                # >= 2023-01-01，本阶段禁止读取
VALIDATION_DIR = SHARD_ROOT / "validation_2022"            # 仅 2022，一次性验证
ROLL_DIR = DATA_ROOT / "roll_dates"

# 因子缓存与运行留痕
FACTOR_DAILY_DIR = DATA_ROOT / "factor_daily_v3"
UNIVERSE_DIR = DATA_ROOT / "universe"
RUNS_DIR = Path(_os.environ.get("TFCTA_RUNS_ROOT", PROJECT_ROOT / "runs"))
CONFIG_DIR = Path(_os.environ.get("TFCTA_CONFIG_DIR", PROJECT_ROOT / "config"))
# 研究输入费用与 tick 表；实验结果只写独立 runs/ 目录。
RESEARCH_OUT_DIR = DATA_ROOT / "research"

# --------------------------------------------------------------------------
# 时间切分（设计文档第 5 节）
# --------------------------------------------------------------------------
HOLDOUT_START = _dt.date(2022, 1, 1)      # 研究期硬边界：>= 此日期禁止接触
VALIDATION_END = _dt.date(2022, 12, 31)
VALIDATION_YEAR = HOLDOUT_START.year
STRICT_OOS_START = _dt.date(2023, 1, 1)
DEFAULT_OOS_END = _dt.date(2025, 12, 31)
RESEARCH_END = _dt.date(2021, 12, 31)

STUDY_START = _dt.date(2016, 1, 1)        # 第一个计入绩效的信号日

# 日频因子研究逐年诊断；ML 的训练/测试折由 intraday.ml 明确指定。
# 2016/2017 两年的滚动阈值与标准化窗口落在夜盘未全面铺开的 2014-2016，报告中须标注

# --------------------------------------------------------------------------
# 品种
# --------------------------------------------------------------------------
# 与 backtest_cta_pack/data/merge_minute_ranges.py 的 NAMES 完全一致（83 个，顺序保留）
ALL_SYMBOLS = [
    'AG', 'AL', 'AU', 'BC', 'BU', 'CU', 'FU', 'HC', 'LU', 'NI', 'NR', 'PB', 'RB',
    'RU', 'SC', 'SN', 'SP', 'SS', 'WR', 'ZN', 'A', 'B', 'BB', 'C', 'CS', 'EB',
    'EG', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'LH', 'M', 'P', 'PG', 'PP', 'RR', 'V',
    'Y', 'AP', 'CF', 'CJ', 'CY', 'FG', 'JR', 'LR', 'MA', 'OI', 'PF', 'PK', 'PM',
    'RI', 'RM', 'RS', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR', 'WH', 'ZC', 'LC',
    'IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TS', 'TL', 'AO', 'EC', 'SH', 'AD', 'BR',
    'LG', 'PR', 'PX', 'SI', 'PS',
]

# 金融期货，一律剔除（设计文档第 4.1 节第一步）
FINANCIAL_SYMBOLS = ['IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TS', 'TL']
COMMODITY_SYMBOLS = [s for s in ALL_SYMBOLS if s not in FINANCIAL_SYMBOLS]

# 品种池门槛（设计文档第 4.1 节）
MIN_DAILY_TURNOVER = 30e8      # 日均成交额 >= 30 亿元（按 trading_date 日度求和后取中位数）
MIN_VALID_DAY_RATIO = 0.90     # 当年 closew 非 NaN 的交易日占比
MIN_VALID_DAYS = 200           # 当年有效交易日下限

# 需要输出品种池的年份。2015 年只用于预热，第一个计入绩效的年份是 STUDY_START。
UNIVERSE_YEARS = list(range(2015, RESEARCH_END.year + 1))

# 时点有效性：第 y 年的池子用 [y - UNIVERSE_LOOKBACK_YEARS, y-1] 的统计量判定。
# 取 1 而非 3 是刻意的：ZC 这类僵尸品种的成交额是断崖式塌缩，回看窗口越长，
# 塌缩后仍被留在池子里的年数越多。回看 1 年反应最快，且完全没有前视。
# 代价是新上市品种要晚一年才能进池——这个方向的保守是可以接受的。
UNIVERSE_LOOKBACK_YEARS = 1

# 夜盘 bar 数中位数的归类门槛（与 EXPECTED_BARS_PER_DAY 对应）
NIGHT_BARS_MIN = 5             # 少于此数视为当日无夜盘（节后零星 bar 不算）
NIGHT_BARS_2300 = 165          # 21:00-23:00 约 120 分钟
NIGHT_BARS_0100 = 285          # 21:00-01:00 约 240 分钟

# --------------------------------------------------------------------------
# 字段
# --------------------------------------------------------------------------
# 因子计算实际需要的字段（设计文档第 3.1 节）。丢掉 dominant_id 这个 object 列是省内存的关键。
FACTOR_FIELDS = [
    'closew',          # 加法复权收盘价，用于价格形状与位置对照
    'close',           # FP 与 DFP 分母（比例量一律用原始价）
    'highw', 'loww',   # 日内极值时间戳
    'volume',          # 成交量持续期 / 量峰时间戳
    'total_turnover',  # 成交额峰值时间戳 + 流动性筛选
    'trading_date',    # 唯一合法的"日"定义（夜盘归属次日）
]
# 收益口径需要的开盘价（设计文档第 8.3 节）。因子计算不用它们，
# 但 day_ret 要用，而且分片一旦落盘就补不回这两列——所以 step1 一并保留。
PRICE_FIELDS = ['open', 'openw']
ROLL_FIELD = 'dominant_id'     # 仅 step1 用于提取换月日，提完即丢

# --------------------------------------------------------------------------
# 时段定义（设计文档第 6.1 节）
# --------------------------------------------------------------------------
# 用 (起, 止] 的分钟墙钟区间判定；NIGHT 跨自然日，单独处理。
SESSION_NIGHT = 'NIGHT'
SESSION_AM = 'AM'
SESSION_PM = 'PM'
SESSIONS = [SESSION_NIGHT, SESSION_AM, SESSION_PM]

NIGHT_START_HOUR = 20          # 夜盘 bar 的墙钟小时 >= 20 或 <= 4 即判为夜盘
NIGHT_END_HOUR = 4
AM_START = _dt.time(8, 30)
AM_END = _dt.time(11, 30)
PM_START = _dt.time(13, 0)
PM_END = _dt.time(15, 30)

# 预期的每日分钟 bar 数（用于 step1 验收，允许半日市等偏差）
EXPECTED_BARS_PER_DAY = {'no_night': 225, 'night_2300': 345, 'night_0100': 465, 'night_0230': 555}

# --------------------------------------------------------------------------
# Time-factor definition (fixed before research; no parameter sweep).
THRESHOLD_LOOKBACKS = [250]
THRESHOLD_PCTS = [55.0]
FP_TOP_NS = [1, 3]

IC_REFERENCE_LOOKBACK = 250
IC_REFERENCE_PCT = 55.0
IC_MIN_OBS = 60
# Diagnostics evaluate 1/3/5/10-day horizons separately from rebalancing.
IC_PERIOD = 'ME'
IC_PERIOD_MIN_OBS = 10      # 一期至少这么多个有效 (因子, 收益) 配对才算一个观测
IC_PERIOD_MIN_COUNT = 6     # 少于这么多期不给 t 值，宁可留空也不给一个假精度
# 事前标准化窗口。因子 z 与收益的波动都只用 t 日已知的数据。
IC_Z_WINDOW = 252
IC_VOL_WINDOW = 60
IC_Z_MIN = 20

# 方向验收判「与先验相反」所需的最小 |t|（时序 t，不是横截面 t）。
# 没有这道门槛，IC = +0.0008、t = 0.22 会被判成 flip 并拦在第 4 步——那不是"方向相反"，
# 那是"这个因子在商品上没有可测的 IC"。两件事的处置完全不同：前者要去查实现，
# 后者是一个研究结论（论文的因子没迁移过来），应当记录并继续，而不是假装查到了 bug。
# 注意这**不是**放松闸门：显著的反向依然是 flip 并且照样拦。
SIGN_T_MIN = 2.0

# --------------------------------------------------------------------------
# 因子方向。事前先验，不在 2016–2021 的分钟 IC 上重估或翻号。
# +1 越大越看多，-1 越大越看空。
# dfp：尾盘相对日内均衡价超跌则看多。
# ts_high：高点越晚越看空；ts_low：低点越晚越看多。
# pmt / vmt：最长持续期越晚越看空。vd_ratio：早盘比午后更平稳则看多。
# ts_volume / ts_turnover / vol_pm：放量越靠近午后越看多。
# ts_high_pm：午后高点越晚越看空；ts_low_pm：午后低点越晚越看多。
# spike_am：早盘触碰全日高点的次数越多越看多。
# 方向是报告给出的次日含义，不在研究期收益上翻号。
# --------------------------------------------------------------------------
FACTOR_SIGNS = {
    'dfp_max': +1,
    'dfp_top3': +1,
    'pmt': -1,
    'vmt': -1,
    'vd_ratio': +1,
    'ts_high': -1,
    'ts_low': +1,
    'ts_volume': +1,
    'ts_turnover': +1,
    'ts_high_pm': -1,
    'ts_low_pm': +1,
    'spike_am': +1,
    'vol_pm': +1,
}
DURATION_FACTORS = ['dfp_max', 'dfp_top3']
TIMESTAMP_FACTORS = ['ts_high', 'ts_low']
PRIOR_FACTORS = list(FACTOR_SIGNS)


# --------------------------------------------------------------------------
# 样本外守卫
# --------------------------------------------------------------------------
class HoldoutViolation(RuntimeError):
    """研究代码试图接触 2022-01-01 及以后的数据。"""


OOS_LEDGER = DATA_ROOT / "oos" / "ledger.jsonl"
_final_evaluation: dict | None = None   # 只由 final_evaluation() 在 with 块内设置


def assert_oos_research_locked(end=None) -> None:
    """研究代码不能靠日期、环境变量或调用参数打开样本外；只有 final_evaluation() 的 with 块内放行，
    且读取不得越过登记的截止日。"""
    if _final_evaluation is None:
        raise HoldoutViolation('2023–2025 严格样本外保持封存；尚未冻结策略，禁止读取')
    if end is not None and to_date(end) > _final_evaluation["oos_end"]:
        raise HoldoutViolation(f"样本外读取越过登记的截止日 {_final_evaluation['oos_end']}: {to_date(end)}")


def final_evaluation_end() -> _dt.date:
    assert_oos_research_locked()
    return _final_evaluation["oos_end"]


def _sha256(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_ledger(entry: dict, ledger=None) -> None:
    import json
    path = Path(ledger or OOS_LEDGER)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def read_ledger(ledger=None) -> list[dict]:
    import json
    path = Path(ledger or OOS_LEDGER)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class final_evaluation:
    """唯一能打开 2023 年以后数据的入口。

    条件：策略配置已写明 chosen 与 frozen_at；测试窗口已经结束；窗口不超过已分片的 DEFAULT_OOS_END。
    进入时先把配置文件哈希、窗口和脚本写进台账，再放行读取；离开 with 块立即重新上锁。
    """

    def __init__(self, spec: dict, files, oos_end, strategy: str, note: str = "",
                 today=None, ledger=None):
        self.spec, self.files, self.strategy, self.note = spec, list(files), strategy, note
        self.oos_end, self.today, self.ledger = to_date(oos_end), today, ledger

    def __enter__(self) -> dict:
        global _final_evaluation
        if _final_evaluation is not None:
            raise HoldoutViolation("已有一个打开的最终评估")
        if not self.spec.get("chosen") or not self.spec.get("frozen_at"):
            raise HoldoutViolation("配置没有冻结（缺 chosen 或 frozen_at），不能打开样本外")
        if self.oos_end < STRICT_OOS_START or self.oos_end > DEFAULT_OOS_END:
            raise HoldoutViolation(
                f"样本外截止日须在 {STRICT_OOS_START}..{DEFAULT_OOS_END}（已分片范围），收到 {self.oos_end}")
        assert_test_window_closed(self.oos_end, today=self.today)
        hashes = {Path(p).name: _sha256(p) for p in self.files}
        entry = {
            "kind": "final_evaluation", "status": "opened",
            "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "strategy": self.strategy, "chosen": self.spec["chosen"], "frozen_at": self.spec["frozen_at"],
            "oos_start": str(STRICT_OOS_START), "oos_end": str(self.oos_end),
            "sha256": hashes, "note": self.note,
        }
        append_ledger(entry, self.ledger)
        _final_evaluation = {**entry, "oos_end": self.oos_end}
        return entry

    def __exit__(self, *exc) -> None:
        global _final_evaluation
        _final_evaluation = None


def assert_research_only(path) -> None:
    """拒绝任何指向 holdout_locked/ 的读取。"""
    p = Path(path).resolve()
    if any(root.resolve() in p.parents or p == root.resolve()
           for root in (HOLDOUT_DIR, VALIDATION_DIR)):
        raise HoldoutViolation(
            f"本阶段禁止读取样本外数据: {p}\n"
            "研究入口禁止读取验证期和样本外路径。"
        )


def assert_validation_only(path) -> None:
    """验证读取只允许来自专门的 2022 分片目录。"""
    p = Path(path).resolve()
    root = VALIDATION_DIR.resolve()
    if root not in p.parents and p != root:
        raise HoldoutViolation(f"2022 验证数据必须来自 {root}: {p}")


def assert_holdout_only(path) -> None:
    """锁定文件只能从 holdout_locked 目录读取。"""
    p = Path(path).resolve()
    root = HOLDOUT_DIR.resolve()
    if root not in p.parents and p != root:
        raise HoldoutViolation(f"样本外日历只能来自 {root}: {p}")


def assert_validation_2022_dates(index_or_series, what: str = "data") -> None:
    """验证面板必须严格限于 2022 年，拒绝混入训练期或严格 OOS。"""
    import pandas as pd

    ts = pd.to_datetime(pd.Index(index_or_series))
    if len(ts) == 0:
        return
    lo, hi = pd.Timestamp(HOLDOUT_START), pd.Timestamp(STRICT_OOS_START)
    if ts.min() < lo or ts.max() >= hi:
        raise HoldoutViolation(
            f"{what} 必须只含 2022 年数据，实际范围 {ts.min().date()}..{ts.max().date()}"
        )


def assert_strict_oos_dates(index_or_series, what: str = "data") -> None:
    """样本外日期必须在 2023 年及以后。"""
    import pandas as pd

    ts = pd.to_datetime(pd.Index(index_or_series))
    if len(ts) and ts.min() < pd.Timestamp(STRICT_OOS_START):
        raise HoldoutViolation(
            f"{what} 必须只含 {STRICT_OOS_START} 及以后的日期，"
            f"实际最早为 {ts.min().date()}"
        )


def to_date(value) -> _dt.date:
    """字符串、datetime、date、Timestamp 统一成 date。"""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value)[:10])


def assert_test_window_closed(end, today=None) -> None:
    """测试期的最后一天必须已经过去。"""
    today = _dt.date.today() if today is None else to_date(today)
    end = to_date(end)
    if today <= end:
        raise HoldoutViolation(
            f"测试期尚未结束（截止 {end}，今天 {today}），拒绝执行。"
        )


def assert_no_holdout_dates(index_or_series, what: str = "data") -> None:
    """校验时间索引未越过 HOLDOUT_START。"""
    import pandas as pd

    ts = pd.to_datetime(pd.Index(index_or_series))
    if len(ts) == 0:
        return
    hi = ts.max()
    if hi >= pd.Timestamp(HOLDOUT_START):
        raise HoldoutViolation(
            f"{what} 含样本外日期 {hi.date()} (>= {HOLDOUT_START})，本阶段禁止使用。"
        )


def ensure_dirs() -> None:
    for d in (SHARD_ROOT, RESEARCH_DIR, ROLL_DIR,
              FACTOR_DAILY_DIR, UNIVERSE_DIR, RUNS_DIR, CONFIG_DIR,
              RESEARCH_OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def report_step(step: int, *, passed: bool, paths, next_step: int | None = None,
                note: str = "") -> None:
    """终端只报结论：是否通过、每个结果文件是什么、能不能进下一步。"""
    print(f"第 {step} 步{'通过' if passed else '未通过'}。")
    if note:
        print(note)
    print("结果：")
    for path, desc in paths:
        print(f"  {desc}")
        print(f"    {path}")
    if passed and next_step is not None:
        print(f"可以进入第 {next_step} 步。")
    elif not passed:
        print("不要进入下一步。")
