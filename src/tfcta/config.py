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

# 数据根目录。设置环境变量 TFCTA_DATA_ROOT 可整体重定向——用于在合成数据上演练
# 整条管道而不污染真实分片。生产运行不要设置它。商品与股指分开放。
DATA_ROOT = Path(_os.environ.get("TFCTA_DATA_ROOT", PROJECT_ROOT / "data"))
COMMODITY_DATA = DATA_ROOT / "commodity"
INDEX_DATA_ROOT = DATA_ROOT / "index"
COMMODITY_OOS_DATA_DIR = COMMODITY_DATA / "oos"
INDEX_OOS_DATA_DIR = INDEX_DATA_ROOT / "oos"

# 分钟频原始单体 pickle（扩展名是 .txt，内容是 pickle），在 data/commodity/data_min/ 下。
# 可用环境变量 TFCTA_MINUTE_DIR 指向别处；默认数据目录受 /data/ 忽略规则保护。
MINUTE_RAW_DIR = Path(_os.environ.get("TFCTA_MINUTE_DIR", COMMODITY_DATA / "data_min"))
MINUTE_MONOLITH = MINUTE_RAW_DIR / "future_all1mdata_20100101-20251231.txt"

# 商品分钟分片。研究期与 OOS 分开存放；OOS 根目录合并保存 2022–2025。
SHARD_ROOT = COMMODITY_DATA / "minute_shards"
RESEARCH_DIR = SHARD_ROOT / "research"
OOS_DIR = SHARD_ROOT / "oos"
ROLL_DIR = COMMODITY_DATA / "roll_dates"

# 股指期货分钟分片。研究期与样本外分开，读取时按目录拦截。
INDEX_SHARD_ROOT = INDEX_DATA_ROOT / "minute_shards"
INDEX_RESEARCH_DIR = INDEX_SHARD_ROOT / "research"
INDEX_OOS_DIR = INDEX_SHARD_ROOT / "oos"
INDEX_EXTERNAL_DIR = INDEX_DATA_ROOT / "external_rqdata"
INDEX_EXTERNAL_OOS_DIR = INDEX_EXTERNAL_DIR / "oos"
INDEX_CONTRACTS_DIR = INDEX_EXTERNAL_DIR / "contracts"
INDEX_ROLL_YIELD_DIR = INDEX_EXTERNAL_DIR / "roll_yield"
INDEX_SPOT_DIR = INDEX_EXTERNAL_DIR / "spot"
INDEX_MARKET_FLOWS_PATH = INDEX_EXTERNAL_DIR / "flows" / "market_flows.csv"
INDEX_MEMBER_RANK_PATH = INDEX_EXTERNAL_DIR / "member_rank" / "member_rank.csv"
INDEX_VOLATILITY_DIR = INDEX_EXTERNAL_DIR / "volatility"
INDEX_QVIX_RESEARCH_PATH = INDEX_VOLATILITY_DIR / "qvix_50etf.csv"
INDEX_QVIX_OOS_DIR = INDEX_EXTERNAL_OOS_DIR / "volatility"
INDEX_L1_CLOSE_DIR = INDEX_EXTERNAL_DIR / "l1_close"
INDEX_L1_TICKS_DIR = INDEX_EXTERNAL_DIR / "l1_ticks"
INDEX_CONSTITUENTS_DIR = INDEX_EXTERNAL_DIR / "constituents"

# 因子缓存与运行留痕
FACTOR_DAILY_DIR = COMMODITY_DATA / "factor_daily_v3"
UNIVERSE_DIR = COMMODITY_DATA / "universe"
RUNS_DIR = Path(_os.environ.get("TFCTA_RUNS_ROOT", PROJECT_ROOT / "runs"))
CONFIG_DIR = Path(_os.environ.get("TFCTA_CONFIG_DIR", PROJECT_ROOT / "config"))
# 研究输入费用与 tick 表；实验结果只写独立 runs/ 目录。
RESEARCH_OUT_DIR = COMMODITY_DATA / "research"

# --------------------------------------------------------------------------
# 时间切分
# --------------------------------------------------------------------------
OOS_START = _dt.date(2022, 1, 1)
DEFAULT_OOS_END = _dt.date(2025, 12, 31)
RESEARCH_END = _dt.date(2021, 12, 31)

STUDY_START = _dt.date(2016, 1, 1)        # 第一个计入绩效的信号日
EARLY_START = _dt.date(2011, 1, 1)        # 用户 2026-10-23：2011–2015 也算研究期，账本可以从这里起算；2010 只做预热
TRADING_DAYS_PER_YEAR = 242

# --------------------------------------------------------------------------
# 品种
# --------------------------------------------------------------------------
# 商品与金融期货代码。金融期货不进商品池。
ALL_SYMBOLS = [
    'AG', 'AL', 'AU', 'BC', 'BU', 'CU', 'FU', 'HC', 'LU', 'NI', 'NR', 'PB', 'RB',
    'RU', 'SC', 'SN', 'SP', 'SS', 'WR', 'ZN', 'A', 'B', 'BB', 'C', 'CS', 'EB',
    'EG', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'LH', 'M', 'P', 'PG', 'PP', 'RR', 'V',
    'Y', 'AP', 'CF', 'CJ', 'CY', 'FG', 'JR', 'LR', 'MA', 'OI', 'PF', 'PK', 'PM',
    'RI', 'RM', 'RS', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR', 'WH', 'ZC', 'LC',
    'IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TS', 'TL', 'AO', 'EC', 'SH', 'AD', 'BR',
    'LG', 'PR', 'PX', 'SI', 'PS',
]

# 金融期货，不进商品池
FINANCIAL_SYMBOLS = ['IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TS', 'TL']
COMMODITY_SYMBOLS = [s for s in ALL_SYMBOLS if s not in FINANCIAL_SYMBOLS]

# 品种池门槛
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
# 因子计算实际需要的字段。丢掉 dominant_id 这个 object 列是省内存的关键。
FACTOR_FIELDS = [
    'closew',          # 加法复权收盘价，用于价格形状与位置对照
    'close',           # FP 与 DFP 分母（比例量一律用原始价）
    'highw', 'loww',   # 日内极值时间戳
    'volume',          # 成交量持续期 / 量峰时间戳
    'total_turnover',  # 成交额峰值时间戳 + 流动性筛选
    'trading_date',    # 唯一合法的"日"定义（夜盘归属次日）
]
# 收益口径需要的开盘价。因子计算不用它们，
# 但 day_ret 要用，而且分片一旦落盘就补不回这两列——所以 step1 一并保留。
PRICE_FIELDS = ['open', 'openw']
ROLL_FIELD = 'dominant_id'     # 仅 step1 用于提取换月日，提完即丢

# --------------------------------------------------------------------------
# 时段定义
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

# 持续期因子缓存用的阈值：过去 250 日、第 55 百分位（兴业口径的中心参数）。
IC_REFERENCE_LOOKBACK = 250
IC_REFERENCE_PCT = 55.0
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
TIMESTAMP_FACTORS = ['ts_high', 'ts_low']
# --------------------------------------------------------------------------
# 样本边界
# 行情按 research/ 与 oos/ 分目录存放。回测命令直接选择区间，读研究期只打开
# research/，读样本外只打开 oos/；跨过 2021/2022 时两边各读各的，再按日期接上。
# 研究期到 2021-12-31，样本外是 2022-01-01 至 2025-12-31。没有 2026 分区。
# --------------------------------------------------------------------------
class HoldoutViolation(RuntimeError):
    """日期或路径越出声明的分区。"""


def backtest_dir(market: str, strategy_id: str, folder: str) -> Path:
    """账本目录：runs/<市场>/strategies/<策略>/<区间>/。"""
    if market not in ("commodity", "index"):
        raise ValueError(f"未知市场 {market}")
    return RUNS_DIR / market / "strategies" / strategy_id / folder


def study_dir(market: str, name: str) -> Path:
    """研究输出：runs/<市场>/research/<名称>/。"""
    if market not in ("commodity", "index"):
        raise ValueError(f"未知市场 {market}")
    return RUNS_DIR / market / "research" / name


def assert_research_only(path) -> None:
    """拒绝任何指向样本外目录的研究读取。"""
    p = Path(path).resolve()
    if any(root.resolve() in p.parents or p == root.resolve()
           for root in (OOS_DIR, INDEX_OOS_DIR, INDEX_EXTERNAL_OOS_DIR)):
        raise HoldoutViolation(
            f"本阶段禁止读取样本外数据: {p}\n"
            "研究入口只允许读取研究期路径。"
        )


def assert_under(path, root, what: str) -> None:
    """路径必须落在给定根目录里。"""
    p = Path(path).resolve()
    root = Path(root).resolve()
    if root not in p.parents and p != root:
        raise HoldoutViolation(f"{what}必须来自 {root}: {p}")


def assert_oos_only(path) -> None:
    """商品样本外文件只允许来自商品 OOS 分区。"""
    assert_under(path, OOS_DIR, "商品样本外数据")


def assert_index_oos_only(path) -> None:
    """股指样本外文件只允许来自股指 OOS 分区。"""
    assert_under(path, INDEX_OOS_DIR, "股指样本外数据")


def assert_oos_dates(index_or_series, what: str = "data") -> None:
    """样本外日期不得早于 2022 年。"""
    import pandas as pd

    ts = pd.to_datetime(pd.Index(index_or_series))
    if len(ts) and ts.min() < pd.Timestamp(OOS_START):
        raise HoldoutViolation(
            f"{what} 必须只含 {OOS_START} 及以后的日期，"
            f"实际最早为 {ts.min().date()}"
        )


def to_date(value) -> _dt.date:
    """字符串、datetime、date、Timestamp 统一成 date。"""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value)[:10])


def assert_no_holdout_dates(index_or_series, what: str = "data") -> None:
    """校验时间索引未越过 OOS_START。"""
    import pandas as pd

    ts = pd.to_datetime(pd.Index(index_or_series))
    if len(ts) == 0:
        return
    hi = ts.max()
    if hi >= pd.Timestamp(OOS_START):
        raise HoldoutViolation(
            f"{what} 含样本外日期 {hi.date()} (>= {OOS_START})，本阶段禁止使用。"
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
