"""全局配置：路径、时间切分、时段定义、参数网格。

本模块是唯一的常量来源。任何脚本都不应硬编码路径或日期。

关键纪律（见 docs/Design.md 第 0 节）：
    2022-01-01 及以后的数据在本阶段一次都不看。
    HOLDOUT_DIR 由 step1 写入后即锁定，任何研究代码读取它都算 bug。
    load_shard() 会主动拒绝越界访问。
"""
from __future__ import annotations

import datetime as _dt
import os as _os
from pathlib import Path

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
# src/tfcta/config.py -> src/tfcta -> src -> time_factor_cta
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 分钟频原始单体 pickle（扩展名是 .txt，内容是 pickle），在本项目根目录下。
# 可用环境变量 TFCTA_MINUTE_DIR 指向别处，但默认就是仓库内的 data_min/——
# 原始数据跟着项目走，不依赖同级的 backtest_cta_pack 是否存在。
MINUTE_RAW_DIR = Path(_os.environ.get("TFCTA_MINUTE_DIR", PROJECT_ROOT / "data_min"))
MINUTE_MONOLITH = MINUTE_RAW_DIR / "future_all1mdata_20100101-20251231.txt"

# 数据根目录。设置环境变量 TFCTA_DATA_ROOT 可整体重定向——用于在合成数据上演练
# 整条管道而不污染真实分片。生产运行不要设置它。
DATA_ROOT = Path(_os.environ.get("TFCTA_DATA_ROOT", PROJECT_ROOT / "data"))

# step1 产出
SHARD_ROOT = DATA_ROOT / "minute_shards"
RESEARCH_DIR = SHARD_ROOT / "research"                     # <= 2021-12-31，可自由使用
HOLDOUT_DIR = SHARD_ROOT / "holdout_locked"                # >= 2022-01-01，本阶段禁止读取
ROLL_DIR = DATA_ROOT / "roll_dates"

# 因子缓存与运行留痕
FACTOR_DAILY_DIR = DATA_ROOT / "factor_daily"
UNIVERSE_DIR = DATA_ROOT / "universe"
RUNS_DIR = Path(_os.environ.get("TFCTA_RUNS_ROOT", PROJECT_ROOT / "runs"))
CONFIG_DIR = Path(_os.environ.get("TFCTA_CONFIG_DIR", PROJECT_ROOT / "config"))
# 第 5-11 步的研究产物（IC、参数扫描、选参、组合、费率）。
# runs/ 是每次运行的留痕快照；这里是下游步骤接着读的最新一份。
RESEARCH_OUT_DIR = DATA_ROOT / "research"

# --------------------------------------------------------------------------
# 时间切分（设计文档第 5 节）
# --------------------------------------------------------------------------
HOLDOUT_START = _dt.date(2022, 1, 1)      # 研究期硬边界：>= 此日期禁止接触
RESEARCH_END = _dt.date(2021, 12, 31)

WARMUP_START = _dt.date(2014, 7, 1)       # 阈值+信号双层滚动预热（约 360 个交易日）
STUDY_START = _dt.date(2016, 1, 1)        # 第一个计入绩效的信号日
STUDY_END = RESEARCH_END

BACKWARD_OOS_START = _dt.date(2010, 1, 1)  # 向后时间外检验（只看符号，不得选参）
BACKWARD_OOS_END = _dt.date(2014, 12, 31)

# walk-forward：训练 3 年 / 测试 1 年 / 步进 1 年
WF_TRAIN_YEARS = 3
WF_TEST_YEARS = 1
WF_TEST_YEARS_LIST = [2016, 2017, 2018, 2019, 2020, 2021]
# 2016/2017 两折的训练窗口落在夜盘未全面铺开的 2013-2015，报告中须标注
WF_FOLDS_WITH_SPARSE_NIGHT = [2016, 2017]

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
    'closew',          # 价格持续期 / FP / DFP
    'close',           # 归一化分母（框架惯例：信号用复权价，量纲用原始价）
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
# 参数网格（设计文档第 7 节，数值取自论文原文）
# --------------------------------------------------------------------------
THRESHOLD_LOOKBACKS = [200, 250, 300]                 # 阈值回看期 N（交易日）
THRESHOLD_PCTS = [50.0, 52.5, 55.0, 57.5, 60.0]       # 阈值分位数 M（%）
SIGNAL_WINDOWS = [20, 25, 30, 35, 40, 45, 50, 55, 60] # 信号滚动期 W（交易日）
SIGNAL_BANDS = [(30.0, 70.0), (25.0, 75.0), (20.0, 80.0)]  # (低轨, 高轨) 百分位

FP_TOP_NS = [1, 3]             # 公允均衡价格取持续期前 N 大（论文用 1 和 3）
EXTREME_SIGMA = 2.0            # 极端持续期判定：均值 + 2 倍标准差

# 手续费敏感性（框架默认 0.00025）
FEE_GRID = [0.00025, 0.0005, 0.001]
# 第 6 步回测默认用这一档；另外两档只做敏感性，不参与选参。
FEE_BASE = FEE_GRID[0]

# 滑点：按"每次换手穿越几个最小变动价位"计，比例成本由 research/costs.py 从数据
# 估出的 tick / 当年价位中位数换算。写成 tick 数而不是一个比例数，是因为同样穿一个
# tick，铁矿（价位 ~640、tick 0.5）要付 9.9bp，锡（~142000、tick 10）只付 0.7bp——
# 本样本实测相差 13.7 倍（中位 3.3bp）。固定比例会把这个横截面差异抹平，而等权组合的
# 成本恰恰由低价位品种主导。
#
# 基准取 1 个 tick：日频调仓、次日开盘成交，挂在对手价上一个 tick 是这个频率下
# 偏保守但不夸张的假设。0 档只作为对照，**不能**用它选参或写结论——无滑点等于
# 假装换手免费，选出来的信号窗口会系统性偏短。
SLIPPAGE_TICKS = 1.0
SLIPPAGE_TICK_GRID = [0.0, 1.0, 2.0]

# 框架 -std mad：滚动中位数，除以 5×MAD，clip 到 [-1, 1]。
# 窗口含当日——当日因子值在收盘时已知，标准化可以用它；信号分位轨则不含当日。
STD_WINDOW = 1000
STD_MAD_MULT = 5.0
STD_CLIP = 1.0

# 第 4 步 IC 的参照阈值。取论文默认 N=250、M 区间中点，在选参之前做符号检查。
# 这里不扫网格，避免用 IC 挑参数。
IC_REFERENCE_LOOKBACK = 250
IC_REFERENCE_PCT = 55.0
IC_MIN_OBS = 60

# IC 显著性的**时序**口径（research/stats.py::ic_period_series / timeseries_t）。
# 一期（默认一个自然月）先在品种内算 Spearman、再在期内对品种取平均，得到一条
# IC 时间序列，t 值是这条序列均值的 Newey-West t。分母来自时间上的变异，商品之间
# 的同期相关性被期内平均吸收掉——跨品种口径把高度相关的商品当独立样本，t 会虚高。
IC_PERIOD = 'ME'            # 月末重采样；一个测试年约 12 个观测
IC_PERIOD_MIN_OBS = 10      # 一期至少这么多个有效 (因子, 收益) 配对才算一个观测
IC_PERIOD_MIN_COUNT = 6     # 少于这么多期不给 t 值，宁可留空也不给一个假精度

# 方向验收判「与先验相反」所需的最小 |t|（时序 t，不是横截面 t）。
# 没有这道门槛，IC = +0.0008、t = 0.22 会被判成 flip 并拦在第 4 步——那不是"方向相反"，
# 那是"这个因子在商品上没有可测的 IC"。两件事的处置完全不同：前者要去查实现，
# 后者是一个研究结论（论文的因子没迁移过来），应当记录并继续，而不是假装查到了 bug。
# 注意这**不是**放松闸门：显著的反向依然是 flip 并且照样拦。
SIGN_T_MIN = 2.0

# 第 9.2 节第三层的裁决标准。本阶段只写进 frozen_config，不执行。
OOS_SR_DECAY_MAX = 0.40
OOS_MDD_RATIO_MAX = 1.5

# --------------------------------------------------------------------------
# 因子方向（设计文档第 6 节，全部来自论文先验，不得事后按数据翻转）
# --------------------------------------------------------------------------
# +1 表示"因子值越大越看多"；-1 表示"越大越看空"，实现时统一在因子定义处取负，
# 使下游信号逻辑只有一套。任何对本表的修改都必须写进 frozen_config.yaml 并说明依据。
FACTOR_SIGNS = {
    # --- 持续期族主力 ---
    'dfp_max': +1,      # 公允均衡价格高于收盘 -> 非理性超跌 -> 次日看多
    'dfp_top3': +1,
    'pmt': -1,          # 价格稳态时点越晚 -> 信息消化越慢 -> 次日看空
    'vr': +1,           # 午前量能持续期相对午后越高 -> 早盘消化越充分 -> 看多
    'vr_night': +1,     # 夜盘版（商品扩展，同向假设）
    'vmt': -1,          # 量能稳态时点越晚 -> 看空
    # --- 持续期族基础聚合（对照组，无强先验，暂设 +1 并在报告中标注为无先验） ---
    'dur_mean': +1, 'dur_std': +1, 'dur_max': +1, 'dur_gap': +1, 'dur_extreme': +1,
    'vdur_mean': +1, 'vdur_std': +1, 'vdur_max': +1, 'vdur_gap': +1, 'vdur_extreme': +1,
    # --- 时间戳族 ---
    'ts_high': -1,      # 上一轮小时频实测 IC 为负（t=-5.2），与论文预测一致
    'ts_low': +1,       # 实测 IC 为正（t=+5.2）
    'ts_vmax': -1,      # 实测 IC 为负（t=-3.8）
    'ts_tomax': -1,     # 未实测，沿用量峰的同族先验
    'night_vol_share': +1,   # 上一轮实测 t=+3.05，跨制度最稳
    'night_day_range': +1,   # 上一轮实测 t=+2.23
}

# 探索性时间戳因子：论文列举但未在国债上给出方向，上一轮小时频也未实测。
# 一律暂定 +1 并计入 NO_PRIOR_FACTORS——它们**不参与主结论**，只在因子层诊断中出现。
# 严禁把这些因子事后按 IC 符号翻转后再放进主合成，那等于用样本内信息定向。
EXPLORATORY_SIGNS = {
    'ts_high_am': +1, 'ts_high_pm': +1, 'ts_low_am': +1, 'ts_low_pm': +1,
    'ts_high_night': +1, 'ts_low_night': +1,
    'cnt_high_am': +1, 'cnt_high_pm': +1, 'is_high_am': +1,
}

# 无强先验的对照组因子，报告中须单独标注（避免把"事后定向"混同为"先验定向"）
NO_PRIOR_FACTORS = {
    'dur_mean', 'dur_std', 'dur_max', 'dur_gap', 'dur_extreme',
    'vdur_mean', 'vdur_std', 'vdur_max', 'vdur_gap', 'vdur_extreme',
} | set(EXPLORATORY_SIGNS)

# 全部已知因子的方向表（主力 + 探索性）
ALL_SIGNS = {**FACTOR_SIGNS, **EXPLORATORY_SIGNS}

# 因子分组（设计文档第 8.2 节：分族合成，不要一锅端）
COMBO_GROUPS = {
    'COMBO_DUR': ['dfp_max', 'dfp_top3', 'pmt', 'vr', 'vr_night', 'vmt'],
    'COMBO_TS': ['ts_high', 'ts_low', 'ts_vmax', 'ts_tomax',
                 'night_vol_share', 'night_day_range'],
    'COMBO_BASE': sorted(NO_PRIOR_FACTORS),   # 对照组：应显著弱于上面两组
}
COMBO_GROUPS['COMBO_ALL'] = COMBO_GROUPS['COMBO_DUR'] + COMBO_GROUPS['COMBO_TS']
# 基础同质化聚合，第 8.2 节的对照组。COMBO_BASE 还含无先验的探索性时间戳因子，
# 那些因子不是「平凡统计量」，不拿来回答「超额是不是来自均值/波动」。
COMBO_GROUPS['COMBO_AGG'] = [
    'dur_mean', 'dur_std', 'dur_max', 'dur_gap', 'dur_extreme',
    'vdur_mean', 'vdur_std', 'vdur_max', 'vdur_gap', 'vdur_extreme',
]

# 持续期族依赖 (N, M)；时间戳族不依赖。两张表的并集必须等于 ALL_SIGNS。
DURATION_FACTORS = [
    'dur_mean', 'dur_std', 'dur_max', 'dur_gap', 'dur_extreme',
    'vdur_mean', 'vdur_std', 'vdur_max', 'vdur_gap', 'vdur_extreme',
    'dfp_max', 'dfp_top3', 'pmt', 'vr', 'vr_night', 'vmt',
]
TIMESTAMP_FACTORS = [
    'ts_high', 'ts_low', 'ts_vmax', 'ts_tomax',
    'ts_high_am', 'ts_high_pm', 'ts_low_am', 'ts_low_pm',
    'ts_high_night', 'ts_low_night',
    'cnt_high_am', 'cnt_high_pm', 'is_high_am',
    'night_vol_share', 'night_day_range',
]
# 有论文或上一轮小时频先验、允许做符号验收的因子。无先验的对照组不在此列。
PRIOR_FACTORS = [k for k in FACTOR_SIGNS if k not in NO_PRIOR_FACTORS]

# 依赖夜盘的因子：无夜盘品种上必须为 NaN，绝不填 0
# （上一轮小时频踩过的坑：结构性零值把 t=3.05 的真因子压成 t=1.1）
#
# 本表必须**穷尽**所有"只在有夜盘时才有定义"的因子，两个用途都依赖它的完整性：
#   1. 验收时按有无夜盘分组统计缺失率——漏登记的因子会被整池缺失率误判为不合格；
#   2. 反向检查"无夜盘品种上是否真的是 NaN 而不是 0"——漏登记就等于没检查。
# ts_high_night / ts_low_night 是第 3 步演练时补上的：它们在 timestamp_factors 里
# 走的是 `if tag == 'night' and not has_night: continue` 分支，和 vr_night 一样是
# 结构性缺失，只是当初写这张表时只想到了显式带 night_ 前缀的那几个。
NIGHT_DEPENDENT_FACTORS = {
    'vr_night', 'night_vol_share', 'night_day_range',
    'ts_high_night', 'ts_low_night',
}


# --------------------------------------------------------------------------
# 样本外守卫
# --------------------------------------------------------------------------
class HoldoutViolation(RuntimeError):
    """研究代码试图接触 2022-01-01 及以后的数据。"""


def assert_research_only(path) -> None:
    """拒绝任何指向 holdout_locked/ 的读取。在所有加载函数入口调用。"""
    p = Path(path).resolve()
    if HOLDOUT_DIR.resolve() in p.parents or p == HOLDOUT_DIR.resolve():
        raise HoldoutViolation(
            f"本阶段禁止读取样本外数据: {p}\n"
            "见 docs/Design.md 第 0 节约束一。"
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
    for d in (SHARD_ROOT, RESEARCH_DIR, HOLDOUT_DIR, ROLL_DIR,
              FACTOR_DAILY_DIR, UNIVERSE_DIR, RUNS_DIR, CONFIG_DIR,
              RESEARCH_OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
