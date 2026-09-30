# 实验指导：高频时间维度因子的商品期货时序 CTA 复现

面向第一次上手这个仓库的人。项目借鉴兴业证券《基于高频时间维度的国债期货择时因子》，
在商品期货分钟数据上构造因子，并提供研究期筛选、一次性验证和样本外评估流程。
2016–2021 是研究期，2022 是一次性验证期，2023–2025 是严格样本外，后两段都只在策略冻结后各读一次。

设计依据见 [Design.md](Design.md)，本文只讲怎么做。
验证 / 样本外登记簿由台账自动生成，写在 `docs/ResearchNotes.md`（第一次写台账时创建）。
文档里提到"第 N 节"时，指的都是那份设计文档。

---

## 一、规矩与环境

### 0. 三条不能破的规矩

这三条是整个实验可信度的全部来源。破了任何一条，后面所有数字都失去意义，
而且**不会有任何报错**——这才是危险的地方。

**规矩一：策略筛选止于 2021；2022 与 2023 起的数据各只看一次。**
研究期代码拒绝读取 `holdout_locked/`，2022 验证只读取 `validation_2022/`，验证 loader
拒绝非 2022 日期。验证分片只能由第 1 步写：要么从切研究期的同一个单体文件按 `trading_date`
取 2022（`--validation-from-monolith`，与切 holdout 同一处过滤，2023+ 的行不落盘、不参与计算），
要么来自独立的 2022-only 源文件。任何其他代码都不得从 `holdout_locked/` 或混合文件拼出 2022。
看过 2022 结果之后不许再改配置重测；台账按指纹和 `book_key` 拦截重复验证，不要删除台账绕过。

**规矩二：因子方向是先验，不许事后按数据翻转。**
方向先验分别来自论文、已有实证或因子定义，见 `config.FACTOR_SIGNS`、
`config.TECH_PRIOR_SIGNS` 与 `factors.library.SIGNED_PRIORS`。第 4 步发现核心先验因子的 IC 符号显著相反时，脚本返回 1 并停下——
正确的反应是**回去查实现**，不是把符号改过来。改了就等于用样本内信息定向，
全部 t 值作废。没有方向先验的因子登记在因子目录中，只报 IC；进入 step5 时必须显式写 `:+1` 或 `:-1`，
且方向需由经济逻辑事前确定，不能根据研究期 IC 事后定向。

**规矩三：区分因子构造参数与策略筛选。**
step3 默认构造 N=250、M=55 的分钟因子；step5 可指定因子、板块、持有批数和风险目标，
只读 2016–2021 研究期。step5 结果是同一研究样本上的筛选，不能当成独立验证；
先按经济假设限定候选与比较规则，不要根据多轮回测结果不断扩展搜索。

---

### 1. 环境与数据

#### 1.1 解释器

先激活环境，之后本文所有命令都直接用 `python`：

```bash
conda activate factor-mining
python -m pytest
python scripts/step3_build_factors.py
```

分片是这个环境里的 pyarrow 写出来的，换到 base 环境去读会报
`OSError: Repetition level histogram size mismatch`。自检也在这个环境里跑。

中文输出在 GBK 终端下会乱码，命令前加 `PYTHONIOENCODING=utf-8`。

基础流程依赖 `pandas`、`numpy`、`pyarrow`（分片格式；缺少时 step1 会退回 pickle）和
`pytest`（自检）。ORB 研究模型另需 `scipy`、`scikit-learn`。

#### 1.2 原始数据

`data/data_min/` 下五个文件全是 **pickle，只是扩展名写成了 `.txt`**，
必须用 `pickle.load` 读，`read_csv` 会直接乱码。本项目只用最后一个：

```
data/data_min/future_all1mdata_20100101-20251231.txt   10.7 GB
```

两层 MultiIndex 列 `(品种, 字段)`，index 是分钟墙钟时间戳，83 品种 × 15 字段。
`config.MINUTE_MONOLITH` 指向它；要挪到别处，设环境变量 `TFCTA_MINUTE_DIR`。

**载入这个文件的峰值内存约 12-15 GB**，且 `pickle.load` 无法分块，
只有第 1 步需要它，之后全部读分片。机器内存不够时，先用
`--symbols` 分批跑第 1 步（但每批都要重新载入一次单体文件，代价是时间换内存）。

#### 1.3 先跑自检

任何实验之前先确认代码是好的：

```bash
python -m pytest
```

以当前工作区运行结果为准，不依赖固定测试数量或耗时；不要把输出管道接到 `grep`，以免隐藏退出状态。
测试按功能放在 `tests/data`、`tests/factors`、`tests/research`、`tests/download`。
这套测试不测"跑得通"，测的是几件**做错了也不会报错**的事：
夜盘因子在无夜盘品种上是 NaN 而非 0、阈值网格与逐个调用逐元素相同、
时间戳族不随 (N, M) 变化、抽查统计不混入预热年、因子表的每行对应一个
`trading_date`、信号分位轨不含当日、仓位晚一天成交、手续费与滑点按同一份换手扣、
IC 一折必须同时切品种池和时间、显著性用时序 t 而非横截面 t、
中心点距离先 z-score、tick 估计取频繁最小档而非众数。
这里红了就不要往下走。

---

## 二、因子与时间轴

### 2. 因子到底在算什么

一页版。细节在设计文档第 6-7 节。

#### 2.1 持续期族

核心量是**持续期**：第 i 根分钟 bar 的持续期 = `i − j`，其中 j 是**当日**、
i 之前、**最近**一根满足 `|Value_i − Value_j| ≥ Threshold` 的 bar。

一直找不到这样的 j，持续期就**从当日开盘累积**（0 基下标下 `dur = i`）。
这一条很容易写错成 0 或 NaN——写错了就等于把"整个上午价格没动过"
这种最有信息量的情形抹掉。

`Threshold` 是**动态**的：过去 N 个交易日**全部**日内一阶绝对差分池化后的第 M 分位数，
**不含当日**（含了就是前视）。当前策略只用 N=250、M=55，只在价格上算阈值。

由持续期派生、并且进了周频书的因子：

| 因子                       | 含义                                                                | 方向 |
| -------------------------- | ------------------------------------------------------------------- | ---- |
| `dfp_max` / `dfp_top3` | 持续期最长的 1 / 3 根 bar 的价格（公允均衡价格 FP）相对收盘价的偏离 | +1   |

DFP 必须除以收盘价归一化，否则铜和玻璃差几个数量级，等权组合会被高价品种主导。

#### 2.2 时间戳族

日内极值出现的**时刻**本身就是因子，且**完全不依赖 (N, M)**，所以单独存一份。
周频书只用 `ts_high`（方向 −1）和 `ts_low`（方向 +1）。

时刻一律归一化为 `gamma_norm = (γ − 1) / (N_t − 1) ∈ [0, 1]`，
因为商品每天的 bar 数不一样（无夜盘 225、到 23:00 是 345、到 01:00 是 465、
到 02:30 是 555）。不归一化就没法跨品种等权。

#### 2.3 商品相对论文的三处扩展

1. **夜盘**。论文的国债只有"午前/午后"，商品要扩成"夜盘/午前/午后"三段。
   `trading_date` 是唯一合法的"日"定义：21:00 的夜盘 bar 属于**次一**交易日。
2. **无夜盘品种**（如 JD）的夜盘类因子必须是 **NaN，绝不是 0**。
   `config.NIGHT_DEPENDENT_FACTORS` 登记了全部 5 个这类因子。
   上一轮小时频就是被结构性零值把一个 t=3.05 的真因子压成了 t=1.1。
3. **时序而非截面**。论文做单标的择时，这里 75 个品种各自出多空信号，再等权合成。

---

### 3. 时间轴

| 区间               | 用途           | 允许做什么                                                                        |
| ------------------ | -------------- | --------------------------------------------------------------------------------- |
| 2010-01 .. 2014-12 | 向后时间外检验 | **只看 IC 符号**，不得选参、不得据此调整任何设定                            |
| 2014-07 .. 2015-12 | 预热           | 阈值（N 最长 300 日）+ 信号分位轨（W 最长 60 日）双层滚动的启动期，不计绩效       |
| 2016-01 .. 2021-12 | 研究期         | 全部可用。6 折 walk-forward：训练 3 年 / 测试 1 年 / 步进 1 年，测试年 2016…2021 |
| 2022-01 .. 2022-12 | 一次性验证     | 冻结策略后运行一次，不调参                                                        |
| 2023-01 起         | 严格 OOS       | 策略冻结前不读取、不统计、不调参                                                  |

2016 和 2017 这两折的训练窗口落在 2013-2015，那时夜盘尚未全面铺开，
`config.WF_FOLDS_WITH_SPARSE_NIGHT` 记着这件事，最终报告里必须标注。

收益口径（第 8.3 节）：`day_ret[t] = (openw[t+1] − openw[t]) / open[t]`，
即**今天收盘出信号、明天开盘进、后天开盘出**。链条是
`forward_return = day_ret.shift(-1)`、`execute_position = signal.shift(1)`，
两处滞后都不许省。

---

## 三、操作步骤

### 4. 全流程一览

主流程 7 个脚本，编号 1–7 连续、不重复。**必须按顺序跑**，
每一步都靠前一步的落盘产物。退出码约定：`0` 通过，`1` 验收不过或被闸门拒绝（要处理），
`2` 前置条件不满足（通常是上一步还没跑）。
终端只报是否通过、结果文件路径、能不能进下一步；表格和逐品种明细都在落盘文件里，不要从终端抄数字。
`--list-pools` / `--list-factors` 仍会把目录打到终端。

| 步 | 脚本                           | 做什么                                                  | 主要产物                                                                 |
| -- | ------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1  | `step1_shard_minutes.py`     | 切分片，落盘后立刻验收                                  | `data/minute_shards/`、`data/roll_dates/`                            |
| 2  | `step2_universe.py`          | 时点有效品种池                                          | `data/universe/universe_by_year.json`                                  |
| 3  | `step3_build_factors.py`     | 构造**全部**因子：分钟级、外部数据、日频装配，写因子目录 | `data/factor_daily/`、`data/research/factor_catalog.csv`             |
| 4  | `step4_factor_ic.py`         | 符号闸门、全部因子 IC、逻辑组合、稳定性与板块异质性     | `ic_by_fold.csv`、`factor_ic_all.csv`、`sector_factor_ic.csv` 等 |
| 5  | `step5_backtest_research.py` | 研究期回测，自选因子和板块                              | `data/research/backtest_research.csv`                                  |
| 6  | `step6_validate_2022.py`     | 2022 验证期测试，自选因子和板块，每本书只测一次         | `data/validation_2022/ledger.jsonl`                                    |
| 7  | `step7_oos_test.py`          | 严格样本外测试，验证未通过或测试期未结束则拒绝          | `data/oos/ledger.jsonl`                                                |

日内开盘区间突破（ORB）元标签是另一条独立的线，只依赖第 1–3 步的产物：
`research_orb_ml.py` 做研究期模型比较，`validate_orb_ml.py` 执行 2022 / 样本外流程，见第 5 节末尾。
`research_intraday_slate.py` / `validate_intraday_slate.py` 是日内多策略清单的研究与一次性检验，见第 5 节末尾。
第 6、7 步、`validate_orb_ml.py` 和 `validate_intraday_slate.py` 每次写台账后，会同步刷新 `docs/ResearchNotes.md` 末尾的登记簿。

以后新增因子一律在第 3 步构造，不再另开脚本：分钟级因子写进 `factors/intraday.py`
（落盘在 `factors/cache.py`），外部数据因子写进 `factors/external.py`，日频量价/慢信号写进
`factors/daily.py` 并在 `factors/library.py::assemble` 登记（已定向的在 `SIGNED_PRIORS` 登记先验符号）。
三处的因子都会自动进入因子目录，第 5–7 步按名字直接选用。

第 5–7 步共用 `research/backtest/strategy.py`：同一套因子写法、板块划分、执行口径（五批错开 + 波动率目标）和配置指纹。

包结构：

| 层 | 模块 | 内容 |
| --- | --- | --- |
| `data` | `shard_io`、`bars`、`sessions`、`universe`、`sectors` | 分片读写与守卫、日线与收益口径、日内时段、时点品种池、五大板块 |
| `factors` | `intraday`、`daily`、`external`、`cache`、`library` | 分钟级时间因子、日频指标与慢变量、外部数据因子、缓存、因子库装配 |
| `research/analysis` | `stats`、`screen` | 时序 IC 与绩效、板块异质性筛选 |
| `research/backtest` | `engine`、`costs`、`strategy` | 调仓成交扣费等权、tick 滑点、自选因子与板块的书 |
| `research/workflow` | `context`、`history`、`ledger` | 运行留痕与上下文、验证期/样本外数据准备、使用台账 |

#### 外部数据下载（可选）

外部数据下载与主流程隔离，按研究期、2022 验证期、2023+ 严格 OOS 分目录缓存，
默认品种池与 `config.COMMODITY_SYMBOLS` 取交集。原始接口响应位于
`data/external_rqdata/`。当前 RQData 服务不支持会员排名、near-main roll yield 和
front/next-month 连续合约；社会库存/产业利润也未接入。

账号写在 `config/rqdata.env`（已 git 忽略）。
`RQDATAC_LICENSE` 和用户名密码二选一，已 export 的环境变量优先。

```bash
conda run -n gu python download/getRiceQuantExternalData.py \
   --start-date 20100101 --end-date 20211231 --symbols all
```

下载完由第 3 步构造外部因子（默认研究期和 2022 两个分区）。

#### 板块

商品期货分五大类，定义在 `data/sectors.py`，每个品种只属于一类，
`--list-pools` 可随时打印：

| 板块     | 说明                                                                     |
| -------- | ------------------------------------------------------------------------ |
| 有色金属 | 铜铝锌铅镍锡、国际铜、氧化铝、铸造铝，以及碳酸锂、工业硅、多晶硅        |
| 黑色金属 | 螺纹、热卷、铁矿、焦煤焦炭、硅铁锰硅、不锈钢、线材、动力煤              |
| 贵金属   | 黄金、白银                                                               |
| 能源化工 | 原油、燃油、沥青、LPG、聚酯链、烯烃、甲醇、橡胶、尿素，以及玻璃、纯碱、集运指数 |
| 农产品   | 油脂油料、谷物、软商品、养殖、果品，以及纸浆、原木、胶合板、纤维板      |

有判断成分的归类：玻璃、纯碱放能源化工（化工-建材链，与煤化工共振），
林产品放农产品，集运指数 EC 不是实物商品、暂放能源化工。要改就改这一张表，
第 4 步异质性和第 5–7 步的池子会同时生效。另有特殊池 `全部`，即不分板块。

每一步都会在 `runs/{时间戳}_step{N}/` 下留一份快照（产物 + `params.json`/`manifest.json`），
`data/research/` 下是下游接着读的**最新**一份。想复盘某次运行看 `runs/`，
想接着跑看 `data/research/`。

---

### 5. 逐步操作

以下把工作目录设为仓库根 `D:\liushengqi\time_factor_cta`，并且已经
`conda activate factor-mining`。

#### 第 1 步　切分片并验收

```bash
python scripts/step1_shard_minutes.py
```

默认切 75 个商品品种。先用 `--dry-run` 空跑一遍确认面板结构和品种齐全，再正式跑。

| 参数 | 含义 |
| --- | --- |
| `--symbols RB CU ...` | 只切这些品种；缺省为全部商品期货 |
| `--all` | 连 8 个金融期货一起切（共 83 个），本实验不需要 |
| `--dry-run` | 只读单体文件、报告结构，不落盘、不验收 |
| `--monolith <路径>` | 分钟单体文件，缺省 `config.MINUTE_MONOLITH` |
| `--format auto\|parquet\|pickle` | 分片格式；`auto` 有 pyarrow 用 parquet，否则 pickle |
| `--validation-from-monolith` | 从 `--monolith` 按 `trading_date` 取 2022 年，只写 `validation_2022/` |
| `--validation-source <路径>` | 独立的 2022-only 单体 pickle，只写 `validation_2022/` |

```bash
python scripts/step1_shard_minutes.py --dry-run
python scripts/step1_shard_minutes.py
python scripts/step1_shard_minutes.py --symbols RB CU --dry-run   # 只看两个品种
```

做了四件事：按品种取出 `FACTOR_FIELDS + PRICE_FIELDS` 八列、
把 `trading_date` 规范化、丢掉全字段皆空的行（未上市期）、
按 `HOLDOUT_START` 切成两份分别落盘。顺带从 `dominant_id` 的跳变提取换月日
写进 `data/roll_dates/{品种}.csv`，提完就把这个 object 列丢掉（省内存的关键）。

**这是全仓库唯一允许写 `holdout_locked/` 的地方。写完就当它不存在。**

注意两条警告：某品种缺 `open`/`openw` 时仍然会落盘（因子能算），
但第 8.3 节的 `day_ret` 算不出来，脚本末尾会把这些品种列出来；
分片一旦落盘就补不回这两列，所以发现了要在这一步解决。

落盘结束后会立刻跑验收：时间边界、夜盘归属不过就停；bar 数、换月日、清单有问题也停。dry-run 不验收。全量分片时验收全部品种。逐项结果写在当次 `runs/*_step1/verify.csv`。

**2022 验证分片另走一条路。** 上面的切分只写 `research/` 和 `holdout_locked/`，不写 `validation_2022/`。
默认从同一个单体文件按 `trading_date` 取 2022 年（复权基准与研究期天然一致）：

```bash
python scripts/step1_shard_minutes.py --validation-from-monolith --dry-run
python scripts/step1_shard_minutes.py --validation-from-monolith
```

也可以给一份独立导出的 2022-only 源文件（同格式：宽面板 pickle，列为 品种 × 字段）：
`--validation-source data/data_min/<2022 文件>`。两种模式的差别只在第一条：

- 独立源文件整份检查，不做过滤：墙钟时间戳超出 [2021-12-31, 2023-01-01)、或任何 `trading_date` 不在 2022，整份拒收。
  2021-12-31 的夜盘归属 2022-01-04，导出时要包含它，否则 2022 第一个交易日缺夜盘。
- 复权衔接：逐品种比较研究期最后一根与 2022 第一根的 `closew − close`。
  同一主力下这个偏移不变；个别品种不等按边界换月处理并列出，超过两成不等说明复权基准变了，整份拒收。
  基准不同会让 step6 在跨年那一天凭空多一笔收益。
- 全部检查通过才落盘；`validation_2022/` 已有分片时不覆盖。落盘后按验证 loader 读回，
  逐品种查日期、夜盘归属、bar 数、NaN。

#### 第 2 步　品种池

```bash
python scripts/step2_universe.py
```

逐年筛：日均成交额 ≥ 30 亿、当年 `closew` 非空的交易日占比 ≥ 90%、
有效交易日 ≥ 200。金融期货在 `config` 层面就已剔除。

| 参数 | 含义 |
| --- | --- |
| `--symbols ...` | 只筛这些品种；缺省为全部商品期货 |
| `--lookback N` | 回看年数，缺省 `C.UNIVERSE_LOOKBACK_YEARS = 1` |
| `--min-turnover-yi X` | 日均成交额门槛（亿元），缺省 30 |
| `--quiet` | 不打印逐年进出池 |

主结论一律用缺省参数；门槛是写死的规矩（见第 7 节），改了就要同步改设计文档。

关键是**时点有效**：第 y 年的池子只用 `[y − 1, y − 1]` 的统计量判定，
绝不用当年或以后的数据。回看窗口取 1 年而不是 3 年是刻意的——
ZC（动力煤）这类僵尸品种的成交额是断崖式塌缩，回看越长，塌缩之后还被留在池里的
年数越多。代价是新上市品种要晚一年进池，这个方向的保守可以接受。

产物除了 `universe_by_year.json`，还有 `turnover_trajectory.csv`（逐年成交额轨迹）
和 `entries_exits.csv`（进出池记录）。**跑完看一眼 `entries_exits.csv`**：
如果某年进出池品种数暴增，通常是成交额单位或字段出了问题，不是市场变了。

`universe_fixed.txt` 是研究期最后一年（2021）的池子，只给冒烟测试用，
**不要拿它跑主结论**——用研究期末的池子回测全研究期就是前视。主结论一律用逐年池。

#### 第 3 步　持续期抽查并计算因子

```bash
python scripts/step3_build_factors.py
```

先对 RB、CU、M 做持续期形态抽查，不过就停止，不会进入因子计算。

这一步不产生因子，只回答"持续期算出来的东西像不像持续期"。
默认抽查 3 个品种（RB、CU、M）× 1 列（`closew`）× 3 个年份
（2016、2018、2021），用 N=250、M=55。抽查范围写死在脚本的 `PROBE_SYMBOLS` / `PROBE_YEARS`，
`--symbols` 不影响抽查，只影响后面的因子计算。

验收标准只有一条：**`p95 / p50 ≥ 3`** 且非全 NaN。
持续期的分布本来就是右偏的长尾——大部分 bar 一两分钟就突破阈值，
少数几根能拖很久，而正是那几根携带信息。偏度塌了说明阈值算错了。

**刻意没有设"中位数持续期 ≥ 10 分钟"这类门槛**，原因见设计文档 §6.2.1：
商品价格是离散到 tick 的，当某品种"零变动分钟"的占比超过 M%，
第 M 分位数就直接落在 1 个 tick 上，阈值**退化**。
此时价格持续期的实际含义变成"距上次 tick 跳动几分钟"——
它仍然是一个有效的、甚至更干净的信息变量，但**经济解释变了**，
所以抽查表里会记下"阈值退化样本占比"，超过一半时要在报告里写明。
这件事必须写进最终报告，也意味着对 M 的敏感性会集中在高分位一侧
（要到 60% 才开始脱离退化）。成交量持续期不受影响。

脚本内部有个容易做错且做错也不报错的地方：阈值要 N 个交易日预热，
所以每次都多读两年数据，**汇总时必须把预热年排除**（`keep` 掩码）。
混进去的话，预热年那些 NaN 会把缺失率稀释，看起来更干净，实则是错的。
预热排除由第 3 步抽查脚本按 `keep` 掩码执行；运行时应确认输出统计仅覆盖指定抽查年份。

#### 因子计算

```bash
python scripts/step3_build_factors.py
```

品种池 × 一组阈值（N=250、M=55）+ 1 份时间戳族。落盘结构：

```
data/factor_daily/
  timestamp/{品种}.parquet        # ts_high、ts_low，不依赖 (N, M)
  N250_M55/{品种}.parquet         # dfp_max、dfp_top3
```

每个品种的分钟数据**只读一次**，日内坐标**只算一次**。
若一次算多组 (N, M)，共用一个 `rolling_threshold_grid`。
`tests/factors/test_factor_cache.py` 证明网格结果与逐个调用逐元素相同。

**按 (组合, 品种) 粒度可续跑**：已存在的文件直接跳过，中断后重跑即可。
要强制重算加 `--overwrite`。只跑一部分用
`--combos N250_M55 --symbols RB CU`。

跑完自动做两项验收：

1. **因子健康度**（默认用中间组合 N250_M55，可用 `--health-combo` 改）：
   每个因子非空率 > 90%、不能全为同一个值、时点类因子必须落在 [0, 1]。
   夜盘类因子的缺失率**按有无夜盘分组统计**——不分组的话，
   一个完全正确的结果会因为混了无夜盘品种而被判不合格，
   进而诱导出 `fillna(0)` 这个致命修法。
2. **夜盘因子反向检查**：在无夜盘的 (品种, 交易日) 上，5 个
   `NIGHT_DEPENDENT_FACTORS` 必须**全是 NaN 且 0 值个数为 0**。
   出现任何非 NaN 值就返回 1，必须先修掉。

   这里的口径曾经错过一次，值得记住：**"有没有夜盘"不是品种属性，是挂牌历史事实**。
   2014 年之前整个市场都没有夜盘；C / CS / FU / L / PP / V 是 2019 年才有的；
   JD / AP 至今没有。所以判定必须逐 (品种, **交易日**) 做，
   按品种整体判会让 2019 年之前的那些年悄悄算出非 NaN 值。

产物快照在 `runs/*_step3/`，包括 `duration_probe.csv`（持续期抽查）、
`factor_health.csv`（因子健康度）、`build_log.csv`（品种构建/跳过状态）和 `manifest.json`。

#### 外部因子与因子目录

分钟级因子之后，同一个脚本接着做两件事：

1. **外部因子**：从 `data/external_rqdata/` 构造期限结构、仓单和主力持仓，
   按分区写进 `data/factor_daily/external/{research,validation_2022,holdout_locked}/`。
   默认只构造 `research` 和 `validation_2022`。样本外分区必须显式要求，并给出**已经过去**的截止日，
   日历只构造到那一天：

   ```bash
   python scripts/step3_build_factors.py --skip-minute \
       --external-partitions holdout_locked --oos-end 2025-12-31
   ```

   截止日未过、或没给截止日，脚本直接拒绝。只有第 7 步要用外部因子时才需要这一步。
2. **因子目录** `data/research/factor_catalog.csv`：装配研究期全部因子，列出来源
   （分钟缓存 / 外部数据 / 日频装配）、是否已定向、先验符号、第 5–7 步的写法、
   2016–2021 池内覆盖率和首个有效日期。第 5–7 步的 `--list-factors` 打印的就是这张表。

第 3 步全部参数：

| 参数 | 含义 |
| --- | --- |
| `--symbols ...` | 只算这些品种（分钟级、外部因子和因子目录都受影响）；缺省为品种池内全部 |
| `--combos N250_M55 ...` | 持续期族的 (N, M) 组合；缺省只算周频书用的 `N250_M55` |
| `--overwrite` | 已有缓存也重算；缺省按 (组合, 品种) 跳过已存在的文件 |
| `--no-timestamp` | 不重算时间戳族，只改了持续期族公式时配合 `--overwrite` 用 |
| `--health-combo N250_M55` | 健康度验收用哪个组合，缺省取中间那组 |
| `--skip-minute` | 跳过持续期抽查和分钟级因子，只重建外部因子和因子目录 |
| `--external-partitions research validation_2022 holdout_locked` | 外部因子分区，缺省前两个；`holdout_locked` 必须同时给 `--oos-end` |
| `--oos-end YYYY-MM-DD` | 样本外外部因子的截止日，必须已经过去 |
| `--no-external` | 跳过外部因子 |
| `--no-catalog` | 跳过因子目录 |

```bash
python scripts/step3_build_factors.py                                   # 全量
python scripts/step3_build_factors.py --symbols RB CU --combos N250_M55 # 小范围试跑
python scripts/step3_build_factors.py --overwrite --no-timestamp        # 只重算持续期族
python scripts/step3_build_factors.py --skip-minute --no-external       # 只重建因子目录
```

分钟级健康度、外部因子任何一项不过，脚本最终返回 1。

#### 第 4 步　单因子 IC、符号闸门与逻辑组合

```bash
python scripts/step4_factor_ic.py
python scripts/step4_factor_ic.py --symbols RB CU M --no-combos --no-heterogeneity   # 快速看符号闸门
```

| 参数 | 含义 |
| --- | --- |
| `--symbols ...` | 只用这些品种；缺省为品种池并集 |
| `--no-combos` | 跳过第 3 部分的逻辑组合回测 |
| `--no-heterogeneity` | 跳过第 4 部分的稳定性与板块异质性 |

符号闸门（第 1 部分）总会跑；缩小品种范围得到的 IC 只用来检查流程，不作结论。

四部分输出：

1. `ic_by_fold.csv`：有先验的四个时间戳/持续期因子逐折 IC，做符号闸门；
2. `factor_ic_all.csv`：全部因子（时间组合、量价、时序动量、carry、反转代理、
   无方向指标）的事前 IC 与逐年 `ic_ts`。量价因子方向取 `config.TECH_PRIOR_SIGNS`
   （趋势先验），`er / vol_ratio / atr_pct / pv_corr` 无方向，只报 IC；
3. `logic_combo_ic.csv`：按经济逻辑分组的等权组合，与第 5 步同一执行口径
   （五批错开、波动率目标）扣费回测，报年化收益、年化波动、收益风险比、最大回撤、换手，按 Sharpe 排序；
4. `factor_stability.csv`、`sector_factor_ic.csv`、`symbol_factor_ic.csv`：
   年度 IC 六年同号、|年均 IC| > 0.01、时序 |t| > 2 的稳定性筛选，以及通过筛选的因子
   （外加期限结构因子）在五大板块和单品种上的同一套检验。单品种至少 4 个有效年份。
   这部分只是候选线索，多重比较很重，不能据此直接缩池。

在 `N=250, M=55`（论文默认 N、M 区间中点）这**一组**参数上算时序 IC，
按 walk-forward 折分开，再给一行 `mean_of_folds`。

**IC 的收益口径是持仓期**：`ic_ts` / `t` 用未来 `C.IC_HORIZON = 5` 个交易日的累计持有收益
（`bars.holding_forward_return`），与书的持有期一致；次日收益 IC 另列 `ic_ts_1d` / `t_1d` 备查。
符号闸门、稳定性筛选都看 5 日口径。两个口径差别很大时说明信号衰减快：
时间因子和反转代理在 5 日口径下明显变弱，慢信号（tsmom、截面动量）反而更稳。
**这一步刻意不扫网格**——用 IC 挑参数就是选参。第 5 步用的就是这一组阈值和这里核过的符号。

**一折 = 一段时间 × 一个品种池**，两者都要切。这里曾经只切品种池、时间传全历史，
于是"逐折 IC"其实是同一个全样本 IC 的六个品种池变体，折间一致性好得离谱，
只切品种池、不切时间，向后窗口也会把研究期算进去。时序 t 值的前提就是一折是一段时间。

##### 显著性用时序 t，不用横截面 t

表里有两列 t，别读错：

| 列          | 口径                     | 回答的问题               |
| ----------- | ------------------------ | ------------------------ |
| `t`       | **时序**（主口径） | 这个因子在时间上稳不稳   |
| `t_cross` | 跨品种截面（仅留痕）     | 这个因子在多少品种上同向 |

`t_cross` 不能当显著性用：它的分母是品种间 IC 的标准误，品种越同质分母越小。
商品同期高度相关（同一波宏观冲击推动整个板块），照这个口径"再加一个高度相关的
品种"就能把 t 抬上去，这显然不是显著性。

时序 t 的算法：**先在品种内按月求 `mean(z_t · r̃_{t+1})`，再在月内跨品种平均**，
一个月只贡献一个观测（`IC_PERIOD='ME'`，月内至少 `IC_PERIOD_MIN_OBS=10` 个观测，
至少 `IC_PERIOD_MIN_COUNT=6` 个月才给 t）。`z` 是用截至当日的 252 日均值、标准差
标准化的因子，`r̃` 是除以事前 60 日波动的未来收益，所以 `ic_ts` 量级与相关系数可比。
顺序反了就等于把约 40 个相关品种当成独立样本。标准误用 **Newey-West**
（Bartlett 权重，截断滞后 `floor(4·(n/100)^(2/9))`，落在 `nw_lag` 列）。

**不要在月内算相关系数。** 旧口径就是月内 Spearman，它要在月内去均值；对 RSI、
均线乖离这类日间高度持续的因子，20 个观测的月内去均值有 Stambaugh 型负偏差——
纯随机游走上的 RSI 能得到 ic_ts≈−0.20、t≈−49。量价因子当初"t≈−30 的显著反转"
全部是这个假象。`tests/research/test_research.py::test_timeseries_ic_has_no_small_sample_bias_on_persistent_factor`
钉住了这一条。`ic` 列（逐品种全年 Spearman）对持续性因子同样有偏，只看方向时也要谨慎。

修正后四个时间因子的 t 在 2.3–3.4 之间，等权组合 t=3.8，六年全为正。

##### 符号闸门

闸门只对 `PRIOR_FACTORS`（有先验的那些，对照组不参与）生效，三档：

- `ok`：方向与先验一致。
- `flip`：方向相反**且时序 t 显著**（`|t| ≥ SIGN_T_MIN = 2.0`）→ 返回 1，
  打印 `flipped` 并停下。**不要翻符号**，回去查实现。
- `flip_weak`：方向相反但**测不出来** → 只提示，不阻断。

分开这两档是必要的：`pmt` 实测 IC=+0.0008、t=0.22，六折里四折反向两折同向，
量级全在 0.016 以内。判成 `flip` 等于宣称"发现了方向错误"并把人送去查实现，
而真实结论是"论文的这个因子没迁移到商品上"。`vmt`（t=1.19）同类。
不给 t 值时退回严格口径，防止有人忘了传 t 而悄悄放松闸门。

`ic_by_fold.csv` 里 `mean_of_folds` 那一行的 `n_symbols` 是**有效折的品种数均值**，
折数另有 `n_folds` 一列（逐折行这一格为空）。折之间品种池会变，
所以不要把它当成累加值读。

#### 第 5–7 步共用的配置

三个脚本接受同一组参数，也可以写进 JSON 文件用 `--config` 读入（命令行覆盖文件）。
下面是配置结构示例；仓库当前不附带 `config/strategy_example.json`：

```json
{
  "factors": {"tsmom": 1, "carry_roll": 1},
  "pools": [],
  "merge_pools": false,
  "symbols": null,
  "fee_rate": 0.00025,
  "slippage_ticks": 1.0,
  "tranches": 5,
  "vol_target": 0.20,
  "pass_criteria": {"min_net_sharpe": 0.5, "min_net_ann_return": 0.0}
}
```

| 参数                                   | 含义                                                                                     |
| -------------------------------------- | ---------------------------------------------------------------------------------------- |
| `--config <json>`                    | 读 JSON 配置；键名同上例，命令行给出的同名参数覆盖文件                                      |
| `--factors`                          | 等权因子。`名字` 取先验方向；`名字:+1` / `名字:-1` 把符号乘在**原始值**上（与冻结方案 JSON 同口径）。无方向因子必须写符号 |
| `--pools`                            | 板块，可多选，每个板块单独成一本书；`全部` 表示不分板块。缺省即 `全部`         |
| `--merge-pools`                      | 多个板块之外，再把它们合成一本                                                           |
| `--symbols`                          | 与板块取交集；不给板块时这些品种合成一本                                                 |
| `--fee-rate`、`--slippage-ticks`    | 手续费率、滑点 tick 数，缺省 0.00025 与 1                                                |
| `--tranches`                         | 错开调仓批数，缺省 5（每批持有 5 个交易日）；`0` = 旧的每周最后一个交易日单批调仓       |
| `--vol-target`                       | 单品种年化波动目标，缺省 0.20，杠杆上限 `C.VOL_TARGET_CAP = 2.5`；`0` = 不缩放            |
| `--min-sharpe`、`--min-ann-return`  | 第 5、6 步同一套门槛：扣费后算术 Sharpe ≥ 0.5 且扣费后年化 > 0。Sharpe = 日均收益 / 日波动 × √252，不是几何年化 / 波动 |
| `--list-pools`、`--list-factors`    | 打印板块分类 / 第 3 步因子目录后退出                                                     |

写法示例（三个脚本通用）：

```bash
# 先看有哪些因子、哪些已定向、写法是什么
python scripts/step5_backtest_research.py --list-factors
python scripts/step5_backtest_research.py --list-pools
# 已定向因子写名字即可；无方向因子（如 er、warehouse_low）必须带符号
--factors time_combo neg_clv carry_ms:+1 er:-1
# 两个板块各成一本，再合成第三本；板块名可用简称（黑色 / 能化 / 农产品 ...）
--pools 黑色金属 农产品 --merge-pools
# 只在几个品种上合成一本
--symbols RB HC I J JM
# 旧口径对照：周五单批、不做波动率缩放
--tranches 0 --vol-target 0
```

普通因子乘上符号后按 252 日窗口（至少 120 个有效值）做事前 z 分数，等权后截到 [−1, 1]。
`time_combo` 和 `time_combo_trend` 已经标准化，入书时不再标准化第二次。
`time_combo_trend` = `time_combo × 1{sign(time_combo) = sign(tsmom)}`：只保留与 tsmom 同向的
时间因子观点，反向记 0、任一缺失记 NaN；同样不再标准化。它是独立因子名，book_key 与 `time_combo` 不同。

执行链 `strategy.book_position`：收盘信号 → **波动率缩放**（× 目标波动 / 事前 60 日年化波动，
缺省目标 20%，杠杆截到 2.5；波动为 0 或缺失则该格 NaN）→ **按 `--tranches` 分批错开调仓**
（缺省 5 批，每批持有 5 个交易日；仓位取各批平均；某批调仓日信号缺失则该批持有期内空仓）→
次日开盘成交。当年池内等权，扣手续费和 tick 滑点。研究期没过门槛的池子，第 6 步不测。

为什么不用旧的周五单批：同一信号换成周一到周四调仓，研究期毛 Sharpe 从 1.14 掉到 −0.31～0.57，
五个交易日相位都在 0.2～0.6。旧口径的收益大半押在"周五收盘取信号"这个日历相位上，
错开调仓把这部分相位运气平均掉。

**配置指纹**：等效因子权重、池子、品种过滤、费率、滑点、阈值参数、调仓批数、波动率目标
共同决定一本书的指纹。`名字` 和 `名字:先验符号` 算同一本；`time_combo:+1` 展开成四个时间因子
各 1/4 的等效权重，与"四个时间因子等权"是同一本；合并池按板块名排序（`黑色金属+农产品`
与 `农产品+黑色金属` 相同）。通过门槛不进指纹，测过之后放宽门槛也不能重测同一本书。

**book_key**：只含等效因子权重 + 池 + 品种，不含成本和执行口径。第 6 步按 book_key 拦截：
同一组因子在同一池子上看过 2022 后，换调仓节奏、波动率目标或成本也不能再验证。

#### 第 5 步　研究期回测

```bash
python scripts/step5_backtest_research.py                      # 默认四个时间因子，5 批错开、不分板块
python scripts/step5_backtest_research.py --factors tsmom carry_roll --tranches 10
python scripts/step5_backtest_research.py --factors cs_mom_ra_250 --tranches 20
python scripts/step5_backtest_research.py --factors time_combo neg_clv \
  --pools 黑色金属 农产品 --merge-pools
python scripts/step5_backtest_research.py --config <研究配置.json>
```

默认因子取 `config.FACTOR_SIGNS`（四个时间戳/持续期因子），默认 5 批错开、单品种年化波动目标 20%。
默认值只是基线，不代表已选定的最佳策略。研究期门槛通过只表示可作为研究候选，不表示已验证或可以直接进入第 6 步。
拼接 2016–2021 六个 walk-forward 测试年，
每个池子输出整段一行（毛/净年化、Sharpe、最大回撤、换手、合成信号的时序 IC 与 t）和逐年各一行。
结果写 `data/research/backtest_research.csv`，快照在 `runs/*_step5_backtest/`。

有池子过门槛返回 0，全部没过返回 1。
这一步只用研究期，不读取验证/OOS 分区；但反复按收益挑因子、板块、持有期或成本仍会过拟合研究期。
执行 step6 前先冻结全部配置。

#### 第 6 步　2022 验证期测试

```bash
python scripts/step6_validate_2022.py --factors time_combo neg_clv neg_ret_day \
    --pools 黑色金属 农产品
```

只读 `research/` 和 `validation_2022/` 分片，2023 年及以后不在输入路径上。
2022 的品种池用第 2 步的研究期统计按同一口径筛出（第 y 年只用第 y−1 年）。
时间因子在研究期 + 2022 的分钟数据上用同一套公式重算，外部因子拼接研究期和 2022 两个分区，
滚动标准化因此是连续预热的。滑点按加载的分钟数据逐年估 tick（含 2022）。

研究期整段没过同一套门槛的池子，这一步直接拒绝，不读 2022 分片、不写入验证台账。
**每本书在 2022 上只测一次。** 结果与是否通过写进 `data/validation_2022/ledger.jsonl`，
同一指纹或同一 book_key 再跑会直接拒绝。
有池子通过返回 0；没有可运行的候选时返回 1；前置条件或数据缺失时返回 2。
写完台账会刷新 `docs/ResearchNotes.md` 末尾的登记簿（标记之间的内容自动生成，不要手改）。

第 6 步的参数就是上面第 5–7 步共用的那一组，**必须与第 5 步研究时用的完全一致**，
最稳妥是把配置写成 JSON，第 5、6、7 步都用 `--config` 读同一份。

#### 第 7 步　严格样本外测试

```bash
python scripts/step7_oos_test.py --factors time_combo neg_clv neg_ret_day \
    --pools 农产品 --oos-end 2025-12-31
```

两道闸门，任何一道不过都不读样本外数据：

1. **测试期必须已经结束**：`--oos-end`（缺省 2025-12-31）必须早于今天；
2. **这本书必须通过了 2022 验证**：同一指纹在验证台账里有记录，且用台账里的指标
   按**默认门槛**（Sharpe ≥ 0.5、年化 > 0）重算仍通过；配置文件里放宽的门槛不算数。没验证过、没通过的池子逐个拒绝，全部被拒就直接退出。

同一本书在样本外只测一次：同一指纹在 `data/oos/ledger.jsonl` 里有任何记录就拒绝，
换 `--oos-end` 也不行（否则可以逐年延长窗口反复看）。
样本外窗口从 2023-01-01 起。各年品种池用 2022 验证分片和样本外分片按第 2 步口径逐年重筛，
新上市品种（如碳酸锂、工业硅）满一年统计后才进池。选了外部因子时，
先按第 3 步的说明构造 `holdout_locked` 分区。

**价格口径**：分片的 `closew` 是加法复权（`close − closew` 日内恒定、只在换月日跳变），
比例类指标必须先经 `data/bars.py::multiplicative_prices` 转换，收益写成 `Δclosew / close`；
DFP 的持续期用 `closew`，FP 与分母用 `close`。

参数：第 5–7 步共用的那一组，外加 `--oos-end YYYY-MM-DD`（缺省 `C.DEFAULT_OOS_END`，必须已经过去）。
通过的书写进样本外台账后同样刷新登记簿。有池子测完返回 0，闸门拒绝返回 1 或 2。
样本外结果只用来报告，不许据此回头挑策略或调参。

#### 日内 ORB 元标签（独立流程）

与第 5–7 步的日频书分开，只依赖第 1–3 步的产物（研究期 / 2022 / 样本外分钟分片、tick 表、日频因子缓存）。
这里只讲口径和怎么跑。

- 候选交易：交易日最前面 30 根 1 分钟 K 线做开盘区间，之后突破入场，当日最后一根按收盘价平；
  多空两条单边路径分别撮合。`orb30|eod` 从交易日第一根算起（有夜盘的品种含夜盘），
  `orb30day|eod` 从日盘第一根算起。
- 元标签：目标 = 主口径单笔净收益 / ATR%，截在 ±5；模型预测值 > 0 才做这笔。
- 主口径：历史交易所手续费（经纪商加收）+ 每边 1 tick。

**研究期网格**（只读 2016–2021，不读取验证期；重复网格仍会增加研究期选择偏差）：

```bash
python scripts/research_orb_ml.py                                        # 全网格
python scripts/research_orb_ml.py --models ridge10 huber --groups core core+carry+time
python scripts/research_orb_ml.py --bases "orb30|eod" --rebuild          # 重建候选交易缓存
```

| 参数 | 含义 |
| --- | --- |
| `--models ...` | 模型，缺省全部：`ridge1` `ridge10` `ridge100` `ridge1000`（岭回归四档正则）、`huber`、`logit`（按目标正负做逻辑回归）、`hgb`、`hgb_d2`（深度 2）、`rf`、`et`（树模型） |
| `--groups ...` | 特征组，缺省全部（见下表） |
| `--bases ...` | 开盘定义，缺省 `orb30\|eod` 和 `orb30day\|eod` 两个；含竖线，命令行要加引号 |
| `--rebuild` | 重建候选交易缓存 `data/research/orb_candidates.pkl`；改了撮合、成本或上下文特征后要加 |

| 特征组 | 内容 |
| --- | --- |
| `core` | 开盘区间本身：区间宽度 / ATR、跳空、入场位置、区间成交量、成本占 ATR 等 |
| `trend` | 滞后一日的 `tsmom`、`tsmom_20`、`cs_mom_ra_250`（按交易方向取号），及 `er`、`vol_ratio` |
| `carry` | 滞后一日的 `carry_ms`、`cs_carry_ms`（按交易方向取号） |
| `time` | 滞后一日的 `time_combo`、`neg_clv`、`neg_ret_day`（按交易方向取号） |
| 组合 | `carry`、`time`、`carry+time`、`core`、`core+trend`、`core+carry`、`core+time`、`core+trend+carry`、`core+carry+time`、`all`（四组全用） |

训练用预测年之前三年（不早于 2016），测试年 2019–2021，2018 只用于嵌套选择
（预测 T 年只按 2018..T−1 的走步结果挑配置）。产物在 `runs/*_orb_ml_research/`：
`grid.csv`（每格逐年与整段净 Sharpe、保留率）、`nested.csv`（嵌套选择）、`daily_net_top.csv`（前几格的日净收益）。

**2022 验证与样本外**（一本书 = 一个模型 + 一个特征组及方向范围，开盘定义固定 `orb30|eod`）：

```bash
python scripts/validate_orb_ml.py --model ridge10 --groups core                # 只测 2022
python scripts/validate_orb_ml.py --model ridge10 --groups core --long-only     # 事后探索变体，只测 2022
```

| 参数 | 含义 |
| --- | --- |
| `--model` | 必填，取值同上面的模型列表 |
| `--groups` | 必填，取值同上面的特征组列表（一次一个） |
| `--long-only` | 仅保留多头预测的探索变体；不允许继续读取 OOS |
| `--oos` | 读 2023–2025。要求 2022 已测过且通过（净 Sharpe ≥ 0.5 且净年化 > 0），沿用 2022 那次保存的模型，不重训 |

模型只用 2019–2021 的池内交易训练。指纹 = 规格串的 sha256 前 16 位，2022 结果写验证台账，
样本外写样本外台账，两处都会刷新登记簿。产物：`runs/*_orb_ml_validation2022/`
（`performance.csv`、`daily_net.csv`、`model.pkl`）和 `runs/*_orb_ml_oos/`。
退出码：0 跑完且通过（`--oos` 时为跑完），1 未通过 / 还没测 2022 / 2022 未通过，
2 这本已经测过。已测过哪些书看登记簿；同一本不能重测，换一个特征组或正则再测 2022 也要算进验证次数。

#### 日内多策略清单（独立流程）

十几条池级、多空对称的规则放在 `intraday/walk_forward.py` 的 `SLATE` 里，每条只有一小格参数：

- 日内：`orb30`、`orb30_trend`、`orb30_compress`、`dual_thrust`、`intraday_momentum`、`vwap_fade`、`gap_fade`；
- 隔夜：`overnight_momentum`；
- 多日持仓（小时或日线重采样）：`donchian_hourly`、`ema_hourly`、`tsmom_daily`、`bollinger_reversion`、
  `carry_daily`、`tsmom_carry_daily`。

所有书都按同一套逐腿记账（毛收益、滑点、手续费），与 ORB 元标签口径相同：主口径为历史手续费 + 每边 1 tick，
另报 slip2 / fee_x2 / fee_2026 / close_yday。

```bash
python scripts/research_intraday_slate.py [--rebuild] [--no-freeze]   # 研究期走步，只读 2016–2021
python scripts/validate_intraday_slate.py                             # 全部冻结书一次性测 2022
python scripts/validate_intraday_slate.py --oos                       # 2022 通过的书读 2023–2025
python scripts/validate_intraday_slate.py --void <书> --reason "..."   # 数据缺陷时作废 2022 记录
```

- 研究：每个测试年（2019–2021）用前三年的主口径 Sharpe 选格。冻结时用 2019–2021 选格，写进
  `config/intraday_slate.json`，文件已存在时不改写。组合书 `slate_combo` 由走步 Sharpe > 0 的书组成，
  按 2019–2021 波动的倒数定权。
- 检验：开跑前先重算 2019–2021 的 Sharpe，与冻结值相差超过 0.02 就拒绝。指纹由规格串算出。
- 作废：数据有缺陷时，比如外部分区没覆盖 2022，用 `--void` 追加作废记录，然后不带参数再跑一次。
  这一次只重跑作废的书，规格不变。作废清单列在登记簿末尾。
- 带 carry 的书要求外部分区覆盖检验窗口：
  - 2022 要先下载 `validation_2022` 的 roll_yield / dominant，再跑 `step3 --external-partitions validation_2022`；
  - 样本外同理，用 `holdout_locked` 加 `--oos-end`。

---

## 四、判读、禁区与排错

### 6. 怎么判断复现成功

按重要性排序，四条都要看。

**一、方向站得住。** 第 4 步四个因子的折间平均 IC 符号与 `FACTOR_SIGNS` 一致。
这条不过，第 5 步的绩效不必看。

**二、周频书扣费后仍为正。** 第 5 步 `backtest_research.csv` 整段那一行的扣费后 Sharpe 是这条策略的结果。

**三、跨年不塌。** 第 5 步逐年各行的 `net_sharpe` 里，
不能是"某一年赚光全部、其余五年平的"。尤其留意 2016、2017 两折
（训练窗口的夜盘稀疏），这两年特别好或特别差都要在报告里解释。

指标本身的量级不设硬门槛。商品时序 CTA 在 0.00025 费率下，
拼接测试年的收益风险比落在 0.5-1.0 是合理区间；
超过 1.5 先别高兴，回去查有没有前视（最常见的两处：
分位轨忘了 `shift(1)`、以及 `execute_position` 少了一层滞后）。

---

### 7. 不许做的事

一张清单，都是"做了不会报错但结论作废"的操作：

- 绕过研究/验证专用 loader 读取 `holdout_locked/`；在 OOS 策略冻结前读取或统计 2023 年及以后数据。
- 根据 2022 验证结果反复修改参数并重跑验证；手工删改 `ledger.jsonl` 来重测同一本书。
- 看了第 6 步哪个板块没通过，就把因子或板块微调成"新的一本书"再去验证。台账挡不住这种事，
  每一次验证都会留在台账里，报告时要如实给出一共验证了多少本书。
- 看了第 4 步的 IC 符号之后去改 `FACTOR_SIGNS`。
- 在第 5 步以外另开地方扫参数，或按收益风险比改成员和符号。
- 把无先验因子按研究期 IC 符号事后定向后放进主合成。
- 给夜盘类因子 `fillna(0)`。
- 把分位轨窗口改成含当日。
- 为了让某个验收通过而放宽门槛（`SKEW_MIN`、`MIN_NON_NULL`、
  `MIN_DAILY_TURNOVER` 都是写死的；要改必须同步改设计文档并说明依据）。

---

### 8. 从零重跑的完整序列

确认 `data/data_min/future_all1mdata_20100101-20251231.txt` 在位之后，
按顺序执行。每一步都要看退出码：非 0 就停下处理，不要往后跑。
这是从干净数据状态重建的示例，不是日常刷新命令：step1 会重写已有研究/锁定分片，后续步骤也会更新下游产物。
已有分片和缓存时不要盲目整段重跑；先备份需要保留的产物，只补跑确实缺失或因公式版本需要重建的步骤。
验证分片已有时，step1 的验证写入会拒绝覆盖；不要通过删除目录绕过。

```bash
conda activate factor-mining
python -m pytest
python scripts/step1_shard_minutes.py --dry-run
python scripts/step1_shard_minutes.py
python scripts/step1_shard_minutes.py --validation-from-monolith   # 2022 验证分片
python scripts/step2_universe.py
python scripts/step3_build_factors.py
python scripts/step4_factor_ic.py
python scripts/step5_backtest_research.py --factors <因子 ...>
# 配置冻结后才跑 step6，2022 通过的书才跑 step7
python scripts/step6_validate_2022.py --config <冻结配置.json>
python scripts/step7_oos_test.py --config <冻结配置.json>
# 日内 ORB 元标签（独立于第 4–7 步）
python scripts/research_orb_ml.py
python scripts/validate_orb_ml.py --model <模型> --groups <特征组>
# 日内多策略清单
python scripts/research_intraday_slate.py
python scripts/validate_intraday_slate.py
```

想在合成数据上先把整条管道跑通一遍而不碰真实分片，设
`TFCTA_DATA_ROOT` 指向一个临时目录即可（`src/tfcta/data/synth.py`
能造出结构与真实面板一致的分钟数据，测试就是用它）。生产运行不要设这个变量。

第 3 步最慢：品种数 × 一组阈值 × 每品种的分钟行数。第 5 步只读日频缓存，几秒钟。
第 6、7 步要在研究期 + 验证期（+ 样本外）的分钟数据上重算时间因子，全部品种约一两分钟。
第 3 步支持 `--symbols` / `--combos` / `--skip-minute`，第 4 步支持 `--symbols` / `--no-combos` /
`--no-heterogeneity`，可先缩小范围验证流程。

---

### 9. 排错

| 症状                           | 原因与处置                                                                                    |
| ------------------------------ | --------------------------------------------------------------------------------------------- |
| 敲`python` 无输出、退出码 49 | 还没激活环境，命中了应用商店占位程序。先`conda activate factor-mining`                      |
| 任何脚本返回 2，提示"分片为空" | 第 1 步还没跑，或`TFCTA_DATA_ROOT` 指错了地方                                               |
| 返回 2，提示"找不到品种池"     | 先跑`step2_universe.py`                                                                     |
| 返回 2，提示"请先运行 step3"   | 因子缓存缺对应的 (N, M) 目录。检查`data/factor_daily/` 下有没有 `N250_M55` 这类目录       |
| `HoldoutViolation`           | **这是保护，不是故障**。有代码在读 2022 年之后的数据，去找调用链，不要绕过它            |
| 落盘成了 pickle 而不是 parquet | 没装 pyarrow。装上体积更小、能按列读；不装也能跑                                              |
| 载入单体文件时内存爆           | `pickle.load` 无法分块。分批 `--symbols`，或换内存更大的机器                              |
| 第 5 步报缺少因子列            | 第 3 步的缓存里没有`ts_high` / `ts_low` / `dfp_max` / `dfp_top3`。补跑第 3 步         |
| 第 5–7 步报"没有事前方向"     | 选了无方向因子，写成`名字:+1` 或 `名字:-1`；`--list-factors` 看哪些已定向             |
| 第 6 步"已经测过" | 同一指纹或 book_key 已在 `data/validation_2022/ledger.jsonl` 里，这本书不能再测 2022 |
| 读因子缓存报版本不符           | 缓存由旧版公式生成（`cache.CACHE_VERSION`），`step3_build_factors.py --overwrite` 重建 |
| 第 7 步"没有做过 2022 验证"    | 第 7 步的参数必须与第 6 步完全一致（同一配置文件最稳妥），指纹才对得上                |
| 第 7 步"测试期尚未结束"        | `--oos-end` 不能是今天或以后                                                            |
| 读 CSV 中文乱码                | 产物一律`utf-8-sig`，用 Excel 直接打开即可；命令行看用 `iconv` 或直接 `pandas.read_csv` |

---

### 10. 已知的口径局限

不是 bug，是做了取舍的地方。记在这里免得下次重新发现一遍，也免得误读数字。

- **第 4 步要看的 t 是时序 t，不是 `t_cross`**（见第 4 步小节）。
  `t_cross` 的分母是品种之间 IC 的标准误，把相关性很高的商品当成了独立样本，
  只能用来排序，不能当显著性。
- **某品种某天 `day_ret` 是 NaN 但有成交时，成本照扣**，收益记 0，不随收益一起变成 NaN。
- **tick 滑点按上一年的收盘价中位数折算**（第 y 年用第 y−1 年，首年回填），
  不用当年价格，避免用到当年后段的价位。
- **波动率目标是单品种的**：信号截到 ±1 再乘 20%/σ，品种间不相关的部分互相抵消，
  组合年化波动只有 2%–6%，远低于 20%。比较书时看 Sharpe，不要直接比年化收益。
- **滑点按 1 个 tick 计，不是冲击成本模型**。同样穿一个 tick，低价位品种付出的
  比例远高于高价位品种（本样本约 0.7–9.9bp，中位 3.3bp）。第 5 步按 1 个 tick 计。
  日频分位轨换手很高时，这 1 个 tick 足以把毛收益为正的因子做成扣费后为负。
