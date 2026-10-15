# 商品期货时间因子 CTA 研究

从米筐分钟数据出发，研究商品期货的日频因子和日内规则，并在严格隔离的样本外上做最终检验。起点是兴业证券《基于高频时间维度的国债期货择时因子及个股 CTA 研究》（`docs/commodity/` 下的 PDF）：价格停留时间、高低点出现的先后能否提供可交易的信息。

当前结论（2026-10-11）：**还没有做过 2023–2025 的最终测试。** 唯一保留的策略是早盘元策略（主策略·亏损反手·30笔）。研究期夏普 1.15，2022 诊断夏普 0.56；这两段都不能代替样本外。其他方案（日频核心 + 早盘卫星、四条日频腿、方向判别、时间因子、机器学习、持仓量、降成本等）都已放弃并删除，已提交过的可从 git 历史找回。早盘主策略见 [MorningOverreactionReversal.md](docs/commodity/MorningOverreactionReversal.md)。

## 目录

```
config/rqdata.env                 米筐凭证（不入库）
config/commodity/                 商品策略与研究配置
config/index/                     股指期货配置
download/commodity/               商品分钟行情、外部数据、手续费
download/index/                   股指期货分钟行情
scripts/commodity/                商品流程脚本
scripts/index/                    股指复现脚本
src/tfcta/                        共用库：路径、分片、因子、回测、凭证
  index/gtja_time.py              国泰君安时间维度因子
tests/commodity/                  商品单元测试，只用合成数据
tests/index/                      股指单元测试
docs/commodity/                   商品研究文档
docs/index/                       股指研报
data/commodity/                   商品数据（不入库）
data/index/                       股指数据（不入库）
runs/commodity/                   商品研究产物（不入库）
runs/index/                       股指研究产物（不入库）
```

## 环境

```bash
conda activate factor-mining
```

```bash
pip install -r requirements.txt rqdatac tqdm
```

`rqdatac` 和 `tqdm` 只有下载脚本需要。米筐凭证写在 `config/rqdata.env`（已在 `.gitignore` 里），二选一：

```
RQDATAC_LICENSE=...
```

```
RQDATAC_USERNAME=...
RQDATAC_PASSWORD=...
```

所有命令都在仓库根目录运行。可选环境变量：`TFCTA_MINUTE_DIR`（原始分钟目录）、`TFCTA_DATA_ROOT`、`TFCTA_RUNS_ROOT`、`TFCTA_CONFIG_DIR`。

## 数据边界

| 分区 | 日期 | 用途 | 位置 |
|---|---|---|---|
| 预热 | 2016 年以前 | 只用于滚动窗口和 2016 年品种池，不计绩效 | `data/commodity/minute_shards/research/` |
| 研究期 | 2016–2021 | 选因子、定规则、定参数、定分配 | `data/commodity/minute_shards/research/` |
| 2022 | 2022 | 已被多次看过，只能作压力诊断 | `data/commodity/minute_shards/validation_2022/` |
| 严格样本外 | 2023–2025 | 只给冻结的策略跑最终测试 | `data/commodity/minute_shards/holdout_locked/` |

硬性规则：

- 研究接口读到 2022 年及以后的日期会抛 `HoldoutViolation`。2022 只能通过 `--validation-2022` 这类专门入口读取。
- 样本外分片、样本外外部数据、样本外手续费的读取接口全部加锁，年份结束也不会自动解锁。只有 `C.final_evaluation(...)` 块内才放行，而且读取不能越过登记的截止日。
- `final_evaluation` 的进入条件：配置里写明 `chosen` 和 `frozen_at`；测试窗口已结束；截止日在 2023-01-01..2025-12-31。进入时先往 `data/commodity/oos/ledger.jsonl` 追加一条 `opened` 记录（含配置文件 sha256），再读数据。台账只追加，不改不删。
- 当前早盘元策略还没有 2023–2025 的台账结果。这三年都可以计分。进入 `final_evaluation` 时才会往 `data/commodity/oos/ledger.jsonl` 追加记录。

## 全流程

### 0. 下载

分钟行情（主力连续 88、889、99 和价差 agio）。总表默认是 `data/commodity/data_min/future_all1mdata_20100101-20251231.txt`。下载 2026 年以后的数据时加 `--standalone`，不然会合并进总表并改名：

```bash
python download/commodity/getRiceQuantData_MinFreq.py -t 20100101-20251231
```

```bash
python download/commodity/getRiceQuantData_MinFreq.py -t 20260101-20261008 --standalone
```

外部数据会按日期自动分到研究期（`data/commodity/external_rqdata/`）、`validation_2022/` 和 `holdout_locked/` 三个目录，并按 `coverage.json` 增量补齐。加 `--dry-run` 只列出缺口：

```bash
python download/commodity/getRiceQuantExternalData.py --start-date 20100101 --end-date 20251231
```

只补某一类数据用 `--datasets`，例如 2022 年的仓单（2026-10-10 已补齐；2023-01-01 到 2026-10-08 此前已在 `holdout_locked/`）：

```bash
python download/commodity/getRiceQuantExternalData.py --start-date 20220101 --end-date 20221231 --datasets warehouse
```

历史手续费按分区下载。研究期需要先有品种池（第 2 步），样本外则直接下载所有有合约乘数的品种：

```bash
python download/commodity/getFeeHistory.py --partition research
```

```bash
python download/commodity/getFeeHistory.py --partition validation_2022
```

```bash
python download/commodity/getFeeHistory.py --partition oos
```

最小变动价位表按年从分钟原始收盘价估计（非零价格变动里占比至少 1% 的最小步长，一年不到 100 次变动的留空），第 y 年的值用于第 y+1 年。分片之后生成研究期和 2022 两张表（`data/commodity/research/tick_size.csv`、`data/commodity/validation_2022/tick_size.csv`），2023 年以后的在最终评估里现算：

```bash
python scripts/commodity/build_ticks.py
```

早盘特征构建时，池内品种在池内年份算不出成本（缺手续费或 tick）会直接报错。郑商所 2017-06-22 以前米筐没有平今费，按平仓费和之后第一次出现的平今费中较高的一个补。

### 1. 分片

把分钟总表按品种切开：2021 年及以前写 `research/`，2022 年及以后写 `holdout_locked/`（写完即锁）。同时提取换月日到 `data/commodity/roll_dates/`，并做验收（时段、夜盘、复权、换月）：

```bash
python scripts/commodity/step1_shard_minutes.py
```

2022 单独切一份，写 `validation_2022/`：

```bash
python scripts/commodity/step1_shard_minutes.py --validation-from-monolith
```

其他参数：`--symbols RB HC` 只切部分品种，`--all` 包含金融期货，`--dry-run` 只报告不落盘，`--format parquet|pickle` 指定分片格式。

股指期货分钟行情单独下载和预处理，不合并进商品池；当前准备 IF/IH/IC 的研究期数据与 OI 分片，且不触碰 2022 验证期和 2023 年后锁定期：

```bash
python download/index/getRiceQuantIndexFutures.py --symbols IF IH IC --start-date 20150101 --end-date 20211231
```

数据写入 `data/index/minute_shards/research/`。股指没有夜盘，相关研究须采用独立的日盘窗口和次日收益定义，不能套用商品的夜盘时钟。

若要预先下载但暂不查看股指样本外数据，单独运行：

```bash
python download/index/getRiceQuantIndexFutures.py --partition oos_locked
```

该命令将 IF/IH/IC 的 2022-01-01 至 2025-12-31，以及 IM 自 2022-07-22 上市日至 2025-12-31，写入 `data/index/minute_shards/holdout_locked/`。锁定目录受研究读取守卫保护；未冻结正式检验前不得载入或查看。

国泰君安时间维度研报的复现放在股指研究期上，不并进商品早盘。规则、阈值和 60 日滚动标准化写在 `config/index/gtja_time.json`，计算在 `src/tfcta/index/gtja_time.py`：

```bash
python scripts/index/gtja_time.py
```

结果在 `runs/index/gtja_time/`。绩效只到 2021-12-31，用来对照研报方法，不把研报里筛出来的组合当成新策略。

为检查连续合约走势和复权拼接，可生成 IF/IH/IC 的 2016–2021 复权日线 K 线图（不读取 2022 及以后）：

```bash
python scripts/index/plot_kline.py
```

图表写入 `runs/index/gtja_time/plots/`；日线 OHLC 由分钟分片中的复权价格聚合，成交量按交易日求和。

### 2. 品种池

第 y 年的品种池只用第 y−1 年的日均成交额和有效交易日决定，写到 `data/commodity/universe/universe_by_year.json`，报告在 `runs/commodity/<时间>_step2/`：

```bash
python scripts/commodity/step2_universe.py
```

门槛可以调，比如 `--min-turnover-yi 5`（单位亿元）、`--lookback 250`。2022 和样本外的品种池按同一规则在运行时计算，不落盘。

### 3. 因子构建

只用研究期数据，算两类：

- 分钟时间因子：持续期 DFP、高低点时间戳、报告类因子，写到 `data/commodity/factor_daily_v3/`。
- 期限结构：主力/次主力年化价差，写到 `data/commodity/factor_daily_v3/external/`。

```bash
python scripts/commodity/build_factors.py
```

`--symbols RB HC` 只算部分品种，`--overwrite` 重算已有缓存。日频价格类因子（tsmom、波动等）在研究时从日线现算，不需要缓存。

### 4. 因子研究

在 2016–2021 上跑 `config/commodity/research_candidates.json` 里预先写好的因子组合，只做评估，不自动挑选。每个候选有 `id`、`factors`（因子名 → 方向）、`days`（持有天数）、`hypothesis`。输出 IC、分年 IC、绩效、换仓相位敏感性和相关性，写到 `runs/commodity/<时间>_research_v3/`：

```bash
python scripts/commodity/research.py --specs config/commodity/research_candidates.json
```

`--slippage-ticks 2` 用 2 跳做压力测试，`--skip-ic` 跳过 IC 诊断。

### 5. 早盘策略

早盘策略统一从 `scripts/commodity/morning.py` 运行，配置在 `config/commodity/morning.json`，结果在 `runs/commodity/morning/`。v1 是冻结原策略；v2 是双端 1/5 时钟规则并保留夜盘同向；benchmark 只用低点时钟减高点时钟，绝对差阈值为 0.6。`--compare` 一次运行三者并输出全板块/三板块对比图。

先生成逐日特征表，决策时点是 09:16 收盘。不带参数时生成研究期和 2022 两份，写到 `runs/commodity/morning/features/`。样本外特征只能在 `C.final_evaluation(...)` 锁内生成：

```bash
python scripts/commodity/morning_features.py
```

```bash
python scripts/commodity/morning_features.py research
```

v1 是冻结原策略（30 笔亏损反手）；v2 使用高/低点前后 1/5 时钟规则并保留夜盘同向过滤。benchmark 不用偏离或夜盘方向，采用 `low_clock-high_clock` 的正负方向，只在绝对值大于等于 0.6 时交易，并沿用成本/量能等可交易过滤。运行研究期：

```bash
python scripts/commodity/morning.py --v1
python scripts/commodity/morning.py --v2
python scripts/commodity/morning.py --benchmark
python scripts/commodity/morning.py --compare
```

验证期可分别运行：
```bash
python scripts/commodity/morning.py --v1 --validation-2022
python scripts/commodity/morning.py --v2 --validation-2022
python scripts/commodity/morning.py --benchmark --validation-2022
python scripts/commodity/morning.py --compare --validation-2022
```

输出在 `runs/commodity/morning/morning_v1/`、`morning_v2/`、`morning_benchmark/`；三策略并列结果和图在 `runs/commodity/morning/comparison/`。`morning.py` 同时保留 OOS 所需兼容函数。

回看窗口的参数面（2026-10-10，样本外用完之后的探索）：亏损反手，10–90 笔或 20–120 个交易日，在 2016–2022 上算夏普、DSR（扣除尝试次数的夏普显著性，`stats.deflated_sharpe`）和 PBO（组合对称交叉验证的过拟合概率，`stats.pbo_cscv`）。网格写在同一份配置的 `grid` 里，输出在 `runs/commodity/morning/meta_grid/`：

```bash
python scripts/commodity/morning.py --grid
```

高低点出现位置和 09:17→11:30 收益的相关，只在 2016–2021 上算，并分板块做按日聚类的显著性检验。统一说明（含早期偏离样本结果）写在 `docs/commodity/高低点时钟的经济解释与下一步.md`，输出在 `runs/commodity/morning/clock_return/`：

```bash
python scripts/commodity/morning_clock_return.py
```

除了每年合并全部品种日的 pooled Spearman（`yearly.csv`、`plots/yearly.png`），还输出逐日截面 IC：每个交易日在品种之间算一次 Spearman（每天至少 5 个品种），再按年平均并做 Newey-West t 检验，去掉了“当天整个市场一起涨跌”的成分。文件是 `ic_daily.csv`、`ic_yearly.csv`、`plots/ic_yearly.png`、`plots/ic_cumulative.png` 和分板块的 `plots/ic_sectors/`。

加 `--balance` 先做涨跌各半重采样：去掉收益恰为 0 的品种日，在每个“年份 × 板块”里把涨、跌多的一边随机抽到和少的一边一样多（固定种子），再跑同一套分析，输出到 `runs/commodity/morning/clock_return_balanced/`；另换 200 个种子重复抽样，分布在 `balance_draws.csv`，每格抽样前后的笔数在 `balance_check.csv`：

```bash
python scripts/commodity/morning_clock_return.py --balance
```

平均路径、含成交量的典型正例/反例 K 线、时钟和价格位置的分解、年度异质性和事前状态分组，输出到 `runs/commodity/morning/clock_cases/`，解读和下一步见同一份统一说明：

```bash
python scripts/commodity/morning_clock_cases.py
```

### 统一运行与绘图

```bash
python scripts/commodity/run_strategy.py morning_v1 research
python scripts/commodity/run_strategy.py morning_v2 research
python scripts/commodity/run_strategy.py morning_benchmark research
python scripts/commodity/run_strategy.py morning_compare research
```

统一入口也可直接使用：

```bash
python scripts/commodity/morning.py --v1
python scripts/commodity/morning.py --v2
python scripts/commodity/morning.py --benchmark
python scripts/commodity/morning.py --compare
```

`--compare` 输出全板块和三板块两张净值/回撤图；单版本命令输出各自结果目录。验证期加 `--validation-2022`。图表均写到对应结果目录的 `plots/` 下。要按已有结果挑序列重画，用：

```bash
python scripts/commodity/plot_strategy.py morning_v1 research --series morning_v1
```

```bash
python scripts/commodity/plot_strategy.py oos 2023-2025
```

### 7. 样本外最终测试

口径写在 `config/commodity/oos_protocol.json`：可计分年份为 2023、2024、2025，这三年尚未跑过。主口径 1 跳，参考 2 跳和不切换的主策略。`--years` 中明确指定的年份计入绩效；运行较晚年份时，之前的样本外历史仍用于连续信号和滚动状态。先检查，这一步只重算研究期和 2022 并与已落盘结果逐日核对，不读样本外：

```bash
python scripts/commodity/oos_final.py --check
```

检查通过后正式运行，年份可选 `2023`、`2024`、`2025` 或相应连续区间，例如 `2023-2025`：

```bash
python scripts/commodity/oos_final.py --years 2023-2025 --rerun
```

脚本的执行顺序：

1. 确认配置已冻结、研究期和 2022 结果、样本外手续费都已就绪。
2. 写台账 `data/commodity/oos/ledger.jsonl`，打开 `C.final_evaluation(...)`。
3. 算样本外品种池，生成样本外早盘特征。
4. 研究期、2022、样本外接成连续序列（元策略的最近 30 笔跨年连续），核对研究期和 2022 部分与已落盘结果一致。
5. 输出到 `runs/commodity/oos/<年份>/`：summary.csv、yearly.csv、morning_trades.csv、daily.csv、plots/，并把结果追加到台账。中途出错也会在台账里记一条 aborted。

同一配置、同一年份已有结果时，必须加 `--rerun` 才会重跑。

## 回测假设

- 日频因子：收盘后才知道的信号在下一交易日开盘成交，收益为 `(openw[t+1] − openw[t]) / open[t]`（`openw` 是加法复权开盘价）。换月日按历史费率另计平旧开新的成本。
- 日内规则：决策 09:16 收盘，09:17 开盘进场，11:30 收盘离场。开仓用历史开仓费，平仓用平今费。
- 手续费：按金额收费的 ×1.01，按手数收费的 +0.01 元。滑点为每边 1 个上一年最小变动价位，压力口径 2 跳。
- 这是连续名义仓位的研究模型，没有整数手、保证金约束、涨跌停和盘口冲击。

## 新策略的流程

1. 在研究期提出假设并写成配置（`config/*.json`），写明 `declared_at`，配置必须先于结果。
2. 只在 2016–2021 上选规则、参数和分配，选择标准也事先写进配置。
3. 选定后写 `chosen` 和 `frozen_at`，此后不再改动。2022 只作诊断，不用它回头调参。
4. 写样本外口径文件和脚本，读数据部分包在 `C.final_evaluation(...)` 里（参考 `config/commodity/oos_protocol.json` 和 `scripts/commodity/oos_final.py`）。先确认台账里目标年份没有被用过。
5. 样本外只跑一次，结果不论好坏都留在台账里。

## 测试

```bash
python -m pytest
```

测试只用合成数据（`tfcta.data.make_symbol`），不读真实的样本外数据。覆盖范围包括：时段与分片、品种池、因子计算、回测成本、各分区的读取锁、`final_evaluation` 的门槛与台账、早盘规则、元策略账面与多腿缩放的防前视、样本外脚本的年份与台账。
