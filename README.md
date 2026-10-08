# 商品期货时间因子 CTA 研究

从米筐分钟数据出发，研究商品期货的日频因子和日内规则，并在严格隔离的样本外上做最终检验。起点是兴业证券《基于高频时间维度的国债期货择时因子及个股 CTA 研究》（`docs/` 下的 PDF）：价格停留时间、高低点出现的先后能否提供可交易的信息。

当前结论（2026-10-08）：**没有通过样本外检验、可实盘的策略。** 最近一个方案“日频核心 + 早盘卫星”研究期夏普 1.85，2023 严格样本外 −10.4%（夏普 −1.46），已放弃。详见 [CorePlusMorningPortfolio.md](docs/CorePlusMorningPortfolio.md) 和 [MorningOverreactionReversal.md](docs/MorningOverreactionReversal.md)。

## 目录

```
config/        策略与研究配置（json），米筐凭证 rqdata.env（不入库）
download/      米筐下载：分钟行情、外部数据（合约/主力/展期收益/仓单/现货）、手续费
scripts/       流程脚本，按下文顺序运行
src/tfcta/     库代码
  config.py    路径、时间边界、样本外锁和台账
  data.py      分钟分片读写、交易时段、日线、品种池、复权衔接检查、合成数据
  factors.py   分钟时间因子、日频因子、外部数据因子、因子缓存与装配
  research.py  成本、回测引擎、IC 与绩效统计、研究流程
  rq_auth.py   读取米筐凭证
tests/         单元测试，只用合成数据
docs/          研究文档
data/          数据（不入库）
runs/          研究产物（不入库）
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
| 预热 | 2016 年以前 | 只用于滚动窗口和 2016 年品种池，不计绩效 | `data/minute_shards/research/` |
| 研究期 | 2016–2021 | 选因子、定规则、定参数、定分配 | `data/minute_shards/research/` |
| 2022 | 2022 | 已被多次看过，只能作压力诊断 | `data/minute_shards/validation_2022/` |
| 严格样本外 | 2023–2025 | 只给冻结的策略跑最终测试 | `data/minute_shards/holdout_locked/` |

硬性规则：

- 研究接口读到 2022 年及以后的日期会抛 `HoldoutViolation`。2022 只能通过 `--validation-2022` 这类专门入口读取。
- 样本外分片、样本外外部数据、样本外手续费的读取接口全部加锁，年份结束也不会自动解锁。只有 `C.final_evaluation(...)` 块内才放行，而且读取不能越过登记的截止日。
- `final_evaluation` 的进入条件：配置里写明 `chosen` 和 `frozen_at`；测试窗口已结束；截止日在 2023-01-01..2025-12-31。进入时先往 `data/oos/ledger.jsonl` 追加一条 `opened` 记录（含配置文件 sha256），再读数据。台账只追加，不改不删。
- 当前台账记录：
  - 2026-09-30，`vwap_fade`（旧日内策略），样本外 2023–2025，夏普 −0.25。
  - 2026-10-08，日频核心 + 早盘卫星，样本外 2023，夏普 −1.46；同日追加一条 `decision` 记录，标明该方案已放弃。
- 样本外是否干净，以台账为准。2023–2025 已被 vwap_fade 用过一次，2023 又被组合用过一次。新策略要拿到干净的样本，只能等 2026 年结束，或者做模拟盘。

## 全流程

### 0. 下载

分钟行情（主力连续 88、889、99 和价差 agio）。总表默认是 `data/data_min/future_all1mdata_20100101-20251231.txt`。下载 2026 年以后的数据时加 `--standalone`，不然会合并进总表并改名：

```bash
python download/getRiceQuantData_MinFreq.py -t 20100101-20251231
```

```bash
python download/getRiceQuantData_MinFreq.py -t 20260101-20261008 --standalone
```

外部数据会按日期自动分到研究期（`data/external_rqdata/`）、`validation_2022/` 和 `holdout_locked/` 三个目录，并按 `coverage.json` 增量补齐。加 `--dry-run` 只列出缺口：

```bash
python download/getRiceQuantExternalData.py --start-date 20100101 --end-date 20251231
```

历史手续费按分区下载。研究期需要先有品种池（第 2 步），样本外则直接下载所有有合约乘数的品种：

```bash
python download/getFeeHistory.py --partition research
```

```bash
python download/getFeeHistory.py --partition validation_2022
```

```bash
python download/getFeeHistory.py --partition oos
```

最小变动价位表 `data/research/tick_size.csv`（品种 × 年）是早先生成的现成文件，仓库里没有生成它的脚本。样本外沿用 2021 年的值。

### 1. 分片

把分钟总表按品种切开：2021 年及以前写 `research/`，2022 年及以后写 `holdout_locked/`（写完即锁）。同时提取换月日到 `data/roll_dates/`，并做验收（时段、夜盘、复权、换月）：

```bash
python scripts/step1_shard_minutes.py
```

2022 单独切一份，写 `validation_2022/`：

```bash
python scripts/step1_shard_minutes.py --validation-from-monolith
```

其他参数：`--symbols RB HC` 只切部分品种，`--all` 包含金融期货，`--dry-run` 只报告不落盘，`--format parquet|pickle` 指定分片格式。

### 2. 品种池

第 y 年的品种池只用第 y−1 年的日均成交额和有效交易日决定，写到 `data/universe/universe_by_year.json`，报告在 `runs/<时间>_step2/`：

```bash
python scripts/step2_universe.py
```

门槛可以调，比如 `--min-turnover-yi 5`（单位亿元）、`--lookback 250`。2022 和样本外的品种池按同一规则在运行时计算，不落盘。

### 3. 因子构建

只用研究期数据，算两类：

- 分钟时间因子：持续期 DFP、高低点时间戳、报告类因子，写到 `data/factor_daily_v3/`。
- 期限结构：主力/次主力年化价差，写到 `data/factor_daily_v3/external/`。

```bash
python scripts/build_factors.py
```

`--symbols RB HC` 只算部分品种，`--overwrite` 重算已有缓存。日频价格类因子（tsmom、波动等）在研究时从日线现算，不需要缓存。

### 4. 因子研究

在 2016–2021 上跑 `config/research_candidates.json` 里预先写好的因子组合，只做评估，不自动挑选。每个候选有 `id`、`factors`（因子名 → 方向）、`days`（持有天数）、`hypothesis`。输出 IC、分年 IC、绩效、换仓相位敏感性和相关性，写到 `runs/<时间>_research_v3/`：

```bash
python scripts/research.py --specs config/research_candidates.json
```

`--slippage-ticks 2` 用 2 跳做压力测试，`--skip-ic` 跳过 IC 诊断。

候选可以带 `overlay`（叠加层）：主信号照常计算，`overlay.factors` 的组合与主信号反向时按 `action` 处理（`veto` 置 0，`halve` 减半），可选 `threshold` 只在控制信号绝对值达到阈值时处理。时间因子叠加在日频核心上的预先声明候选和结论见 `config/overlay_candidates.json`（研究期未通过，未采用）：

```bash
python scripts/research.py --specs config/overlay_candidates.json --skip-ic
```

### 5. 策略研究（以早盘规则为例）

先生成逐日特征表，决策时点是 09:16 收盘。不带参数时生成研究期和 2022 两份，写到 `runs/morning_features/`。样本外特征只能由第 7 步的脚本在锁内生成：

```bash
python scripts/morning_features.py
```

```bash
python scripts/morning_features.py research
```

规则、候选和参数写在 `config/morning_oor.json`，研究期上冻结后，2022 只跑一次：

```bash
python scripts/research_morning_oor.py
```

```bash
python scripts/research_morning_oor.py --validation-2022
```

输出在 `runs/morning_oor/<分区>/`：candidates.csv、yearly.csv、stress.csv、neighborhood.csv、trades.csv、daily.csv、nav.png。

开盘偏离的方向判别（流动性冲击逆向、新信息顺向）：特征、规则和选择标准写在 `config/morning_direction.json`，只跑研究期，输出到 `runs/morning_oor/direction/`（diagnostics.csv、spread_by_year.csv、rules.csv、legs.csv、ridge_coef.csv、yearly.csv、daily.csv、nav.png）。研究期没有规则通过：

```bash
python scripts/research_morning_direction.py
```

### 6. 组合构建

`config/portfolio.json` 写定核心、卫星、波动缩放、候选分配和选择标准。研究期按标准选出方案，人工写入 `chosen` 和 `frozen_at` 后，再跑 2022：

```bash
python scripts/research_portfolio.py
```

```bash
python scripts/research_portfolio.py --validation-2022
```

输出在 `runs/portfolio/<分区>/`：schemes.csv、daily.csv、nav.png。2022 的运行会核对研究期部分与已落盘结果逐日一致。

### 7. 样本外最终测试

口径事先写在 `config/oos_protocol.json`，包括主口径、参考口径、允许的年份、费用、品种池和期限结构的处理。所选年份之前的样本外年份也会读入，用于信号和乘数的连续历史，但不计入绩效：

```bash
python scripts/oos_portfolio.py --years 2023
```

```bash
python scripts/oos_portfolio.py --years 2023-2025
```

```bash
python scripts/oos_portfolio.py --years 2023,2025
```

脚本的执行顺序：

1. 检查配置已冻结、研究期结果和样本外手续费都已就绪。
2. 写台账。
3. 读样本外数据，做 2022 → 样本外的复权衔接检查。
4. 核对研究期和 2022 部分与已落盘结果一致。
5. 输出到 `runs/oos/<年份>/`：summary.csv、yearly.csv、两个熔断口径各一份 daily_*.csv、nav.png，并把结果追加到台账。

同一配置、同一年份已有结果时，必须加 `--rerun` 才会重跑。配置自上次样本外运行后有改动，也会记入台账。

## 回测假设

- 日频因子：收盘后才知道的信号在下一交易日开盘成交，收益为 `(openw[t+1] − openw[t]) / open[t]`（`openw` 是加法复权开盘价）。换月日按历史费率另计平旧开新的成本。
- 日内规则：决策 09:16 收盘，09:17 开盘进场，11:30 收盘离场。开仓用历史开仓费，平仓用平今费。
- 手续费：按金额收费的 ×1.01，按手数收费的 +0.01 元。滑点为每边 1 个上一年最小变动价位，压力口径 2 跳。
- 这是连续名义仓位的研究模型，没有整数手、保证金约束、涨跌停和盘口冲击。

## 新策略的流程

1. 在研究期提出假设并写成配置（`config/*.json`），写明 `declared_at`，配置必须先于结果。
2. 只在 2016–2021 上选规则、参数和分配，选择标准也事先写进配置。
3. 选定后写 `chosen` 和 `frozen_at`，此后不再改动。2022 只作诊断，不用它回头调参。
4. 写样本外口径文件，比照 `config/oos_protocol.json`，脚本的读数据部分包在 `C.final_evaluation(...)` 里（参考 `scripts/oos_portfolio.py`）。先确认台账里目标年份没有被用过。
5. 样本外只跑一次，结果不论好坏都留在台账里。

## 测试

```bash
python -m pytest
```

测试只用合成数据（`tfcta.data.make_symbol`），不读真实的样本外数据。覆盖范围包括：时段与分片、品种池、因子计算、回测成本、各分区的读取锁、`final_evaluation` 的门槛与台账、早盘规则和组合缩放。
