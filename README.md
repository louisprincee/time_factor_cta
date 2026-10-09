# 商品期货时间因子 CTA 研究

从米筐分钟数据出发，研究商品期货的日频因子和日内规则，并在严格隔离的样本外上做最终检验。起点是兴业证券《基于高频时间维度的国债期货择时因子及个股 CTA 研究》（`docs/` 下的 PDF）：价格停留时间、高低点出现的先后能否提供可交易的信息。

当前结论（2026-10-10）：**没有可实盘的策略。** 唯一保留的策略是早盘元策略（主策略·亏损反手·30笔），2023–2025 严格样本外夏普 0.04（2023 +1.9%，2024 +1.5%，2025 −3.3%），2 跳滑点 −0.35。其他方案（日频核心 + 早盘卫星、四条日频腿、方向判别、时间因子、机器学习、持仓量、降成本等）都已放弃并删除，已提交过的可从 git 历史找回。2016–2025 的数据都已看过，台账见 `data/oos/ledger.jsonl`。早盘主策略见 [MorningOverreactionReversal.md](docs/MorningOverreactionReversal.md)。

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
- 样本外历史访问以台账为准：旧 `vwap_fade` 曾使用 2023–2025，旧“核心 + 卫星”方案曾使用 2023。当前早盘元策略的冻结协议允许将 2023 作为计分年份；历史访问记录保留，解释结果时应一并参考。

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

只补某一类数据用 `--datasets`，例如 2022 年的仓单（2026-10-10 已补齐；2023-01-01 到 2026-10-08 此前已在 `holdout_locked/`）：

```bash
python download/getRiceQuantExternalData.py --start-date 20220101 --end-date 20221231 --datasets warehouse
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

### 5. 策略研究（以早盘规则为例）

先生成逐日特征表，决策时点是 09:16 收盘。不带参数时生成研究期和 2022 两份，写到 `runs/morning_features/`。样本外特征只能在 `C.final_evaluation(...)` 锁内生成：

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

输出在 `runs/morning_oor/<分区>/`：candidates.csv、yearly.csv、stress.csv、neighborhood.csv、trades.csv、daily.csv。

早盘主策略是元策略的基础单，它自己在 2022 和 2023 都失败了，不单独作为策略。

### 6. 早盘元策略

配置 `config/morning_meta.json`（已冻结），脚本 `research_morning_meta.py`：主策略的反向单，最近 30 笔已平仓单（只用前一交易日及以前）的账面均值为负时改为顺势。研究期 2016–2021 夏普 1.15，2022 夏普 0.56；设计受“2022 年反向变成延续”启发，2022 不算检验。配置里还保留了停手、偏离反向等对照和结论（`result`、`result_2022`）。研究期和 2022 各跑一次：

```bash
python scripts/research_morning_meta.py
```

```bash
python scripts/research_morning_meta.py --validation-2022
```

输出在 `runs/morning_oor/meta/<分区>/`。

元策略参数面（2026-10-10，样本外用完之后的探索）：窗口 10–90 笔或 20–120 个交易日 × 反手 / 停手 × 主策略 / 偏离反向，在 2016–2022 上算夏普、DSR（扣除尝试次数的夏普显著性，`stats.deflated_sharpe`）和 PBO（组合对称交叉验证的过拟合概率，`stats.pbo_cscv`），网格写在 `config/meta_grid.json`，输出在 `runs/morning_oor/meta_grid/`：

```bash
python scripts/research_meta_grid.py
```

### 统一运行与绘图

```bash
python scripts/run_strategy.py morning-meta research
```

可用策略名：`morning-oor`、`morning-meta`；阶段为 `research` 或 `validation-2022`。每个阶段回测完会自动出图（`tfcta.research.context.plot_performance`，净值和回撤在同一张图），写到结果目录的 `plots/<阶段>/performance.png`。早盘元策略的图只画 30 笔主设定。要挑几条序列重画，用：

```bash
python scripts/plot_strategy.py morning-meta validation-2022 --series 主策略·亏损反手·30笔
```

```bash
python scripts/plot_strategy.py oos 2023-2025
```

### 7. 样本外最终测试

口径写在 `config/oos_protocol.json`：可计分年份为 2023、2024、2025（2026-10-09 已全部跑过，见台账），主口径 1 跳，参考 2 跳和不切换的主策略。`--years` 中明确指定的年份计入绩效；运行较晚年份时，之前的样本外历史仍用于连续信号和滚动状态。先检查，这一步只重算研究期和 2022 并与已落盘结果逐日核对，不读样本外：

```bash
python scripts/oos_final.py --check
```

检查通过后正式运行，年份可选 `2023`、`2024`、`2025` 或相应连续区间，例如 `2023-2025`：

```bash
python scripts/oos_final.py --years 2023-2025 --rerun
```

脚本的执行顺序：

1. 确认配置已冻结、研究期和 2022 结果、样本外手续费都已就绪。
2. 写台账 `data/oos/ledger.jsonl`，打开 `C.final_evaluation(...)`。
3. 算样本外品种池，生成样本外早盘特征。
4. 研究期、2022、样本外接成连续序列（元策略的最近 30 笔跨年连续），核对研究期和 2022 部分与已落盘结果一致。
5. 输出到 `runs/oos/<年份>/`：summary.csv、yearly.csv、morning_trades.csv、daily.csv、plots/，并把结果追加到台账。中途出错也会在台账里记一条 aborted。

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
4. 写样本外口径文件和脚本，读数据部分包在 `C.final_evaluation(...)` 里（参考 `config/oos_protocol.json` 和 `scripts/oos_final.py`）。先确认台账里目标年份没有被用过。
5. 样本外只跑一次，结果不论好坏都留在台账里。

## 测试

```bash
python -m pytest
```

测试只用合成数据（`tfcta.data.make_symbol`），不读真实的样本外数据。覆盖范围包括：时段与分片、品种池、因子计算、回测成本、各分区的读取锁、`final_evaluation` 的门槛与台账、早盘规则、元策略账面与多腿缩放的防前视、样本外脚本的年份与台账。
