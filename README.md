# 时间因子 CTA

根据兴业和国泰君安两份研报（`docs/` 下）开展商品和股指期货研究。回测使用整数手数账本，扣历史手续费和每边 1 跳滑点。

- 商品登记两条冻结策略：商品原策略和商品收盘位置策略。规则和实现分别见 [`strategies/commodity_original`](strategies/commodity_original/) 和 [`strategies/commodity_close_location`](strategies/commodity_close_location/)。
- 股指只登记一条冻结策略：股指收盘位置熊市空头（`index_close_location_bear_short`，收盘位置低于 0.4 做多，熊市且收盘位置不低于 0.4 才做空，再加波动率缩放），只含 IF/IH/IC；单独列于 [`strategies/index_catalog.py`](strategies/index_catalog.py)。它的 2022 已按台账打开；2023–2025 和 2026 不打开。
- 冻结只表示规则固定，不代表干净验证或实盘优势。商品两条的 2022 年及以后窗口仍须用户明确授权后才能开启。
- 熊市和品种自身趋势共用 `TREND_WINDOWS = (20, 60, 120)`，并与冻结规格交叉校验。
- 商品策略和全部研究记录见 [`docs/guide.md`](docs/guide.md)；股指策略的规则和证据见 [`docs/index_guide.md`](docs/index_guide.md)，`--market index` 不点名就跑它。

## 环境

```bash
conda activate factor-mining
```

命令都在仓库根目录运行。米筐凭证在 `config/rqdata.env`，不入库。

## 数据边界

| 分区 | 日期 | 商品 | 股指 |
|---|---|---|---|
| 预热 | 2015 | 研究分片里，不计分 | 研究分片里，不计分 |
| 研究期 | 商品 2011–2021；股指 IF/IH/IC 从 2015-04-16 起 | `data/commodity/minute_shards/research/`；账本从 2011 起算，2011–2015 品种池按同一门槛重建，2010 只做因子预热 | `data/index/minute_shards/research/`；三品种共同研究期从 IF、IH、IC 均已上市的首日开始，实际评分日期还需满足数据覆盖和因子预热 |
| 样本外 | 2022 | `validation_2022/`；冻结策略尚未运行，用户说开始之前不打开 | 锁在 `holdout_locked/`，不读 |
| 样本外 | 2023 | `holdout_locked/`；同上 | 同上 |
| 样本外 | 2024 | `holdout_locked/`；同上 | 同上 |
| 样本外 | 2025 | `holdout_locked/`；同上 | 同上 |
| 前推 | 2026-01-01 起 | `data/commodity/data_min/` 里单独下载的 2026 文件，未分片，不读 | — |

研究代码读到 2022 年及以后会抛 `HoldoutViolation`。策略冻结之后才能打开 2022 及以后，打开前先写台账：2022 记一条 `year_test`，2023 起通过 `final_evaluation`。台账在 `data/commodity/oos/ledger.jsonl`，作废记录用 `kind: void`。不读 2026。

2026-10-26 用户决定：2022–2025 是当前策略的样本外，当前策略从未在这几年运行过；用户说开始看样本外之前一律不打开。台账（`data/commodity/oos/ledger.jsonl`）目前是空的，打开时先写台账再读。

2026-10-26 用户决定：商品新实验、拉数和回测默认研究日历区间为 2011–2021；股指可比组合排除 IM，IF/IH/IC 的共同研究期从 2015-04-16 开始（IF 于 2010 年上市，IH 和 IC 于 2015-04-16 上市）。滚动因子需按规格预热，结果文件与图表从首个非零目标日开始，不输出更早的预热日期。其他已登记研究结果保留原有样本标注。

## 数据准备

从分钟总表到回测输入，按顺序跑一次。`holdout_locked/` 已经有分片时，第一步会拒绝覆盖。

```bash
python src/tfcta/prepare/shard.py                      # 分钟总表切成研究期 / 锁定分片和换月表
python src/tfcta/prepare/shard.py --validation-from-monolith   # 只写 2022 验证分片
python src/tfcta/prepare/universe.py                   # 逐年流动性池，第 y 年只看第 y−1 年
python src/tfcta/prepare/ticks.py                      # 逐年最小变动价位，第 y 年用于第 y+1 年
python src/tfcta/prepare/factors.py                    # 研究期时间因子缓存（N=250、M=55 持续期阈值）
python download/commodity/getFeeHistory.py --partition research        # 另有 validation_2022、oos 两个分区
python download/commodity/getRiceQuantExternalData.py --datasets roll_yield --end-date 20251231   # 主力对次主力展期收益，按研究期 / 2022 / 锁定分区存
```

展期收益缺失时回测会报出是哪个品种哪一年，不会悄悄当成不入选。补某几个品种用 `--symbols` 和 `--start-date/--end-date`。

## 回测

商品回测默认运行 [`strategies/commodity_catalog.py`](strategies/commodity_catalog.py) 中的两条冻结策略；股指策略在独立注册表中管理，但正式账本同样由 `backtest.engine` 执行。结果直接写入 `runs/strategies/<策略>/<区间>/`。

```bash
python -m backtest.engine --research                 # 2011–2021
python -m backtest.engine --start 2016 --end 2021     # 冻结时的研究期
python -m backtest.engine --research --by-year       # 每年一本新的 1 亿元账
python -m backtest.engine --start 2018 --end 2020
python -m backtest.engine 商品收盘位置策略 --research
python -m research.report.strategy_report       # 商品两本冻结策略的图表报告，先跑上面的账本
python -m backtest.engine --market index --research    # 股指熊市空头含波动率缩放，只跑 2015–2021
python -m backtest.engine --market index --validation  # 它的 2022 已打开，只打印保存的结果
python -m backtest.engine --year 2024                # 单年，冻结后
python -m backtest.engine --oos [--by-year]          # 2023–2025，冻结后
```

正式账本在 `runs/strategies/<策略>/<区间>/`；商品两本冻结策略的图表报告在 `runs/reports/commodity/`，股指熊市空头的图在它自己账本的 `plots/` 下。账本细节见 [`backtest/README.md`](backtest/README.md)，规则见 [`strategies/README.md`](strategies/README.md)。

## 下载

商品分钟总表和股指分钟分片分开下，不混进同一个池子。目录里已经有不止一份商品总表时（例如 2010–2025 和 2026 各一份），增量合并会拒绝运行，新区间请加 `--standalone` 单独下：

```bash
python download/commodity/getRiceQuantData_MinFreq.py -t 20100101-20251231
python download/index/getRiceQuantIndexFutures.py --symbols IF IH IC --start-date 20150101 --end-date 20211231
python -m download.index.getIndexRollYield             # IF / IH / IC 研究期展期收益；米筐 get_roll_yield 对股指返回空，目前下不到
```

股指没有夜盘。时点按当天日盘自己的分钟从第一根归一化到最后一根，不用商品夜盘网格。

## 还在用的研究代码

已经做完的机制检验脚本在 2026-10-29 删除，结论留在 [`docs/guide.md`](docs/guide.md) 和 [`docs/index_guide.md`](docs/index_guide.md)。现在 `research/` 只留两类：

- `research/commodity/`：商品冻结策略的研究期账本，由 `backtest.engine` 调用。
- `research/report/`：商品两本冻结策略的图表，`python -m research.report.strategy_report`。

数据下载还在 `download/`：

```bash
python -m download.index.getIndexSpot              # 沪深300、上证50、中证500 现货日线，2015-04-16–2021
python -m download.index.getIndexSpot --minute     # 同上三只指数的 1 分钟线，同一研究期
python -m download.index.getMemberRank             # 中金所前 20 名会员多空持仓，IF / IH / IC，2015-04-16–2021
python -m download.index.getIndexRollYield        # 用合约日线自算 IF / IH / IC 展期收益，2015–2021
python -m download.commodity.getTickL1Close --pool # 商品研究池主力，2011–2021，只存每日 L1 汇总，可续跑
python -m download.index.getTickL1Close            # IF/IH/IC 主力，2015-04-16–2021，保存全部 L1 tick 与日汇总，可续跑
python -m download.index.getQvix                   # 50ETF 波动率指数；研究期到 2021，2022 及以后写入锁定目录且不读
python -m download.index.getConstituentPanel        # 沪深300/中证500 成分股价格、融资、股本、北向，2015–2021
python -m download.index.getMarketFlows             # 两市融资余额和北向成交，2015–2021
python -m download.index.getTickL1Close --partition oos_locked --start 2022-01-01 --end 2025-12-31 # 只存锁定期原始 tick，不生成汇总或分析
```

`runs/` 不进仓库，结构如下：

| 目录 | 内容 |
|---|---|
| `runs/strategies/<策略ID>/<区间>/` | `backtest.engine` 写的正式账本（summary/annual/daily/positions/events 和 `plots/performance.png`） |
| `runs/reports/commodity/` | `research.report.strategy_report`：商品两本冻结策略的净值回撤、对比买入持有、组成部分图 |
| `runs/research/<研究名>/` | 商品研究检验，各带 `protocol.json` |
| `runs/index/<研究名>/` | 股指研究检验，各带 `protocol.json` |
| `runs/archive/<日期>_superseded/` | 被取代的旧输出，只留档不再引用 |

商品 L1 流动性检验的预登记规格为 `config/research/commodity_l1_liquidity.json`，结果在 `runs/research/commodity_l1_liquidity/`。42 个品种文件覆盖不一；仅在各品种属于年度研究池的日期上分析，样本为 2011–2021，34 个品种达到至少 500 个有效日、跨 3 年的门槛，5 个板块达到板块检验门槛。相对尾盘价差扩大在 32/34 个合格品种及全部 5 个可检验板块上显著（BH q≤0.10）；相对深度变薄只在 4/34 个品种和农产品、化工两个板块显著。低收盘位置后的次日盘中反转在 CU、FG、JD、ZN 四个品种及黑色、有色、化工三个板块通过多重检验。控制尾盘买卖不平衡后，反转仍在 15 个品种、4 个板块显著；买卖不平衡本身在农产品、化工、黑色板块预测同方向延续，而非流量反转。流动性恶化与反转增强的交互没有一致证据：价差交互没有合格显著项，深度交互仅化工板块呈预期方向（q=0.074），有色板块反向（q=0.074），品种层面没有显著交互。对中间价 CLV 截断到 [0,1] 的事后稳健性复算没有改变任何 q≤0.10 的合格显著结论；42 个源文件共发现 109 条越界记录，其中 BB 有 18 条极端值，但其样本仅 220 天，不满足品种检验门槛。日汇总不含原始高低价 tick 序列，无法诊断所有越界原因。结果支持尾盘流动性压力代理变差，但不确证“流动性提供导致反转”的机制。以上均为重叠研究期内观察性检验，不是因果识别或样本外验证；仅使用 L1 盘口汇总，不能代表 L2 深度。

## 测试

```bash
python -m pytest tests/test_research.py tests/test_index.py
```

测试用合成数据，不读真实的 2022 及以后行情。日常改动只跑覆盖改动的必要测试；共享账本、成本或数据分区契约发生变化时才运行全套。
