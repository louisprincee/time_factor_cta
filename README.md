# 时间因子 CTA

根据兴业和国泰君安两份研报（`docs/` 下）开展商品和股指期货研究。回测使用整数手数账本，扣历史手续费和每边 1 跳滑点。

- 商品主线保留一条已冻结策略：按池宽1倍加三窗口趋势空头（`market_clock_slow_short_20_60_120`，2026-10-23 冻结）。多头和空头逻辑、因子与展期收益处理都在 [`strategies/market_clock_slow_short_20_60_120`](strategies/market_clock_slow_short_20_60_120/)。
- 股指另有一条已登记、尚未冻结的研究候选：`index_close_location_reversal`，仅针对 IF/IH/IC；单独列于 [`strategies/index_catalog.py`](strategies/index_catalog.py)，不混入商品回测注册表。
- 商品另有一条探索期策略：商品收盘位置反转加趋势空头（`commodity_close_location_reversal`），多头只看池内收盘位置 < 0.5，空头腿同冻结策略；只能跑研究期（`python -m backtest.engine commodity_close_location_reversal --research`，变体对照 `python -m research.commodity_close_location_reversal`）。
- 熊市和品种自身趋势共用 `TREND_WINDOWS = (20, 60, 120)`，并与冻结规格交叉校验。
- 策略规则、实现、研究期结果和研究局限见 [`docs/guide.md`](docs/guide.md)。

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
| 样本外 | 2022 | `validation_2022/`；当前策略从未运行过，用户说开始之前不打开 | 锁在 `holdout_locked/`，不读 |
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

当前只有一条策略，运行时不需策略名。

```bash
python -m backtest.engine --research                 # 2011–2021
python -m backtest.engine --start 2016 --end 2021     # 冻结时的研究期
python -m backtest.engine --research --by-year       # 每年一本新的 1 亿元账
python -m backtest.engine --start 2018 --end 2020
python -m backtest.engine --validation               # 2022，冻结后
python -m backtest.engine --year 2024                # 单年，冻结后
python -m backtest.engine --oos [--by-year]          # 2023–2025，冻结后
```

结果在 `runs/strategies/<id>/commodity/<区间>/`。账本细节见 [`backtest/README.md`](backtest/README.md)，规则见 [`strategies/README.md`](strategies/README.md)。

## 下载

商品分钟总表和股指分钟分片分开下，不混进同一个池子。目录里已经有不止一份商品总表时（例如 2010–2025 和 2026 各一份），增量合并会拒绝运行，新区间请加 `--standalone` 单独下：

```bash
python download/commodity/getRiceQuantData_MinFreq.py -t 20100101-20251231
python download/index/getRiceQuantIndexFutures.py --symbols IF IH IC --start-date 20150101 --end-date 20211231
python -m download.index.getIndexRollYield             # IF / IH / IC 研究期展期收益；米筐 get_roll_yield 对股指返回空，目前下不到
```

股指没有夜盘。时点按当天日盘自己的分钟从第一根归一化到最后一根，不用商品夜盘网格。

## 因子检验

```bash
python -m research.time_factor_screen          # 两份研报的时间戳和公允价偏离，商品截面和股指，2016–2021
python -m research.market_clock_slow_short_factors  # 当前策略的信号层检验，研究期
python -m research.index_market_clock              # IF / IH / IC 当前策略回测；仅输出首个信号日及之后的账本和净值/回撤图
python -m research.index_market_clock --overwrite  # 同研究协议重写已有股指结果
python -m research.index_close_location_candidate  # 已登记的 CLV<0.5 股指候选，只跑 2016–2021 研究期；未冻结、不读 OOS
python -m research.index_close_location_candidate --robustness  # 候选的预登记稳健性：起始日、等敞口常数长仓、容量
python -m research.index_liquidity_mechanism       # 股指 L1 逐笔加现货：期现基差、成交流量冲击分解、反弹时点
python -m download.index.getIndexSpot              # 沪深300、上证50、中证500 现货日线，2015-04-16–2021
python -m download.index.getMemberRank             # 中金所前 20 名会员多空持仓，IF / IH / IC，2015-04-16–2021
python -m research.index_positioning_mechanism     # 会员持仓和持仓量变化与收盘位置反转的关系
python -m research.index_positioning_mechanism --unabsorbed  # 没被大会员接住的卖压能否单独预测反弹
python -m research.index_close_location_candidate --short-leg  # 股指候选加空头腿（展期版和纯趋势版），2015–2021
python -m research.market_clock_mechanism         # 组成部分贡献图、振幅和收盘位置检验、分钟流动性代理，2011–2021
python -m research.commodity_close_location_reversal --three-book-comparison  # 三策略严格替换对照及贡献图
python -m download.index.getIndexRollYield        # 用合约日线自算 IF / IH / IC 展期收益，2015–2021
python -m research.close_location_mechanism [--l1] # 收盘位置反转的机制检验；--l1 用 L1 盘口汇总
python -m download.commodity.getTickL1Close --pool # 商品研究池主力，2011–2021，只存每日 L1 汇总，可续跑
python -m download.index.getTickL1Close            # IF/IH/IC 主力，2015-04-16–2021，保存全部 L1 tick 与日汇总，可续跑
python -m download.index.getTickL1Close --symbols IF # 股指按品种分批下载；IH、IC 可用同样命令
python -m research.index_close_location_mechanism   # 股指 L1 收盘位置与流动性机制诊断；使用预登记样本
python -m download.index.getTickL1Close --partition oos_locked --start 2022-01-01 --end 2025-12-31 # 只存锁定期原始 tick，不生成汇总或分析
```

检验规则先写在 `config/research/` 下的 JSON 里，再跑。结果在 `runs/research/<名字>/` 和 `runs/index/<名字>/`。
三策略对照的规格为 `config/research/three_strategy_clv_comparison.json`，入口并入 `research/commodity_close_location_reversal.py`，结果和高清图在 `runs/research/three_strategy_clv_comparison/`。它只替换五因子时间分数，保留原 carry 门槛、趋势空头和账本风控；结果属于重叠研究期的样本内分析，不是样本外验证。

## 测试

```bash
python -m pytest tests/test_research.py tests/test_index.py
```

测试用合成数据，不读真实的 2022 及以后行情。日常改动只跑覆盖改动的必要测试；共享账本、成本或数据分区契约发生变化时才运行全套。
