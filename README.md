# 时间因子 CTA

根据兴业和国泰君安两份研报（`docs/` 下）构建的商品期货时间因子策略。回测都是整数手数账本，扣历史手续费和每边 1 跳滑点。

- 当前只保留一条已冻结策略：按池宽1倍加慢趋势空头（`market_clock_slow_short`，2026-10-22 冻结）。多头和空头逻辑、因子与展期收益处理都合并在 [`strategies/market_clock_slow_short`](strategies/market_clock_slow_short/)。
- 熊市和品种自身趋势共用 `TREND_WINDOWS = (60, 120, 250)`，并与冻结规格交叉校验。
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
| 研究期 | 2011–2021 | `data/commodity/minute_shards/research/`；用户 2026-10-23 决定账本从 2011 起算，2011–2015 品种池按同一门槛重建，2010 只做因子预热。2016–2021 是冻结时用的研究期 | `data/index/minute_shards/research/`，IF / IH / IC，2016–2021 |
| 2022 | 2022 | `validation_2022/`；台账清空前已打开过，不是干净样本 | 锁在 `holdout_locked/`，不读 |
| 2023 | 2023 | `holdout_locked/`；台账清空前已打开过，不是干净样本 | 同上 |
| 2024 | 2024 | `holdout_locked/`；台账清空前已打开过，不是干净样本 | 同上 |
| 2025 | 2025 | `holdout_locked/`；台账清空前没有打开记录 | 同上 |
| 前推 | 2026-01-01 起 | `data/commodity/data_min/` 里单独下载的 2026 文件，未分片，不读 | — |

研究代码读到 2022 年及以后会抛 `HoldoutViolation`。策略冻结之后才能打开 2022 及以后，打开前先写台账：2022 记一条 `year_test`，2023 起通过 `final_evaluation`。台账在 `data/commodity/oos/ledger.jsonl`，作废记录用 `kind: void`。不读 2026。

2026-10-22 按用户决定清空了商品 OOS 台账（`data/commodity/oos/ledger.jsonl`）和 2022 年及以后的结果文件。清空前，2022–2024 已经为当时的策略打开过，2025 没有打开记录。空台账不让任何年份变干净：2022–2024 重跑只算复现。台账空了以后，`--validation`、`--year`、`--oos` 会重新读数据、写新台账，不会只打印旧结果。

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
python -m research.gtja_time                   # 国泰君安持续期和行为时间戳，IF / IH / IC，2016–2021
python -m research.market_clock_slow_short_factors  # 当前策略的信号检验；是已看过的研究期诊断
python -m research.index_market_clock              # 慢趋势策略及多头基线迁移到 IF / IH / IC；缺股指展期收益，目前跑不了
python -m research.index_kline                 # IF / IH / IC 研究期日 K 线图
```

检验规则先写在 `config/research/` 下的 JSON 里，再跑。结果在 `runs/research/<名字>/` 和 `runs/index/<名字>/`。

## 测试

```bash
python -m pytest
```

测试用合成数据，不读真实的 2022 及以后行情。
