# 时间因子 CTA

根据兴业和国泰君安两份研报（`docs/` 下）构建的商品期货时间因子策略。回测都是整数手数账本，扣历史手续费和每边 1 跳滑点。

- 当前策略：当天分数大于 0 且滞后一日展期收益大于 0 才做多，[`market_clock_carry`](strategies/market_clock_carry/)。五日符号已换掉。只做多全市场时钟的做法见 [`docs/全市场时钟做法与表现.md`](docs/全市场时钟做法与表现.md)。
- 四条纯多头尾部策略的代码和结果已于 2026-10-20 删除。做法和当时的数字留在 [`docs/四策略做法与表现.md`](docs/四策略做法与表现.md)。

## 环境

```bash
conda activate factor-mining
```

命令都在仓库根目录运行。米筐凭证在 `config/rqdata.env`，不入库。

## 数据边界

| 分区 | 日期 | 商品 | 股指 |
|---|---|---|---|
| 预热 | 2015 | 研究分片里，不计分 | 研究分片里，不计分 |
| 研究期 | 2016–2021 | `data/commodity/minute_shards/research/` | `data/index/minute_shards/research/`，IF / IH / IC |
| 2022 | 2022 | `validation_2022/`，商品上已经看过；全市场时钟 2026-10-20 已打开 | 和 2023–2025 一起锁在 `holdout_locked/`，还没打开 |
| 样本外 | 2023–2025 | `holdout_locked/`，商品上已经读过多次（早盘策略、四板块、20 日和 10 日窗口）；全市场时钟只按年打开，2023 已打开，2024、2025 还没有 | 同上，还没打开 |
| 前推 | 2026-01-01 起 | `data/commodity/data_min/` 里单独下载的 2026 文件，未分片，不读 | — |

研究代码读到 2022 年及以后会抛 `HoldoutViolation`。2023–2025 只有 `final_evaluation` 块里能读。商品台账在 `data/commodity/oos/ledger.jsonl`，股指台账在 `data/index/oos/ledger.jsonl`。

## 数据准备

从分钟总表到回测输入，按顺序跑一次。`holdout_locked/` 已经有分片时，第一步会拒绝覆盖。

```bash
python src/tfcta/prepare/shard.py                      # 分钟总表切成研究期 / 锁定分片和换月表
python src/tfcta/prepare/shard.py --validation-from-monolith   # 只写 2022 验证分片
python src/tfcta/prepare/universe.py                   # 逐年流动性池，第 y 年只看第 y−1 年
python src/tfcta/prepare/ticks.py                      # 逐年最小变动价位，第 y 年用于第 y+1 年
python src/tfcta/prepare/factors.py                    # 研究期时间因子和期限结构缓存
python download/commodity/getFeeHistory.py --partition research
python download/commodity/getRiceQuantExternalData.py  # 展期收益、仓单、持仓等外部数据
```

## 回测

当前跑当日分数且展期为正。研究期、2022 和 2023–2025 都可以整段或分年。2022 及以后先记台账，以前看过，不是干净样本外。

```bash
python -m backtest.engine --research
python -m backtest.engine --validation
python -m backtest.engine --oos
python -m backtest.engine --year 2024
python -m backtest.engine --start 2018 --end 2020
```

结果在 `runs/strategies/market_clock_carry/commodity/` 下。账本细节见 [`backtest/README.md`](backtest/README.md)，规则见 [`strategies/README.md`](strategies/README.md)。

股指研究期手续费：

```bash
python download/commodity/getFeeHistory.py --market index --partition research
```

## 下载

商品分钟总表和股指分钟分片分开下，不混进同一个池子。目录里已经有不止一份商品总表时（例如 2010–2025 和 2026 各一份），增量合并会拒绝运行，新区间请加 `--standalone` 单独下：

```bash
python download/commodity/getRiceQuantData_MinFreq.py -t 20100101-20251231
python download/index/getRiceQuantIndexFutures.py --symbols IF IH IC --start-date 20150101 --end-date 20211231
```

股指没有夜盘。时点按当天日盘自己的分钟从第一根归一化到最后一根，不用商品夜盘网格。

## 因子检验

```bash
python -m research.time_factor_screen   # 两份研报的时间戳和公允价偏离，商品截面和股指，2016–2021
python -m research.gtja_time            # 国泰君安持续期和行为时间戳，IF / IH / IC，2016–2021
python -m research.index_kline          # IF / IH / IC 研究期日 K 线图
python -m research.market_clock_carry      # 当天分数且展期收益为正，商品 2016–2021
```

检验规则先写在 `config/research/` 下的 JSON 里，再跑。这些脚本不构造冻结策略的持仓。

## 测试

```bash
python -m pytest
```

测试用合成数据，不读真实的 2022 和 2023–2025 行情。
