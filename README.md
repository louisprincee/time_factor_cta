# 时间因子 CTA

商品期货和股指期货上的四条纯多头策略。信号用日内高低点时戳，收盘决定，下一交易日开盘进，持有 5 个有效交易日。回测是整数手数账本。

做法、样本边界和已经跑出来的数字写在 [`docs/四策略做法与表现.md`](docs/四策略做法与表现.md)。

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
| 2022 | 2022 | `validation_2022/`，商品上已经看过 | 和 2023–2025 一起锁在 `holdout_locked/`，还没打开 |
| 样本外 | 2023–2025 | `holdout_locked/`，商品上已经读过多次（早盘策略、四板块、20 日和 10 日窗口），不能再用 | 同上，还没打开 |
| 前推 | 2026-01-01 起 | `data/commodity/data_min/` 里单独下载的 2026 文件，未分片 | — |

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

```bash
python -m backtest.engine --research
python -m backtest.engine --market index --research
python -m backtest.engine --validation
python -m backtest.engine --market index --validation
python -m backtest.engine --oos
python -m backtest.engine --market index --oos
```

不写 `--market` 就是商品。一次跑 `strategies/catalog.py` 里的四条。`python -m backtest.engine --experimental-research` 单独跑 `strategies/research_catalog.py` 里的三条商品探索策略，只读 2016–2021。结果在 `runs/strategies/<策略>/commodity/` 或 `index/` 下，再分 `research/`、`validation/`、`oos/`。账本细节见 [`backtest/README.md`](backtest/README.md)，规则见 [`strategies/README.md`](strategies/README.md)。

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
```

检验规则先写在 `config/research/` 下的 JSON 里，再跑。这些脚本不构造冻结策略的持仓。

## 测试

```bash
python -m pytest
```

测试用合成数据，不读真实的 2022 和 2023–2025 行情。
