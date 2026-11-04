# 时间因子 CTA

根据兴业和国泰君安两份研报（`docs/` 下）开展商品和股指期货研究。回测使用整数手数账本，扣历史手续费和每边 1 跳滑点。

- 商品登记两条冻结策略：商品原策略和商品收盘位置策略。规则和实现分别见 [`strategies/commodity_original`](strategies/commodity_original/) 和 [`strategies/commodity_close_location`](strategies/commodity_close_location/)。
- 股指唯一注册策略是已冻结但尚未做样本外验证的 **`index_clv_rally_base_long_2x`**：21/63/126 日多头票数决定0至半仓底仓，剩余预算只在 pooled CLV<0.4 时启用，冲高z>0时只将CLV部分乘2，风控前目标最高2倍，完全不做空。原1.5倍版本已移除实现，仅保留历史对照结果。冻结只锁定规则，不代表已通过样本外或可部署验证；规则、研究登记和结果见 [`docs/index_guide.md`](docs/index_guide.md)。
- `index_clv_rally_equal_up_long` 已按用户要求删除实现和配置，历史研究结果仍保留。`index_clv_rally_equal_long` 与 `index_clv_adaptive_long` 的实现也已移除。历史记录见 [`docs/index_research_review.md`](docs/index_research_review.md)。
- 熊市和品种自身趋势共用 `TREND_WINDOWS = (20, 60, 120)`，并与冻结规格交叉校验。
- 商品策略和全部研究记录见 [`docs/guide.md`](docs/guide.md)；股指当前规则和参数研究见 [`docs/index_guide.md`](docs/index_guide.md)。`--market index --research` 运行当前冻结的股指策略。
- 股指历史研究在 [`docs/index_research_review.md`](docs/index_research_review.md)；当前策略规则、研究期结果和证据边界见[`股指策略说明`](docs/index_guide.md)。

## 环境

```bash
conda activate gu
```

命令都在仓库根目录运行。米筐凭证在 `config/rqdata.env`，不入库。

## 数据边界

| 分区 | 日期 | 商品 | 股指 |
|---|---|---|---|
| 研究期 | 商品 2011–2021；股指 IF/IH/IC 从 2015-04-16 起 | `data/commodity/minute_shards/research/`；账本从 2011 起算，2011–2015 品种池按同一门槛重建，2010 只做因子预热 | `data/index/minute_shards/research/`；三品种共同研究期从 IF、IH、IC 均已上市的首日开始，实际评分日期还需满足数据覆盖和因子预热 |
| 样本外 | 2022–2025 | `data/commodity/minute_shards/oos/` | `data/index/minute_shards/oos/` |

`data/` 顶层仅保留 `commodity/` 与 `index/`。每个市场下，研究期在 `research/`，样本外在 `oos/`。回测按请求的年份读取对应分区。研究脚本仍只读研究期，读到样本外路径会抛出 `HoldoutViolation`。样本外尚未出结果。

研究日历区间：商品 2011–2021；股指排除 IM，IF/IH/IC 的共同研究期从 2015-04-16 开始（IF 于 2010 年上市，IH 和 IC 于 2015-04-16 上市）。滚动因子需按规格预热，结果文件与图表从首个有效计分日开始，不输出更早的预热日期。

## 数据准备

从分钟总表到研究期回测输入，按顺序运行：

```bash
python src/tfcta/prepare/shard.py                      # 分钟总表切成 research/OOS 分片和换月表
python src/tfcta/prepare/universe.py                   # 逐年流动性池，第 y 年只看第 y−1 年
python src/tfcta/prepare/ticks.py                      # 逐年最小变动价位，第 y 年用于第 y+1 年
python src/tfcta/prepare/factors.py                    # 研究期时间因子缓存（N=250、M=55 持续期阈值）
python -m download.commodity.getFeeHistory --partition research
python -m download.commodity.getCommodityExternal --datasets roll_yield --end-date 20211231
```

展期收益缺失时回测会报出是哪个品种哪一年，不会悄悄当成不入选。补某几个品种用 `--symbols` 和 `--start-date/--end-date`。

## 回测

商品回测默认运行 [`strategies/commodity_catalog.py`](strategies/commodity_catalog.py) 中的两条冻结策略；股指策略在独立注册表中管理，正式账本同样由 `backtest.engine` 执行。结果写入 `runs/<市场>/strategies/<策略>/<区间>/`。

```bash
python -m backtest.engine --research                 # 2011–2021
python -m backtest.engine --start 2016 --end 2021     # 冻结时的研究期
python -m backtest.engine --research --by-year       # 每年一本新的 1 亿元账
python -m backtest.engine --start 2018 --end 2020
python -m backtest.engine 商品收盘位置策略 --research
python -m research.report.strategy_report       # 商品两本冻结策略的图表报告，先跑上面的账本
python -m backtest.engine --market index --research # 股指冻结策略，2015–2021；固定1000万、L1；固定1000万、L1
python -m backtest.engine index_clv_rally_base_long_2x --research
python -m backtest.engine --oos
python -m backtest.engine --market index --oos
```

正式账本在 `runs/<市场>/strategies/<策略>/<区间>/`。商品图表报告在 `runs/commodity/research/strategy_report/`。研究输出在 `runs/<市场>/research/<名称>/`。账本细节见 [`backtest/README.md`](backtest/README.md)，规则见 [`strategies/README.md`](strategies/README.md)。

## 下载

商品分钟总表和股指分钟分片分开下，不混进同一个池子。目录里已经有不止一份商品总表时（例如 2010–2025 和 2026 各一份），增量合并会拒绝运行，新区间请加 `--standalone` 单独下：

```bash
python -m download.commodity.getCommodityMinutes -t 20100101-20211231
python -m download.index.getIndexFutures --symbols IF IH IC --start-date 20150101 --end-date 20211231
python -m download.index.getIndexRollYield             # IF / IH / IC 研究期展期收益；米筐 get_roll_yield 对股指返回空，目前下不到
```

股指没有夜盘。时点按当天日盘自己的分钟从第一根归一化到最后一根，不用商品夜盘网格。

## 还在用的研究代码

商品账本和图表：

- `research/commodity/`：商品冻结策略的研究期账本，由 `backtest.engine` 调用。
- `research/report/`：商品两本冻结策略的图表，`python -m research.report.strategy_report`。

股指测试统一维护在 `tests/test_index.py`，不再按研究任务新增测试文件。日常只维护 `main` 分支。

股指研究统一放在 `research/index/`。当前策略位于 `strategies/index_clv_rally_base_long_2x/`，账本由 `backtest.engine` 执行；逐年曲线在 `runs/index/strategies/index_clv_rally_base_long_2x/research/plots/`。历史部件拆分：`python -m research.index.index_adaptive_attribution`；历史底仓对照用 `--base-momentum`，预算转移历史对照与仓位审计用 `--budget-transfer`。

```bash
python -m research.index.index_allocation_regimes # 历史配置研究与状态诊断，不登记可运行策略
python -m research.index.index_adaptive_signal_grid # 预登记门槛网格；仅读研究期
python -m research.index.market_mechanisms liquidity
python -m research.index.market_mechanisms positioning
python -m research.index.market_mechanisms positioning --unabsorbed
python -m research.index.market_mechanisms qvix
python -m research.index.market_mechanisms intraday-cost
python -m research.index.spot_reversal long-sample
python -m research.index.spot_reversal open-pressure
python -m research.index.weak_close_diagnostics close
python -m research.index.weak_close_diagnostics price-extremes --shape
python -m research.index.weak_close_diagnostics path-volume
```

机制子研究写入 `runs/index/research/<名称>/`。历史候选结果仅供查阅，不代表相关策略实现仍可运行。

数据下载按市场分开放在 `download/commodity/` 和 `download/index/`。手续费脚本在商品目录，用 `--market index` 下股指。

```bash
python -m download.index.getIndexSpot              # 沪深300、上证50、中证500 现货日线，2015-04-16–2021
python -m download.index.getIndexSpot --early      # 同上三只指数的日线，只下 2005-01-04 至 2015-04-15，单独文件
python -m download.index.getIndexSpot --minute     # 同上三只指数的 1 分钟线，同一研究期
python -m download.index.getMemberRank             # 中金所前 20 名会员多空持仓，IF / IH / IC，2015-04-16–2021
python -m download.index.getIndexRollYield        # 用合约日线自算 IF / IH / IC 展期收益，2015–2021
python -m download.commodity.getCommodityL1 --pool # 商品研究池主力，2011–2021，只存每日 L1 汇总，可续跑
python -m download.index.getIndexL1            # IF/IH/IC 主力，2015-04-16–2021，保存全部 L1 tick 与日汇总，可续跑
python -m download.index.getQvix                   # 50ETF 波动率指数研究数据
python -m download.index.getConstituents        # 沪深300/中证500 成分股价格、融资、股本、北向，2015–2021
python -m download.index.getMarketFlows             # 两市融资余额和北向成交，2015–2021
python -m download.commodity.getFeeHistory --market index --partition research
```

`runs/` 不进仓库，结构如下：

| 目录 | 内容 |
|---|---|
| `runs/<市场>/strategies/<策略ID>/<区间>/` | `backtest.engine` 写的正式账本。区间是 `research`、`oos`、单年或 `2018-2020` |
| `runs/<市场>/research/<名称>/` | 机制检验和图表报告。商品策略报告在 `runs/commodity/research/strategy_report/` |
| `runs/commodity/prepare/` | 分片和品种池准备的验收记录 |

商品 L1 流动性检验的预登记规格为 `config/research/commodity_l1_liquidity.json`，结果在 `runs/commodity/research/commodity_l1_liquidity/`。42 个品种文件覆盖不一；仅在各品种属于年度研究池的日期上分析，样本为 2011–2021，34 个品种达到至少 500 个有效日、跨 3 年的门槛，5 个板块达到板块检验门槛。相对尾盘价差扩大在 32/34 个合格品种及全部 5 个可检验板块上显著（BH q≤0.10）；相对深度变薄只在 4/34 个品种和农产品、化工两个板块显著。低收盘位置后的次日盘中反转在 CU、FG、JD、ZN 四个品种及黑色、有色、化工三个板块通过多重检验。控制尾盘买卖不平衡后，反转仍在 15 个品种、4 个板块显著；买卖不平衡本身在农产品、化工、黑色板块预测同方向延续，而非流量反转。流动性恶化与反转增强的交互没有一致证据：价差交互没有合格显著项，深度交互仅化工板块呈预期方向（q=0.074），有色板块反向（q=0.074），品种层面没有显著交互。对中间价 CLV 截断到 [0,1] 的事后稳健性复算没有改变任何 q≤0.10 的合格显著结论；42 个源文件共发现 109 条越界记录，其中 BB 有 18 条极端值，但其样本仅 220 天，不满足品种检验门槛。日汇总不含原始高低价 tick 序列，无法诊断所有越界原因。结果支持尾盘流动性压力代理变差，但不确证“流动性提供导致反转”的机制。以上均为重叠研究期内观察性检验，不是因果识别或样本外验证；仅使用 L1 盘口汇总，不能代表 L2 深度。

## 测试

```bash
python -m pytest tests/test_research.py tests/test_index.py
```

测试用合成数据，不读真实的 2022 及以后行情。日常改动只跑覆盖改动的必要测试；共享账本、成本或数据分区契约发生变化时才运行全套。
