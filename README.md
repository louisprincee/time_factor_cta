# 商品与股指时间因子研究 Guide

本仓库从米筐期货分钟数据构造商品及股指时间因子，复现并检验兴业证券与国泰君安相关研究思路；当前策略主线为商品 `ts_low` 纯多头整数手回测。此文档是当前入口说明，所有命令均从仓库根目录执行。

## 1. 目录和主线

| 路径                                                 | 用途                                                      |
| ---------------------------------------------------- | --------------------------------------------------------- |
| `src/tfcta/`                                       | 数据读取、因子公式、商品手续费/tick、预测性统计和区间守卫 |
| `src/tfcta/prepare/`                               | 分区分片、品种池、因子缓存和 tick 缓存                    |
| `src/tfcta/index/gtja_time.py`                     | 国泰君安股指时间因子定义                                  |
| `research/time_factor_screen.py`                   | 商品与股指时间因子研究筛查（兴业相关主线）                |
| `research/gtja_time.py`                            | 国泰君安股指时间因子预测性分析                            |
| `research/index_kline.py`                          | IF/IH/IC 复权日线核验图                                   |
| `strategies/ts_low.py`、`strategies/ts_low.json` | 当前唯一策略：商品 `ts_low` 纯多头                      |
| `backtest/engine.py`                               | 整数手数组合回测、成本、风控、图表                        |
| `config/research/`                                 | 预先声明的因子、参数、样本及判定规则                      |
| `runs/`                                            | 本机生成的研究和回测产物，不作为代码提交                  |

不再保留旧早盘研究、多空策略和旧板块扫描代码及其相关结果。

## 2. 环境和路径

```bash
conda activate gu
pip install -r requirements.txt
```

下载行情另需米筐 `rqdatac` 与 `tqdm`。将凭证放在不入库的 `config/rqdata.env`，
格式如下（或使用 `RQDATAC_USERNAME` / `RQDATAC_PASSWORD`）：

```dotenv
RQDATAC_LICENSE=...
```

可用环境变量：`TFCTA_MINUTE_DIR`、`TFCTA_DATA_ROOT`、`TFCTA_RUNS_ROOT`、
`TFCTA_CONFIG_DIR`。不要把凭证、原始行情、分片、审计台账或 runs 结果加入 Git。

## 3. 数据和预处理

分钟数据命令的 `-t` 接受 `YYYYMMDD-YYYYMMDD`。本项目需要研究期前的滚动预热数据；
下载或使用已有 2010–2025 原始历史数据后，预处理器负责按研究/验证/锁定分区拆分。

```bash
python download/commodity/getRiceQuantData_MinFreq.py -t 20100101-20251231
python download/commodity/getRiceQuantExternalData.py --start-date 20100101 --end-date 20251231
python download/commodity/getFeeHistory.py --partition research
python download/commodity/getFeeHistory.py --partition validation_2022
```

手续费分区分别写入研究期和 2022 验证期缓存；`getFeeHistory.py --partition oos`
会获取样本外手续费，**本 Guide 不建议执行**。股指研究数据单独下载，仅用于 IF/IH/IC：

```bash
python download/index/getRiceQuantIndexFutures.py --partition research --symbols IF IH IC --start-date 20150101 --end-date 20211231
```

商品原始分钟数据完成后，依次运行：

```bash
python src/tfcta/prepare/shard.py
python src/tfcta/prepare/universe.py
python src/tfcta/prepare/factors.py
python src/tfcta/prepare/ticks.py
```

- `shard.py` 按配置生成研究、验证和锁定分区；如果只有单独的 2022 来源，可运行
  `python src/tfcta/prepare/shard.py --validation-from-monolith`。
- `universe.py` 默认按配置的成交额和有效日条件生成逐年品种池；可选参数为
  `--symbols`、`--lookback`、`--min-turnover-yi`、`--quiet`。
- `factors.py` 默认构建全部适用品种的因子缓存；`--symbols A RB` 限定品种，
  `--overwrite` 显式覆盖缓存。
- `ticks.py` 生成研究期及验证期的最小变动价位缓存。锁定期不会通过该命令写入 tick。

已完成数据准备时不要重复下载或覆盖缓存。商品研究命令要求手续费、tick、逐年品种池、
因子缓存和行情分片齐备；缺失成本不会静默跳过。

**数据/规格抽样核对（仅研究分区，2026-10-18）**：对 RB、CU、TA 的研究分片检查
2016–2021 换月日与交易日历，分别有 18/18、71/71、18/18 个换月日能在对应分片中找到。
从这三份研究分钟分片按 `estimate_ticks` 重算 2015–2021 年 tick，和落盘缓存中的 tick
及非零变动数完全一致（RB/CU/TA 的估计 tick 分别为 1/10/2 个报价单位）。三者手续费
缓存每个品种在 2016–2021 六年合计各有 1,461 条日记录；TA 的 357 条平今费填补都落在
2016-01-04 至 2017-06-22，符合代码限定的 CZCE 切换日前范围。未发现这些内部一致性检查
的差异；它们不是独立来源认证。

仍未完成交易所级独立核验：`MULTIPLIER` 是无生效日期的静态乘数表；tick 是从同一行情
源估算，并非交易所规格表对照；手续费来自 RiceQuant 历史缓存，TA 缺失平今费按保守规则
估算填补。当前没有保存相应年份的交易所合约规格/费率公告或独立行情对照，故不能据此
声称乘数、历史费率、复权衔接或换月合约价格已认证。后续若取得带日期的交易所公告或
独立数据源，应逐品种记录来源、生效日期、实测差异和处理决定；在此之前保持现有估算，
并继续标为研究级。

## 4. 研究期因子实验

### 4.1 兴业相关商品/股指时间因子筛查

```bash
python -m research.time_factor_screen
```

使用 `config/research/time_factor_screen.json`，信号期为 2016–2021，不读 2022 及之后数据。
持续期主规格：过去 250 个交易日（不含当天）分钟价格绝对变动的 50% 分位数作为阈值；
商品不使用固定点数，股指另有固定 3 点对照。目标包括决策时点后开盘到再下一交易日开盘，
并记录预先声明的日内时点目标。配置声明的因子族包括高/低点时戳与差值、价格偏离
（DFP）、成交量/成交额时戳，以及报告持续期、量价持续期比和早盘冲击等变量。结果为
预测性筛查，不直接生成策略信号或组合。

当前配置对应 130 个市场/决策时点/因子/输出品种筛查单元（商品 31、股指合约 99），
每个单元有单因子和控制后两个 `|t|` 门槛，共 260 次配置门槛比较。因子、决策时点、
股指合约及共享交易日样本存在依赖；HAC 仅处理各自统计量内的序列相关，不解决跨检验
依赖。因此结果明确标为**未作多重检验校正的探索性发现**，不报告 FDR/家族错误率控制
后的显著性，也不把“达到筛查门槛”解释为确认性显著。

输出目录 `runs/research/time_factor_screen/`：

- `summary.csv`：逐因子/决策时点统计及预设门槛结果；
- `path_diagnostics.csv`：路径诊断；
- `panel.parquet`：逐观测研究面板；
- `protocol.json`：本次配置快照。

这里的研究屏幕是按配置实际实现的候选集合；不能仅凭因子名称相似，就把它描述为研报中所有变量均已完整、严格同口径复现，也不能把屏幕结果等同于共享因子缓存。

### 4.2 国泰君安股指时间因子

```bash
python -m research.gtja_time
```

只处理 IF、IH、IC；2015 年用于滚动标准化预热，预测检验期为 2016-01-04 至
2021-12-31。配置位于 `config/research/gtja_time_factors.json`：价格持续阈值 3 点、
成交量持续阈值 50、60 日滚动标准化、60 个最小有效观测、5 阶 Newey–West/HAC。
计算 28 个报告定义因子，预测目标是收盘因子对应的下一交易日开盘至再下一交易日开盘收益；
输出逐品种逐年 Spearman 相关与标准化线性斜率/HAC 统计，不产生仓位或策略组合。

输出在 `runs/index/gtja_time/`：`factor_panel.parquet`、`factor_summary.csv`、
`factor_yearly.csv` 和 `factor_protocol.json`。

### 4.3 股指日线拼接核验

```bash
python -m research.index_kline
```

只对 IF/IH/IC 的研究分片绘制配置日期范围内的复权日 K 线和成交量图，写入
`runs/index/gtja_time/plots/`。该命令用于拼接/数据核验，不做因子检验。

### 4.4 趋势与日内时间因子过滤器比较（探索性实验）

```bash
python -m research.cta_micro_filter_experiments --period both
```

该比较使用 `config/research/cta_micro_filter_experiments.json` 中预先声明的设定，
不修改已冻结的 `strategies/ts_low.json`，也不用于最终样本外评估。趋势基准是收盘时已知的
20 日时序动量符号，纯多头、固定持有 5 个有效交易日；对照包括现有 `ts_low` 规则、
趋势单独运行、趋势加单日/3 日 EMA/5 日 EMA 的 `ts_low - ts_high` 上尾过滤、最长价格
持续期时点（`pmt`）稳定度过滤，以及 80/50 分位迟滞入场。所有变体走同一整数手账本、
品种池、资金和费用假设，比较毛盈亏、费用、净收益、Sharpe、回撤和换手。

追加的状态比较固定使用 14 日 ADX：`<=20` 作为震荡、`>=25` 且 20 日动量同向作为趋势，
中间区间不分类；另以 ATR/收盘价相对其过去 250 日中位数区分高低波动。只在信号日状态
满足条件时允许新入场，不因之后状态变化而强制清仓。输出 `*_regime_diagnostics.csv`
按信号日的 ADX 状态报告各候选后续 5 日毛收益；这是未扣成本的重叠信号诊断，不能当作
可实现组合净收益。还比较“单日微观上尾 + 原始 AM/PM 成交量持续期比高于其过去 250 日
中位数”的 AND 过滤；另计算分钟价格、成交量持续期的日内均值和样本标准差，比较持续期
均值高于、标准差低于各自过去 250 日中位数的单项和双项过滤。持续期计算使用研究配置
相同的过去 250 日、55% 阈值；成交量持续期比和原始持续期统计均不代表主动买卖量或订单流。

平滑值只使用当日及此前观测；所有分位阈值只以此前 250 个有效日计算；迟滞在微观因子
回落至过去 250 日中位数以下后才重新允许下一次入场。独立的 `pmt` 5 日波动过滤仍只是
“最长价格持续期出现时点”的稳定度代理，不代表持续期绝对水平、订单流或资金建仓。
所有状态与过滤结果均为探索性研究，不应据此作经济机制断言。

`--period research` 只运行 2016–2021；`--period validation` 只运行 2022，并为持续期因子
使用研究期最后 251 个交易日预热（包含首个有效因子日之前所需的 250 日阈值历史）；
`both` 先完成研究期再运行 2022 验证。脚本不读取
2023–2025。输出写到 `runs/research/cta_micro_filter_experiments/`，包括 protocol、
总表、分年、逐日账本、持仓和事件。该实验包含多个预先声明变体，研究期胜出者仍有多重
比较偏差；本次已同时评估所有变体的 2022 结果，不能再将其中挑出的赢家说成独立确认，
也不能据此反复调参。冻结策略文件保持不变；任何新选策略需重新登记研究/验证流程，
且不得读取 2023–2025。

### 4.5 已有的 2016–2022 IC 诊断产物

`runs/research/time_factor_ic_2016_2022/` 保存了此前单次 IC 诊断的
`daily_ic.csv`、`factor_ic_summary.csv`、`factor_ic_heatmap.png`、
`factor_panel_2022.csv` 和 `protocol.json`。目前仓库没有与该产物一一对应、
受版本控制的独立可重跑 CLI；不要虚构可重现命令，也不要把它与上面只读到 2021 的
两个筛查程序混为一谈。

## 5. 商品策略与整数手数回测

策略定义和参数在 `strategies/ts_low.py`、`strategies/ts_low.json`。信号为当日
`ts_low - ts_high`；按单品种过去有效交易日因子分布计算分位阈值，信号日收盘后确定，
下一可交易日开盘做多，达到持有期后退出。只做多，不做空、不反向。

### 5.1 允许运行的研究与验证命令

```bash
python -m backtest.engine --research
python -m backtest.engine --validation
```

两者均为真实整数手数模拟，均使用同一个固定组合设定，而不是遍历策略配置的 27 种
信号参数：

| 参数           | 当前设定                                                             |
| -------------- | -------------------------------------------------------------------- |
| 计分区间       | `--research`：2016–2021；`--validation`：2022                   |
| 因子           | `ts_low - ts_high`                                                 |
| 方向           | 只做多，收盘信号、下一可交易日开盘入场                               |
| 回看/阈值/持有 | 250 个有效交易日 / 严格高于历史 80% 分位（最高 20%）/ 5 个有效交易日 |
| 初始权益       | 每个区间独立从人民币 1 亿元起算                                      |
| 名义敞口       | 目标上限 1.5 倍；在当年流动性池与四个策略板块的交集内分配            |
| 合约手数       | 按原始主力开盘价 × 静态乘数换算，向下取整整数手                     |
| 保证金         | 固定按权益的 20% 假设，不代表历史逐日交易所保证金                    |
| 成本           | 历史手续费；以前一年度估算 tick 作为滑点                             |
| 风控           | 组合峰值回撤达到 20% 后停止开仓、尝试清仓，并暂停至下一自然年        |
| 年化           | 年化收益、波动率与 Sharpe 使用 242 个交易日                          |
| 品种池         | 年度流动性筛选再与农产品、黑色钢铁、有色金属、化工四类取交集         |

每个区间只计本区间收益；区间前行情只用于回看，持仓和权益不跨区间继承。
当日日线高低价相同仅作为锁价代理，不能模拟真实涨跌停排队、逐日保证金、结算价
或换月结算现金流。因此这是含整数手数、成本与风控的研究估算，不是交易所级成交回测。
账本的持仓损益仍由连续主力加法复权开盘价变化估算；换月日只按旧主力前一日收盘、
新主力当日开盘估算交易成本，并不结转真实合约现金流。持仓跨自然年继续计持有日数；
若样本结束时仍有持仓，不会虚构强平，输出的期末权益不含假定的末日清算费用。

输出统一在 `runs/strategies/ts_low/`：

| 文件                                                          | 内容                                   |
| ------------------------------------------------------------- | -------------------------------------- |
| `integer_lot_research.csv` / `integer_lot_validation.csv` | 区间汇总绩效                           |
| `*_annual.csv`                                              | 每年净收益、Sharpe、回撤和交易笔数     |
| `*_daily.csv`                                               | 逐日权益、净收益、敞口、成本与风控状态 |
| `*_positions.csv`                                           | 品种、合约和整数手持仓                 |
| `*_events.csv`                                              | 开平仓、换月、阻单与熔断事件           |
| `plots/integer_lot_<period>/<strategy>/`                    | 净值曲线与最大回撤图                   |

图表由 `backtest.engine` 的 `plot_performance` 自动生成；无需另跑绘图脚本。
`--research` 对应 `<period>=research`，`--validation` 对应 `<period>=validation`。

### 5.2 冻结与样本外限制

`python -m backtest.engine --oos` 是已冻结参数下的 2023–2025 整数手最终评估入口，
需要 `strategies/ts_low.json` 中唯一的 `chosen` 参数、`frozen_at` 和
`tfcta.config.final_evaluation` 审计守卫。当前配置冻结参数为回看 250 日、quantile
0.8、持有 5 日。根据项目维护者确认，截至 2026-10-18 尚未读取 2023–2025 行情；
这段数据保留为未触碰的最终样本外窗口。不要在研究或验证过程中运行该命令；
只有策略与实现冻结、明确批准最终评估后，才可通过审计守卫运行该命令。

## 6. 测试

运行全部测试：

```bash
python -m pytest
```

针对本轮清理后的策略、成本与回测入口做快速检查：

```bash
python -m pytest tests/data/test_costs.py tests/strategies/test_ts_low.py tests/backtest/test_metrics.py
```

大多数测试用合成数据；研究/验证回测需要完整本地行情缓存，且不能以测试通过替代数据边界检查。
