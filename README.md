# 商品与股指时间因子

从米筐分钟数据出发，检验兴业证券、国泰君安两篇时间维度研报里能迁过来的因子：高低点时点、价格和成交量持续期、早盘路径。当前只做到因子研究。没有策略候选，也没有样本内或样本外回测。

## 目录

```
src/tfcta/            数据、因子公式、手续费和预测性统计
  prepare/            分片、品种池、因子缓存、最小变动价位
  index/gtja_time.py  国泰君安 28 个股指时间因子的公式
research/             因子研究。一个题目一个模块，不构造持仓
strategies/           策略候选。现在是空的
backtest/             回测引擎。现在没有注册策略，不能跑 IS / OOS
config/research/      研究开始前写定的因子和判定
download/             米筐下载
docs/                 研报和已经做过的检验记录
tests/                与上面四层对应
```

研究代码不引用 `backtest`。策略以后只从 `strategies/` 进入 `backtest/`。因子图留在 `research/`。

## 环境

```bash
conda activate factor-mining
pip install -r requirements.txt
```

下载脚本另外需要 `rqdatac` 和 `tqdm`。米筐凭证写在 `config/rqdata.env`（不入库）：

```
RQDATAC_LICENSE=...
```

或用户名密码 `RQDATAC_USERNAME` / `RQDATAC_PASSWORD`。命令都在仓库根目录运行。可选环境变量：`TFCTA_MINUTE_DIR`、`TFCTA_DATA_ROOT`、`TFCTA_RUNS_ROOT`、`TFCTA_CONFIG_DIR`。

## 数据边界

| 分区 | 日期 | 用途 |
|---|---|---|
| 预热 | 2016 年以前 | 滚动窗口，不计结果 |
| 研究期 | 2016–2021 | 因子检验 |
| 2022 | 2022 | 只能走专门的验证入口，不能用来改规则 |
| 严格样本外 | 2023–2025 | 锁定。只有冻结策略进入 `final_evaluation` 才能读 |

研究接口读到 2022 年及以后会抛 `HoldoutViolation`。现在没有冻结策略，这三年保持锁定。

## 数据

分钟行情、外部数据和手续费：

```bash
python download/commodity/getRiceQuantData_MinFreq.py -t 20100101-20251231
python download/commodity/getRiceQuantExternalData.py --start-date 20100101 --end-date 20251231
python download/commodity/getFeeHistory.py --partition research
```

2022 和样本外手续费把 `--partition` 换成 `validation_2022` 或 `oos`。股指研究期单独下载，不并进商品池：

```bash
python download/index/getRiceQuantIndexFutures.py --symbols IF IH IC --start-date 20150101 --end-date 20211231
```

然后做商品预处理。分片、品种池、因子缓存、最小变动价位：

```bash
python src/tfcta/prepare/shard.py
python src/tfcta/prepare/universe.py
python src/tfcta/prepare/factors.py
python src/tfcta/prepare/ticks.py
```

2022 验证分片：`python src/tfcta/prepare/shard.py --validation-from-monolith`。

## 因子研究

都只读 2016–2021。配置在 `config/research/`，结果在 `runs/`。

```bash
python -m research.time_factor_screen
python -m research.gtja_time
python -m research.index_kline
python -m research.morning_clock features
python -m research.morning_clock return
python -m research.morning_clock cases
python -m research.morning_clock path
python -m research.morning_clock flow
python -m research.morning_clock state
python -m research.cross_market
python -m research.incremental
python -m research.timing_cutoff
```

`morning_clock` 是早盘高低点时钟的一次研究：特征表、位置与收益、个例、路径效率、开盘量能、期限结构与仓单。`return --exit 1500` 把持有改到 15:00，`return --balance` 做涨跌各半重采样。

股指因子公式在 `src/tfcta/index/gtja_time.py`，筛查在 `research/gtja_time.py`。K 线只用来看 2016–2021 的复权拼接。

检验记录在 `docs/research/` 和 `docs/commodity/高低点时钟的经济解释与下一步.md`。

## 策略和回测

还没开始。候选写进 `strategies/`，冻结之后才用 `backtest/engine.py` 算样本内绩效和净值。样本外沿用 `tfcta.config.final_evaluation`：配置里有 `chosen` 和 `frozen_at`，测试窗口已结束，读取包在这个上下文里，并往 `data/commodity/oos/ledger.jsonl` 追加记录。

## 测试

```bash
python -m pytest
```

测试用合成数据，不读锁定的样本外行情。
