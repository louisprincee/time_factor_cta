<!-- 验证登记簿：由 ledger.write_validation_log 自动生成，勿手改 -->
### 2022 验证

防重复以 `data/validation_2022/ledger.jsonl` 为准，本表是可读索引。
Fingerprint = 因子与方向、池、费率、滑点及执行配置；step6 另按 book_key（等效因子权重 + 池 + 品种）拦截，
同一组因子在同一池子上看过 2022 后，换成本或执行口径也不能再验证。
通过标准：净算术 Sharpe >= 0.5 且净年化 > 0。

| 时间/来源 | 池 | 因子（乘在原始值上的方向） | 品种数 | 费率 | 滑点(tick) | 净年化 | 净 Sharpe | 净最大回撤 | 通过 | Fingerprint | 结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- | --- |
| 2026-09-30T15:06:45 | 全部 | `slate:orb30:m30:{}` | 40 | - | 1.0 | -4.91% | -1.974 | -4.89% | 未通过 | `4b3413b406e8547f` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:orb30_trend:L120:{"lookback": 120}` | 40 | - | 1.0 | -2.01% | -0.911 | -3.94% | 未通过 | `888e1d1e25f6b5b5` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:orb30_compress:pr0.8:{"rule": 0.8}` | 40 | - | 1.0 | -0.93% | -0.871 | -1.37% | 未通过 | `857724abda61e41b` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:dual_thrust:k0.7:{"k": 0.7}` | 40 | - | 1.0 | -0.39% | -0.554 | -0.91% | 未通过 | `a2458929d076c542` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:intraday_momentum:last30:{}` | 40 | - | 1.0 | -7.18% | -9.353 | -6.85% | 未通过 | `06983bafeda4810d` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:vwap_fade:k1.0:{"k": 1.0}` | 40 | - | 1.0 | 0.23% | 0.634 | -0.28% | 通过 | `5af7b55f5e3aaf47` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:gap_fade:g0.6:{"g": 0.6}` | 40 | - | 1.0 | -0.19% | -0.404 | -0.41% | 未通过 | `32b84e6880fb0c45` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:overnight_momentum:day:{"window": "day"}` | 40 | - | 1.0 | -5.41% | -3.856 | -5.24% | 未通过 | `8a13226ece97cb65` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:donchian_hourly:n120:{"n": 120}` | 40 | - | 1.0 | -1.98% | -0.453 | -5.84% | 未通过 | `1d806d2693d96fb7` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:ema_hourly:20/120:{"fast": 20, "slow": 120}` | 40 | - | 1.0 | -0.93% | -0.191 | -5.40% | 未通过 | `4c83bff37bd4d26b` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:tsmom_daily:L250:{"lookback": 250}` | 40 | - | 1.0 | -1.22% | -0.314 | -5.76% | 未通过 | `b18efe17a14c526e` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:06:45 | 全部 | `slate:bollinger_reversion:z2.0:{"entry": 2.0}` | 40 | - | 1.0 | 0.29% | 0.105 | -3.51% | 未通过 | `e3e3824ea2764916` | [运行结果](../runs/20260930_150645_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:15:24 | 全部 | `slate:carry_daily:band0.05:{"band": 0.05}` | 40 | - | 1.0 | -1.03% | -0.316 | -4.55% | 未通过 | `2be795b51934a3e1` | [运行结果](../runs/20260930_151524_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:15:24 | 全部 | `slate:tsmom_carry_daily:L250:{"lookback": 250}` | 40 | - | 1.0 | -1.21% | -0.378 | -4.95% | 未通过 | `1ac26d0f468f491e` | [运行结果](../runs/20260930_151524_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:15:24 | 全部 | `slate:slate_combo:carry_daily=band0.05@0.118161,donchian_hourly=n120@0.073859,ema_hourly=20/120@0.071892,tsmom_carry_daily=L250@0.095868,tsmom_daily=L250@0.076936,vwap_fade=k1.0@0.563286` | 40 | - | 1.0 | -0.40% | -0.318 | -1.84% | 未通过 | `e1a2b5a4105c77cc` | [运行结果](../runs/20260930_151524_intraday_slate_validation2022/performance.csv) |
| 2026-09-30T15:18:25 | 全部 | `ml:huber:orb30\|eod:core+carry+time` | 40 | - | 1.0 | -0.09% | -0.046 | -2.03% | 未通过 | `b1976f98096cab6e` | [运行结果](../runs/20260930_151825_orb_ml_validation2022/performance.csv) |

### 样本外（2023 起）

台账 `data/oos/ledger.jsonl`，只有 2022 通过的书才会出现在这里，每本书在同一窗口只测一次。

| 时间 | 池 | 因子 | 窗口 | 净年化 | 净 Sharpe | 净最大回撤 | Fingerprint | 结果 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-09-30T15:19:20 | 全部 | `slate:vwap_fade:k1.0:{"k": 1.0}` | 2023-01-01..2025-12-31 | -0.10% | -0.254 | -0.80% | `5af7b55f5e3aaf47` | [运行结果](../runs/20260930_151920_intraday_slate_oos/performance.csv) |

### 作废记录

输入数据有缺陷的运行，作废后按同一规格重跑一次，上面两张表只列重跑结果。

| 时间 | Fingerprint | 原因 | 作废的运行 |
| --- | --- | --- | --- |
| 2026-09-30T15:14:39 | `2be795b51934a3e1` | carry_daily：2022 展期外部分区只到 2022-01-11（原始 roll_yield 未下载），之后信号缺失；已补下载 41 品种 2022 数据并重建分区 | D:\liushengqi\time_factor_cta\runs\20260930_150645_intraday_slate_validation2022 |
| 2026-09-30T15:14:39 | `1ac26d0f468f491e` | tsmom_carry_daily：2022 展期外部分区只到 2022-01-11（原始 roll_yield 未下载），之后信号缺失；已补下载 41 品种 2022 数据并重建分区 | D:\liushengqi\time_factor_cta\runs\20260930_150645_intraday_slate_validation2022 |
| 2026-09-30T15:14:39 | `e1a2b5a4105c77cc` | slate_combo：2022 展期外部分区只到 2022-01-11（原始 roll_yield 未下载），之后信号缺失；已补下载 41 品种 2022 数据并重建分区 | D:\liushengqi\time_factor_cta\runs\20260930_150645_intraday_slate_validation2022 |
<!-- 验证登记簿结束 -->
