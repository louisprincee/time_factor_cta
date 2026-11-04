# Repository instructions

Read this file before changing the repository. If it conflicts with the code, trust the code and update this file in the same change.

## Project and active strategies

This project researches time-of-day factors for Chinese commodity and index futures.

- Commodity strategies are registered in `strategies/commodity_catalog.py`. Their default research window is 2011-2021.
- The only registered index strategy is frozen `index_clv_rally_base_long_2x`. The prior 1.5x comparison strategy implementation and config were removed at user request; its research results and ledger remain historical records. The active strategy uses long-only base b = 0.5 times the fraction of positive 21/63/126-session compounded equal-weight close-return windows. Complete zero/negative windows get no long vote; incomplete windows retain a long vote (first reduction needs 21 actual returns). Remaining budget 1-b goes to CLV<0.4 longs only; otherwise it stays cash. Rally z>0 multiplies only that dynamic CLV allocation by 2.0, so pre-vol targets are 0..2.0. No shorts. Same 10M/L1/RV20/breaker basis. Half the budget is a maximum pre-risk base allocation, not a minimum actual holding. The 2x study (18.95% annualized / Sharpe 1.49 / -14.87% max drawdown, mean actual exposure 50.13%, max exposure 201.49%, max margin occupancy 40.30%) passed its predeclared comparison with the historical 1.5x baseline (16.82% / 1.44 / -14.50%) on the 2015-2021 research sample only. Frozen at user request on 2026-11-04; OOS is eligible but has not been run, and must not run without the user's explicit request. Freezing records the rule, not evidence of OOS validity or deployability.
- `index_clv_rally_equal_up_long` was explicitly deleted at user request; its historical result (21.59% / 1.60 / -11.39%) and caveats remain in `docs/index_research_review.md`. The retired `index_clv_rally_equal_long` and `index_clv_adaptive_long` implementations stay removed. Preserve historical economic validation studies; do not restore retired rules.
- The 3×3 body/CLV threshold study is preregistered in `config/research/index_adaptive_signal_grid.json`. The selected body>0.7, pooled CLV>=0.8 variant ranked highest by total return among nine variants but lower by Sharpe and with deeper drawdown than the Sharpe winner. Report all trials and selection bias. Freezing records the chosen rule; it does not establish an edge. The grid was run on a different base strategy, so do not call it the best setting for the current rally base.
- Historical component attribution: `python -m research.index.index_adaptive_attribution` (spec `config/research/index_adaptive_attribution.json`, research only). Its adaptive components are held in research-only helpers; core CLV reversal and rally ×1.5 carry most of that historical book's return. Index `summary.csv`/`annual.csv` report equal-weight buy-and-hold comparison.
- Nothing in this repository is a proven deployable edge. Historical research is observational and often post-selected. Read `docs/guide.md`, `docs/index_guide.md`, and `docs/index_research_review.md` for strategy rules, caveats, and prior findings. Do not restore retired strategies or deleted candidate runners.

## Data permissions and sample boundaries

The top level of `data/` contains only `commodity/` and `index/`.

- Commodity research uses 2011-2021. Index research uses the common IF/IH/IC period beginning 2015-04-16 and ending in 2021; exclude IM from comparable panels.
- Both markets use `research/` for the research window and one `oos/` root for 2022-2025; do not split OOS files into year directories. Partition by data type only. Index minute shards keep price, open interest, and dominant-contract fields together in one file per symbol.
- Index external data is grouped by type under `data/index/external_rqdata/` (`contracts/`, `roll_yield/`, `spot/`, `flows/`, `member_rank/`, `volatility/`, `l1_close/`, `l1_ticks/`, and `constituents/`). OOS external files use the matching type beneath `oos/`. OOS tick costs are estimated from prior-year minute bars at runtime; do not add a competing static tick table.
- Research and out-of-sample are separate data directories. A backtest reads `research/` or `oos/` according to the requested years. There is no ledger and no per-strategy unlock. Out-of-sample results have not been produced; do not invent or restore them.
- Research loaders still refuse out-of-sample paths. `assert_research_only`, `assert_no_holdout_dates`, and `HoldoutViolation` in `src/tfcta/config.py` enforce that boundary. Do not add a second unlock path. Years after 2025 are out of range.
- Before a study using external data such as carry, open interest, or warehouse data, verify complete coverage for the requested sample. Missing required data or costs must raise explicitly.

## Research and backtest protocol

1. Before producing new research numbers, preregister the hypothesis, parameters, sample boundaries, exclusions, and pass criteria in the appropriate JSON under `config/research/` or the strategy directory. State explicitly which dates are not read.
2. Use only information available at the signal time. Close signals execute at the next tradable open; external data is lagged by one trading day unless its specification explicitly says otherwise. Rolling statistics exclude the current day unless declared otherwise.
3. Report every declared variant and account for parameter trials. Do not select signs or thresholds from returns, tune per symbol, or present overlapping research as independent evidence.
4. Default costs are historical exchange fees (amount fees multiplied by 1.01 plus CNY 0.01 per lot) and one tick of slippage per side, based on the prior year's estimated tick. Missing fee, tick, or multiplier coverage must fail explicitly.
5. Only the user may freeze a strategy. Do not edit frozen trading parameters; a changed rule requires a new strategy ID. Research and out-of-sample backtests use the same command and the matching data directory.
6. Do not launch batches of exploratory backtests or add thresholds, filters, or permanent-stop rules without a clearly scoped request.

## Code, tests, and files

- Match surrounding naming, formatting, helpers, and localization. Existing comments and docstrings are usually short and Chinese; identifiers are English.
- Prefer precise changes in existing modules. Search for prior art before adding a helper or file. Remove code only when it is demonstrably dead; research utilities supporting retained studies are not dead merely because the active strategy does not call them.
- Keep tests focused on material behavior and high-risk invariants such as look-ahead, sample guards, accounting, execution, and frozen strategy rules. Use synthetic data only. Tests must never read real OOS data. Keep index tests in `tests/test_index.py`; do not add a test file without an explicit request.
- Use existing linters, builds, and test commands. Run the smallest relevant test set first; run the full suite when shared core behavior changes or focused results indicate broader risk.
- Do not add dependencies beyond `requirements.txt` without asking. Do not store large artifacts in the repository; `data/` and `runs/` are local data/output areas.
- Maintain only the `main` branch. Do not create branches, commit, or push unless explicitly asked. If committing, stage only in-scope files.
- RiceQuant credentials are in `config/rqdata.env`; never print, commit, or move them.

## Useful commands

Run commands from the repository root:

```bash
python -m pytest tests/test_research.py tests/test_index.py
python -m backtest.engine --research
python -m backtest.engine --market index --research
python -m backtest.engine index_clv_rally_base_long_2x --research
python -m research.index.index_adaptive_signal_grid
python -m backtest.engine --oos
python -m backtest.engine --market index --oos
```

`--research` uses the research partition. `--oos` uses 2022–2025. A custom `--start/--end` must stay inside 2011–2025 for commodities and 2015–2025 for index futures. Out-of-sample has not been run.
