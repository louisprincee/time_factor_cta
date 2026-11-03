# Repository instructions

Read this file before changing the repository. If it conflicts with the code, trust the code and update this file in the same change.

## Project and active strategies

This project researches time-of-day factors for Chinese commodity and index futures.

- Commodity strategies are registered in `strategies/commodity_catalog.py`. Their default research window is 2011-2021.
- The index strategies are registered in `strategies/index_catalog.py`: default `index_clv_rally_fixed_long`, `index_clv_individual_equal_long`, `index_clv_pooled_equal_long`, `index_clv_return_rank_20d_long`, and `index_clv_inverse_vol_20d_long`. All trade IF/IH/IC, not IM. Their research calendar starts on 2015-04-16; actual scoring begins after required data and factor warmup. `python -m backtest.engine --market index --research` runs every registered book; a named strategy runs only that book.
- The index rule is frozen: pooled CLV below 0.4 opens a long position; positive rally z-score scales it by 1.5; fixed IF/IH/IC multipliers are 0.891744548286604, 0.623831775700935, and 1.484423676012461; a 20-day volatility cap limits scaling to 1; there is no short leg. The basis is CNY 10 million with L1 30-minute execution and a 10% drawdown breaker that halts trading for 10 sessions.
- `index_clv_individual_equal_long` instead signals each instrument on its own CLV below 0.4 and splits the daily budget equally among signaled instruments. It retains the rally multiplier, 20-day volatility scaling, and the same execution and breaker basis. The original book remains the default and is unchanged.
- `index_clv_pooled_equal_long` uses the equal-weight pooled CLV below 0.4 as a common signal, then gives IF/IH/IC equal target weights. It retains the rally multiplier, 20-day volatility scaling, and the same execution and breaker basis. The original book remains the default and is unchanged.
- `index_clv_return_rank_20d_long` and `index_clv_inverse_vol_20d_long` keep the pooled CLV signal and total risk budget, but allocate product multipliers by 20-session relative-return rank or inverse realized volatility. Both use an equal-weight warmup and the same execution and breaker basis.
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
python -m backtest.engine index_clv_rally_fixed_long --research
python -m backtest.engine index_clv_individual_equal_long --research
python -m backtest.engine index_clv_pooled_equal_long --research
python -m backtest.engine --oos
python -m backtest.engine --market index --oos
python -m backtest.engine index_clv_return_rank_20d_long --start 2022 --end 2025
```

`--research` uses the research partition. `--oos` uses 2022–2025. A custom `--start/--end` must stay inside 2011–2025 for commodities and 2015–2025 for index futures. Out-of-sample has not been run.
