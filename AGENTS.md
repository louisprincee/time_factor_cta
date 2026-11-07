# Repository instructions

Read this file before changing the repository. If it conflicts with the code, trust the code and update this file in the same change.

## Project and active strategies

This project researches time-of-day factors for Chinese commodity and index futures.

- Commodity strategies are registered in `strategies/commodity_catalog.py`. Their default research window is 2011-2021.
- Only three index books remain in `INDEX_BOOKS`, the formal registry, and the OOS registry: (1) `index_tactical_clv_only` (default), no momentum; when pooled CLV<0.4, equal-weight the full tactical budget, otherwise cash. (2) `index_tactical_clv_factor_select`, no momentum; when pooled CLV<0.4, allocate across all products proportional to 1+ favorable votes (IF `dfp_top3`, IC `dfp_top3`, IC `ts_high`). These factor cells were selected from the same sample and were not significant; report selection bias. (3) `index_clv_breakout_sleeve`, unchanged pure-CLV sleeve plus a fixed 0.5 long-only budget using a prior-55-session close breakout and prior-20-session low exit, equally split across active products. All trade IF/IH/IC, do not short, and use the same 10M/L1/RV20/breaker basis; rally z>0 multiplies only the CLV sleeve by 2.0. They have only 2015-2021 research results. Pure CLV had the best Sharpe/drawdown (1.687 / -9.77%); the standard breakout had higher cumulative return but lower Sharpe/deeper drawdown (1.614 / -15.16%) and no significant paired increment (HAC(5) t=0.62). The factor book is retained as the one user-selected factor strategy, not because its selected cells are independent evidence. Former momentum, bull-half, MACD-half, and EMA20/60-half strategy packages and unused protocols have been removed. The nonweak-breakout historical runner and its protocol remain solely to reproduce that variant; historical rules and findings for removed variants remain documented, not runnable strategies. The prior comparison protocol has been removed; future comparison of the active set must follow `config/research/index_selected_strategy_comparison.json`. The engine writes each active book's benchmark charts and, after a multi-book run, one board of current strategies plus equal-weight buy-and-hold at `runs/index/research/index_strategy_comparison/`. OOS has not been run and must not run without the user's explicit request. Freezing records rules, not evidence of OOS validity or deployability.
- `index_weak_close_mechanism_classifier` remains a research diagnostic, not a trading strategy or sleeve; it joins cross-index weak-close breadth, high-open reversal, and L1 tail pressure in an expanding-year classifier. Classifier results failed the preregistered gate (AUC improved but Brier worsened); it was not included in the user's freeze request.
- The standard `index_clv_breakout_sleeve` is the only retained trend strategy. The nonweak-breakout variant additionally required pooled CLV>=0.4 at signal close for new entries; it is now retired after its post-hoc 2015–2021 comparison had worse risk-adjusted results. Preserve that study and its limitations. Existing price/L1 indicators do not causally label information shocks versus liquidity pressure; the prior classifier did not meet its preregistered gate.
- `index_clv_rally_equal_up_long` was explicitly deleted at user request; its historical result (21.59% / 1.60 / -11.39%) and caveats remain in the historical section of `docs/index_guide.md`. The retired `index_clv_rally_equal_long` and `index_clv_adaptive_long` implementations stay removed. Preserve historical economic validation studies; do not restore retired rules.
- The 3×3 body/CLV threshold study is preregistered in `config/research/index_adaptive_signal_grid.json`. The selected body>0.7, pooled CLV>=0.8 variant ranked highest by total return among nine variants but lower by Sharpe and with deeper drawdown than the Sharpe winner. Report all trials and selection bias. Freezing records the chosen rule; it does not establish an edge. The grid used a different base strategy and does not establish the best setting for any currently retained book.
- Historical component attribution: canonical entry `python -m research.index strategy attribution` (specs `config/research/index_adaptive_attribution.json`, `index_base_momentum.json`, and `index_base_budget_transfer.json`, research only; old module commands remain available). Its adaptive components are held in research-only helpers; core CLV reversal and rally ×1.5 carry most of that historical book's return. Index `summary.csv`/`annual.csv` report equal-weight buy-and-hold comparison.
- Time-factor increment test (run 2026-11-05; all three factors "not supported": pooled t 1.50/0.41/0.92, Holm p 0.40/0.72/0.72; early IF `ts_low_pm` t 2.49): canonical entry `python -m research.index factor time-increment` (spec `config/research/index_time_factor_increment.json`; old module command remains available). It tests `ts_low_pm`, `dfp_top3` and `ts_high` beyond CLV, return and volume on the pooled IF/IH/IC panel (signals 2015-04-16..2021-12-29, Holm over 3) and replicates on early IF (2011-01-04..2015-04-14), on a unified 09:31–11:30/13:01–15:00 clock. It must reproduce the time-factor screen per product from 2016 before reporting. Signal-level only; it does not change the frozen strategy.
- Nothing in this repository is a proven deployable edge. Historical research is observational and often post-selected. Read `docs/guide.md` and `docs/index_guide.md` for strategy rules, caveats, and prior findings. Do not restore retired strategies or deleted candidate runners.

## Data permissions and sample boundaries

The top level of `data/` contains only `commodity/` and `index/`.

- Commodity research uses 2011-2021. Index research uses the common IF/IH/IC period beginning 2015-04-16 and ending in 2021; exclude IM from comparable panels.
- Both markets use `research/` for the research window and one `oos/` root for 2022-2025; do not split OOS files into year directories. Partition by data type only. Index minute shards keep price, open interest, and dominant-contract fields together in one file per symbol.
- Index external data is grouped by type under `data/index/external_rqdata/` (`contracts/`, `roll_yield/`, `spot/`, `flows/`, `member_rank/`, `volatility/`, `l1_close/`, `l1_ticks/`, and `constituents/`). OOS external files use the matching type beneath `oos/`. OOS tick costs are estimated from prior-year minute bars at runtime; do not add a competing static tick table.
- Research and out-of-sample are separate data directories. A backtest reads `research/` or `oos/` according to the requested years. There is no ledger and no per-strategy unlock. Out-of-sample results have not been produced; do not invent or restore them.
- Research loaders still refuse out-of-sample paths. `assert_research_only`, `assert_no_holdout_dates`, and `HoldoutViolation` in `src/tfcta/config.py` enforce that boundary. Do not add a second unlock path. Years after 2025 are out of range.
- The frozen index factor-tilt books have an OOS factor-input path: it uses the final 250 research trading days to warm up rolling DFP thresholds, then generates factor panels only for requested 2022-2025 OOS dates. This implementation has not been run against OOS data or backtested; use synthetic tests only unless the user explicitly requests an OOS run.
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
python -m backtest.engine index_tactical_clv_only --research
python -m research.index strategy signal-grid
python -m backtest.engine --oos
python -m backtest.engine --market index --oos
```

`--research` uses the research partition. `--oos` uses 2022–2025. A custom `--start/--end` must stay inside 2011–2025 for commodities and 2015–2025 for index futures. Out-of-sample has not been run.
