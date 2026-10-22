# AGENTS.md — working rules for this repository

Read this before changing anything. It reflects the repo on 2026-10-22 (two live books: `market_clock_pool` and `market_clock_slow_short`).
If the code and this file disagree, trust the code, then update this file in the same change.

## 1. What the project is

We build time-dimension ("when did it happen inside the day") factor strategies for Chinese futures.
They are based on two broker reports in `docs/`:

| Report | Market | Core idea |
|---|---|---|
| 国泰君安期货 2025-02-25, 股指策略系列五 (GTJA) | IF / IH / IC index futures | Price/volume **duration** factors with a **fixed** threshold (3 points, 50 lots), plus behavioural timestamps: `time_to_high/low`, `time_to_max_volume/turnover`, `time_to_half_volume`, `afternoon_high_time`. Each factor is rolling z-scored, and the signal is a sign conjunction of **two** factors. Next-open to next-open return, 2 bp round trip, everything in-sample. |
| 兴业证券 2026-06-09, 高频时间维度国债期货择时 (XY) | T / TL treasury futures (+ single-stock CTA overlay) | Duration uses a **dynamic** threshold: the M-th percentile (50–60) of absolute minute changes over the past N≈250 days. Factors: DFP (fair price at the longest duration minus close, top-1/top-3), PMT/VMT (minute of the longest price/volume duration), VR (AM/PM volume-duration ratio), and timestamps of high, low, max volume and turnover, PM high/low and AM spike count. Each factor gets its own rolling-quantile band, then an equal-weight combo. Robust parameters are taken from the centroid of the grid, not the best cell. No cost is stated. |

Priors the reports agree on:
- A late low is bullish.
- A late high, especially a late PM high, is bearish.

Where they conflict:
- On max-volume/turnover timing the two reports point opposite ways. `FACTOR_SIGNS` in `src/tfcta/config.py` holds our fixed priors.
- **Signs are priors.** Never flip them based on research-period returns.

Factor code lives in `src/tfcta/factors.py`. Key functions: `duration_series`, `dfp_factors`, `timestamp_factors`, `report_factors`, `rolling_threshold`. The GTJA index variants are in `src/tfcta/index/gtja_time.py`.

## 2. Where the research stands (do not re-discover this)

Be honest about this in every report. Nothing in this repo is a proven deployable edge yet; the mainline is being tested year by year.

**Factor screens** (`research/time_factor_screen.py`, `research/gtja_time.py`, outputs in `runs/research/`, `runs/index/`):
- On commodities, `ts_low`, `ts_low − ts_high` and `dfp_*` have cross-sectional t > 3 at the close decision.
- Only 1 of 115 cells passes the full predeclared gate (t, 5/6 years, 4/6 sectors), and 0 are allowed to pair.
- The GTJA factors on IF/IH/IC, 2016–2021: no single factor reaches |NW t| ≥ 2.

`strategies/catalog.py` lists the two live books. The four long-only tail rules, their write-up and `backtest/index_market.py` were removed. Do not restore them.

**Live books (two, both frozen).** Write-up `docs/当日分数且展期为正.md`. Both books share these rules:
- Budget: |weight| × 1 × equity / pool width.
- Risk: 10% drawdown → liquidate and halt 10 trading days.
- Execution: on the next trading day's first bar (night open if there is a night session, else day open). Unchanged weight stays on; only the lot difference trades.

1. **按池宽1倍, `market_clock_pool`** (`strategies/market_clock_carry/`, frozen 2026-10-21).
   - Signal: at the close, the five-factor market score must be > 0, and that name's main-versus-sub `annualized_yield` lagged one trading day must be strictly > 0. Weight 1, otherwise 0.
2. **按池宽1倍加慢趋势空头, `market_clock_slow_short`** (`strategies/market_clock_slow_short/`, frozen 2026-10-22).
   - The long leg is identical to `market_clock_pool` and reuses its code.
   - The short leg opens only when the market slow trend is < 0.
   - Market trend = mean sign of the pool's equal-weight close-to-close return, summed over 60/120/250 days. With < 250 days of history a day is not a bear day, so 2016 has no shorts.
   - On a bear day, a name gets weight −1 when both hold:
     - its own trend (mean sign of 60/120/250-day returns, (closew_t − closew_{t−n}) / close_{t−n}) is < 0;
     - its lagged carry is < 0.

- **OOS records reset (user decision 2026-10-22).** The commodity OOS ledger (`data/commodity/oos/ledger.jsonl`) and every 2022+ result folder were cleared.
  - Before the reset, 2022–2024 had been opened for the books of that time; 2025 had no recorded open.
  - An empty ledger does not make any year clean. Treat 2022–2024 as seen; rerunning them is only a reproduction.
  - With the ledger empty, `--validation`, `--year` and `--oos` read the data again and write new entries; they no longer just print an old result.

- **Roll yield.** RiceQuant `get_roll_yield(type="main_sub")`. `yield` = (main settlement − sub settlement) / sub settlement, verified against contract dailies on 3,840 research days (error 1e-17). 99.7% of rows have the main maturing before the sub; LR, JR, RI, RS, B, SF, PM, BB occasionally invert.
- **Research 2016–21** (1 tick + fees):

  | Book | Sharpe | Annual | Max DD | Beta | Alpha | NW t |
  |---|---|---|---|---|---|---|
  | 按池宽1倍 | 1.36 | 7.2% | −5.7% | 0.25 | 3.7% | 2.22 |
  | 加慢趋势空头 | 1.37 | 7.8% | −5.7% | 0.16 | 5.5% | 2.48 |

  - The slow-short book holds shorts on 294 days (2017-05 to 2020-07).
  - The pool's "bear" days in 2016–21 returned +11.6%/yr, so the research period only prices the insurance.
  - Declared sibling variants in the same bear test, with Sharpe:
    - short leg always on: 1.10;
    - stop long and go short in a bear: 1.12;
    - vol de-risking: 1.42.
- **Signal tests** (`research/market_clock_carry_factors.py`):
  - Timing: score>0 minus else, next-bar pool return +12.7bp/day, NW t 3.22, 6/6 years.
  - Carry: cross-sectional IC 0.022, t 3.20, 5/6 years (2021 negative).
  - Interaction: t 2.25.
  - These numbers were seen in scratch before being declared.
  - Deflated Sharpe at 100 trials is about 0.06 for both books; PBO over the 7 ledger series is 0.53. Do not call the research Sharpe evidence.
- **Dual short failed** (score < 0 and lagged carry < 0 → short).
  - On score < 0 days the next-bar pool return is −0.3bp (t −0.10).
  - Short-only net Sharpe is −0.52.
  - The score's information is one-sided, so the short leg does not use time factors.
- Other variants, the dual-short test and the bear-comparison test were deleted; git history has them.
- **Index transfer test** (`research/index_market_clock.py`, `config/research/index_market_clock_transfer.json`): both frozen books on IF/IH/IC, 2016–2021, fixed three-name pool (`market_score(min_symbols=3)`). Not run yet: RiceQuant `get_roll_yield` returns None for IF/IH/IC (it works for RB and T), so `download/index/getIndexRollYield.py` cannot fill `data/index/external_rqdata/roll_yield/`. Count it as another trial of the same rule family.

**Retired mainline: `market_clock_long`** (code, results and write-up deleted; the numbers are kept only here).
- **Score.** The five report factors are z-scored on each symbol's own 250-day history, averaged per symbol, then averaged over the pool to give a market score.
- **Signal.** Long-only. The whole pool is long when the score is > 0; otherwise flat.
- **Target weight.** Bullish share of the last 5 closes × account multiplier × inverse-vol relative weight (capped at 3).
  - The account multiplier is min(1, 250-day / 20-day pool vol), halved when the 250-day pool return is < 0.
  - The target is held until the bullish share changes.
  - Execution is netted via `simulate_target_lots`.
- **Research period** (2016–21): net Sharpe 1.43, annual 10.2%, vol 7.0%, max drawdown −10.4%. Against the equal-weight pool: beta 0.48, alpha 3.6%/yr, down-capture 0.35.
  - The sector gate was relaxed **post hoc** from 35% to 40% (black sector 36.2%) by the user before freezing. The robustness JSON and the rerun script were deleted with the strategy.
  - Mechanism tests (co-movement, volume surge, OI change) all failed. Treat the strategy as "commodity long, adding after weak market closes", not as a proven liquidity-provision story.
  - Earlier variants were seen and deleted: long-short, equal-weight long (1.31), 3-day, tail entry and passive benchmark.
- **Finding behind it** (research period only): commodity time factors carry information only at the **market** level. The idiosyncratic and sector-rotation parts lose money after costs, and the slow volume factors are just sector bets.

**Older work**, removed from the tree but recoverable from git history:
- The morning-overreaction strategy (兴业 morning rule) and the intraday slate are retired.
- Lesson from the retired books: research Sharpe of 1–1.5 has repeatedly collapsed to roughly 0 out of sample.

## 3. Data partitions and what is already spent (most important section)

| Partition | Dates | Commodity | Index |
|---|---|---|---|
| Warmup | 2015 | in research shards, not scored | same |
| Research | 2016–2021 | `data/commodity/minute_shards/research/` | `data/index/minute_shards/research/` (IF/IH/IC) |
| Validation | 2022 | `validation_2022/`; opened before the 2026-10-22 ledger reset, not clean | inside `holdout_locked/`; **unread** |
| 2023 | 2023 | `holdout_locked/`; opened before the reset, not clean | inside `holdout_locked/`; **unread** |
| 2024 | 2024 | `holdout_locked/`; opened before the reset, not clean | **unread** |
| 2025 | 2025 | `holdout_locked/`; no recorded open before the reset | **unread** |
| Forward | 2026-01-01..2026-10-08 | `data/commodity/data_min/future_*_20260101-20261008.txt`, not sharded; **never read** | — |

Rules:
1. **Sample lock (user decision 2026-10-21).**
   - A new commodity strategy must first pass its predeclared research-period gate and have an economic story. Only then is it frozen.
   - **Freeze gate (user decision 2026-10-21):** `run_span` refuses any window ending in 2022 or later unless every requested book's JSON is `status: frozen` with `frozen_at` and `chosen == parameters`. It refuses before writing the ledger or reading data.
   - Once frozen, `--validation`, `--year Y` and `--oos [--by-year]` open a window. The ledger is written first: 2022 gets a `year_test` entry; 2023+ goes through `final_evaluation`, with `oos_start`, `frozen_at`, `chosen` and file hashes (keyed by repo-relative path, `C.file_hashes`, over `book.FILES`) recorded.
   - A window recorded as opened or finished only prints its saved result. If a recorded result is gone, append a `kind: void` entry with a reason before reopening. The 2026-10-22 reset (§2) removed those records, so this check no longer covers 2022–2024.
   - Do not run any 2022+ window unless the user asks for it in the same turn.
   - Never read 2026. `run_span`, `load_through_year` and `roll_yield` raise before loading it. The `holdout_locked` roll-yield pickles contain rows through 2026-10-08; `roll_yield` drops everything after the window, and reads that partition only inside `final_evaluation`.
   - Never read index 2022+.
2. Research code must go through the guards in `src/tfcta/config.py`: `assert_research_only`, `assert_no_holdout_dates`, `HoldoutViolation`, and the `final_evaluation` context plus ledger. Do not weaken, bypass or monkeypatch them, and do not add an env var or flag that unlocks data.
3. Anything tuned, selected or "fixed" after seeing a commodity window is **in-sample** for that window. An empty ledger does not restore a window's clean status.
4. A new hypothesis is first tested on commodity 2016–2021. A 2022+ window is opened only by an explicit `--year` or `--oos` request in that turn. Never use 2026. Name which window you are using.
5. Before any test that uses external data (carry/roll_yield, warehouse, OI), verify that the partition covers the whole window. A truncated carry file once produced a fake pass. `targets()` calls `require_carry_coverage`, which raises if a pool symbol-year has more than 5% missing roll yield.

## 4. Repository layout

```
src/tfcta/          library: config (paths, partitions, guards, factor signs), data (shards, universe,
                    rolls), factors, costs (fees, ticks, multipliers), prepare/ (sharding, ticks,
                    universe, factor caches), index/ (GTJA index factors)
strategies/         market_clock_carry/: score, roll yield, carry coverage, Book base class, market_clock_pool.json;
                    market_clock_slow_short/: trend and bear helpers, short leg, market_clock_slow_short.json;
                    catalog.py: BOOKS (the two live books) and book_named
backtest/engine.py  simulate_target_lots ledger, metrics (performance, sharpe, max_drawdown, newey_west_t,
                    deflated_sharpe, pbo_cscv), load_research / load_through_year / ledger_inputs, write_results, CLI
research/           market_clock_carry (spans, beta/alpha, ledger, freeze gate), market_clock_carry_factors
                    (signal-level tests), index_market_clock (both books on IF/IH/IC, research period),
                    time_factor_screen, gtja_time; index_kline is only a chart utility
config/research/    predeclared JSON specs for research studies and exploratory strategies
download/           RiceQuant downloaders (commodity minute/external/fees, index minute, index roll yield)
tests/              pytest, synthetic data only (generator in tests/synthetic.py)
docs/               the two reports + 当日分数且展期为正.md (why the two live books)
data/, runs/        gitignored; data inputs and all outputs
```

Outputs:
- Strategy runs go to `runs/strategies/<id>/commodity/<research|oos|YYYY|YYYY-YYYY>/`. Each run writes `summary.csv`, `annual.csv`, `daily.csv`, `positions.csv` and `events.csv`, plus `plots/<name>/`.
- Research studies go to `runs/research/<study>/` or `runs/index/<study>/`, with a `protocol.json`.

## 5. Environment and commands

- Python: `/c/Users/GU494/anaconda3/envs/factor-mining/python.exe` (conda env `factor-mining`). Set `PYTHONIOENCODING=utf-8` on Windows.
- Run everything from the repo root. `pytest.ini` puts `src` and `.` on the path.
- RiceQuant credentials live in `config/rqdata.env`. Never print, commit or move them.

```bash
python -m pytest                                      # must stay green
python -m backtest.engine --research                 # 2016–21，不点名则两条都跑
python -m backtest.engine 按池宽1倍 --research       # 只跑点名的那一条
python -m backtest.engine --research --by-year        # 不点名则每年两条，各一本新的 1e8
python -m backtest.engine --start 2018 --end 2020     # 研究区间；不点名则两条
python -m backtest.engine --validation               # 2022；不点名则两条。先记台账。不是干净样本
python -m backtest.engine --year 2023                # 单年；不点名则两条。先记台账
python -m backtest.engine --oos                      # 2023–2025；不点名则两条。先记台账
python -m backtest.engine --oos --by-year            # 2023、2024、2025 各一本；不点名则每年两条
python -m research.market_clock_carry_factors       # signal-level tests, DSR, PBO
python -m research.index_market_clock                # both books on IF/IH/IC; needs index roll yield (not available)
python -m research.time_factor_screen
python -m research.gtja_time
```

Every 2022+ command above is refused for any book that is not frozen. Both live books are frozen, so only run those commands when the user asks for that window. A window already opened only prints its saved result.

`TFCTA_DATA_ROOT`, `TFCTA_RUNS_ROOT`, `TFCTA_CONFIG_DIR` and `TFCTA_MINUTE_DIR` redirect paths. Use them only for tests and reproduction checks.

Data preparation order (README §数据准备): `prepare/shard.py` → `prepare/universe.py` → `prepare/ticks.py` → `prepare/factors.py`, plus fee and external downloads. `shard.py` refuses to overwrite `holdout_locked/`; the minute downloader refuses incremental merge when more than one monolith exists.

## 6. Research protocol (follow on every change that produces numbers)

1. **Predeclare.**
   - Write the hypothesis, parameters, sample and pass criteria in a JSON under `config/research/` or `strategies/<id>/<id>.json` **before** running.
   - The `sample` text must state what is not read. The loaders check for it: exploratory strategy specs need "不读取2022及2023年以后" and "不读 2026"; research configs need "不读 2022" and "不读 2026".
   - Code validates the JSON against its constants, so the two cannot drift silently.
2. **No parameter sweeps for selection without accounting.**
   - If you test a grid, report every cell, the number of trials, and DSR/PBO where relevant (`deflated_sharpe` and `pbo_cscv` in `backtest/engine.py`).
   - Prefer the 兴业 approach: pick centroid or robust parameters, not the best cell.
   - No per-symbol tuning. Parameters are pool-level.
3. **Report all variants that were declared**, not just winners. Do not drop a losing sibling after seeing results. Keep candidates with positive Sharpe as forward candidates, and give the lower bound as information.
4. **Costs are realistic by default:**
   - Historical exchange fees ×1.01 + 0.01 yuan/lot.
   - Slippage of 1 tick per side, using the prior year's estimated tick (`src/tfcta/costs.py`).
   - No 2-tick sensitivity is required (user decision 2026-10-20): report 1 tick plus fees.
   - Never use RiceQuant's current `tick_size` for history.
   - Missing fee or tick data must raise, not silently skip a symbol or reuse stale values: `require_multipliers`, `fee_tables` (ffill ≤ 31 days), `slippage_tables` (ffill ≤ 1 year), NaN cost raises inside `simulate_target_lots`, `_require_cost_coverage` for the 2022 pool in `load_through_year`.
5. **Timing:**
   - Signals use data known at the decision time. Close signals enter at the next tradable open.
   - External data such as carry is lagged one trading day.
   - Rolling thresholds and quantiles exclude the current day unless the spec says otherwise.
6. **Freezing:** a book is frozen when its JSON has `status: frozen`, `frozen_at` and `chosen == parameters`. Only the user freezes. Unfrozen books cannot open 2022+. Never edit a frozen JSON; a changed rule is a new book with a new ID.
7. **Ledger:**
   - Any 2022+ opening is written to the ledger first.
   - A repeat needs a `kind: void` entry with a reason first.
   - Abandoned strategies get a `kind="decision"` entry.

## 7. Known data and implementation pitfalls

- **Prices:**
  - `closew/highw/loww/openw` are **additive back-adjusted**; for example RU 2016 `closew` is about 510 versus `close` 10315.
  - Never take ratios of w-columns. Use differences of w-columns divided by the raw price, as `load_research` does for open-to-open returns.
- **Trading day:**
  - `trading_date` is the only valid day key, and the night session belongs to the next day.
  - Night length changed over time: DCE and ZCE went from 150 to 120 bars in 2019.
  - Some symbols have no night session before 2019 (C, CS, V, L, PP, EG). SC, FU and SP start in 2018.
- **Clock normalisation:**
  - Commodity timestamps are normalised over the whole trading day **including night**, so the scale shifts on days without a night session.
  - Index futures normalise over the day session only.
  - When you build a new clock, state which scope it uses.
- **Warmup gaps:**
  - The 250-day rolling windows mean there are no signals early on. That is warmup, not a bug.
- **Universe:** the year-y pool uses only year y−1 turnover and valid days (`screen_year`, `load_universe`). Check the symbol count after loading; a truncated fee file once shrank the pool to RB alone.
- **Rolls:** the target ledger charges fees and slippage at roll, using the old dominant's prior close and the new dominant's open. It does not carry real contract settlement. Results are research estimates, not exchange-grade fills.
- **Target ledger (`simulate_target_lots`):**
  - Weights run from −`TARGET_CAP` to `TARGET_CAP` (3); negative is short. `market_clock_pool` uses 0/1, the slow-short book −1/0/1.
  - Notional = |weight| × gross × start-of-day equity / pool width. The selected-count split was deleted with the books that used it.
  - Lots are not resized while the weight is unchanged. The whole book has one breaker, which halts `halt_days` trading days (required argument).
  - `熔断口径回撤` in `daily.csv` resets its peak when a halt ends. Reported max drawdown (`max_drawdown`) does not.
- **No-night days:** about 100 of the 1460 research days have no night session (post-holiday days, and the 2020 night suspension). On those days the full-day `ts_low`/`ts_high` market mean shifts by roughly −0.23/+0.18 sd, because the normalisation grid is shorter.
- **Year-test factors:** `load_through_year` recomputes the five factors from 320 research days before 2022 onward. It asserts that the recomputed values equal the research cache on the last 60 research days. Keep `duration_factors`, `report_factors` and the cache on the same N250/M55 definition.
- **Flow:** `backtest.engine.main` maps switches to `research.market_clock_carry.run_span(start, end, by_year, book)`, which loads data, calls `book.targets(data, columns)` on each book from `strategies.catalog.BOOKS`, runs `simulate_target_lots`, writes results with `write_results` and prints beta/alpha from `summarize`.

## 8. Coding conventions

- Match the surrounding style.
  - Docstrings and comments are short and in **Chinese**. Identifiers are English. Output CSV column names are Chinese.
  - Comment density is low; comments explain "why" and sample boundaries.
- Prefer extending existing modules (`factors.py`, `engine.py`, `costs.py`) over new files.
  - Use one directory per strategy, with `__init__.py` plus `<id>.json`.
  - Delete dead code instead of leaving parallel versions. Git history is the archive.
- No new dependencies beyond `requirements.txt` without asking: numpy, pandas, scipy, matplotlib, pyarrow, pytest.
- Every new factor, strategy or ledger rule gets a test in `tests/<area>/` on **synthetic data**. Tests must never read real 2022+ data. Cover look-ahead (signal on day t cannot use t+1), guard behaviour, and config/code consistency.
- Do not write large artifacts into the repo. `data/` and `runs/` are gitignored, and scratch work goes in `runs/scratch/`, which you delete when done.
- Commit only when asked, on a branch, staging specific files.

## 9. Checklist for any modification

1. Read the relevant module, its JSON spec and its tests. Re-read §3 to decide which partitions the change may touch.
2. If the change produces new numbers, write or adjust the predeclared spec first.
3. Implement it, then add or adjust tests.
4. Run `python -m pytest`. Run only the research-period entry point that the change needs.
5. Update docs that describe changed behaviour or numbers: `README.md` (Chinese user manual), `backtest/README.md`, `strategies/README.md`. Update this file if rules or layout changed.
6. Report back with:
   - what changed;
   - which data partitions were read;
   - net numbers with cost assumptions, including all declared variants;
   - what is in-sample versus clean;
   - what was not verified.

   Keep it short. Do not call research-period Sharpe "evidence" when the same rule family already failed OOS.
