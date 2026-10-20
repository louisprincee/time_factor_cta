# AGENTS.md — working rules for this repository

Read this before changing anything. It reflects the repo on 2026-10-20.
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

**Frozen catalog** (`strategies/catalog.py`, frozen 2026-10-19):
- Four long-only rules, all built on the `ts_low − ts_high` tail: strictly above its own prior 250-day 80th percentile, with the 10-day trend sign, entry at the next open, held 5 days.

| Rule | Commodity Sharpe, research → OOS 2023–25 |
|---|---|
| `trend_ema3` | 0.81 → 0.08 |
| `trend_high_atr` | 1.05 → 0.18 |
| `trend_adx_range` | 1.06 → −1.06 |
| `regime_two_leg` | 1.37 → −0.47 |

- On index futures, research period only, the ranking reverses. See `docs/四策略做法与表现.md`.

**Mainline: `market_clock_long`** (frozen 2026-10-20; code and frozen JSON in `strategies/market_clock_long/`; registered in `strategies/research_catalog.py`; write-up in `docs/全市场时钟做法与表现.md`).
- **Score.** The five report factors are z-scored on each symbol's own 250-day history, averaged per symbol, then averaged over the pool to give a market score.
- **Signal.** Long-only. The whole pool is long when the score is > 0; otherwise flat.
- **Target weight.** Bullish share of the last 5 closes × account multiplier × inverse-vol relative weight (capped at 3).
  - The account multiplier is min(1, 250-day / 20-day pool vol), halved when the 250-day pool return is < 0.
  - The target is held until the bullish share changes.
  - Execution is netted via `simulate_target_lots`.
- **Research period** (2016–21): net Sharpe 1.43, annual 10.2%, vol 7.0%, max drawdown −10.4%. Against the equal-weight pool: beta 0.48, alpha 3.6%/yr, down-capture 0.35.
  - Six variants are registered in `config/research/market_clock_long_robustness.json`; rerun them with `python -m research.market_clock_long`.
  - The sector gate was relaxed **post hoc** from 35% to 40% (black sector 36.2%) by the user before freezing.
  - Mechanism tests (co-movement, volume surge, OI change) all failed. Treat the strategy as "commodity long, adding after weak market closes", not as a proven liquidity-provision story.
  - Earlier variants were seen and deleted: long-short, equal-weight long (1.31), 3-day, tail entry and passive benchmark.
- **Year-by-year tests**, opened in order with `python -m backtest.engine --year Y`. Each year is a fresh 1e8 account; earlier years are warmup only.

| Year | Net return | Sharpe | Max drawdown | EW pool | beta | Alpha vs pool | Down-capture |
|---|---|---|---|---|---|---|---|
| 2022 | +2.1% | 0.25 | −17.1% | +7.5% | 0.57 | −2.3% | 0.75 (Jun–Jul drawdown) |
| 2023 | +4.0% | 0.81 | −4.7% | +3.8% | 0.42 | +2.3% | 0.28 |

- 2024 and 2025 are **not opened**.
- **Finding behind it** (research period only): commodity time factors carry information only at the **market** level. The idiosyncratic and sector-rotation parts lose money after costs, and the slow volume factors are just sector bets.

**Older work**, removed from the tree but recoverable from git history:
- The morning-overreaction strategy (兴业 morning rule) failed 2022 and 2023.
- The intraday slate failed 2023–25.
- Lesson from both: research Sharpe of 1–1.5 has repeatedly collapsed to roughly 0 out of sample.

## 3. Data partitions and what is already spent (most important section)

| Partition | Dates | Commodity | Index |
|---|---|---|---|
| Warmup | 2015 | in research shards, not scored | same |
| Research | 2016–2021 | `data/commodity/minute_shards/research/` | `data/index/minute_shards/research/` (IF/IH/IC) |
| Validation | 2022 | `validation_2022/`; **seen, used for rule selection**; `market_clock_long_2022` opened 2026-10-20 | inside `holdout_locked/`; **unread** |
| OOS | 2023–2025 | `holdout_locked/`; **consumed several times** (morning strategy, four-sector run, 20-day and 10-day runs). `market_clock_long_2023` opened 2026-10-20; 2024 and 2025 are not opened for it. See `data/commodity/oos/ledger.jsonl` | **unread**, no ledger yet |
| Forward | 2026-01-01..2026-10-08 | `data/commodity/data_min/future_*_20260101-20261008.txt`, not sharded; **never read** (user decision 2026-10-20) | — |

Rules:
1. **Validation plan (user decision 2026-10-20).**
   - A new commodity strategy must first pass its predeclared research-period gate and have an economic story. Only then is it frozen.
   - After freezing, open commodity 2022, 2023, 2024 and 2025 **one year at a time**, with a separate ledger entry for each year.
   - Those years' market data and other strategies' results have been seen before. Label the results "rule not fit on these years", not "clean OOS".
   - Never read 2026 data.
   - Never read index 2022+.
   - Run an OOS or year command only when the user explicitly asks for that year in the same turn.
   - Year-by-year opening is `python -m backtest.engine --year Y` (`run_market_clock_year`).
     - It refuses out-of-order years and any year already in the ledger.
     - 2022 writes a `year_test` entry; 2023 and later go through `final_evaluation`.
     - A diagnostic read of an opened year (for example, the pool comparison) must also go through `final_evaluation`, under a `*_diagnostic` strategy name.
2. Research code must go through the guards in `src/tfcta/config.py`: `assert_research_only`, `assert_no_holdout_dates`, `HoldoutViolation`, and the `final_evaluation` context plus ledger. Do not weaken, bypass or monkeypatch them, and do not add an env var or flag that unlocks data.
3. Anything tuned, selected or "fixed" after looking at commodity 2022 or 2023–25 is **in-sample** for those periods. Say so explicitly. Do not label it validation or OOS.
4. A new hypothesis can only be tested cleanly on commodity 2016–2021 (research), on index 2022+ once frozen, or on 2026+ forward data. Name which one you are using.
5. Before any test that uses external data (carry/roll_yield, warehouse, OI), verify that the partition covers the whole window. A truncated carry file once produced a fake pass.

## 4. Repository layout

```
src/tfcta/          library: config (paths, partitions, guards, factor signs), data (shards, universe,
                    rolls), factors, costs (fees, ticks, multipliers), prepare/ (sharding, ticks,
                    universe, factor caches), index/ (GTJA index factors)
strategies/         one dir per strategy: __init__.py (ID, NAME, DETAIL, signal); frozen ones add load_config + <id>.json
                    book.py = shared tail/ATR/ADX/entry_schedule; catalog.py = frozen; research_catalog.py = exploratory
backtest/engine.py  integer-lot ledger, performance / DSR / PBO metrics, research/validation/oos entry points, CLI
backtest/index_market.py  index-futures data loading (day session only, IM pool rules)
research/           predeclared studies (time_factor_screen, gtja_time, market_clock_long); index_kline is only a chart utility
config/research/    predeclared JSON specs for research studies and exploratory strategies
download/           RiceQuant downloaders (commodity minute/external/fees, index minute)
tests/              pytest, synthetic data only (generator in tests/synthetic.py)
docs/               the two reports + 全市场时钟做法与表现.md (mainline) + 四策略做法与表现.md (frozen four)
data/, runs/        gitignored; data inputs and all outputs
```

Outputs:
- Strategy runs go to `runs/strategies/<id>/<commodity|index>/<research|validation|oos>/`. Each run writes `summary.csv`, `annual.csv`, `daily.csv`, `positions.csv` and `events.csv`, plus a `plots/` directory.
- Research studies go to `runs/research/<study>/` or `runs/index/<study>/`, with a `protocol.json`.

## 5. Environment and commands

- Python: `/c/Users/GU494/anaconda3/envs/factor-mining/python.exe` (conda env `factor-mining`). Set `PYTHONIOENCODING=utf-8` on Windows.
- Run everything from the repo root. `pytest.ini` puts `src` and `.` on the path.
- RiceQuant credentials live in `config/rqdata.env`. Never print, commit or move them.

```bash
python -m pytest                                      # must stay green (187 tests on 2026-10-20)
python -m backtest.engine --research                  # frozen four, commodity 2016–21
python -m backtest.engine --market index --research   # frozen four, IF/IH/IC 2016–21
python -m backtest.engine --validation                # 2022 (commodity already seen)
python -m backtest.engine --experimental-research     # research_catalog (market_clock_long), commodity 2016–21 only
python -m backtest.engine --year 2024                 # next year of the frozen mainline; only when the user asks
python -m research.market_clock_long                  # its predeclared variants, robustness grid and mechanism tests
python -m research.time_factor_screen
python -m research.gtja_time
```

`TFCTA_DATA_ROOT`, `TFCTA_RUNS_ROOT`, `TFCTA_CONFIG_DIR` and `TFCTA_MINUTE_DIR` redirect paths. Use them only for tests and reproduction checks.

Data preparation order (README §数据准备): `prepare/shard.py` → `prepare/universe.py` → `prepare/ticks.py` → `prepare/factors.py`, plus fee and external downloads. `shard.py` refuses to overwrite `holdout_locked/`; the minute downloader refuses incremental merge when more than one monolith exists.

## 6. Research protocol (follow on every change that produces numbers)

1. **Predeclare.**
   - Write the hypothesis, parameters, sample and pass criteria in a JSON under `config/research/` or `strategies/<id>/<id>.json` **before** running.
   - The `sample` text must state what is not read. The loaders check for it, for example `accept_lock` requires "不读 2023".
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
   - Missing fee or tick data must raise, not silently skip a symbol or reuse stale values: `require_multipliers`, `fee_tables` (ffill ≤ 31 days), `slippage_tables` (ffill ≤ 1 year), NaN cost raises inside `simulate_integer_lots`, `_require_cost_coverage` for validation.
5. **Timing:**
   - Signals use data known at the decision time. Close signals enter at the next tradable open.
   - External data such as carry is lagged one trading day.
   - Rolling thresholds and quantiles exclude the current day unless the spec says otherwise.
6. **Freezing:**
   - A strategy enters `catalog.py` only with `frozen_at` and `chosen == parameters`.
   - Exploratory strategies go in `research_catalog.py` and are never allowed through OOS entry points.
7. **Ledger:**
   - Any OOS opening is written to the ledger first, via `final_evaluation`.
   - A repeat requires an explicit rerun and a note.
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
  - The 250-day tail and ATR-median warmups mean there are no signals early on; index 2016 `trend_high_atr` has zero trades for this reason. That is warmup, not a bug.
  - ADX NaN is neither "range" nor "trend".
- **Universe:** the year-y pool uses only year y−1 turnover and valid days (`screen_year`, `load_universe`). Check the symbol count after loading; a truncated fee file once shrank the pool to RB alone.
- **Rolls:** the integer-lot ledger charges fees and slippage at roll, using the old dominant's prior close and the new dominant's open. It does not carry real contract settlement. Results are research estimates, not exchange-grade fills.
- **Long-only ledgers:** both simulators are long-only. Short and weighted-entry paths were removed together with the carry strategies and the external carry cache.
- **Target execution:** a strategy with `EXECUTION = "target"` provides `targets(signed, bars, universe, index, columns)` and goes through `simulate_target_lots`.
  - Weights run from 0 to `TARGET_CAP` = 3; equal-weight full is 1.
  - Notional = weight × 1.5 × equity / pool width.
  - Only the lot difference is traded, and lots are not resized while the weight stays the same. The whole book has one breaker.
  - `ledger_inputs()` returns aligned ledger inputs for scripts that call the simulator directly.
- **No-night days:** about 100 of the 1460 research days have no night session (post-holiday days, and the 2020 night suspension). On those days the full-day `ts_low`/`ts_high` market mean shifts by roughly −0.23/+0.18 sd, because the normalisation grid is shorter.
- **Year-test factors:** `load_market_clock_year` recomputes the five factors from 320 research days before 2022 onward. It asserts that the recomputed values equal the research cache on the last 60 research days. Keep `duration_factors`, `report_factors` and the cache on the same N250/M55 definition.
- **Engine dispatch:** `_run_integer_lot_period` sends `EXECUTION = "target"` strategies to `targets()`/`simulate_target_lots`. Everything else must have the frozen-four signature `signal(trend, micro, high_vol, ranging, above_range)`.

## 8. Coding conventions

- Match the surrounding style.
  - Docstrings and comments are short and in **Chinese**. Identifiers are English. Output CSV column names are Chinese.
  - Comment density is low; comments explain "why" and sample boundaries.
- Prefer extending existing modules (`factors.py`, `book.py`, `engine.py`, `costs.py`) over new files.
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
5. Update docs that describe changed behaviour or numbers: `README.md` (Chinese user manual), `backtest/README.md`, `strategies/README.md`, `docs/四策略做法与表现.md`. Update this file if rules or layout changed.
6. Report back with:
   - what changed;
   - which data partitions were read;
   - net numbers with cost assumptions, including all declared variants;
   - what is in-sample versus clean;
   - what was not verified.

   Keep it short. Do not call research-period Sharpe "evidence" when the same rule family already failed OOS.
