# AGENTS.md — working rules for this repository

Read this before changing anything. It reflects the repo on 2026-10-29 (three frozen books: two commodity and one index, the bear-short book).
If the code and this file disagree, trust the code, then update this file in the same change.

**Research sample default (user decision 2026-10-26):** New commodity experiments, data pulls, and backtests default to the full 2011–2021 research window. For a comparable IF/IH/IC index panel, the research calendar starts on 2015-04-16, when all three are listed; exclude IM. Actual scored dates still depend on available data and required factor warmup; declare and report those limits. Existing preregistered results retain their original sample labels.

## 1. What the project is

We build time-dimension ("when did it happen inside the day") factor strategies for Chinese futures.
They are based on two broker reports in `docs/`:

| Report | Market | Core idea |
|---|---|---|
| Guotai Junan Futures, 2025-02-25, Index Strategy Series 5 (GTJA) | IF / IH / IC index futures | Price/volume **duration** factors with a **fixed** threshold (3 points, 50 lots), plus behavioural timestamps: `time_to_high/low`, `time_to_max_volume/turnover`, `time_to_half_volume`, `afternoon_high_time`. Each factor is rolling z-scored, and the signal is a sign conjunction of **two** factors. Next-open to next-open return, 2 bp round trip, everything in-sample. |
| Industrial Securities, 2026-06-09, High-Frequency Time-of-Day Treasury Futures Timing (XY) | T / TL treasury futures (+ single-stock CTA overlay) | Duration uses a **dynamic** threshold: the M-th percentile (50–60) of absolute minute changes over the past N≈250 days. Factors: DFP (fair price at the longest duration minus close, top-1/top-3), PMT/VMT (minute of the longest price/volume duration), VR (AM/PM volume-duration ratio), and timestamps of high, low, max volume and turnover, PM high/low and AM spike count. Each factor gets its own rolling-quantile band, then an equal-weight combo. Robust parameters are taken from the centroid of the grid, not the best cell. No cost is stated. |

Priors the reports agree on:
- A late low is bullish.
- A late high, especially a late PM high, is bearish.

Where they conflict:
- On max-volume/turnover timing the two reports point opposite ways. `FACTOR_SIGNS` in `src/tfcta/config.py` holds our fixed priors.
- **Signs are priors.** Never flip them based on research-period returns.

Factor code lives in `src/tfcta/factors.py`. Key functions: `duration_series`, `dfp_factors`, `timestamp_factors`, `report_factors`, `rolling_threshold`.

## 2. Where the research stands (do not re-discover this)

Be honest about this in every report. Nothing in this repo is a proven deployable edge yet; the mainline is being tested year by year.

**Factor screens** (`research/time_factor_screen.py`, outputs in `runs/research/`):
- On commodities, `ts_low`, `ts_low − ts_high` and `dfp_*` have cross-sectional t > 3 at the close decision.
- Only 1 of 115 cells passes the full predeclared gate (t, 5/6 years, 4/6 sectors), and 0 are allowed to pair.

`strategies/commodity_catalog.py` lists two frozen commodity books; `strategies/index_catalog.py` lists one frozen IF/IH/IC book. The four long-only tail rules, their write-up and `backtest/index_market.py` were removed. Do not restore them.

**Original commodity book (frozen 2026-10-23).** Write-up `docs/guide.md`. This strategy combines these two legs:
- Long leg: at the close, the five-factor market score must be > 0, and that name's main-versus-sub `annualized_yield` lagged one trading day must be strictly > 0. Weight +1.
- Short leg, only on bear days: a name gets weight −1 when its own trend (mean sign of 20/60/120-day returns, (closew_t − closew_{t−n}) / close_{t−n}) is < 0 and its lagged carry is < 0.
- Budget: |weight| × 1 × equity / pool width. A name is never in both legs on the same day (long needs carry > 0, short carry < 0).
- Risk: 10% drawdown → liquidate and halt 10 trading days.
- Execution: on the next trading day's first bar (night open if there is a night session, else day open). Unchanged weight stays on; only the lot difference trades.

The market bear rule and symbol short rule share `TREND_WINDOWS = (20, 60, 120)` in `strategies/commodity_original/__init__.py`. Both frozen JSON window fields are checked against this constant. Scoring begins on 2011-01-14; 2010 is factor warmup only.

**Index books removed 2026-10-29 (user decision).** The frozen book `index_close_location_bear_short` remains (CLV < 0.4 long even in a bear, short only when bear and CLV ≥ 0.4, 20-day vol scale ≤ 1). The strict-swap index book and the long-only `index_close_location_reversal` were deleted with their specs, engine routes (`run_index_research`, `run_index_2022`), tests and the research module `research/index_close_location_candidate.py` (its bear-overlay study moved to `research/index_bear_overlays.py`); `research/index_long_day_split.py` and the configs of the removed studies were deleted too. The ledger `data/index/oos/ledger.jsonl` has a `kind: decision` entry for each. `strategies/index_close_location/` now holds only `SYMBOLS` and `market_close_location`, so the bear-short book's files and their 2022 ledger hash are unchanged. A backup of everything removed is in `runs/archive/2026-10-29_removed_index_books/`. Research-log entries below that mention the removed books and scripts are history; their commands no longer run.

**Close-location strategies (frozen 2026-10-27).** The commodity close-location strategy uses equal-weight pool CLV < 0.5 for its long trigger and retains the original lagged positive-carry filter and short leg. The index strategy uses equal-weight IF/IH/IC CLV < 0.5 and retains its carry filter and short leg. Their frozen rules are in the strategy JSON files. Both were frozen after overlapping research results had already been inspected; their research reruns are in-sample, not independent validation. Neither freeze authorizes OOS. Do not open 2022+ windows until the user explicitly names one.

- **OOS records reset (user decision 2026-10-22).** The commodity OOS ledger (`data/commodity/oos/ledger.jsonl`) and every 2022+ result folder were cleared.
  - **User decision 2026-10-26: 2022–2025 is the out-of-sample window for the current strategy, which has never been run on it. Do not open any 2022+ window until the user says to start OOS.**
    - The two CLV-based books were frozen by explicit user decision on 2026-10-27 after overlapping research results had already been inspected. They also have no OOS results; this freeze does not satisfy the user's separate authorization to open an OOS window.
  - With the ledger empty, `--validation`, `--year` and `--oos` read the data again and write new entries; they no longer just print an old result.

- **Roll yield.** RiceQuant `get_roll_yield(type="main_sub")`. `yield` = (main settlement − sub settlement) / sub settlement, verified against contract dailies on 3,840 research days (error 1e-17). 99.7% of rows have the main maturing before the sub; LR, JR, RI, RS, B, SF, PM, BB occasionally invert.
- **Research window (user decision 2026-10-23): `--research` is now 2011–2021.**
  - **User decision 2026-10-26:** commodity research defaults to 2011–2021. The comparable IF/IH/IC index research calendar starts 2015-04-16, the first date all three instruments are listed; exclude IM. A later first scored date is allowed only where data coverage or predeclared factor warmup requires it; document the reason. Existing index transfer results retain their original 2016–2021 scored sample.
  - Pools for 2011–2015 are rebuilt with the same one-prior-year screen (`load_research(first_year=2011)`); the rebuilt 2015 must equal the stored 2015. Widths are 11/10/12/11/17, versus 22–37 from 2016.
  - Scoring starts 2011-01-14, the first day all five factors exist; 2010 is factor warmup only. The trend windows are 20/60/120 days.
  - Fees: DCE has real RiceQuant fees from 2010 (downloaded 2026-10-23). SHFE fees exist only from 2013-10-08. For 2011-01-01..2013-10-07, AG/AL/AU/CU/RB/RU/ZN reuse their 2013-10-08 rate (`backfill_shfe_fees`, user decision). This is an assumption: setting that stretch to ×0 or ×2 moves 2011–2013 annual net returns by < 0.2 pt.
  - Multipliers follow the held contract (`CONTRACT_MULTIPLIER_RANGES`, checked per contract against RiceQuant instruments): RU ≤ RU1207 is 5 t, FU1202–FU1812 is 50 t, PB1109–PB1308 is 25 t. A roll into a different multiplier keeps notional (RU 2012-04-06: 67 → 33 lots). No 2016–2021 position was affected.
  - **2011–2021, one 1e8 book, 1 tick + fees:**

    | Book | Sharpe | Annual | Max DD | Calmar |
    |---|---|---|---|---|
    | Current strategy | 1.03 | 5.6% | −6.5% | 0.87 |
- **Research 2016–21** (frozen-period window, `--start 2016 --end 2021`, 1 tick + fees):

  | Book | Sharpe | Annual | Max DD | Beta | Alpha | NW t |
  |---|---|---|---|---|---|---|
  | Current strategy | 1.25 | 7.0% | −5.7% | 0.15 | 5.0% | 2.25 |

  - The pool's "bear" days in 2016–21 returned +11.6%/yr, so the research period only prices the insurance.
- **Signal tests** (`research/market_clock_slow_short_factors.py`):
  - Timing: score>0 minus else, next-bar pool return +12.7bp/day, NW t 3.22, 6/6 years.
  - Carry: cross-sectional IC 0.022, t 3.20, 5/6 years (2021 negative).
  - Interaction: t 2.25.
  - These numbers were seen in scratch before being declared.
- **Market-score sector diagnostic** (2016–2021, no costs, report only, run 2026-10-23). The user decided not to split by sector, so the script, spec and test were deleted; results and the predeclared spec are in `runs/research/market_score_sectors/`.
  - A sector-balanced score (sector means, then an equal-weight mean across sectors) is 12.6bp/day, t 3.27, versus the frozen score's 12.7bp, t 3.22. Their correlation is 0.98.
  - Sector-own scores add nothing once the market score is controlled: t 0.2–1.25 in every sector.
  - The market score times ferrous metals (t 3.67, 6/6 years) and chemicals (t 2.84, 6/6). It does not time agricultural products (t 0.61) and is weak on non-ferrous metals (t 1.52).
  - No single sector drives it: dropping a sector leaves t 2.2–3.4. Dropping precious metals costs the most (t 2.18).
  - This confirms the market-level finding. Do not build sector-split scores. Dropping agricultural products would be a post-hoc selection.
- **Carry normalisation (checked 2026-10-26).** RiceQuant `annualized_yield` already equals `yield × 365 / |maturity gap in days|` (error < 1e-15, with maturities in `data/commodity/external_rqdata/contract_maturity.csv`), so it is already divided by the contract-month gap.
  - Only the 492 rows (0.41%) where the main matures *after* the sub keep the "main − sub" sign, which there means far above near.
  - Inside the pool this flips 9 lagged symbol-days (M and Y, 2011). The frozen book keeps `annualized_yield`.
- **Mechanism study** (`research/market_clock_mechanism.py`, `config/research/market_clock_mechanism.json`, 2011–2021, run 2026-10-26). Results are in `runs/research/market_clock_mechanism/`, plots in `plots/`.
  - **Components** (each its own 1e8 book, 1 tick + fees):

    | Book | Sharpe | Annual | Positive years |
    |---|---|---|---|
    | full | 1.03 | 5.6% | 11/11 |
    | long leg | 0.63 | 2.8% | 8/11 |
    | short leg | 0.58 | 2.6% | 7/11 |
    | carry>0 only | 0.71 | 5.3% (beta 0.51) | 7/11 |
    | score>0 only | −0.14 | — | 7/11 |
    | whole pool | 0.18 | — | 6/11 |

    The two legs take turns. The short leg earns +7% to +11% a year in 2011–2015 and loses 0–2% in 2016–2021. The long leg is the reverse. The full book's stability is diversification between them, not a stable single signal.
  - **The score is a close-location proxy.** Its daily correlation with the pool's mean close location ((closew − loww)/(highw − loww)) is −0.925, and −0.65 with the same-day pool return. Score timing over 2011–2021 is +7.7bp/day (t 2.56). Adding close location drops it to +0.3bp (t 0.06); same-day return alone leaves +5.6bp (t 1.48); range leaves it unchanged. Close location alone predicts the next-bar pool return with t −3.42 and a negative slope in 11/11 years. It is market level only: the cross-sectional IC is t −0.91. **The time-of-day content adds nothing beyond "the market closed near its low".** The effect is a robust market-level short-term reversal, not a time-dimension effect.
  - **Range.** The pool's mean range predicts the next-bar pool return with t 2.24, positive in 7/11 years. It is ~uncorrelated with the score (0.05) and has no cross-sectional effect.
  - **Liquidity proxies from minute bars** (Abdi–Ranaldo spread, tick-rule volume imbalance). On score>0 days the tail imbalance is far more negative (t −30), which is mechanical. Timing is no stronger on illiquid days (interaction t −0.26), and tail imbalance has no predictive power of its own (t −0.24). **The liquidity-provision story is not supported by these proxies.** An L1 tick pilot (5 symbols, 2016–2021) is sized but not run.
- **Close-location mechanism** (`research/close_location_mechanism.py`, `config/research/close_location_mechanism.json`, run 2026-10-26). Results are in `runs/research/close_location_mechanism/`.
  - Commodities 2011–2021, market level.
  - L1 pilot: RB/CU/M/TA/I main contracts 2016–2021, 7,305 contract-days of tick data. `download/commodity/getTickL1Close.py` stores only daily close-window summaries in `data/commodity/external_rqdata/l1_close/`.
  - **Bid-ask bounce is rejected.**
    - The reversal sits in the next day's session (−32bp per unit CLV, t −4.2, 11/11 years). The close→next-open gap moves the other way (t 1.9).
    - CLV from tick mid-quotes predicts exactly as CLV from trade prices does (t −2.63 vs −2.68).
    - Close-vs-mid is 1.1bp against a 19bp Q1–Q5 effect.
  - **Information is not supported.** The reversal is stronger, not weaker, on heavy volume.
  - **Liquidity / price pressure is partly supported.**
    - On low-close days the last 30 minutes have wider spreads relative to the day (t 12.1), thinner L1 depth (t −2.4) and net selling (t −15.8; partly mechanical).
    - The reversal is larger in high-vol (t −2.9 vs −1.3) and high-volume (t −3.1 vs −1.3) states, but neither interaction is significant (t ≈ −1.1 to −1.2).
    - It is one-sided: the lowest CLV quintile is +8.9bp (t 2.4); the highest is −4.6bp (n.s.).
  - **Not a flow-compensation story.** Signed tail flow continues rather than reverses (t +2.4 with CLV in the regression), while CLV gets stronger (t −3.5). What reverts is price displacement beyond the flow, consistent with transient impact in a thin book.
  - **Indistinguishable:** close vs minute VWAP (≈ the settlement price) correlates 0.88 with CLV.
  - **Follow-ups (`--followup`, run 2026-10-26) do not confirm a flow mechanism:**
    - *Scaling with tail liquidity.* The CLV slope does not get steeper when the tail spread widens or depth thins. L1 interactions have the wrong sign (t +0.9, +1.4), and so does the minute Amihud proxy on the full pool (t +1.7).
    - *Proxy quality.* The minute proxy barely tracks L1 (rank corr −0.06 with the spread ratio, −0.31 with depth), so it cannot stand in for L1.
    - *Open interest.* The reversal is stronger on pool OI-decline days: −42.8bp (t −2.9) vs −20.0bp (t −1.7). But the interaction is t −1.2, and on down days the liquidation share predicts the rebound only at t 1.9. Cross-sectionally, liquidated versus new-short names show no difference (t −0.3), and low-close days are not more liquidation-heavy (52% vs 52%).
    - *Roll and delivery.* The reversal is *weaker* in roll-heavy and pre-delivery-exit-heavy periods (wrong direction).
    - **Conclusion:** the reversal is robust and not mechanical, but its cause is still unidentified. Price pressure from liquidity, liquidation or rolling is not supported as the driver.
    - Per the predeclared rule, 2011–2015 L1 is **not** pulled: F1a has the wrong sign and the minute proxy does not track L1.
- **Commodity L1 liquidity follow-up** (`research/close_location_mechanism.py --all-l1`, `config/research/commodity_l1_liquidity.json`, run 2026-10-27). Uses 42 downloaded commodity product L1 summaries plus research minute bars, restricted to each product's annual research-pool dates in 2011–2021. No 2022+ or 2026 data were read. 34 products meet the minimum 500 observations and 3 calendar years; 5 sectors meet the same effective-date threshold with at least two products per date.
  - Low midpoint close location coincides with a wider last-30-minute spread relative to the day in 32/34 eligible products and all 5 eligible sectors after within-family Benjamini–Hochberg correction (q ≤ 0.10). Relative depth is significantly thinner in only 4/34 products and 2/5 eligible sectors.
  - Next-session intraday reversal survives correction in CU, FG, JD and ZN, and in the ferrous-metals, non-ferrous-metals and chemicals sector aggregates. The weak-close × wider-spread interaction has no significant eligible result. The weak-close × thinner-depth interaction is significant only for chemicals in the expected direction (q = 0.074), while non-ferrous metals is significant in the opposite direction (q = 0.074); there are no eligible product-level interaction findings.
  - Controlling for tail signed-volume imbalance, the close-location reversal remains significant in 15 products and 4 sectors. Tail signed-volume imbalance itself predicts same-direction continuation in agriculture, chemicals, and ferrous metals, not flow reversal.
  - A post-hoc data-quality sensitivity clips midpoint close location to [0, 1] and reruns every model. The 42 source files contain 109 out-of-range rows; BB contains 18 extreme rows but is ineligible (220 pool observations, below the 500-day threshold). Clipping changes none of the eligible q ≤ 0.10 findings. The daily summaries do not retain the raw high/low tick series needed to diagnose every excursion.
  - **Conclusion:** the L1 evidence supports close-time spread widening as a liquidity-stress correlate, but does not robustly confirm that liquidity provision causes the subsequent reversal. These are observational, overlapping research-period findings, not causal identification or OOS validation. The downloader stores daily L1 summaries only; they do not measure L2 depth.
- **Index transfer test** (`research/index_market_clock.py`, `config/research/index_market_clock_transfer.json`, IF/IH/IC 2016–2021, run 2026-10-26). IM listed 2022-07-22, so it is outside the index research window and was not tested (user decision).
  - Carry is computed from contract dailies by `download/index/getIndexRollYield.py`. That rule reproduces RiceQuant commodity main_sub exactly on RB/CU/M 2018–2019: sub contract 1461/1461 days, annualized error < 2e-15. Main = `get_dominant(rule=0)`, which equals the shard dominant 100%. Sub = the largest same-day OI among contracts maturing after the main.
  - IC carry is > 0 on 99.6% of days; IF 83%, IH 73%.
  - Results:

    | Book | Sharpe | Annual | Max DD | Beta | Alpha | NW t |
    |---|---|---|---|---|---|---|
    | long leg alone | 1.09 | 12.7% | −9.5% | 0.38 | 7.8% | 2.32 |
    | current strategy | 1.03 | 11.9% | −10.4% | 0.34 | 7.6% | 2.15 |

    Both books lost only in 2018.
  - On index too the score is subsumed by close location: alone 15.1bp (t 2.37, 6/6 years); with close location −5.4bp (t −0.63); correlation −0.89. The three names move almost together, so this is a timing test on one asset; the rule was set on commodities and the index run is a transfer test.
- **Index L1 mechanism diagnostic** (`research/index_close_location_mechanism.py`, predeclared in `config/research/index_close_location_mechanism.json`, run 2026-10-26). IF/IH/IC main-contract historical L1 ticks cover 2015-04-16..2021-12-31 (1,638 days each; 96,537,464 rows total); the mechanism sample is 2016-01-13..2021-12-31 (1,454 days). Trade-price and midpoint close-location slopes for next-session intraday return are both about −0.00285 (NW t −2.72/−2.71); the close-to-next-open gap is weak (t −1.37). Low-close days have a wider tail/day spread ratio (difference +0.0331, t 6.60) and thinner tail/day L1 depth (−0.0163, t −3.82); the close-location × thin-depth term is negative (t −2.24), while spread and signed-imbalance interactions are not significant. Daily range has no clear relation to same-day or next-session returns (|t| < 0.8). This is partial liquidity-pressure evidence, not causal proof or OOS confirmation. All a2-a5/b2-b5 prices and sizes are zero across the downloaded ticks; usable history is L1, not L2.
- **Index positioning mechanism** (`research/index_positioning_mechanism.py`, `config/research/index_positioning_mechanism.json`, run 2026-10-27).
  - Data: futures data only, IF/IH/IC 2015-04-16..2021-12-31. CFFEX daily top-20 member long/short positions come from `download/index/getMemberRank.py` (`data/index/external_rqdata/member_rank.csv`; product level equals the sum over contracts), plus total open interest from contract dailies. Expiry days and the days before them (162) are excluded. Results are in `runs/index/index_positioning_mechanism/`.
  - **Hedging pressure is rejected.** On low-close days the top-20 members' net long position *rises* (net-flow slope on CLV −46bp, t −5.8; 2016+: t −6.4; 6 of 7 years). The next day they unwind it (t +2.9; 2016+ t +4.9).
  - The selling at a weak close therefore comes from outside the top 20 (smaller brokers). The top-20 brokers' clients absorb it and sell back into the rebound, which is a liquidity-provision pattern at the position level.
  - This flow does not cleanly predict returns. With CLV in the regression, top-20 net flow is t +1.4 (2016+: −0.8).
  - The reversal is concentrated on days when the top 20 also net-sold (CLV slope −43bp, t −2.8, versus +8bp when they absorbed). The interaction is t −2.1 on the full sample but only t −1.4 for 2016+. This reads as "the reversal happens when selling exceeds what the large members absorb".
  - **Liquidation versus new positions is not distinguished.** Total-OI-up and OI-down days show no difference (interaction t ±0.7).
  - Caveats: members are brokers aggregating clients, not individual traders. The top 20 hold 62–69% of OI. 2015-09..2019-04 had CFFEX opening limits. On this sample (expiry days excluded) the baseline next-session slope is t −1.8; the ledger next-open-to-open slope is t −2.75.
- **Unabsorbed selling pressure** (`python -m research.index_positioning_mechanism --unabsorbed`, `config/research/index_unabsorbed_pressure.json`, run 2026-10-27, proposed after the positioning results).
  - The originally proposed "outside-top-20 net selling minus top-20 net buying" is unusable. By the open-interest identity, outside-top-20 net change equals minus the top-20 net change (checked on IF: correlation 1.000000).
  - Definition used: U = max(0, 0.5 − CLV) on days the top 20 did not net-buy, and B is the same on days they did.
  - **Not robust:**
    - Full sample: U alone t −0.06; the U-minus-B difference t −0.7.
    - 2016+: U alone t 2.2 / 2.5 (next session / ledger), but B also predicts (t 1.4 / 2.2) and U minus B is only t 1.2 / 0.7.
    - "The reversal needs selling the large members did not absorb" is therefore not supported. Low closes predict the rebound whether or not the top 20 absorbed.
- **Index candidate plus short leg** (`python -m research.index_close_location_candidate --short-leg`, `config/research/index_short_leg_2015_2021.json`, run 2026-10-27).
  - Same ledger and 1 tick plus fees; the 2015 tick is 0.2, the CFFEX fixed tick.
  - The commodity short leg (bear, own trend < 0, lagged carry < 0) barely trades on index, because IC carry is > 0 on 99.6% of days. It holds shorts on 106 days, all IF/IH in 2018–2021, and **lowers returns**:
    - 2015-04-16 start: Sharpe 0.97 → 0.84, annual 14.9% → 12.2%, paired t −1.2;
    - 2016-01-13 start: 1.47 → 1.34.
  - A trend-only short leg (no carry filter) is worse: Sharpe 0.52 / 0.96, paired t −1.4 / −1.2. It earns only in 2018 (+8.4% gross on equity) and loses 6–10% a year in 2016, 2019 and 2020. It does cut beta (0.38 → 0.16; −0.02 for 2016+) and max drawdown for 2016+ (−16.8% → −12.9%).
  - Neither variant improves the index candidate.
- **Commodity close-location long leg (not built).** Replacing the frozen five-factor score with CLV would be a new book ID chosen after seeing CLV subsume the score, so its research result would be post hoc. Commodity work is paused at the user's request.
- **Commodity close-location book** (`strategies/commodity_close_location/`, `exploratory_predeclared`, runner `research/commodity_close_location_reversal.py`, run 2026-10-27). User request: the index rule on commodities. Long: pool equal-weight CLV < 0.5 → +1 in every pool name, no carry filter. Short: identical to the frozen leg; a conflict goes to the short. Post hoc (proposed after CLV was seen to subsume the score). 1 tick + fees, results in `runs/research/commodity_close_location_reversal/`:

  | Window | Book | Sharpe | Annual | Max DD | Beta | Paired t vs frozen |
  |---|---|---|---|---|---|---|
  | 2011–21 | frozen | 1.03 | 5.6% | −6.5% | −0.01 | — |
  | 2011–21 | main | 0.70 | 5.0% | −9.9% | 0.12 | −0.39 |
  | 2011–21 | long_carry (CLV replaces score, carry > 0 kept) | 1.12 | 6.2% | −6.0% | −0.01 | 0.83 |
  | 2011–21 | long_only | 0.01 | −0.3% | −38.1% | 0.45 | −2.19 |
  | 2016–21 | frozen | 1.25 | 7.0% | −5.7% | 0.15 | — |
  | 2016–21 | main | 0.90 | 6.7% | −9.9% | 0.27 | −0.08 |
  | 2016–21 | long_carry | 1.19 | 6.6% | −6.0% | 0.15 | −0.36 |

  Without carry the CLV long is just pool beta (long_only loses 2011–2015). The original unfiltered exploratory book is not frozen. The later strict one-for-one replacement is a distinct, frozen ID; it remains in-sample and cannot open 2022+ without explicit user authorization.
- **Three frozen-book report** (`research/strategy_report.py`, `config/research/strategy_report.json`, run 2026-10-28). Research partitions only; overlaps inspected samples, in-sample. It rebuilds each full strategy from its frozen rule and asserts it equals the engine ledger day by day, then runs component books, an unrebalanced 1× buy-and-hold (no breaker) and IF/IH/IC single-product books, all with fees + 1 tick. Outputs `runs/reports/commodity/` and `runs/reports/index/` (plots: `nav_drawdown`, `vs_buy_hold`, `components`, plus `products_*` for index). Annual / Sharpe / Max DD:
  - Commodity 2011-01-14..2021-12-30: original 5.6% / 1.03 / −6.5%; CLV swap 6.2% / 1.12 / −6.0%; time signal alone −1.5% / −0.14 (score) and −0.3% / 0.01 (CLV); carry>0 alone 5.3% / 0.71; short leg alone 2.6% / 0.58; pool buy-and-hold 3.6% / 0.29 / −56%.
  - Index 2015-04-20..2021-12-30: joint 11.9% / 0.88 / −26.4%; CLV alone 14.9% / 0.97; carry alone 11.1% / 0.68; short alone −0.9%; buy-and-hold 5.8% / 0.37. Single products (same pooled signal): IF 0.73, IH 0.62, IC 0.96.
  - **The frozen index long trigger is the equal-weight IF/IH/IC CLV**, shared by all three. A post-hoc diagnostic using each product's own CLV gives joint Sharpe 0.78 (IF 0.78, IH 0.08, IC 0.85); the frozen rule is unchanged.
  - Component books are separate ledgers, not additive. The earlier `three_strategy_clv_comparison` study, its zero-cost single-index counterfactuals and the engine `--index-individuals` flag were removed; their old outputs were archived and later deleted.
- **Index close-location shape, GTJA replication and variants** (run 2026-10-28, post hoc, research only).
  - Shape (`research/index_close_location_mechanism.py --shape`, `config/research/index_close_location_shape.json`): relative position matters, not distance to the low. 2016+ ledger next segment: CLV −41.6bp (t −3.35), unchanged with range; vol-scaled d_low/d_high lose to CLV (d_high t 0.08 with CLV). One-sided (Q1 +26bp t 3.7, Q5 −5bp n.s.). IC slope > IH (paired t 2.0); weekend/holiday interaction t 0.4; high-vol interaction t −1.6; survives dropping the 2015–16 crash (t −3.0).
  - GTJA 23 pair rules × IF/IH/IC (`research/index_time_factors.py`, `config/research/index_time_factor_replication.json`; factors in `tfcta.factors.gtja_index_factors`, fixed 3-pt / 50-lot thresholds): fees + 1 tick, 2016-01-04..2021-12-30. Report's own-product cells average net Sharpe 0.19, transfers 0.15; IH rules mostly negative on IH. No cell has t ≥ 2 after controlling same-day CLV (max 1.69; min BH q 0.31). The good IC/IF rules correlate 0.3–0.5 with the CLV reversal signal. The report's Volume_duration_open_close formula is sign-reversed versus its text (formula reading loses). Volume_duration_return > 0 is the only consistent, non-significant lead.
  - Two-way regressions (same script): 5 of 78 rule cells have t ≥ 2 alone; all fall to 0.5–1.6 after controlling CLV, while CLV stays t −2.6 to −3.5 on IF/IC after controlling any rule. Rule–CLV correlation is only 0.2–0.5: the rules are encompassed by CLV, not identical to it.
  - Product tilt (`research/index_close_location_candidate.py --tilt`, `config/research/index_product_tilt.json`): with the pooled CLV signal, IF/IH/IC next-segment slopes are −39/−40/−46bp (t ≈ −3.1 to −3.2), identical per unit of vol (IC − IH t 0.1). The earlier "IC strong, IH weak" came from each product's own CLV, a noisier signal for IH. Vol parity and walk-forward strength weights do not beat equal; post-hoc IC tilts add only beta. Keep equal weights.
  - CLV variants (`research/index_close_location_candidate.py --variants`, `config/research/index_close_location_variants.json`): thresholds 0.3–0.5 form a plateau (2016+ Sharpe 1.36–1.46), 0.6–0.7 worse; graded sizing and a range gate cut beta/drawdown but no variant beats clv<0.5 on paired t. Nothing frozen.
- **Index reversal channels and risk-managed candidates** (run 2026-10-28, research only, post hoc).
  - Channels (`research/index_close_location_mechanism.py --channels`, `config/research/index_reversal_channels.json`; margin and northbound data from `download/index/getMarketFlows.py` → `data/index/external_rqdata/market_flows.csv`, 2015-01-05..2021-12-31). 2016+: the decline that reverses is the signal day's 11:30–14:30 segment (t −2.3), not the tail; the rebound is next day 11:30–14:30 (t −3.4), not the last half hour. Auction gap, past-20-day return and the 2018-08-20 SSE closing auction do not modulate it. Low-close days coincide with margin deleveraging (t 8.3) and northbound selling (t 5.4); northbound buys back next day (t −2.4); but neither scales the rebound (interactions t 0.3, 0.5). Realized-vol interaction t −1.9 (Nagel direction). Post hoc: next 11:30–14:30 on same-day 11:30–14:30, t −4.6, CLV adds nothing to that window. Reversal lasts one open-to-open segment.
  - Candidates (`research/index_close_location_candidate.py --candidates`, `config/research/index_strategy_candidates.json`), 2015-04-20 start, annual / vol / MDD / Sharpe / Calmar: frozen 11.9% / 13.9% / −26.4% / 0.88 / 0.45; clv<0.5 14.9% / 15.7% / −33.4% / 0.97 / 0.45; clv + 20% vol target 13.0% / 11.1% / −17.3% / 1.16 / 0.75; frozen + vol target 11.4% / 9.7% / −11.1% / 1.16 / 1.02; afternoon reversal (post hoc) 16.1% / 15.5% / −26.6% / 1.04 / 0.61; IC−IH style spread 0.20 Sharpe (dead). Vol targeting helps only through 2015; from 2016 it lowers return without lowering drawdown. User decision 2026-10-28: keep only A (frozen) and B (clv<0.5 long only). B was frozen the same day as `index_close_location_reversal` and its only opened out-of-sample window is calendar 2022. C–F code removed. 2015 gap A −2.2% vs B −6.8% comes from A's carry filter (IF/IH negative lagged carry on 32/53 pre-crash and crash days), not from a short leg: neither book shorted in 2015 (bear gate first available 2015-10-14).
  - B 2022 year test (`python -m backtest.engine index_close_location_reversal --validation`, ledger `data/index/oos/ledger.jsonl`, run 2026-10-28). Fees plus one tick from the 2021 estimate, 1e8, 10% breaker. Scored 2022-01-04..2022-12-29 (241 days); the last 2022 session is omitted because its exit open is in 2023. The locked minute files also contain 2023–2025; those rows were dropped before the ledger. Result: net −1.0%, Sharpe −0.01, max drawdown −10.7%, Calmar −0.09. Long on 128 days, flat on 113. The breaker halted 10 trading days; equity trough 2022-03-15. 2023–2025 and 2026 were not opened. A and the commodity books were not run. Follow-up (`research/index_long_day_split.py`, after seeing this year): on long days, win and loss sessions have the same signal-day return (−0.71%), range and volume ratio. Five days (2022-03-08, 03-15, 03-25, 04-21, 10-24) sum to about −12% and offset the other long days. Flat-day market gross averages −17 bp in 2022 and −4.6 bp in the 2016–2021 candidate. Not a new rule.
  - Bear-short book (`strategies/index_bear_reversal/`, frozen 2026-10-28). A and B stay at threshold 0.5. Threshold chosen on the full research window, not on 2016–2021 Sharpe: next open-to-open after the signal close, dummy 1{CLV < h} versus the other days, NW t on 2015-04-16..2021-12-29 is 2.45 / 3.03 / 2.76 for 0.3 / 0.4 / 0.5 (2016+ 3.78 / 4.03 / 3.69). Chosen 0.4. Long when equal-weight CLV < 0.4, including in a bear. Short all three when market trend (20/60/120) < 0 and CLV ≥ 0.4. No carry filter and no per-symbol trend filter. Fees + 1 tick, 10% breaker. Research 2015–2021: 16.3% / 1.05 / −32.6% / Calmar 0.50 versus long-only 0.4 at 13.3% / 0.97 / −33.2% / 0.40. The 120-day trend first exists on 2015-10-14, and the first short signal is 2015-10-20, so the 2015 crash was not covered (7 short signals that year). Short-day net sums: 2018 +22.4% on 93 days (long-only that year +2.3%); 2017/2019/2020 about −2% each. Research book after the user added the vol scale the same day: 16.7% / 1.33 / −17.2% / Calmar 0.97. Annual net +5.2 / +11.1 / +14.5 / +26.6 / +32.5 / +25.6 / +1.1 for 2015–2021. The earlier equal-weight 2022 result (+19.7%, Sharpe 1.19, drawdown −11.8%) was voided when the scale was added; those files were deleted. The scaled rule was then opened on 2022 per the ledger (2026-10-28 17:18): +16.8% / vol 15.1% / Sharpe 1.11 / −12.1% / Calmar 1.40, in `2022/`. User statement 2026-10-29: the short leg and vol scale were designed in advance as insurance, not chosen from 2022 results. The frozen JSON's decision_note still says the scaled rule was not rerun; the ledger is authoritative and the frozen file is not edited. 2023–2025 and 2026 stay closed. User decision 2026-10-29: this book is the only index book in use; `--market index` without a name runs it; write-up `docs/index_guide.md`. Later the same day the 10-million trial was removed. The ledger is 1e8, and each open is capped by the CFFEX non-hedge limit in force that day (per product 600 from 2015-08-26, 100 from 2015-08-31, 10 from 2015-09-07, 20 from 2017-02-17; per contract 50 from 2018-12-03, 500 from 2019-04-22). Closes are uncapped; the unfilled part is topped up on later days while the target weight is unchanged. Research was overwritten: 13.0% / 10.1% vol / −11.1% / Sharpe 1.26 / Calmar 1.17, with 1,064 capped opens. 2022 was voided and rerun the same day on this ledger (2026-10-29). No open exceeded the 500-lot cap (largest 105), so the net return, drawdown and positions match the previous 2022 file. The same day, IF alone from its listing 2010-04-16 through the session before IH/IC listed (scored 2010-04-16..2015-04-15, own CLV, own trend, own vol scale, 1e8 on one name) was run from a separate download (`research/if_before_ih_ic.py`); it did not use the research shards. Result: annual −5.4%, vol 15.4%, Sharpe −0.28, max drawdown −42.1%. Not a new book.   - Liquidity provision (`research/index_close_location_mechanism.py --provision`, `config/research/index_liquidity_provision.json`; constituent panel from `download/index/getConstituentPanel.py` → `data/index/external_rqdata/constituents/`, 1,274 CSI300/CSI500 stocks). No option or ETF-unit access, so Nagel's VIX is proxied by futures realized vol. 2016+: CLV × RV interaction t −1.9 at 5/20/60 days (2015+ ≈ −0.4); vol quintile slopes not monotone. Cross-section on weak-close days: stocks with more same-day margin repayment rebound more next session (t 3.2), but also on strong-close days (difference t 1.1); northbound selling does not predict rebound; CSI500 − CSI300 constituents t −0.1. Margin repayment is the strongest evidence for non-informational selling being reversed, not specific to market weak closes.
  - Forced selling at the index (`research/index_close_location_mechanism.py --forced`, `config/research/index_forced_selling.json`, run 2026-10-28). Constituent net margin repayment / constituent turnover, trailing 250-day z, equal-weight of IF↔CSI300, IH↔SSE50, IC↔CSI500. Research constituents only, through 2021-12-31. Primary 2016+ next-session CLV × F1 interaction −1.4 bp, t −0.10 (including 2015: t +0.42, wrong sign). On weak-close days the F1 slope is t −0.49, and the high tercile rebounds 6.2 bp versus 11.3 bp in the low tercile. Same-day F1 on CLV is t −7.60, so weak closes do coincide with more repayment. With within-year QVIX, the quantity interaction stays t −0.10 and the QVIX interaction is t −2.60. Predeclared verdict: uncertain. Link ⑥ (more selling, larger rebound) is not established.
  - QVIX (`download/index/getQvix.py`, `research/index_qvix_reversal.py`, `config/research/index_qvix_reversal.json`, run 2026-10-28). Public 50ETF option-volatility index. Research file `data/index/external_rqdata/qvix_50etf.csv` is 2015-02-09..2021-12-31 (1,680 days). Dates from 2022-01-01 are written to `data/index/external_rqdata/holdout_locked/qvix_50etf.csv` and were not read. Overlap with official 000188 (70 days, 2017-09-12..2017-12-25) has correlation 1.000. 2016+: CLV × same-day QVIX level −25 bp per standard deviation, t −2.16, negative in 6/6 years; 20-day mean t −2.28; gross reversal on QVIX t 2.23. Including 2015 the interaction is t −0.88 and the gross slope t 0.93, signs unchanged. Quintile slopes −6/−29/−5/−36/−57 bp (only the top bucket t −2.1; rank correlation −0.7). Implied vol clears the predeclared |t| ≥ 2 bar on 2016–2021 and does not on the sample that includes the 2015 crash. Post hoc (2026-10-28): within each year the CLV × QVIX interaction is negative in 7/7 years; the weak pooled 2015+ result comes from 2015's much higher QVIX level (mean 40.5) with a flat CLV slope that year. Reversal scales with fear relative to the past year, not the absolute level.
  - Candidate C (clv<0.5 long, no carry filter, plus the frozen short leg; user request 2026-10-28): 2015+ 12.2% / 15.0% / −33.4% / 0.84 / 0.37, 2016+ 14.9% / 12.0% / −14.1% / 1.22 / 1.06; paired t vs B −1.2. 2×2 (2015+ Sharpe): B 0.97, +short 0.84, +carry 0.95, both (A) 0.88. The short leg lowers Sharpe either way; the carry filter cuts 2015 drawdown.
- **Index close-location candidate robustness** (`python -m research.index_close_location_candidate --robustness`, predeclared in `config/research/index_liquidity_mechanism.json` → `candidate_robustness`, run 2026-10-27). The registered gate verdict is unchanged (fail).
  - Same ledger, fees and 1 tick.
  - Candidate minus an exposure-matched constant long (0.48×): +12.5%/yr, NW t 3.31 (2016-01-13 start); t 3.21 (2016-01-04, including the January 2016 circuit-breaker days); t 2.50 (2015-04-16, including the 2015 crash, where the candidate's max drawdown is −33%).
  - Versus a 1× long it stays t 1.2–1.5.
  - So the timing adds return beyond its lower exposure; the failed gate mostly reflects comparing a ~0.5-beta book against a 1× long.
  - **Capacity is not modelled.** A 1e8 book opens up to 23–58 lots per product per day in 2015–2018 (`robustness/capacity.csv`), above CFFEX's daily opening limits of that period (as recalled: 10 lots from 2015-09, relaxed in steps until 2019; verify against exchange notices). Results before 2019 do not hold at that size.
- **Index liquidity / arbitrage mechanism** (`research/index_liquidity_mechanism.py`, `config/research/index_liquidity_mechanism.json`, run 2026-10-27).
  - Data: IF/IH/IC raw L1 ticks 2015-04-16..2021-12-31 (1,638 days each), plus spot CSI300/SSE50/CSI500 dailies from `download/index/getIndexSpot.py` (`data/index/external_rqdata/spot_daily.csv`). Results are in `runs/index/index_liquidity_mechanism/`; 2016+ figures are below.
  - **Limits to arbitrage (A).** The futures-minus-spot basis converges strongly after weak futures closes: Δbasis on futures CLV t −7.1, negative in 7/7 years. Low-close days have the basis below its 20-day mean (t −3.1). But futures and spot CLV correlate 0.92 and cannot be separated (A3).
  - **Post hoc split (`post_hoc_split.csv`, added after seeing A).** The basis convergence happens **overnight**: close→next open t −8.8; Q1 +22bp, Q5 +2bp.
  - In the next session the basis partly re-widens (t +2.5; Q1 −15bp).
  - The spot index gaps further down overnight (t +3.1) and then rebounds during the session (t −3.5 on futures CLV, −4.2 on spot CLV).
  - Reading: at the close the futures already price a decline that spot only shows at its next open, which is futures price discovery. The reversal the book can trade (from the next open) is the **spot market's own intraday rebound after a weak close and a weak open**, not futures-specific mispricing.
  - **Inventory (B).**
    - The order-flow price-impact decomposition is inconclusive. The flow component F = λ̂ × signed volume does not predict the rebound (t −0.6), and neither does the residual (t −0.9). CLV survives both (t −2.6).
    - The rebound is **not** in the first 30 minutes (t +0.6); it comes later in the session (t −3.5), against fast inventory unwinding.
    - The CLV × high-impact interaction is t −1.3.
    - L1 depth recovers more the morning after a low close (t +3.3), but recovery is uncorrelated with the rebound (−0.05).
  - **Overreaction (O).** The non-flow residual does not reverse (t −0.9).
  - **Conclusion:** the tradable index reversal is a slow, next-session rebound of the cash market after a weak close and an overnight continuation. Futures-specific arbitrage limits and fast inventory unwinding do not explain it.
- **Index cash open-pressure test** (`research/index_spot_open_pressure.py`, `config/research/index_spot_open_pressure.json`, run 2026-10-28). CSI300 / SSE50 / CSI500 1-minute bars 2015-04-16..2021-12-31, downloaded by `download.index.getIndexSpot --minute`; futures CLV from the research shards. Primary sample is outcome dates from 2016-01-04 (1,460 days). No 2022+ data. 2020-04-20 IF and IC were dropped because the first minute open disagreed with the daily open (2.5% and 0.65%); the median relative error on other days is 0.
  - Overnight gap replication: slope on prior futures CLV t +3.10, positive in 6/6 years. Q1 gap −15.8 bp, Q5 −1.1 bp.
  - First 30 morning bars (bar 30 timestamp median 10:00): slope +4 bp per unit CLV, t 0.76. Q1 +6.5 bp, Q5 +10.9 bp. The continuous-trading open is not a continuation of the weak close.
  - Rest of session: −36 bp, t −4.26, negative in 6/6 years. Q1 +18.9 bp, Q5 −9.4 bp.
  - Opening-volume share moves the other way (t +4.25, 6/6 years): low-CLV days have a slightly smaller, not larger, share of volume in the first 30 bars (0.24 vs 0.25).
  - Controlling for the opening-window return leaves the CLV slope unchanged (ratio 1.01). The opening return itself continues into the rest of the session (t +2.33).
  - IC minus IH opening slope t 1.94 on the primary sample (t 2.29 including 2015). Uncertain on the predeclared primary window.
  - **Stop rule:** P1 is not supported, so constituent opening-auction data are not pulled. The tradable rebound sits after 10:00. This is an in-sample mechanism result, not a new book.
- **Index close-location candidate** (`research/index_close_location_candidate.py`, preregistered in `config/research/index_close_location_candidate.json`, run 2026-10-26). Fixed rule: when equal-weight IF/IH/IC close location is below 0.5, hold a 1x equal-weight long basket from the next open; otherwise stay in cash. On 2016-01-13..2021-12-30 (1,453 days), with research fees, 1 tick per side and the shared 10% drawdown breaker: net annual return 19.8%, Sharpe 1.47, max drawdown −16.8%, versus risk-matched equal weight 8.7%, 0.57, −28.6%. Paired excess-return NW t is 1.52, below the preregistered 2.0 gate, so the candidate **fails the full gate**. Raw equal-weight return was +118.2% cumulative with no costs or breaker and is not like-for-like. The sample overlaps previously inspected research dates; keep this as an unregistered exploratory candidate, not a frozen book, and do not open OOS.

## 3. Data partitions and what is already spent (most important section)

| Partition | Dates | Commodity | Index |
|---|---|---|---|
| Warmup | 2010 | in research shards; factor warmup only | — |
| Research | 2011–2021 | `data/commodity/minute_shards/research/`. **User decision 2026-10-23:** the book scores from 2011; 2011–2015 pools are rebuilt; SHFE fees before 2013-10-08 are backfilled (§2). | `data/index/minute_shards/research/`; comparable IF/IH/IC calendar starts 2015-04-16, when all three are listed. IF was listed in 2010; IH and IC in 2015; IM in 2022 and is excluded. Shared-pool scoring also requires the declared factor warmup. |
| OOS | 2022 | `validation_2022/`; out-of-sample for the current strategy, never run with it; closed until the user starts OOS | inside `holdout_locked/`; **unread** |
| OOS | 2023 | `holdout_locked/`; same | inside `holdout_locked/`; **unread** |
| OOS | 2024 | `holdout_locked/`; same | **unread** |
| OOS | 2025 | `holdout_locked/`; same | **unread** |
| Forward | 2026-01-01..2026-10-08 | `data/commodity/data_min/future_*_20260101-20261008.txt`, not sharded; **never read** | — |

Rules:
1. **Sample lock (user decision 2026-10-21).**
   - A new commodity strategy must first pass its predeclared research-period gate and have an economic story. Only then is it frozen.
   - **Freeze gate (user decision 2026-10-21):** `run_span` refuses any window ending in 2022 or later unless every requested book's JSON is `status: frozen` with `frozen_at` and `chosen == parameters`. It refuses before writing the ledger or reading data.
   - Once frozen, `--validation`, `--year Y` and `--oos [--by-year]` open a window. The ledger is written first: 2022 gets a `year_test` entry; 2023+ goes through `final_evaluation`, with `oos_start`, `frozen_at`, `chosen` and file hashes (keyed by repo-relative path, `C.file_hashes`, over `book.FILES`) recorded.
   - A window recorded as opened or finished only prints its saved result. If a recorded result is gone, append a `kind: void` entry with a reason before reopening. The ledger is currently empty.
   - Do not run any 2022+ window until the user says to start OOS, and then only the window they name.
   - A user may explicitly authorize prefetching raw index L1 ticks without opening OOS: `python -m download.index.getTickL1Close --partition oos_locked --start 2022-01-01 --end 2025-12-31` writes only to `data/index/external_rqdata/holdout_locked/l1_ticks/`, without summaries or analysis. This does not open an OOS result; never inspect/use those files until the user starts the named OOS window. The downloader refuses dates beyond 2025.
   - Never read 2026. `run_span`, `load_through_year` and `roll_yield` raise before loading it. The `holdout_locked` roll-yield pickles contain rows through 2026-10-08; `roll_yield` drops everything after the window, and reads that partition only inside `final_evaluation`.
   - Never read index 2022+.
2. Research code must go through the guards in `src/tfcta/config.py`: `assert_research_only`, `assert_no_holdout_dates`, `HoldoutViolation`, and the `final_evaluation` context plus ledger. Do not weaken, bypass or monkeypatch them, and do not add an env var or flag that unlocks data.
3. Anything tuned, selected or "fixed" after seeing a commodity window is **in-sample** for that window.
4. A new commodity hypothesis defaults to 2011–2021. A comparable IF/IH/IC panel defaults to the common listed period beginning 2015-04-16; exclude IM. Respect data availability and predeclared warmup, and report the resulting scored dates. A 2022+ window is opened only by an explicit `--year` or `--oos` request in that turn. Never use 2026. Name which window you are using.
5. Before any test that uses external data (carry/roll_yield, warehouse, OI), verify that the partition covers the whole window. A truncated carry file once produced a fake pass. `targets()` calls `require_carry_coverage`, which raises if a pool symbol-year has more than 5% missing roll yield.

## 4. Repository layout

```
src/tfcta/          library: config (paths, partitions, guards, factor signs), data (shards, universe,
                    rolls), factors, costs (fees, ticks, multipliers), prepare/ (sharding, ticks,
                    universe, factor caches)
strategies/         commodity_original/: frozen commodity strategy;
                    commodity_close_location/: exploratory CLV book plus frozen close-location strategy;
                    index_close_location/: shared IF/IH/IC equal-weight close-location function only (no book);
                    index_bear_reversal/: the only index book: frozen CLV<0.4 long plus bear-market short that still requires CLV ≥ 0.4, sized by trailing 20-day vol, never above 1; each open capped by that day's CFFEX limit; 1e8;
                    commodity_catalog.py: commodity books; index_catalog.py: separate index books
backtest/engine.py  simulate_target_lots ledger, metrics (performance, sharpe, max_drawdown, newey_west_t,
                    deflated_sharpe, pbo_cscv), load_research / load_through_year / ledger_inputs, write_results, CLI
research/           commodity/: the commodity book runner (spans, beta/alpha, ledger, freeze gate);
                    report/: charts for the two frozen commodity books.
                    Finished one-off study scripts were removed 2026-10-29; their numbers stay in the log above.
config/research/    strategy_report.json plus the sector map in time_factor_screen.json
download/           RiceQuant downloaders (commodity minute/external/fees, index minute, index roll yield)
tests/              pytest, synthetic data only (generator in tests/synthetic.py)
docs/               the two broker reports + guide.md (commodity and research history) + index_guide.md (the default index book)
data/, runs/        gitignored; data inputs and all outputs
```

`download/index/getTickL1Close.py` saves historical main-contract L1 ticks for IF/IH/IC during the research period to `data/index/external_rqdata/l1_ticks/` and produces daily L1 summaries. Its `--partition oos_locked` mode only prefetches raw 2022–2025 ticks to `data/index/external_rqdata/holdout_locked/l1_ticks/`, without summarizing or analyzing them. RiceQuant's historical index-futures API does not provide usable multi-level L2 depth. The index close-location mechanism write-up is in the research log above; the script was removed with the other finished studies.

The two frozen commodity strategies are registered in `strategies/commodity_catalog.py`. The single frozen IF/IH/IC book (bear-short, frozen 2026-10-28) is registered separately in `strategies/index_catalog.py`. Both markets' formal strategy ledgers run through `backtest.engine`. Index research remains 2015–2021. The bear-short book’s equal-weight 2022 run was voided when the vol scale was written into the rule; the scaled rule then opened 2022 (ledger 2026-10-28 17:18). It is the default and only index book in use. The original unfiltered `commodity_close_location_reversal` candidate remains exploratory and is not in the frozen registry. Both commodity books still require explicit user authorization before any 2022+ window is opened. 2023–2025 and 2026 stay closed.

Outputs:
- Strategy runs go to `runs/strategies/<id>/<research|oos|YYYY|YYYY-YYYY>/`. Each run writes `summary.csv`, `annual.csv`, `daily.csv`, `positions.csv`, `events.csv` and `plots/performance.png`.
- The commodity chart report goes to `runs/reports/commodity/`; the bear-short research run writes its own `plots/vs_buy_hold.png` and `side_days.csv`.
- Research studies go to `runs/research/<study>/` (commodity) or `runs/index/<study>/` (index), with a `protocol.json`.
- Superseded outputs are moved to `runs/archive/<date>_superseded/`, never referenced by code.

## 5. Environment and commands

- Python: `/c/Users/GU494/anaconda3/envs/factor-mining/python.exe` (conda env `factor-mining`). Set `PYTHONIOENCODING=utf-8` on Windows.
- Run everything from the repo root. `pytest.ini` puts `src` and `.` on the path.
- RiceQuant credentials live in `config/rqdata.env`. Never print, commit or move them.

```bash
python -m pytest tests/test_research.py tests/test_index.py # focused example; use paths for changed behavior
python -m backtest.engine --research                 # 2011–2021; runs all registered strategies unless one is named
python -m backtest.engine --market index --research # 熊市空头含波动率缩放，只跑 2015–2021
python -m backtest.engine --start 2016 --end 2021     # Frozen-period research window; results go to 2016-2021/ and are used by signal-level tests
python -m backtest.engine market_clock_slow_short_20_60_120 --research  # Run only the named strategy
python -m backtest.engine --research --by-year        # A separate 1e8 book for each year
python -m backtest.engine --start 2018 --end 2020     # Research window; runs all registered strategies unless one is named
python -m backtest.engine --validation               # 2022; run only after the user starts OOS, and record it in the ledger first
python -m backtest.engine --year 2023                # Single year; name a frozen strategy and record it in the ledger first
python -m backtest.engine --oos                      # 2023–2025; name a frozen strategy and record it in the ledger first
python -m backtest.engine --oos --by-year            # 2023, 2024, and 2025; name a frozen strategy
python -m backtest.engine index_close_location_bear_short --validation # 冻结旧规则已打开的 2022，只打印
python -m research.report.strategy_report           # two commodity books: NAV/drawdown, vs buy-and-hold, components; run the engine ledgers first
python -m download.index.getIndexRollYield           # IF/IH/IC carry from contract dailies, 2015–2021
```

Every 2022+ command above is refused for any book that is not frozen. Three books are currently frozen; still, run none of them until the user says to start OOS. A window already opened only prints its saved result.

`TFCTA_DATA_ROOT`, `TFCTA_RUNS_ROOT`, `TFCTA_CONFIG_DIR` and `TFCTA_MINUTE_DIR` redirect paths. Use them only for tests and reproduction checks.

Data preparation order (README, Data preparation): `prepare/shard.py` → `prepare/universe.py` → `prepare/ticks.py` → `prepare/factors.py`, plus fee and external downloads. `shard.py` refuses to overwrite `holdout_locked/`; the minute downloader refuses incremental merge when more than one monolith exists.

## 6. Research protocol (follow on every change that produces numbers)

1. **Predeclare.**
   - Write the hypothesis, parameters, sample and pass criteria in a JSON under `config/research/` or `strategies/<id>/<id>.json` **before** running.
   - The `sample` text must state what is not read. The loaders check for the required sample-boundary phrases: exploratory strategy specs must exclude 2022 onward and 2026; research configs must exclude 2022 and 2026.
   - Code validates the JSON against its constants, so the two cannot drift silently.
2. **No parameter sweeps for selection without accounting.**
   - If you test a grid, report every cell, the number of trials, and DSR/PBO where relevant (`deflated_sharpe` and `pbo_cscv` in `backtest/engine.py`).
   - Prefer the Industrial Securities approach: pick centroid or robust parameters, not the best cell.
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
  - Weights run from −`TARGET_CAP` to `TARGET_CAP` (3); negative is short. Both live books use −1/0/1.
  - Notional = |weight| × gross × start-of-day equity / pool width. The selected-count split was deleted with the books that used it.
  - Multipliers follow the held contract (`contract_multipliers` from `ledger_inputs`). Fee rows are converted with their own contract's multiplier.
  - Lots are not resized while the weight is unchanged. The whole book has one breaker, which halts `halt_days` trading days (required argument).
  - The breaker drawdown metric in `daily.csv` resets its peak when a halt ends. Reported max drawdown (`max_drawdown`) does not.
- **No-night days:** about 100 of the 1460 research days have no night session (post-holiday days, and the 2020 night suspension). On those days the full-day `ts_low`/`ts_high` market mean shifts by roughly −0.23/+0.18 sd, because the normalisation grid is shorter.
- **Year-test factors:** `load_through_year` recomputes the five factors from 320 research days before 2022 onward. It asserts that the recomputed values equal the research cache on the last 60 research days. Keep `duration_factors`, `report_factors` and the cache on the same N250/M55 definition.
- **Flow:** `backtest.engine.main` routes commodity runs to `research.commodity.market_clock_slow_short_20_60_120.run_span(start, end, by_year)` and runs the frozen IF/IH/IC strategy through the shared `simulate_target_lots` ledger and `write_results`. `research.report.strategy_report` rebuilds the full books and asserts equality with those ledgers before drawing.
- **Index L1:** The historical IF/IH/IC tick endpoint exposes usable bid/ask level 1 only; deeper quote columns observed on the probe day were zero. Do not label these data L2 or infer order-level flow.

## 8. Coding conventions

- Match the surrounding style.
  - Docstrings and comments are short and in **Chinese**. Identifiers are English. Output CSV column names are Chinese.
  - Comment density is low; comments explain "why" and sample boundaries.
- **Tests stay minimal.** Add a test only for a material behavior change or a high-risk invariant (look-ahead, sample/holdout guards, accounting, or a live strategy rule). Prefer one concise synthetic test that exercises the changed contract; do not add broad test matrices, duplicate coverage, tests for report wording/formatting, or tests for one-off exploratory analyses. Run the focused existing tests first; run the full suite only when a shared core contract changes or the focused tests indicate wider risk. Do not remove existing tests just to reduce the count without checking what production behavior they protect.
- Prefer extending existing modules (`factors.py`, `engine.py`, `costs.py`) over new files.
- Do not add code files when a related existing module is a suitable home; create a new code file only for a distinct responsibility.
- Before adding a code file, check for a nearby module with the same responsibility; extend it when that keeps responsibilities clear, and create a separate file only for a genuinely distinct component.
  - Use one directory per strategy, with `__init__.py` plus `<id>.json`.
  - Delete dead code instead of leaving parallel versions. Git history is the archive.
- No new dependencies beyond `requirements.txt` without asking: numpy, pandas, scipy, matplotlib, pyarrow, pytest.
- Every new factor, strategy or ledger rule gets a focused test module in `tests/` using **synthetic data**. Tests must never read real 2022+ data. Cover look-ahead (signal on day t cannot use t+1), guard behaviour, and config/code consistency.
- Do not write large artifacts into the repo. `data/` and `runs/` are gitignored, and scratch work goes in `runs/scratch/`, which you delete when done.
- Commit only when asked, on a branch, staging specific files.

## 9. Checklist for any modification

1. Read the relevant module, its JSON spec and its tests. Re-read §3 to decide which partitions the change may touch.
2. If the change produces new numbers, write or adjust the predeclared spec first.
3. Implement it, then add or adjust tests.
4. Run focused tests covering the changed contract, then only the research-period entry point that the change needs. Run the full test suite only when a shared core contract changes or focused results show wider risk.
5. Update docs that describe changed behaviour or numbers: `README.md` (Chinese user manual), `backtest/README.md`, `strategies/README.md`. Update this file if rules or layout changed.
6. Report back with:
   - what changed;
   - which data partitions were read;
   - net numbers with cost assumptions, including all declared variants;
   - what is in-sample versus clean;
   - what was not verified.

   Keep it short.
