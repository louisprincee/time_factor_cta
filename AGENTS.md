# AGENTS.md — working rules for this repository

Read this before changing anything. It reflects the repo on 2026-10-27 (three frozen research books: two commodity and one index).
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

`strategies/catalog.py` lists two frozen commodity books; `strategies/index_catalog.py` lists one frozen IF/IH/IC book. The four long-only tail rules, their write-up and `backtest/index_market.py` were removed. Do not restore them.

**Original commodity book (frozen 2026-10-23).** Write-up `docs/guide.md`. This strategy combines these two legs:
- Long leg: at the close, the five-factor market score must be > 0, and that name's main-versus-sub `annualized_yield` lagged one trading day must be strictly > 0. Weight +1.
- Short leg, only on bear days: a name gets weight −1 when its own trend (mean sign of 20/60/120-day returns, (closew_t − closew_{t−n}) / close_{t−n}) is < 0 and its lagged carry is < 0.
- Budget: |weight| × 1 × equity / pool width. A name is never in both legs on the same day (long needs carry > 0, short carry < 0).
- Risk: 10% drawdown → liquidate and halt 10 trading days.
- Execution: on the next trading day's first bar (night open if there is a night session, else day open). Unchanged weight stays on; only the lot difference trades.

The market bear rule and symbol short rule share `TREND_WINDOWS = (20, 60, 120)` in `strategies/market_clock_slow_short_20_60_120/__init__.py`. Both frozen JSON window fields are checked against this constant. Scoring begins on 2011-01-14; 2010 is factor warmup only.

**Strict close-location replacements (frozen 2026-10-27).** The commodity book `commodity_close_location_reversal_strict_swap` replaces only the original five-factor long trigger with equal-weight pool CLV < 0.5; its lagged positive-carry filter and short leg are unchanged. The index book `index_close_location_reversal_strict_swap` makes the same one-for-one replacement for the equal-weight IF/IH/IC market signal and keeps the original carry filter and short leg. Their complete frozen rules are in their strategy JSON files. Both were frozen after overlapping research results had already been inspected; their research reruns are in-sample, not independent validation. Neither freeze authorizes OOS. Do not open 2022+ windows until the user explicitly names one.

- **OOS records reset (user decision 2026-10-22).** The commodity OOS ledger (`data/commodity/oos/ledger.jsonl`) and every 2022+ result folder were cleared.
  - **User decision 2026-10-26: 2022–2025 is the out-of-sample window for the current strategy, which has never been run on it. Do not open any 2022+ window until the user says to start OOS.**
    - The two strict CLV replacement books were frozen by explicit user decision on 2026-10-27 after overlapping research results had already been inspected. They also have no OOS results; this freeze does not satisfy the user's separate authorization to open an OOS window.
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
- **Commodity close-location book** (`strategies/commodity_close_location_reversal/`, `exploratory_predeclared`, runner `research/commodity_close_location_reversal.py`, run 2026-10-27). User request: the index rule on commodities. Long: pool equal-weight CLV < 0.5 → +1 in every pool name, no carry filter. Short: identical to the frozen leg; a conflict goes to the short. Post hoc (proposed after CLV was seen to subsume the score). 1 tick + fees, results in `runs/research/commodity_close_location_reversal/`:

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
- **Three frozen-book comparison and attribution** (`research/commodity_close_location_reversal.py --three-book-comparison`, run 2026-10-27). All results use research partitions only and overlap previously examined samples. With historical fees and one tick per side:
  - Commodity original (2011-01-14..2021-12-30): net Sharpe 1.03, annual 5.64%, max DD −6.48%.
  - Commodity strict CLV replacement (same sample): net Sharpe 1.12, annual 6.21%, max DD −5.99%.
  - Index strict CLV replacement (2015-04-20..2021-12-30): net Sharpe 0.88, annual 11.94%, max DD −26.35%; 2015 uses the fixed 0.2-point tick.
  - Results and charts are in `runs/research/three_strategy_clv_comparison/`. Commodity signal-only, carry-only, signal-plus-carry, and short-only books are separate counterfactuals, not additive causal contributions. The actual ledger decomposes only into the combined long leg, short leg, and costs. IF/IH/IC are also shown as single-product books and as gross contributors to the shared portfolio; those are not independent replications.
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
  - **Conclusion:** the tradable index reversal is a slow, next-session rebound of the cash market after a weak close and an overnight continuation. Futures-specific arbitrage limits and fast inventory unwinding do not explain it. The mechanism is still not identified; candidate stories to test next are the opening sell pressure of cash-equity holders under T+1 and its absorption during the day.
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
strategies/         market_clock_slow_short_20_60_120/: frozen commodity strategy;
                    commodity_close_location_reversal/: exploratory CLV book plus frozen strict replacement;
                    index_close_location_reversal/: exploratory IF/IH/IC candidate plus frozen strict replacement;
                    catalog.py: commodity books; index_catalog.py: separate index books
backtest/engine.py  simulate_target_lots ledger, metrics (performance, sharpe, max_drawdown, newey_west_t,
                    deflated_sharpe, pbo_cscv), load_research / load_through_year / ledger_inputs, write_results, CLI
research/           market_clock_slow_short_20_60_120 (spans, beta/alpha, ledger, freeze gate), market_clock_mechanism
                    (components, range, close-location and liquidity-proxy tests), market_clock_slow_short_factors
                    (signal-level diagnostics), index_market_clock (IF/IH/IC transfer test),
                    index_close_location_candidate (registered index research strategy),
                    close_location_mechanism, index_close_location_mechanism, time_factor_screen
config/research/    predeclared JSON specs for research studies and exploratory strategies
download/           RiceQuant downloaders (commodity minute/external/fees, index minute, index roll yield)
tests/              pytest, synthetic data only (generator in tests/synthetic.py)
docs/               the two broker reports + guide.md (strategy rules and research results)
data/, runs/        gitignored; data inputs and all outputs
```

`download/index/getTickL1Close.py` saves historical main-contract L1 ticks for IF/IH/IC during the research period to `data/index/external_rqdata/l1_ticks/` and produces daily L1 summaries. Its `--partition oos_locked` mode only prefetches raw 2022–2025 ticks to `data/index/external_rqdata/holdout_locked/l1_ticks/`, without summarizing or analyzing them. RiceQuant's historical index-futures API does not provide usable multi-level L2 depth. The index close-location mechanism diagnostic is `research/index_close_location_mechanism.py`, preregistered in `config/research/index_close_location_mechanism.json`.

The two frozen commodity strategies are registered in `strategies/catalog.py`; the frozen IF/IH/IC strategy is registered separately in `strategies/index_catalog.py`. Both markets' formal strategy ledgers run through `backtest.engine`; index research is restricted to 2015–2021. The original unfiltered `commodity_close_location_reversal` and simple `index_close_location_reversal` candidates remain exploratory and are not in the frozen registries. All frozen books still require explicit user authorization before any 2022+ window is opened.

Outputs:
- Strategy runs go to `runs/strategies/<id>/<commodity|index>/<research|oos|YYYY|YYYY-YYYY>/`. Each run writes `summary.csv`, `annual.csv`, `daily.csv`, `positions.csv` and `events.csv`, plus `plots/<name>/`. The optional `--index-individuals` engine flag writes IF/IH/IC diagnostic ledgers beneath `index/individual/<symbol>/research/`.
- Research studies go to `runs/research/<study>/` or `runs/index/<study>/`, with a `protocol.json`.

## 5. Environment and commands

- Python: `/c/Users/GU494/anaconda3/envs/factor-mining/python.exe` (conda env `factor-mining`). Set `PYTHONIOENCODING=utf-8` on Windows.
- Run everything from the repo root. `pytest.ini` puts `src` and `.` on the path.
- RiceQuant credentials live in `config/rqdata.env`. Never print, commit or move them.

```bash
python -m pytest tests/test_research.py tests/test_index.py # focused example; use paths for changed behavior
python -m backtest.engine --research                 # 2011–2021; runs all registered strategies unless one is named
python -m backtest.engine index_close_location_reversal_strict_swap --research --index-individuals # IF/IH/IC ledgers, research only
python -m backtest.engine --start 2016 --end 2021     # Frozen-period research window; results go to 2016-2021/ and are used by signal-level tests
python -m backtest.engine market_clock_slow_short_20_60_120 --research  # Run only the named strategy
python -m backtest.engine --research --by-year        # A separate 1e8 book for each year
python -m backtest.engine --start 2018 --end 2020     # Research window; runs all registered strategies unless one is named
python -m backtest.engine --validation               # 2022; run only after the user starts OOS, and record it in the ledger first
python -m backtest.engine --year 2023                # Single year; name a frozen strategy and record it in the ledger first
python -m backtest.engine --oos                      # 2023–2025; name a frozen strategy and record it in the ledger first
python -m backtest.engine --oos --by-year            # 2023, 2024, and 2025; name a frozen strategy
python -m research.market_clock_slow_short_factors  # signal-level diagnostics
python -m research.index_market_clock                # current strategy on IF/IH/IC 2016–2021; already run, refuses a rerun
python -m research.index_close_location_candidate --robustness  # predeclared R1–R4 robustness, research only
python -m research.index_liquidity_mechanism         # index L1 + spot mechanism tests (basis, flow impact, timing)
python -m download.index.getIndexSpot                # CSI300/SSE50/CSI500 dailies 2015-04-16..2021
python -m download.index.getMemberRank               # CFFEX top-20 member long/short positions, IF/IH/IC 2015-04-16..2021
python -m research.index_positioning_mechanism       # top-20 member positions and OI change vs the close-location reversal
python -m research.index_close_location_candidate   # exploratory long-only CLV diagnostic; not in the frozen index registry
python -m research.commodity_close_location_reversal # exploratory commodity CLV book, 3 declared variants vs original, 2011–21 and 2016–21
python -m research.commodity_close_location_reversal --three-book-comparison --overwrite # three frozen books, component attribution, and IF/IH/IC plots; research only
python -m research.market_clock_mechanism            # components, range, close location, liquidity proxies, 2011–2021
python -m download.index.getIndexRollYield           # IF/IH/IC carry from contract dailies, 2015–2021
python -m research.time_factor_screen
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
- **Flow:** `backtest.engine.main` routes commodity runs to `research.market_clock_slow_short_20_60_120.run_span(start, end, by_year)` and runs the frozen IF/IH/IC strategy and optional per-symbol diagnostics through the shared `simulate_target_lots` ledger and `write_results`. Research attribution scripts read those strategy outputs; they do not rerun the formal strategy ledger.
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
