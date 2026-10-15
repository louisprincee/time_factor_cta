# Morning timing diagnostics: commodity and index futures

## Scope and information set

This is a discovery-stage factor screen, not a strategy backtest. It does not
select a final signal, optimize a threshold, or claim out-of-sample performance.
The evaluated dates are 2016-01-01 through 2021-12-31.

| Market | Decision | Entry | Exit | Sample |
|---|---|---|---|---|
| Commodity futures | 09:16 close | 09:17 open | 11:30 close | Existing `clock_flow` cache, active-universe observations with a valid night-session return |
| IF/IH/IC | 09:46 close | 09:47 open | 11:30 close | Research minute shards only |

The index window is the first 16 minute bars from 09:31 through 09:46. A short
2015 index warm-up is read from the research shards only to form the trailing
opening-volume baseline; only 2016-2021 targets enter the diagnostics. The
script does not open the 2022+ locked index directory or commodity validation
and holdout partitions.

Commodity has 37,901 symbol-days across 37 products. It is not the full
commodity universe: the source cache contains active-universe observations
with a valid night return, so products/days without a usable night session are
not represented. Index has 4,383 symbol-days: 1,461 dates for each of IF, IH,
and IC.

## Features

All factors are observable at the stated decision time. The common outcome is
the same-session adjusted open-to-close return from entry to 11:30.

- Commodity features reuse the existing 09:16 cache: low-clock minus
  high-clock over the trading session up to 09:16, `dev15`, opening-window
  relative volume, price impact scaled by the square root of relative volume,
  volume-peak time, and aligned open-push.
- Index features are calculated only from 09:31-09:46: low-clock minus
  high-clock, `dev15`, opening-window relative volume, scaled price impact,
  volume-peak time, and price/volume duration summaries. Duration thresholds
  are the report's 3 index points and 50 volume units, applied here to this
  short morning window as exploratory measurements.
- Index overnight gap is the first 09:31 adjusted open relative to the prior
  trading date's final adjusted close. It is available at the first bar, but it
  is not an equivalent input to a commodity night-session clock.
- Opening relative volume uses the prior 20 valid same-window observations'
  median, shifted to exclude the current session, with a 15-observation
  minimum.

The commodity `clock_diff` covers the night session plus the 09:00-09:16
opening. The index `early_clock_diff` covers only 09:31-09:46. Their
coefficients are therefore not directly comparable.

## First-pass results

Correlations are computed per product/instrument over the full development
period. The quintile spread is mean return in the top factor quintile minus
mean return in the bottom quintile, using full-period ranks. This is a
retrospective descriptive spread, not a causal estimate or a walk-forward
strategy. Commodity net spread subtracts the observed round-trip cost in both
tails. Index spread is gross; the displayed break-even cost is the maximum
all-in round-trip cost per leg if the two tails were traded long-short.

### Commodity

| Feature | Mean product Spearman | Products with positive Spearman | Median Q5-Q1 gross (bp) | Median Q5-Q1 after both-leg costs (bp) |
|---|---:|---:|---:|---:|
| Full-session low-minus-high clock | +0.044 | 30/37 | +5.35 | -11.23 |
| `dev15` | -0.066 | 1/37 | -8.63 | -24.08 |
| Aligned open-push | +0.014 | 23/37 | +3.66 | -12.76 |
| Impact / sqrt(relative volume) | +0.011 | 27/37 | +2.38 | -12.92 |
| Volume-peak clock | -0.000 | 19/37 | +0.81 | -16.43 |
| Relative volume | +0.009 | 26/37 | +0.27 | -14.65 |

The daily cross-sectional rank IC for `dev15` was negative in all six years,
from -0.0005 to -0.0594 (mean -0.0306). That is consistent with a tendency for
the recent opening move to reverse, but the univariate full-sample spread does
not establish a tradable edge: costs dominate the retrospective quintile
long-short spread. The full-session clock IC was positive in four of six years
and negative in 2016 and 2021, matching the known instability in the earlier
clock study. The volume proxies show little stable univariate association.

This does not overturn the earlier finding that the clock effect is small and
that the tested volume-state interactions did not pass their predeclared
stability checks. It also does not include day-only commodity observations.

### Index futures

| Feature | Mean instrument Spearman | Instruments with positive Spearman | Median Q5-Q1 gross (bp) | Median per-leg break-even cost (bp) |
|---|---:|---:|---:|---:|
| Overnight gap return | +0.035 | 3/3 | +9.68 | +4.84 |
| Opening-volume surprise | -0.037 | 1/3 | -6.77 | -3.38 |
| Early low-minus-high clock | -0.015 | 1/3 | -6.18 | -3.09 |
| Opening return | +0.012 | 2/3 | +4.06 | +2.03 |
| Price-duration mean | +0.023 | 3/3 | +1.33 | +0.66 |
| Volume-duration slope | +0.009 | 2/3 | +2.79 | +1.39 |

The overnight gap is the clearest descriptive candidate in this screen, but
its top-minus-bottom spread is not directionally consistent in all years:
the median across the three instruments was positive in four of six years.
The clock feature is mixed across instruments and years. The duration results
are exploratory and use thresholds from an all-day report on only 16 morning
bars; they need a scale/sensitivity check before being interpreted. None of
these gross figures demonstrates profitability after actual index execution
costs.

## Interpretation and next research step

1. **Commodity:** the conditional screen below finds a clock association after
   controlling for `dev15`, but the clock-by-`dev15` interaction is weak and
   becomes smaller after adding the aligned-push control. There is not enough
   evidence to promote this interaction to a strategy. Next report single-leg,
   cost-adjusted results by year, sector, and product. Separately bring
   day-only products into the sample rather than generalizing this
   night-session subset to all commodities. Do not keep adding volume filters
   without new, stable evidence.
2. **Index:** the overnight gap adds a small amount of in-sample information
   beyond the first 16-minute open move in this screen, with positive
   conditional gap coefficients for IF/IH/IC over 2016-2021. Annual signs are
   not consistent enough for strategy promotion. Before freezing a candidate,
   check rolling stability and actual index execution costs using only
   research-period data.
3. **Both:** keep this stage as discovery. The commodity development years
   have already been repeatedly examined, and the index sample is also the
   research period; neither result is a new blind validation. Freeze any
   resulting rule before considering locked data. Do not use 2022+ data to
   choose features or thresholds.

## Conditional-information screen

The second-stage runner reads only the first-pass research CSVs and uses
2016-2021 rows. For commodities it ranks the outcome, `dev15`, and clock
within each date and pools those daily cross-sections, clustering standard
errors by date. The `dev15` coefficient remains negative (-0.0233, clustered
t=-3.05) after adding clock; the clock coefficient is +0.0238 (t=3.57).
Adding their interaction changes R-squared only from 0.00130 to 0.00146; the
interaction coefficient is -0.0425 (t=-2.18). When aligned-push and its clock
interaction are added, the `clock × dev15` coefficient shrinks to -0.0253
(t=-1.05). Across annual fits, the interaction sign is negative in all six
years but is individually close to zero in most; the clock main-effect sign
changes by year. These rank-scale effects explain very little variation and
are conditional associations, not trade returns.

For each index contract separately, the runner standardizes the same-session
09:47-11:30 return, overnight gap, and 09:31-09:46 return, and compares
gap-only, open-move-only, and joint regressions. Joint-model conditional gap
coefficients are +0.080 (HAC t=2.22) for IC, +0.094 (t=2.15) for IF, and
+0.064 (t=1.62) for IH. Adding gap to the open-move model raises in-sample
R-squared by 0.00645, 0.00880, and 0.00407, respectively. The conditional gap
coefficient is positive in five of six years for IC, and four of six years for
IF and IH. Those year-by-year changes, modest R-squared increments, and lack
of index execution-cost modeling prevent a profitability claim.

The diagnostics are explanatory screens, not independent validation. Full
period rank transforms/standardization and model comparisons are retrospective;
the annual fits describe stability but do not replace walk-forward evaluation.
The commodity outcome/features remain the active-universe, valid-night-session
subset documented above.

## Reproduction

```bash
python scripts/research/morning_cross_market.py
python scripts/research/morning_incremental.py
```

The runner and exact protocol are in
[`scripts/research/morning_cross_market.py`](../../scripts/research/morning_cross_market.py)
and `runs/research/morning_first_pass/protocol.json`. The conditional models
are in [`scripts/research/morning_incremental.py`](../../scripts/research/morning_incremental.py);
their protocol and outputs are under `runs/research/morning_incremental/`.
The first-pass output directory contains feature rows, full-period and annual
per-instrument diagnostics, and commodity daily cross-sectional rank ICs.
