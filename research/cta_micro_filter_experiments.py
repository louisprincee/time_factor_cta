"""Compare predeclared low-frequency trend and intraday-time-factor variants.

    python -m research.cta_micro_filter_experiments --period both

Only the 2016-2021 research and 2022 validation partitions are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import (
    LONG_ONLY_SECTORS,
    RESEARCH_YEARS,
    StudyData,
    performance,
    sharpe_ratio,
    simulate_integer_lots,
    load_research,
    load_validation,
)
from strategies.ts_low import upper_tail
from tfcta import config as C
from tfcta.costs import MULTIPLIER, require_multipliers
from tfcta.data import (
    add_intraday_coords,
    find_shard,
    load_roll_calendar,
    load_shard,
    load_validation_shard,
)
from tfcta.factors import (
    day_codes_of,
    duration_series,
    intraday_abs_diff,
    raw_price_path,
    report_factors,
    rolling_threshold,
    tsmom_sign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "research" / "cta_micro_filter_experiments.json"
OUTPUT = C.RUNS_DIR / "research" / "cta_micro_filter_experiments"
INITIAL_CAPITAL = 100_000_000.0
HOLDING_DAYS = 5
TREND_WINDOW = 20
WARMUP_DAYS = C.IC_REFERENCE_LOOKBACK
DURATION_WARMUP_DAYS = WARMUP_DAYS + 1
MICRO_LOOKBACK = 250
ENTRY_QUANTILE = 0.8
RESET_QUANTILE = 0.5
DURATION_WINDOW = 5
DURATION_REFERENCE = 250
ADX_PERIOD = 14
ADX_RANGE_MAX = 20.0
ADX_TREND_MIN = 25.0
ATR_PERIOD = 14
ATR_REFERENCE = 250
VOLUME_DURATION_REFERENCE = 250
INTRADAY_DURATION_REFERENCE = 250
SECTORS = ("农产品", "黑色钢铁", "有色金属", "化工")
RESEARCH_START = pd.Timestamp("2016-01-01")
RESEARCH_END = pd.Timestamp("2021-12-31")
VALIDATION_START = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")
MINUTE_COLUMNS = [
    "trading_date", "close", "highw", "loww", "volume", "total_turnover",
]


def _read_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "exploratory_predeclared_comparison":
        raise ValueError("实验配置必须标记为事前声明的探索性比较")
    portfolio = config["portfolio"]
    if (
        portfolio["direction"] != "long_only"
        or portfolio["holding_days"] != HOLDING_DAYS
        or portfolio["trend_window_days"] != TREND_WINDOW
        or portfolio["initial_capital"] != INITIAL_CAPITAL
        or portfolio["exposure_limit"] != 1.5
        or portfolio["margin_rate_assumption"] != 0.2
        or portfolio["drawdown_breaker"] != 0.2
    ):
        raise ValueError("组合设定与当前实验实现不一致")
    micro = config["micro_filter"]
    if (
        micro["factor"] != "ts_low - ts_high"
        or micro["lookback_days"] != MICRO_LOOKBACK
        or micro["entry_quantile"] != ENTRY_QUANTILE
        or micro["smoothing_spans"] != [3, 5]
        or micro["hysteresis_reset_quantile"] != RESET_QUANTILE
        or micro["duration_stability_days"] != DURATION_WINDOW
        or micro["duration_reference_days"] != DURATION_REFERENCE
        or micro["duration_stability_quantile"] != 0.5
        or micro["volume_duration_ratio_reference_days"] != VOLUME_DURATION_REFERENCE
        or micro["volume_duration_ratio_quantile"] != 0.5
        or micro["duration_warmup_days"] != DURATION_WARMUP_DAYS
        or micro["intraday_duration_reference_days"] != INTRADAY_DURATION_REFERENCE
        or micro["intraday_duration_level_quantile"] != 0.5
        or micro["intraday_duration_volatility_quantile"] != 0.5
        or micro["adx_period"] != ADX_PERIOD
        or micro["adx_range_max"] != ADX_RANGE_MAX
        or micro["adx_trend_min"] != ADX_TREND_MIN
        or micro["atr_period"] != ATR_PERIOD
        or micro["atr_reference_days"] != ATR_REFERENCE
        or micro["atr_low_vol_quantile"] != 0.5
    ):
        raise ValueError("微观过滤设定与当前实验实现不一致")
    return config


def _validation_duration_factors(
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    duration_rows = []
    for symbol in symbols:
        validation = load_validation_shard(symbol, columns=MINUTE_COLUMNS)
        if find_shard(C.RESEARCH_DIR, symbol) is None:
            print(
                f"{symbol}: 无研究期持续期预热，该品种持续期变体在验证期不交易",
                flush=True,
            )
            continue
        research = load_shard(
            symbol, directory=C.RESEARCH_DIR, columns=MINUTE_COLUMNS,
        )
        research_days = pd.DatetimeIndex(
            pd.to_datetime(research["trading_date"]).dt.normalize().unique(),
        ).sort_values()
        warmup = research[
            pd.to_datetime(research["trading_date"]).dt.normalize().isin(
                research_days[-DURATION_WARMUP_DAYS:],
            )
        ]
        combined = pd.concat([warmup, validation]).sort_index(kind="mergesort")
        if combined.index.has_duplicates:
            raise ValueError(f"{symbol} 的研究/验证分钟持续期输入有重复时间戳")
        C.assert_no_holdout_dates(
            pd.to_datetime(warmup["trading_date"]), f"{symbol} 持续期研究预热",
        )
        C.assert_validation_2022_dates(
            pd.to_datetime(validation["trading_date"]), f"{symbol} 持续期验证输入",
        )
        factors = report_factors(
            add_intraday_coords(combined),
            lookback=C.IC_REFERENCE_LOOKBACK,
            pct=C.IC_REFERENCE_PCT,
        )
        duration_summary = _intraday_duration_summary(combined)
        duration_summary = duration_summary.loc[
            duration_summary.index.year == 2022,
        ].copy()
        duration_summary["symbol"] = symbol
        duration_rows.append(duration_summary.reset_index())
        factors = factors.loc[factors.index.year == 2022, ["pmt", "vd_ratio"]].copy()
        factors["symbol"] = symbol
        rows.append(factors.reset_index())
    if not rows:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="trading_date"))
        duration_summary = {
            column: empty.copy()
            for column in (
                "price_duration_mean", "price_duration_vol",
                "volume_duration_mean", "volume_duration_vol",
            )
        }
        return empty, empty.copy(), duration_summary
    panel = pd.concat(rows, ignore_index=True)
    pmt = panel.pivot(index="trading_date", columns="symbol", values="pmt").sort_index()
    vd_ratio = panel.pivot(
        index="trading_date", columns="symbol", values="vd_ratio",
    ).sort_index()
    duration_panel = pd.concat(duration_rows, ignore_index=True)
    duration_summary = {
        column: duration_panel.pivot(
            index="trading_date", columns="symbol", values=column,
        ).sort_index()
        for column in (
            "price_duration_mean", "price_duration_vol",
            "volume_duration_mean", "volume_duration_vol",
        )
    }
    return pmt, vd_ratio, duration_summary


def _intraday_duration_summary(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_intraday_coords(frame)
    day_codes, days = day_codes_of(enriched)
    price = raw_price_path(enriched)
    volume = enriched["volume"].to_numpy(dtype="float64")
    price_threshold = rolling_threshold(
        intraday_abs_diff(price, day_codes),
        day_codes,
        C.IC_REFERENCE_LOOKBACK,
        C.IC_REFERENCE_PCT,
    )
    volume_threshold = rolling_threshold(
        intraday_abs_diff(volume, day_codes),
        day_codes,
        C.IC_REFERENCE_LOOKBACK,
        C.IC_REFERENCE_PCT,
    )
    price_duration = duration_series(price, day_codes, price_threshold)
    volume_duration = duration_series(volume, day_codes, volume_threshold)
    boundaries = np.flatnonzero(
        np.r_[True, day_codes[1:] != day_codes[:-1], True],
    )
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        price_values = price_duration[start:end]
        volume_values = volume_duration[start:end]
        price_values = price_values[np.isfinite(price_values)]
        volume_values = volume_values[np.isfinite(volume_values)]
        rows.append({
            "price_duration_mean": (
                float(price_values.mean()) if price_values.size else np.nan
            ),
            "price_duration_vol": (
                float(price_values.std(ddof=1)) if price_values.size >= 2 else np.nan
            ),
            "volume_duration_mean": (
                float(volume_values.mean()) if volume_values.size else np.nan
            ),
            "volume_duration_vol": (
                float(volume_values.std(ddof=1)) if volume_values.size >= 2 else np.nan
            ),
        })
    summary = pd.DataFrame(rows, index=days)
    summary.index.name = "trading_date"
    return summary


def _research_duration_factors(
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    columns = (
        "price_duration_mean", "price_duration_vol",
        "volume_duration_mean", "volume_duration_vol",
    )
    if not symbols:
        empty = pd.DataFrame(
            index=pd.DatetimeIndex([], name="trading_date"),
        )
        return {column: empty.copy() for column in columns}
    feature_rows = []
    for symbol in symbols:
        if find_shard(C.RESEARCH_DIR, symbol) is None:
            raise FileNotFoundError(f"{symbol} 缺少研究期分钟持续期输入")
        frame = load_shard(
            symbol, directory=C.RESEARCH_DIR, columns=MINUTE_COLUMNS,
        )
        trading_dates = pd.to_datetime(frame["trading_date"]).dt.normalize()
        all_days = pd.DatetimeIndex(trading_dates.unique()).sort_values()
        warmup_days = all_days[all_days < RESEARCH_START][-DURATION_WARMUP_DAYS:]
        sample_days = all_days[
            (all_days >= RESEARCH_START) & (all_days <= RESEARCH_END)
        ]
        keep_days = warmup_days.union(sample_days)
        frame = frame.loc[trading_dates.isin(keep_days)]
        C.assert_no_holdout_dates(
            pd.to_datetime(frame["trading_date"]), f"{symbol} 持续期研究输入",
        )
        summary = _intraday_duration_summary(frame)
        summary = summary.loc[
            (summary.index >= RESEARCH_START) & (summary.index <= RESEARCH_END),
        ].copy()
        summary["symbol"] = symbol
        feature_rows.append(summary.reset_index())
    panel = pd.concat(feature_rows, ignore_index=True)
    return {
        column: panel.pivot(
            index="trading_date", columns="symbol", values=column,
        ).sort_index()
        for column in columns
    }


def _signal_inputs(
    data: StudyData,
    micro: pd.DataFrame,
    pmt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars = data.factors.bars
    close = bars["close"]
    closew = bars["closew"]
    trend = tsmom_sign(close, closew, window=TREND_WINDOW)
    return trend, micro, pmt


def market_regimes(bars: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    high = bars["highw"].astype("float64")
    low = bars["loww"].astype("float64")
    close = bars["closew"].astype("float64")
    prior_close = close.shift(1)
    true_range = pd.DataFrame(
        np.fmax.reduce([
            (high - low).to_numpy(),
            (high - prior_close).abs().to_numpy(),
            (low - prior_close).abs().to_numpy(),
        ]),
        index=close.index,
        columns=close.columns,
    )
    upward = high.diff()
    downward = -low.diff()
    plus_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    alpha = 1.0 / ADX_PERIOD
    average_tr = true_range.ewm(
        alpha=alpha, min_periods=ADX_PERIOD, adjust=False,
    ).mean()
    plus_di = 100.0 * plus_dm.ewm(
        alpha=alpha, min_periods=ADX_PERIOD, adjust=False,
    ).mean() / average_tr.where(average_tr > 0)
    minus_di = 100.0 * minus_dm.ewm(
        alpha=alpha, min_periods=ADX_PERIOD, adjust=False,
    ).mean() / average_tr.where(average_tr > 0)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).where(
        (plus_di + minus_di) > 0,
    )
    adx = dx.ewm(alpha=alpha, min_periods=ADX_PERIOD, adjust=False).mean()
    trend = tsmom_sign(bars["close"], bars["closew"], window=TREND_WINDOW)
    regime = pd.DataFrame("transition", index=close.index, columns=close.columns)
    regime = regime.mask(adx.le(ADX_RANGE_MAX), "range")
    regime = regime.mask(adx.ge(ADX_TREND_MIN) & trend.gt(0), "trend_up")
    regime = regime.mask(adx.ge(ADX_TREND_MIN) & trend.lt(0), "trend_down")
    regime = regime.where(adx.notna())

    atr = average_tr / close.where(close > 0)
    prior_atr_median = atr.shift(1).rolling(
        ATR_REFERENCE, min_periods=ATR_REFERENCE,
    ).median()
    volatility = pd.DataFrame(index=close.index, columns=close.columns, dtype=object)
    volatility = volatility.mask(atr.le(prior_atr_median), "low_vol")
    volatility = volatility.mask(atr.gt(prior_atr_median), "high_vol")
    return regime, volatility


def _hysteresis_entries(
    trend: pd.DataFrame,
    micro: pd.DataFrame,
) -> pd.DataFrame:
    entries = pd.DataFrame(False, index=micro.index, columns=micro.columns)
    for symbol in micro.columns:
        values = micro[symbol].to_numpy(dtype="float64")
        trend_values = trend[symbol].reindex(micro.index).to_numpy(dtype="float64")
        finite = np.isfinite(values)
        armed = True
        for position, day in enumerate(micro.index):
            if not finite[position]:
                continue
            history = values[max(0, position - MICRO_LOOKBACK):position]
            history = history[np.isfinite(history)]
            if history.size < MICRO_LOOKBACK:
                continue
            entry_threshold, reset_threshold = np.quantile(
                history, [ENTRY_QUANTILE, RESET_QUANTILE],
            )
            if values[position] <= reset_threshold:
                armed = True
            elif armed and values[position] > entry_threshold:
                if day in entries.index and np.isfinite(trend_values[position]) and trend_values[position] > 0:
                    entries.at[day, symbol] = True
                    armed = False
    return entries


def build_signal_variants(
    trend: pd.DataFrame,
    micro: pd.DataFrame,
    pmt: pd.DataFrame,
    volume_duration_ratio: pd.DataFrame | None = None,
    duration_summary: dict[str, pd.DataFrame] | None = None,
    regime: pd.DataFrame | None = None,
    volatility: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    trend = trend.reindex(index=micro.index, columns=micro.columns)
    pmt = pmt.reindex(index=micro.index, columns=micro.columns)
    trend_long = trend.gt(0)
    daily_tail = pd.DataFrame(
        {
            symbol: upper_tail(micro[symbol], MICRO_LOOKBACK, ENTRY_QUANTILE)
            for symbol in micro.columns
        },
        index=micro.index,
    ).gt(0)
    variants = {
        "trend_20d": trend_long,
        "trend_plus_daily_micro_filter": trend_long & daily_tail,
    }
    for span in (3, 5):
        smoothed = micro.ewm(span=span, min_periods=span, adjust=False).mean()
        tail = pd.DataFrame(
            {
                symbol: upper_tail(smoothed[symbol], MICRO_LOOKBACK, ENTRY_QUANTILE)
                for symbol in smoothed.columns
            },
            index=smoothed.index,
        ).gt(0)
        variants[f"trend_plus_ema{span}_micro_filter"] = trend_long & tail

    duration_clock_volatility = pmt.rolling(
        DURATION_WINDOW, min_periods=DURATION_WINDOW,
    ).std()
    prior_stability_limit = duration_clock_volatility.shift(1).rolling(
        DURATION_REFERENCE, min_periods=DURATION_REFERENCE,
    ).median()
    stable_duration_clock = duration_clock_volatility.le(prior_stability_limit)
    variants["trend_plus_duration_clock_stability"] = trend_long & stable_duration_clock
    variants["trend_plus_micro_hysteresis"] = _hysteresis_entries(
        trend, micro,
    )
    if regime is not None:
        regime = regime.reindex(index=micro.index, columns=micro.columns)
        variants["trend_plus_daily_micro_filter_range_only"] = (
            trend_long & daily_tail & regime.eq("range")
        )
        variants["trend_plus_daily_micro_filter_trend_only"] = (
            trend_long & daily_tail & regime.eq("trend_up")
        )
    if volatility is not None:
        volatility = volatility.reindex(index=micro.index, columns=micro.columns)
        variants["trend_plus_daily_micro_filter_low_atr"] = (
            trend_long & daily_tail & volatility.eq("low_vol")
        )
        variants["trend_plus_daily_micro_filter_high_atr"] = (
            trend_long & daily_tail & volatility.eq("high_vol")
        )
    if volume_duration_ratio is not None:
        volume_duration_ratio = volume_duration_ratio.reindex(
            index=micro.index, columns=micro.columns,
        )
        prior_ratio_median = volume_duration_ratio.shift(1).rolling(
            VOLUME_DURATION_REFERENCE,
            min_periods=VOLUME_DURATION_REFERENCE,
        ).median()
        volume_duration_support = volume_duration_ratio.gt(prior_ratio_median)
        variants["trend_plus_daily_micro_filter_and_volume_duration"] = (
            trend_long & daily_tail & volume_duration_support
        )
    if duration_summary is not None:
        duration_states = {}
        for feature in (
            "price_duration_mean", "price_duration_vol",
            "volume_duration_mean", "volume_duration_vol",
        ):
            values = duration_summary[feature].reindex(
                index=micro.index, columns=micro.columns,
            )
            prior_median = values.shift(1).rolling(
                INTRADAY_DURATION_REFERENCE,
                min_periods=INTRADAY_DURATION_REFERENCE,
            ).median()
            duration_states[feature] = values.gt(prior_median) if feature.endswith(
                "_mean",
            ) else values.lt(prior_median)
        price_stable = (
            duration_states["price_duration_mean"]
            & duration_states["price_duration_vol"]
        )
        volume_stable = (
            duration_states["volume_duration_mean"]
            & duration_states["volume_duration_vol"]
        )
        variants["trend_plus_daily_micro_filter_price_duration_stable"] = (
            trend_long & daily_tail & price_stable
        )
        variants["trend_plus_daily_micro_filter_volume_duration_stable"] = (
            trend_long & daily_tail & volume_stable
        )
        variants["trend_plus_daily_micro_filter_both_durations_stable"] = (
            trend_long & daily_tail & price_stable & volume_stable
        )

    variants["current_ts_low_reference"] = daily_tail
    return variants


def _forward_signal_regime_diagnostics(
    variants: dict[str, pd.DataFrame],
    regime: pd.DataFrame,
    returns: pd.DataFrame,
    universe: dict[int, list[str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    period: str,
) -> pd.DataFrame:
    rows = []
    pools = {int(year): set(names) for year, names in universe.items()}
    for name, signal in variants.items():
        for symbol in returns.columns:
            if symbol not in signal.columns or symbol not in regime.columns:
                continue
            tradable = returns.index[
                np.isfinite(returns[symbol].to_numpy(dtype="float64"))
            ]
            for day in signal.index[signal[symbol].fillna(False)]:
                day = pd.Timestamp(day)
                if day < start or day > end or symbol not in pools.get(day.year, set()):
                    continue
                state = regime.at[day, symbol] if day in regime.index else np.nan
                if pd.isna(state):
                    continue
                later = tradable[tradable > day]
                if not len(later):
                    continue
                entry_pos = returns.index.get_loc(later[0])
                if entry_pos + HOLDING_DAYS > len(returns.index):
                    continue
                holding_days = returns.index[entry_pos:entry_pos + HOLDING_DAYS]
                if len(holding_days) != HOLDING_DAYS:
                    continue
                trade_returns = returns.loc[holding_days, symbol].to_numpy(dtype="float64")
                if len(trade_returns) != HOLDING_DAYS or not np.isfinite(trade_returns).all():
                    continue
                gross = float(np.prod(1.0 + trade_returns) - 1.0)
                rows.append({
                    "策略": name,
                    "区间": period,
                    "信号日": day,
                    "品种": symbol,
                    "市场状态": str(state),
                    "后续5日毛收益": gross,
                })
    if not rows:
        return pd.DataFrame(columns=[
            "策略", "区间", "市场状态", "信号数", "平均后续5日毛收益",
            "中位后续5日毛收益", "胜率",
        ])
    observations = pd.DataFrame(rows)
    return (
        observations.groupby(["策略", "区间", "市场状态"], sort=True)
        .agg(
            信号数=("后续5日毛收益", "size"),
            平均后续5日毛收益=("后续5日毛收益", "mean"),
            中位后续5日毛收益=("后续5日毛收益", "median"),
            胜率=("后续5日毛收益", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )


def _entry_schedule(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe: dict[int, list[str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    index = returns.index
    entries = pd.DataFrame(False, index=index, columns=returns.columns)
    pools = {int(year): set(names) for year, names in universe.items()}
    for symbol in returns.columns:
        if symbol not in signal.columns:
            continue
        valid = np.isfinite(returns[symbol].to_numpy(dtype="float64"))
        tradable = index[valid]
        order = {pd.Timestamp(day): i for i, day in enumerate(tradable)}
        blocked_until = -1
        for day in signal.index[signal[symbol].fillna(False)]:
            day = pd.Timestamp(day)
            if day < start or day > end or symbol not in pools.get(day.year, set()):
                continue
            later = tradable[tradable > day]
            if not len(later):
                continue
            entry_day = pd.Timestamp(later[0])
            entry_position = order[entry_day]
            if (
                symbol not in pools.get(entry_day.year, set())
                or entry_position <= blocked_until
                or entry_position + HOLDING_DAYS >= len(tradable)
            ):
                continue
            entries.at[entry_day, symbol] = True
            blocked_until = entry_position + HOLDING_DAYS
    return entries


def _account_inputs(
    data: StudyData,
    period: str,
    index: pd.DatetimeIndex,
) -> tuple[dict, dict[int, list[str]], list[str]]:
    period_years = tuple(RESEARCH_YEARS) if period == "research" else (2022,)
    end = RESEARCH_END if period == "research" else VALIDATION_END
    symbols = sorted({
        symbol
        for sector in SECTORS
        for symbol in LONG_ONLY_SECTORS[sector]
        if symbol in data.returns.columns
        and any(
            symbol in members and int(year) in period_years
            for year, members in data.universe.items()
        )
    })
    require_multipliers(symbols)
    universe = {
        int(year): sorted(set(members).intersection(symbols))
        for year, members in data.universe.items()
        if int(year) in period_years
    }
    universe = {year: members for year, members in universe.items() if members}
    symbols = sorted({symbol for members in universe.values() for symbol in members})
    if not symbols:
        raise ValueError(f"{period} 实验池为空")

    period_index = index[index.year.isin(period_years)]
    bars = data.factors.bars
    open_price = bars["open"].reindex(index=period_index, columns=symbols)
    roll_calendar = load_roll_calendar(symbols, end.date())
    contract_codes = pd.DataFrame(index=period_index, columns=symbols, dtype=object)
    for symbol in symbols:
        calendar = roll_calendar.get(symbol)
        if calendar is None or calendar.empty:
            raise FileNotFoundError(f"{symbol} 缺少换月合约日历")
        calendar = calendar.drop_duplicates("trading_date", keep="last").set_index(
            "trading_date",
        ).sort_index()
        changes = calendar["new_contract"].copy()
        if calendar.index.min() > period_index.min():
            changes.loc[period_index.min()] = calendar.iloc[0]["prev_contract"]
        contract_codes[symbol] = changes.reindex(
            changes.index.union(period_index),
        ).sort_index().ffill().reindex(period_index)
    if contract_codes.isna().any().any():
        raise ValueError(f"{period} 实验合约代码映射缺失")

    account = {
        "open_price": open_price,
        "close_price": bars["close"].reindex(index=period_index, columns=symbols),
        "adjusted_open": bars["openw"].reindex(index=period_index, columns=symbols),
        "adjusted_open_delta": (
            bars["openw"].shift(-1) - bars["openw"]
        ).reindex(index=period_index, columns=symbols),
        "adjusted_high": bars["highw"].reindex(index=period_index, columns=symbols),
        "adjusted_low": bars["loww"].reindex(index=period_index, columns=symbols),
        "rolls": data.rolls.reindex(index=period_index, columns=symbols, fill_value=False),
        "contract_codes": contract_codes,
        "universe": universe,
        "multipliers": {symbol: MULTIPLIER[symbol] for symbol in symbols},
        "open_fee": data.open_fee.reindex(index=period_index, columns=symbols),
        "close_fee": data.close_fee.reindex(index=period_index, columns=symbols),
        "roll_close_fee": data.roll_close_fee.reindex(index=period_index, columns=symbols),
        "slippage": data.slippage.reindex(index=period_index, columns=symbols),
        "initial_capital": INITIAL_CAPITAL,
        "margin_rate": 0.20,
        "holding_days": HOLDING_DAYS,
        "exposure_limit": 1.5,
        "drawdown_limit": 0.20,
    }
    return account, universe, symbols


def _summarize_variant(
    name: str,
    daily: pd.DataFrame,
    positions: pd.DataFrame,
    events: pd.DataFrame,
) -> dict:
    returns = daily["净收益"].fillna(0.0)
    metrics = performance(returns)
    gross_pnl = float(daily["毛盈亏"].sum())
    fees = float(daily["交易费用"].sum())
    return {
        "策略": name,
        "净累计收益": float(daily["权益"].iloc[-1] / INITIAL_CAPITAL - 1.0),
        "毛盈亏占初始权益": gross_pnl / INITIAL_CAPITAL,
        "费用占初始权益": fees / INITIAL_CAPITAL,
        "费用占毛盈亏": fees / gross_pnl if gross_pnl > 0 else np.nan,
        "费用占毛盈亏绝对值": fees / abs(gross_pnl) if gross_pnl != 0 else np.nan,
        "年化收益": metrics["ann_return"],
        "Sharpe": sharpe_ratio(returns),
        "年化波动": metrics["ann_vol"],
        "最大回撤": float(daily["峰值回撤"].min()),
        "平均敞口倍数": float(daily["敞口倍数"].replace([np.inf, -np.inf], np.nan).mean()),
        "最大敞口倍数": float(daily["敞口倍数"].replace([np.inf, -np.inf], np.nan).max()),
        "开仓笔数": int(daily["开仓笔数"].sum()),
        "平仓笔数": int(daily["平仓笔数"].sum()),
        "换月笔数": int(daily["换月笔数"].sum()),
        "受阻委托数": int(daily["受阻委托数"].sum()),
        "持仓品种日": int(len(positions)),
        "有持仓交易日": int(daily["持仓品种数"].gt(0).sum()),
    }


def _compare_period(
    period: str,
    data: StudyData,
    micro: pd.DataFrame,
    pmt: pd.DataFrame,
    volume_duration_ratio: pd.DataFrame,
    duration_summary: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    years = tuple(RESEARCH_YEARS) if period == "research" else (2022,)
    index = data.returns.index[data.returns.index.year.isin(years)]
    trend, micro, pmt = _signal_inputs(data, micro, pmt)
    account, universe, symbols = _account_inputs(data, period, index)
    micro = micro.reindex(columns=symbols)
    trend = trend.reindex(columns=symbols)
    pmt = pmt.reindex(columns=symbols)
    volume_duration_ratio = volume_duration_ratio.reindex(columns=symbols)
    duration_summary = {
        feature: values.reindex(columns=symbols)
        for feature, values in duration_summary.items()
    }
    regime, volatility = market_regimes(data.factors.bars)
    regime = regime.reindex(columns=symbols)
    volatility = volatility.reindex(columns=symbols)

    variants = build_signal_variants(
        trend, micro, pmt, volume_duration_ratio, duration_summary,
        regime, volatility,
    )
    regime_diagnostics = _forward_signal_regime_diagnostics(
        variants,
        regime,
        data.returns.reindex(columns=symbols),
        universe,
        RESEARCH_START if period == "research" else VALIDATION_START,
        RESEARCH_END if period == "research" else VALIDATION_END,
        period,
    )
    summaries = []
    daily_frames = []
    annual_frames = []
    event_frames = []
    positions_frames = []
    for name, signal in variants.items():
        entries = _entry_schedule(
            signal.reindex(columns=symbols),
            data.returns.reindex(columns=symbols),
            universe,
            RESEARCH_START if period == "research" else VALIDATION_START,
            RESEARCH_END if period == "research" else VALIDATION_END,
        ).reindex(index=index, columns=symbols, fill_value=False).astype("int8")
        daily, positions, events = simulate_integer_lots(entries=entries, **account)
        C.assert_no_holdout_dates(daily.index, f"{name} {period}") if period == "research" else (
            C.assert_validation_2022_dates(daily.index, f"{name} {period}")
        )
        summaries.append(_summarize_variant(name, daily, positions, events))
        tagged_daily = daily.reset_index().assign(策略=name, 区间=period)
        daily_frames.append(tagged_daily)
        if not positions.empty:
            positions_frames.append(positions.assign(策略=name, 区间=period))
        if not events.empty:
            event_frames.append(events.assign(策略=name, 区间=period))
        for year, group in daily.groupby(daily.index.year):
            year_returns = group["净收益"].fillna(0.0)
            annual_frames.append({
                "策略": name,
                "区间": period,
                "年份": int(year),
                "净收益": float(group["权益"].iloc[-1] / group["期初权益"].iloc[0] - 1.0),
                "Sharpe": sharpe_ratio(year_returns),
                "最大回撤": float(group["峰值回撤"].min()),
                "毛盈亏": float(group["毛盈亏"].sum()),
                "费用": float(group["交易费用"].sum()),
                "开仓笔数": int(group["开仓笔数"].sum()),
            })
    summary = pd.DataFrame(summaries)
    trend_baseline = summary.loc[summary["策略"] == "trend_20d"].iloc[0]
    summary["相对20日趋势净收益差"] = summary["净累计收益"] - trend_baseline["净累计收益"]
    summary["相对20日趋势Sharpe差"] = summary["Sharpe"] - trend_baseline["Sharpe"]
    summary["区间"] = period
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / f"{period}_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual_frames).to_csv(
        OUTPUT / f"{period}_annual.csv", index=False, encoding="utf-8-sig",
    )
    pd.concat(daily_frames, ignore_index=True).to_csv(
        OUTPUT / f"{period}_daily.csv", index=False, encoding="utf-8-sig",
    )
    pd.concat(positions_frames, ignore_index=True).to_csv(
        OUTPUT / f"{period}_positions.csv", index=False, encoding="utf-8-sig",
    )
    pd.concat(event_frames, ignore_index=True).to_csv(
        OUTPUT / f"{period}_events.csv", index=False, encoding="utf-8-sig",
    )
    regime_diagnostics.to_csv(
        OUTPUT / f"{period}_regime_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n{period} integer-lot comparison:", flush=True)
    print(summary.to_string(index=False), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", choices=("research", "validation", "both"), default="both")
    args = parser.parse_args(argv)
    config = _read_config()
    (OUTPUT / "protocol.json").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "protocol.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    research_duration_summary = None
    if args.period in ("research", "both"):
        research = load_research(1.0)
        micro = research.factors.raw("ts_low") - research.factors.raw("ts_high")
        pmt = research.factors.raw("pmt")
        volume_duration_ratio = research.factors.raw("vd_ratio")
        duration_symbols = sorted({
            symbol
            for sector in SECTORS
            for symbol in LONG_ONLY_SECTORS[sector]
            if symbol in research.returns.columns
            and any(
                symbol in members and int(year) in RESEARCH_YEARS
                for year, members in research.universe.items()
            )
        })
        research_duration_summary = _research_duration_factors(duration_symbols)
        summaries = [
            _compare_period(
                "research", research, micro, pmt, volume_duration_ratio,
                research_duration_summary,
            )
        ]
    else:
        summaries = []
    if args.period in ("validation", "both"):
        validation, micro = load_validation(1.0)
        symbols = sorted(set(validation.universe.get(2022, ())))
        (
            validation_pmt,
            validation_vd_ratio,
            validation_duration_summary,
        ) = _validation_duration_factors(symbols)
        pmt_history = validation.factors.raw("pmt")
        volume_duration_history = validation.factors.raw("vd_ratio")
        pmt = pd.concat([pmt_history, validation_pmt]).sort_index()
        volume_duration_ratio = pd.concat(
            [volume_duration_history, validation_vd_ratio],
        ).sort_index()
        history_symbols = [
            symbol for symbol in symbols
            if find_shard(C.RESEARCH_DIR, symbol) is not None
        ]
        if research_duration_summary is None:
            historical_duration_summary = _research_duration_factors(history_symbols)
        else:
            cached_symbols = set(
                research_duration_summary["price_duration_mean"].columns,
            )
            missing_history_symbols = [
                symbol for symbol in history_symbols if symbol not in cached_symbols
            ]
            additional_duration_summary = _research_duration_factors(
                missing_history_symbols,
            )
            historical_duration_summary = {}
            for feature, cached in research_duration_summary.items():
                combined = pd.concat(
                    [cached, additional_duration_summary[feature]], axis=1,
                )
                combined = combined.loc[:, ~combined.columns.duplicated()]
                historical_duration_summary[feature] = combined.reindex(
                    columns=history_symbols,
                )
        duration_summary = {}
        for feature in (
            "price_duration_mean", "price_duration_vol",
            "volume_duration_mean", "volume_duration_vol",
        ):
            history = historical_duration_summary[feature]
            duration_summary[feature] = pd.concat(
                [history, validation_duration_summary[feature]],
            ).sort_index()
        summaries.append(
            _compare_period(
                "validation", validation, micro, pmt, volume_duration_ratio,
                duration_summary,
            )
        )
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(OUTPUT / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nComparison artifacts: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
