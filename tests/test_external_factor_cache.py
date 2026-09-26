import json
from datetime import date

import numpy as np
import pandas as pd

from tfcta.factors.external_factor_cache import align_asof, build_symbol_factors


def _save_source(root, dataset, key, data):
    path = root / dataset / f"{key}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(path)
    manifest_path = root / "coverage.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"version": 1, "items": {}}
    manifest["items"][f"{dataset}/{key}"] = {
        "file": f"{dataset}/{key}.pkl",
        "coverage": [["20200101", "20200131"]],
    }
    manifest_path.write_text(json.dumps(manifest))


def test_align_asof_never_backfills_and_lags_one_session():
    calendar = pd.bdate_range("2020-01-01", periods=5)
    source = pd.Series([10.0, 20.0], index=[calendar[1], calendar[3]])

    aligned = align_asof(source, calendar, lag=1)

    assert aligned.iloc[0] != aligned.iloc[0]
    assert aligned.iloc[1] != aligned.iloc[1]
    assert aligned.iloc[2] == 10.0
    assert aligned.iloc[3] == 10.0
    assert aligned.iloc[4] == 20.0


def test_build_symbol_factors_aligns_roll_and_warehouse(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=30)
    calendar = dates[5:]
    roll = pd.DataFrame({
        "yield": np.linspace(0.01, 0.02, len(dates)),
        "annualized_yield": np.linspace(0.1, 0.2, len(dates)),
        "annualized_yield_trading": np.linspace(0.11, 0.21, len(dates)),
    }, index=pd.MultiIndex.from_arrays([[
        "CU"
    ] * len(dates), dates], names=["underlying_symbol", "date"]))
    warehouse = pd.DataFrame({
        "on_warrant": np.arange(100, 130, dtype=float),
    }, index=pd.MultiIndex.from_arrays([dates, ["CU"] * len(dates)],
                                       names=["date", "underlying_symbol"]))
    _save_source(tmp_path, "roll_yield", "CU_main_sub", roll)
    _save_source(tmp_path, "warehouse", "CU", warehouse)

    factors = build_symbol_factors("CU", calendar, "research", tmp_path)

    assert factors.index.equals(calendar[1:])
    assert factors.index.name == "trading_date"
    assert factors["carry_main_sub_annualized_trading"].notna().all()
    assert factors["warehouse_on_warrant"].iloc[0] == 105.0
    assert np.isfinite(factors["warehouse_drawdown_20d"].iloc[-1])


def test_build_symbol_factors_constructs_oi_and_spot_basis(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=25)
    calendar = dates[2:]
    dominant = pd.Series("AU2001", index=dates, name="order_book_id")
    contract = pd.DataFrame({
        "settlement": np.arange(400.0, 425.0),
        "open_interest": np.arange(1000.0, 1025.0),
    }, index=pd.MultiIndex.from_arrays([["AU2001"] * len(dates), dates],
                                       names=["order_book_id", "date"]))
    spot = pd.DataFrame({
        "morning": np.full(len(dates), 390.0),
        "noon": np.full(len(dates), 395.0),
    }, index=pd.MultiIndex.from_arrays([["AU9999.SGEX"] * len(dates), dates],
                                       names=["order_book_id", "date"]))
    _save_source(tmp_path, "dominant", "AU_rank1", dominant)
    _save_source(tmp_path, "contracts", "AU2001", contract)
    _save_source(tmp_path, "spot", "benchmark_AU9999.SGEX", spot)

    factors = build_symbol_factors("AU", calendar, "research", tmp_path)

    assert factors["main_open_interest"].iloc[0] == 1002.0
    assert factors["spot_basis_noon"].iloc[0] == 7.0
    assert factors["spot_basis_noon_pct"].iloc[0] == 7.0 / 395.0
    assert np.isfinite(factors["main_oi_change_20d"].iloc[-1])