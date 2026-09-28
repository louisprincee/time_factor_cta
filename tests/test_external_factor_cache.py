import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import tfcta.factors.external as external
from tfcta.factors.external import align_asof, build_symbol_factors


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
    assert np.isfinite(factors["warehouse_peak_drawdown_20d"].iloc[-1])


def test_non_research_partition_reads_its_own_subdirectory_once(tmp_path):
    dates = pd.bdate_range("2022-01-03", periods=10)
    warehouse = pd.DataFrame({"on_warrant": np.arange(10, 20, dtype=float)},
                             index=pd.MultiIndex.from_arrays(
                                 [dates, ["CU"] * len(dates)],
                                 names=["date", "underlying_symbol"]))
    _save_source(tmp_path / "validation_2022", "warehouse", "CU", warehouse)

    factors = build_symbol_factors("CU", dates, "validation_2022", tmp_path)

    assert factors["warehouse_on_warrant"].iloc[0] == 10.0


def test_build_symbol_factors_uses_main_contract_oi_and_skips_unsynchronized_basis(tmp_path):
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
    assert np.isfinite(factors["main_oi_change_20d"].iloc[-1])
    assert not any(name.startswith("spot_basis_") for name in factors.columns)


def test_warehouse_drawdown_uses_rolling_peak_not_negative_return(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=25)
    inventory = [100.0] * 20 + [200.0, 190.0, 180.0, 170.0, 160.0]
    warehouse = pd.DataFrame({"on_warrant": inventory},
                             index=pd.MultiIndex.from_arrays(
                                 [dates, ["CU"] * len(dates)],
                                 names=["date", "underlying_symbol"]))
    _save_source(tmp_path, "warehouse", "CU", warehouse)

    factors = build_symbol_factors("CU", dates, "research", tmp_path)

    assert factors["warehouse_peak_drawdown_20d"].iloc[-1] == pytest.approx(-0.15)
    assert factors["warehouse_change_20d"].iloc[-1] == pytest.approx(0.7)


def test_load_wide_ignores_unregistered_legacy_columns(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=2)
    directory = tmp_path / "research"
    directory.mkdir()
    pd.DataFrame({
        "spot_basis_noon": [1.0, 2.0],
        "main_open_interest": [100.0, 110.0],
    }, index=dates).to_pickle(directory / "AU.pkl")
    monkeypatch.setattr(external, "EXTERNAL_FACTOR_ROOT", tmp_path)

    panel = external.load_wide(["AU"], "research", dates)

    assert "main_open_interest" in panel
    assert "spot_basis_noon" not in panel


def test_require_partitions_rejects_stale_factor_schema(tmp_path, monkeypatch):
    for partition in ("research", "validation_2022"):
        directory = tmp_path / partition
        directory.mkdir()
        pd.DataFrame({"warehouse_drawdown_20d": [0.0]}).to_pickle(directory / "CU.pkl")
    manifest = {
        "partitions": {
            "research": {"CU": {"columns": ["warehouse_peak_drawdown_20d"]}},
            "validation_2022": {"CU": {"columns": ["warehouse_drawdown_20d"]}},
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(external, "EXTERNAL_FACTOR_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="warehouse_peak_drawdown_20d"):
        external.require_partitions(
            {"warehouse_peak_drawdown_20d": 1.0}, ("research", "validation_2022"))

def test_validation_partition_windows_continue_from_research(tmp_path):
    """2022 的滚动窗口必须接上研究期的原始数据，不能在分区边界上重新预热。

    只读 validation_2022 子目录的话，warehouse_change_20d 前 21 天都是 NaN，
    而且 2022 的 20 日变化会被错算成"从 2022 首日起"的变化。
    """
    research_days = pd.bdate_range("2021-11-01", "2021-12-31")
    validation_days = pd.bdate_range("2022-01-03", periods=10)

    def warehouse(days, start):
        return pd.DataFrame({"on_warrant": np.arange(start, start + len(days), dtype=float)},
                            index=pd.MultiIndex.from_arrays(
                                [days, ["CU"] * len(days)], names=["date", "underlying_symbol"]))

    _save_source(tmp_path, "warehouse", "CU", warehouse(research_days, 100))
    _save_source(tmp_path / "validation_2022", "warehouse", "CU",
                 warehouse(validation_days, 100 + len(research_days)))

    calendar = research_days.append(validation_days)
    factors = build_symbol_factors("CU", calendar, "validation_2022", tmp_path)
    first_2022 = factors.loc[validation_days[0]]
    assert np.isfinite(first_2022["warehouse_change_20d"])
    level = 100.0 + len(research_days) - 1          # 滞后一日：看到的是 2021-12-31 的值
    assert first_2022["warehouse_on_warrant"] == level
    assert first_2022["warehouse_change_20d"] == pytest.approx(level / (level - 20) - 1)

    # 研究期分区不得读到验证期子目录
    research_only = build_symbol_factors("CU", research_days, "research", tmp_path)
    assert research_only.index.max() <= research_days[-1]
    assert research_only["warehouse_on_warrant"].max() < 100 + len(research_days)
