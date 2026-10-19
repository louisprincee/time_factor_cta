"""股指市场切换：日盘时钟、研究期不碰锁定分区。"""
import pandas as pd
import pytest

from tfcta import config as C
from tfcta.data import save_shard
from backtest.index_market import assert_day_session, dominant_calendar, load_research


def _minute(prefix):
    index = pd.to_datetime([
        "2015-01-05 09:31", "2015-01-05 09:32",
        "2016-01-04 09:31", "2016-01-04 09:32",
        "2016-01-05 09:31", "2016-01-05 09:32",
    ])
    close = [100.0, 100.2, 101.0, 101.2, 102.0, 102.2]
    return pd.DataFrame({
        "trading_date": pd.to_datetime(["2015-01-05"] * 2 + ["2016-01-04"] * 2 + ["2016-01-05"] * 2),
        "open": close, "openw": close, "close": close, "closew": close,
        "highw": [value + 1 for value in close],
        "loww": [value - 1 for value in close],
        "volume": 10.0,
        "total_turnover": 1.0e10,
    }, index=index), pd.DataFrame({
        "trading_date": pd.to_datetime(["2015-01-05"] * 2 + ["2016-01-04"] * 2 + ["2016-01-05"] * 2),
        "dominant_id": [f"{prefix}1501"] * 2 + [f"{prefix}1601"] * 4,
    }, index=index)


def _write_book(root):
    minute_dir = root / "minute_shards" / "research"
    for symbol in ("IF", "IH", "IC"):
        prices, oi = _minute(symbol)
        save_shard(prices, minute_dir, symbol)
        save_shard(oi, minute_dir / "open_interest", symbol)
    fees = pd.DataFrame({
        "symbol": ["IF", "IH", "IC"],
        "trading_date": pd.to_datetime(["2015-01-05"] * 3),
        "contract": ["IF1501", "IH1501", "IC1501"],
        "commission_type": ["by_volume"] * 3,
        "open_commission": [30.0] * 3,
        "close_commission": [30.0] * 3,
        "close_commission_today": [30.0] * 3,
    })
    path = root / "research" / "fee_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fees.to_csv(path, index=False)


def test_day_session_rejects_night_clock():
    with pytest.raises(ValueError, match="夜盘"):
        assert_day_session(pd.to_datetime(["2021-01-04 21:01"]), "IF")


def test_dominant_calendar_keeps_the_first_contract_and_later_rolls():
    index = pd.to_datetime(["2021-01-04 09:31", "2021-01-05 09:31"])
    frame = pd.DataFrame({
        "trading_date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
        "dominant_id": ["IF2101", "IF2102"],
    }, index=index)
    calendar = dominant_calendar(frame)
    assert calendar.new_contract.tolist() == ["IF2101", "IF2102"]
    assert calendar.prev_contract.tolist() == ["IF2101", "IF2101"]


def test_index_research_uses_day_session_shards_and_not_the_locked_partition(tmp_path, monkeypatch):
    _write_book(tmp_path)
    monkeypatch.setattr(C, "INDEX_DATA_ROOT", tmp_path)
    monkeypatch.setattr(C, "INDEX_RESEARCH_DIR", tmp_path / "minute_shards" / "research")
    monkeypatch.setattr(C, "MIN_VALID_DAYS", 1)
    monkeypatch.setattr(C, "MIN_DAILY_TURNOVER", 0.0)
    monkeypatch.setattr(C, "MIN_VALID_DAY_RATIO", 0.0)

    def refuse_holdout(*_args, **_kwargs):
        raise AssertionError("研究期不能读股指锁定分区")

    monkeypatch.setattr("backtest.index_market._read_holdout", refuse_holdout)
    data = load_research()
    assert set(data.universe[2016]) == {"IF", "IH", "IC"}
    assert data.returns.index.max().year == 2016
    assert data.roll_calendar["IF"]["new_contract"].tolist()[0] == "IF1501"
    assert data.factors.raw("ts_low").shape[1] == 3


def test_index_research_rejects_a_2022_shard(tmp_path, monkeypatch):
    _write_book(tmp_path)
    minute_dir = tmp_path / "minute_shards" / "research"
    prices, _oi = _minute("IF")
    prices = prices.copy()
    prices.iloc[-1, prices.columns.get_loc("trading_date")] = pd.Timestamp("2022-01-04")
    save_shard(prices, minute_dir, "IF")
    monkeypatch.setattr(C, "INDEX_DATA_ROOT", tmp_path)
    monkeypatch.setattr(C, "INDEX_RESEARCH_DIR", minute_dir)
    with pytest.raises(C.HoldoutViolation, match="样本外"):
        load_research()
