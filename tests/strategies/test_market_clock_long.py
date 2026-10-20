"""只做多时钟的按年顺序还在引擎里，用来挡住乱序打开。策略代码已删除。"""
import pytest

from tfcta import config as C


def test_year_tests_open_in_order_and_only_once(tmp_path):
    from backtest.engine import check_year_order, year_test_name

    ledger = tmp_path / "ledger.jsonl"
    check_year_order(2022, ledger)
    with pytest.raises(C.HoldoutViolation, match="2022"):
        check_year_order(2023, ledger)
    C.append_ledger({"kind": "year_test", "status": "opened", "strategy": year_test_name(2022)}, ledger)
    with pytest.raises(C.HoldoutViolation, match="打开过"):
        check_year_order(2022, ledger)
    with pytest.raises(C.HoldoutViolation):
        check_year_order(2023, ledger)
    C.append_ledger({"kind": "year_test", "status": "finished", "strategy": year_test_name(2022)}, ledger)
    check_year_order(2023, ledger)
    with pytest.raises(C.HoldoutViolation, match="2023"):
        check_year_order(2024, ledger)
    with pytest.raises(C.HoldoutViolation):
        check_year_order(2026, ledger)
