import argparse
import datetime as dt
import json

import pytest

from tfcta import config as C
from tfcta.data import sectors
from tfcta.research.backtest import strategy
from tfcta.research.workflow import ledger


def _args(argv, with_criteria=False):
    parser = argparse.ArgumentParser()
    strategy.add_book_args(parser, with_criteria=with_criteria)
    return parser.parse_args(argv)


def test_every_commodity_belongs_to_exactly_one_of_five_sectors():
    assert set(sectors.SECTOR_BY_SYMBOL) == set(C.COMMODITY_SYMBOLS)
    assert set(sectors.SECTOR_BY_SYMBOL.values()) == set(sectors.SECTORS)
    assert len(sectors.SECTORS) == 5
    assert not set(sectors.SECTOR_BY_SYMBOL) & set(C.FINANCIAL_SYMBOLS)


def test_sector_aliases_and_unknown_names():
    assert sectors.parse_pools(["黑色", "能化", "黑色金属"]) == ["黑色金属", "能源化工"]
    with pytest.raises(ValueError):
        sectors.canonicalize("建材")


def test_bare_factor_uses_prior_and_explicit_sign_multiplies_raw():
    assert strategy.parse_factor_specs(["ts_high", "ts_low"]) == {"ts_high": -1.0, "ts_low": 1.0}
    assert strategy.parse_factor_specs(["ts_high:+1"]) == {"ts_high": 1.0}
    assert strategy.parse_factor_specs({"carry_main_sub_yield": -1}) == {"carry_main_sub_yield": -1.0}


def test_undirected_factor_requires_explicit_sign():
    with pytest.raises(ValueError, match="必须写成"):
        strategy.parse_factor_specs(["warehouse_low"])
    with pytest.raises(ValueError):
        strategy.parse_factor_specs(["ts_high:2"])


def test_fingerprint_ignores_spelling_and_pass_criteria():
    a = strategy.config_from_args(_args(["--factors", "ts_high", "neg_clv"], True))
    b = strategy.config_from_args(_args(["--factors", "neg_clv:+1", "ts_high:-1",
                                      "--min-sharpe", "2"], True))
    assert strategy.fingerprint(a, "农产品") == strategy.fingerprint(b, "农产品")
    assert strategy.fingerprint(a, "农产品") != strategy.fingerprint(a, "黑色金属")


def test_config_file_is_overridden_by_cli(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "factors": {"ts_low": 1}, "pools": ["农产品"], "fee_rate": 0.001,
        "pass_criteria": {"min_net_sharpe": 1.0},
    }), encoding="utf-8")
    cfg = strategy.config_from_args(_args(["--config", str(path), "--pools", "贵金属"], True))
    assert cfg.factors == {"ts_low": 1.0}
    assert cfg.pools == ["贵金属"]
    assert cfg.fee_rate == 0.001
    assert cfg.min_net_sharpe == 1.0


def test_cases_split_by_pool_and_optionally_merge():
    cfg = strategy.config_from_args(_args(["--pools", "黑色金属", "贵金属", "--merge-pools"]))
    cases = strategy.resolve_cases(cfg, ["RB", "I", "AU", "CU"])
    assert cases == {"黑色金属": ["I", "RB"], "贵金属": ["AU"],
                     "黑色金属+贵金属": ["AU", "I", "RB"]}
    assert strategy.resolve_cases(strategy.config_from_args(_args([])), ["RB", "CU"]) == {
        "全部": ["CU", "RB"]}


def test_fingerprint_is_invariant_to_merged_case_order_and_tracks_execution():
    cfg = strategy.config_from_args(_args(["--factors", "tsmom"]))
    assert strategy.fingerprint(cfg, "黑色金属+贵金属") == strategy.fingerprint(cfg, "贵金属+黑色金属")
    weekly = strategy.config_from_args(_args(["--factors", "tsmom", "--tranches", "0"]))
    flat = strategy.config_from_args(_args(["--factors", "tsmom", "--vol-target", "0"]))
    fps = {strategy.fingerprint(c, "全部") for c in (cfg, weekly, flat)}
    assert len(fps) == 3
    assert len({strategy.book_key(c, "全部") for c in (cfg, weekly, flat)}) == 1
    assert flat.vol_target == 0.0 and weekly.tranches == 0


def test_validation_log_lists_factors_costs_and_deduplicates_fingerprints(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "validation_entries", lambda: [
        {"fingerprint": "book-a", "case": "农产品", "factors": "ts_high:-1,ts_low:+1",
         "config": {"fee_rate": 0.00025, "slippage_ticks": 1.0}, "n_symbols": 14,
         "net_ann_return": -0.04, "net_sharpe": -0.5, "net_max_drawdown": -0.08,
         "passed": False, "run_at": "2026-09-27T12:00:00"},
        {"fingerprint": "book-a", "case": "农产品", "factors": "ts_high:-1,ts_low:+1",
         "config": {"fee_rate": 0.00025, "slippage_ticks": 1.0}, "n_symbols": 14,
         "net_ann_return": -0.04, "net_sharpe": -0.5, "net_max_drawdown": -0.08,
         "passed": False, "run_at": "2026-09-27T12:00:00"},
    ])

    monkeypatch.setattr(ledger, "OOS_ROOT", tmp_path / "oos")
    ledger.append(ledger.oos_path(), [{"fingerprint": "book-b", "factors": "carry_ms:+1",
                                       "net_sharpe": 0.2}])
    notes = tmp_path / "ResearchNotes.md"
    notes.write_text("# 笔记\n\n手写内容\n", encoding="utf-8")
    ledger.write_validation_log(notes)
    path = ledger.write_validation_log(notes)          # 第二次只替换标记之间的内容
    content = path.read_text(encoding="utf-8")

    assert content.startswith("# 笔记\n\n手写内容\n") and content.count(ledger.LOG_BEGIN) == 1
    assert "`book-b`" in content
    assert content.count("`book-a`") == 1
    assert "`ts_high:-1,ts_low:+1`" in content
    assert "0.00025" in content and "1.0" in content
    assert "-4.00%" in content and "-0.500" in content and "-8.00%" in content


def test_void_entry_releases_fingerprint_and_is_listed_separately(tmp_path, monkeypatch):
    first = {"fingerprint": "book-v", "factors": "carry", "net_sharpe": 1.8, "passed": True,
             "run_at": "2026-09-30T15:00:00"}
    rerun = {**first, "net_sharpe": 0.3, "passed": False, "run_at": "2026-09-30T17:00:00"}
    entries = [first, ledger.void_entry("book-v", "外部分区缺 2022", "runs/x")]
    assert ledger.lookup(entries, "book-v") == []
    entries.append(rerun)
    assert ledger.lookup(entries, "book-v") == [rerun]

    monkeypatch.setattr(ledger, "validation_entries", lambda: entries)
    monkeypatch.setattr(ledger, "OOS_ROOT", tmp_path / "oos")
    notes = tmp_path / "ResearchNotes.md"
    notes.write_text("# 笔记\n", encoding="utf-8")
    content = ledger.write_validation_log(notes).read_text(encoding="utf-8")
    assert "作废记录" in content and "外部分区缺 2022" in content
    assert "0.300" in content and "1.800" not in content


def test_missing_run_artifacts_are_reported_without_dead_links(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger.C, "PROJECT_ROOT", tmp_path)
    entry = {"run_dir": str(tmp_path / "deleted-run"), "note": "2022 检验"}

    rendered = ledger._result_link(entry)

    assert "2022 检验" in rendered
    assert "运行明细已删除，仅保留台账摘要" in rendered
    assert "[运行结果]" not in rendered


def test_oos_gate_refuses_books_without_passing_validation(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "step7", Path(__file__).resolve().parents[2] / "scripts" / "step7_oos_test.py")
    step7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step7)

    monkeypatch.setattr(ledger, "VALIDATION_ROOT", tmp_path / "v")
    monkeypatch.setattr(ledger, "OOS_ROOT", tmp_path / "o")
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path / "none")
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path / "runs")
    cfg = strategy.config_from_args(_args(["--pools", "农产品", "贵金属"]))
    end = dt.date(2024, 12, 31)

    allowed, refused = step7.gate(cfg, end)
    assert allowed == [] and len(refused) == 2

    ledger.append(ledger.validation_path(), [
        {"fingerprint": strategy.fingerprint(cfg, "农产品"), "passed": True,
         "net_sharpe": 0.8, "net_ann_return": 0.05},
        {"fingerprint": strategy.fingerprint(cfg, "贵金属"), "passed": False,
         "net_sharpe": 0.1, "net_ann_return": 0.01},
    ])
    allowed, _ = step7.gate(cfg, end)
    assert allowed == ["农产品"]

    # 做过一次样本外之后，换个截止日期也不能再测
    ledger.append(ledger.oos_path(), [
        {"fingerprint": strategy.fingerprint(cfg, "农产品"), "oos_end": end.isoformat()}])
    assert step7.gate(cfg, end)[0] == []
    assert step7.gate(cfg, dt.date(2025, 12, 31))[0] == []


def test_oos_gate_recomputes_pass_with_default_criteria(tmp_path, monkeypatch):
    """step6 用 --min-sharpe 0 放宽门槛记成 passed=True 的书，step7 仍按默认门槛拒绝。"""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "step7", Path(__file__).resolve().parents[2] / "scripts" / "step7_oos_test.py")
    step7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step7)
    monkeypatch.setattr(ledger, "VALIDATION_ROOT", tmp_path / "v")
    monkeypatch.setattr(ledger, "OOS_ROOT", tmp_path / "o")
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path / "none")
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path / "runs")
    cfg = strategy.config_from_args(_args(["--pools", "农产品"]))
    ledger.append(ledger.validation_path(), [
        {"fingerprint": strategy.fingerprint(cfg, "农产品"), "passed": True,
         "net_sharpe": 0.2, "net_ann_return": 0.01,
         "criteria": {"min_net_sharpe": 0.0, "min_net_ann_return": 0.0}}])
    assert step7.gate(cfg, dt.date(2024, 12, 31))[0] == []


def test_test_window_must_have_ended():
    with pytest.raises(C.HoldoutViolation):
        C.assert_test_window_closed(dt.date(2025, 12, 31), today=dt.date(2025, 12, 31))
    C.assert_test_window_closed(dt.date(2025, 12, 31), today=dt.date(2026, 1, 1))
