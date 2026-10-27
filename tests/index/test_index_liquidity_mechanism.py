"""股指机制检验：逐笔价格冲击分解自洽，合成数据。"""
import numpy as np
import pandas as pd
import pytest

from research.index_liquidity_mechanism import tick_day_features


def test_flow_and_residual_add_up_and_lambda_recovers_the_planted_impact():
    rng = np.random.default_rng(0)
    stamps = pd.date_range("2018-03-15 09:30:00", periods=7200, freq="1s")
    signs = rng.choice([-1.0, 1.0], size=len(stamps))
    mid = 4000 * np.exp(np.cumsum(2e-6 * signs))             # 每手买单推高中间价 2e-6（对数）
    ticks = pd.DataFrame({"datetime": stamps, "last": mid + 0.1 * signs, "a1": mid + 0.1, "b1": mid - 0.1,
                          "a1_v": 5.0, "b1_v": 5.0, "volume": np.arange(1, len(stamps) + 1, dtype=float)})
    out = tick_day_features(ticks)
    assert out["flow"] + out["resid"] == pytest.approx(out["intraday_mid"])
    assert out["lambda"] == pytest.approx(2e-6, rel=0.25)
    assert out["mid_first30"] == pytest.approx(ticks.loc[ticks["datetime"] <= stamps[0] + pd.Timedelta(minutes=30), "a1"].iloc[-1] - 0.1)
