import numpy as np
import pandas as pd

from research.incremental import _index_confirmation_fit, clustered_ols, hac_ols

def test_clustered_ols_recovers_interaction_effect():
    dates = np.repeat(pd.date_range("2020-01-01", periods=40, freq="B"), 10)
    clock = np.tile(np.linspace(-0.45, 0.45, 10), 40)
    dev = np.tile(np.array([-0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4]), 40)
    interaction = clock * dev
    y = 0.2 * clock - 0.3 * dev - 0.7 * interaction

    result = clustered_ols(
        y,
        {"clock": clock, "dev": dev, "clock_x_dev": interaction},
        dates,
    )

    assert np.isclose(result["coef:clock"], 0.2)
    assert np.isclose(result["coef:dev"], -0.3)
    assert np.isclose(result["coef:clock_x_dev"], -0.7)
    assert result["clusters"] == 40


def test_hac_ols_recovers_standardized_gap_effect():
    rng = np.random.default_rng(21)
    gap = rng.normal(size=300)
    open_move = rng.normal(size=300)
    noise = rng.normal(scale=0.1, size=300)
    y = 0.4 * gap + 0.2 * open_move + noise

    result = hac_ols(y, {"open_move": open_move, "gap": gap}, max_lags=5)

    assert np.isclose(result["coef:gap"], 0.4, atol=0.03)
    assert np.isclose(result["coef:open_move"], 0.2, atol=0.03)
    assert result["t:gap"] > 10


def test_index_confirmation_model_detects_positive_interaction():
    rng = np.random.default_rng(31)
    n = 1200
    gap = rng.normal(size=n)
    open_move = rng.normal(size=n)
    confirmed = (np.sign(gap) == np.sign(open_move)).astype(float)
    centered_confirmation = confirmed - confirmed.mean()
    y = 0.1 * gap + 0.1 * open_move + 0.7 * gap * centered_confirmation
    y += rng.normal(scale=0.12, size=n)
    frame = pd.DataFrame({
        "trading_date": pd.date_range("2018-01-01", periods=n, freq="B"),
        "gross_1130": y,
        "gap_return": gap,
        "open_move": open_move,
    })

    result = _index_confirmation_fit(frame, "IF", "synthetic")

    assert result["gap_x_confirmation_coef"] > 0.3
    assert result["gap_x_confirmation_t_hac"] > 5
    assert result["delta_r2_confirmation"] > 0
