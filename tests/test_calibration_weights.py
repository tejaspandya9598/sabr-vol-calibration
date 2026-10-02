from __future__ import annotations

import numpy as np
import pytest

from sabr.calibration import calibrate
from sabr.model import hagan_lognormal_vol, smile


def _case():
    fwd = 100.0
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
    vols = smile(fwd, strikes, 1.0, 2.0, 0.5, -0.3, 0.4)
    return fwd, strikes, vols


def test_none_weights_equal_explicit_ones():
    fwd, strikes, vols = _case()
    unweighted = calibrate(fwd, strikes, vols, 1.0)
    explicit = calibrate(fwd, strikes, vols, 1.0, weights=np.ones_like(vols))
    for key in ("alpha", "rho", "nu", "rmse"):
        assert unweighted[key] == pytest.approx(explicit[key], abs=1e-12)


@pytest.mark.parametrize("weights", [
    [1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, np.nan],
    [1, 1, 1, 1, 1, 1, 1, 1, np.inf],
    [1, 1, 1, 1, 1, 1, 1, 1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
])
def test_invalid_weights_rejected(weights):
    fwd, strikes, vols = _case()
    with pytest.raises(ValueError):
        calibrate(fwd, strikes, vols, 1.0, weights=weights)


def test_large_weight_pulls_fit_toward_that_strike():
    fwd, strikes, vols = _case()
    noisy = vols.copy()
    noisy[0] += 0.025
    plain = calibrate(fwd, strikes, noisy, 1.0)
    weights = np.ones_like(noisy)
    weights[0] = 1000.0
    focused = calibrate(fwd, strikes, noisy, 1.0, weights=weights)
    plain_err = abs(hagan_lognormal_vol(fwd, strikes[0], 1.0, plain["alpha"], 0.5, plain["rho"], plain["nu"]) - noisy[0])
    focused_err = abs(hagan_lognormal_vol(fwd, strikes[0], 1.0, focused["alpha"], 0.5, focused["rho"], focused["nu"]) - noisy[0])
    assert focused_err < plain_err / 10


def test_rmse_is_unweighted_with_zero_weighted_outlier():
    fwd, strikes, vols = _case()
    noisy = vols.copy()
    noisy[0] += 0.025
    weights = np.ones_like(noisy)
    weights[0] = 0.0
    fit = calibrate(fwd, strikes, noisy, 1.0, weights=weights)
    errors = smile(fwd, strikes, 1.0, fit["alpha"], 0.5, fit["rho"], fit["nu"]) - noisy
    assert fit["rmse"] == pytest.approx(np.sqrt(np.mean(errors**2)), abs=1e-14)
    assert fit["rmse"] > 0.005
