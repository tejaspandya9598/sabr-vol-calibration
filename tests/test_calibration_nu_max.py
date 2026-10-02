from __future__ import annotations

import numpy as np
import pytest

from sabr.calibration import calibrate
from sabr.model import smile


def test_default_nu_max_matches_explicit_five():
    fwd, t = 100.0, 0.5
    strikes = fwd * np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])
    vols = smile(fwd, strikes, t, 2.0, 0.5, -0.3, 0.8)
    default = calibrate(fwd, strikes, vols, t, beta=0.5)
    explicit = calibrate(fwd, strikes, vols, t, beta=0.5, nu_max=5.0)
    for key in ("alpha", "rho", "nu", "rmse"):
        assert default[key] == pytest.approx(explicit[key], abs=1e-12)


@pytest.mark.parametrize("nu_max", [0.0, 1e-4, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_nu_max_is_rejected(nu_max):
    fwd, t = 100.0, 0.5
    strikes = fwd * np.array([0.8, 0.9, 1.0, 1.1])
    vols = smile(fwd, strikes, t, 2.0, 0.5, -0.3, 0.8)
    with pytest.raises(ValueError, match="nu_max"):
        calibrate(fwd, strikes, vols, t, nu_max=nu_max)


def test_short_dated_steep_smile_can_fit_nu_above_five():
    fwd, t = 100.0, 0.01
    strikes = fwd * np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    true = {"alpha": 2.0, "beta": 0.5, "rho": -0.2, "nu": 8.0}
    vols = smile(fwd, strikes, t, true["alpha"], true["beta"], true["rho"], true["nu"])
    bounded = calibrate(fwd, strikes, vols, t, beta=0.5)
    expanded = calibrate(fwd, strikes, vols, t, beta=0.5, nu_max=5.0 / np.sqrt(t))
    assert bounded["nu"] == pytest.approx(5.0, abs=1e-8)
    assert expanded["nu"] > 5.0
    assert expanded["nu"] == pytest.approx(true["nu"], abs=1e-6)
    assert expanded["rmse"] < 1e-8
