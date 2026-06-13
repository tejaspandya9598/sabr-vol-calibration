"""Calibration recovers the data-generating parameters and fits tightly.

The live data layer pulls a real BTC smile from Deribit; this test builds a smile from
known SABR parameters (a unit-test input) so the calibrator can be checked against the
truth, offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sabr.calibration import calibrate
from sabr.model import hagan_lognormal_vol, smile

TRUE = {"alpha": 2.0, "beta": 0.5, "rho": -0.30, "nu": 0.40}
F, T = 100.0, 1.0
STRIKES = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)


def _market_smile(seed: int = 0) -> np.ndarray:
    vols = smile(F, STRIKES, T, TRUE["alpha"], TRUE["beta"], TRUE["rho"], TRUE["nu"])
    rng = np.random.default_rng(seed)
    return vols * (1 + rng.normal(0.0, 0.003, len(STRIKES)))   # ~30bp quote noise


def test_recovers_true_parameters():
    fit = calibrate(F, STRIKES, _market_smile(), T, beta=TRUE["beta"])
    assert abs(fit["alpha"] - TRUE["alpha"]) < 0.05
    assert abs(fit["rho"] - TRUE["rho"]) < 0.05
    assert abs(fit["nu"] - TRUE["nu"]) < 0.05


def test_fit_is_tight():
    fit = calibrate(F, STRIKES, _market_smile(), T, beta=TRUE["beta"])
    assert fit["rmse"] < 1e-3


def test_atm_vol_positive_and_finite():
    v = hagan_lognormal_vol(100, 100, 1.0, 2.0, 0.5, -0.3, 0.4)
    assert 0 < v < 1.0
