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


def test_no_numerical_notch_near_the_forward():
    """z/x(z) is 0/0 at the forward and loses precision either side of it.

    Sampling straight through the ATM point used to produce a notch a few strikes
    wide, which the calibrator would then happily fit. The smile should be smooth:
    no interior point may deviate from the average of its neighbours by more than a
    basis point of vol.
    """
    ks = np.linspace(F - 0.5, F + 0.5, 41)
    vols = smile(F, ks, T, TRUE["alpha"], TRUE["beta"], TRUE["rho"], TRUE["nu"])
    assert np.all(np.isfinite(vols))
    kinks = np.abs(vols[1:-1] - 0.5 * (vols[:-2] + vols[2:]))
    assert kinks.max() < 1e-4, f"smile is not smooth through ATM: max kink {kinks.max():.2e}"


def test_atm_branch_agrees_with_the_general_formula():
    """The exact-ATM shortcut and the general expression must meet in the middle."""
    exact = hagan_lognormal_vol(F, F, T, TRUE["alpha"], TRUE["beta"], TRUE["rho"], TRUE["nu"])
    nudged = hagan_lognormal_vol(F, F * (1 + 1e-9), T,
                                 TRUE["alpha"], TRUE["beta"], TRUE["rho"], TRUE["nu"])
    assert abs(exact - nudged) < 1e-8


def test_rho_outside_the_unit_interval_is_rejected():
    import pytest
    for bad in (-1.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            hagan_lognormal_vol(F, 105.0, T, 2.0, 0.5, bad, 0.4)


def test_expiry_codes_parse_in_utc_and_ignore_locale():
    """Deribit expiries are 08:00 UTC and the month is always English uppercase."""
    from datetime import datetime, timezone
    from sabr.data import _expiry_datetime
    assert _expiry_datetime("27JUN25") == datetime(2025, 6, 27, 8, tzinfo=timezone.utc)
    assert _expiry_datetime("1JAN26") == datetime(2026, 1, 1, 8, tzinfo=timezone.utc)
    assert _expiry_datetime("27jun25") == _expiry_datetime("27JUN25")
    import pytest
    with pytest.raises(ValueError):
        _expiry_datetime("27XXX25")


def test_alpha_bounds_scale_with_the_forward():
    """alpha carries units of vol * F^(1-beta), so its box has to scale with F.

    Only the seed did. On a BTC smile (F ~ 77,000, beta 0.5) the ATM seed is ~139
    against a hardcoded upper bound of 5.0, and scipy refused to start:
    "Initial guess is outside of provided bounds". The bound was written for a
    rates forward near 0.04, where F^(1-beta) is about 0.2 and it never bites.
    """
    F_btc = 77_000.0
    strikes = F_btc * np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15])
    true_alpha = 0.35 * F_btc ** 0.5          # 35% ATM vol in alpha units
    vols = smile(F_btc, strikes, 0.25, true_alpha, 0.5, -0.1, 1.2)

    fit = calibrate(F_btc, strikes, vols, 0.25, beta=0.5)
    assert fit["rmse"] < 1e-3
    assert abs(fit["alpha"] - true_alpha) / true_alpha < 0.05


def test_rates_scale_smile_still_calibrates():
    """The other end of the scale: a 4% forward, where alpha is O(0.2)."""
    F_rate = 0.04
    strikes = F_rate + np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    true_alpha = 0.30 * F_rate ** 0.5
    vols = smile(F_rate, strikes, 2.0, true_alpha, 0.5, -0.2, 0.5)
    fit = calibrate(F_rate, strikes, vols, 2.0, beta=0.5)
    assert fit["rmse"] < 1e-3
    assert abs(fit["alpha"] - true_alpha) / true_alpha < 0.05


def test_non_positive_forward_is_rejected():
    try:
        calibrate(0.0, STRIKES, _market_smile(), T, beta=0.5)
    except ValueError as e:
        assert "forward must be positive" in str(e)
    else:
        raise AssertionError("expected a ValueError on a zero forward")
