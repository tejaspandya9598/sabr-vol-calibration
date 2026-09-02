"""
Calibrate SABR to a market smile.

`beta` is usually fixed by convention (0.5 for rates, ~1 for FX/equity), leaving three
free knobs: alpha (overall level), rho (skew), nu (smile curvature / vol-of-vol). We
fit them by non-linear least squares to the quoted implied vols.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from sabr.model import smile


def calibrate(F: float, strikes, market_vols, T: float, beta: float = 0.5,
              x0: tuple[float, float, float] | None = None) -> dict:
    """Fit (alpha, rho, nu) to the smile. Returns the params + RMSE of the fit."""
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)

    if F <= 0:
        raise ValueError(f"forward must be positive, got {F}")

    # alpha carries units of vol * F^(1-beta), so both the seed and the box it has
    # to sit inside scale with the forward. Only the seed did. On a BTC smile with
    # F = 77,334 and beta = 0.5 the seed is ~139 against a hardcoded upper bound of
    # 5.0, and scipy refused the fit outright: "Initial guess is outside of
    # provided bounds". That bound was written for a rates forward near 0.04, where
    # F^(1-beta) is about 0.2 and the difference never shows.
    scale = F ** (1.0 - beta)

    if x0 is None:
        # Seed alpha from the ATM quote so the fit is scale-invariant; rho/nu neutral.
        atm = int(np.argmin(np.abs(strikes - F)))
        x0 = (max(market_vols[atm] * scale, 1e-3 * scale), -0.2, 0.3)

    def residuals(p):
        alpha, rho, nu = p
        return smile(F, strikes, T, alpha, beta, rho, nu) - market_vols

    # Bounds in vol units: alpha between 0.01% and 500% ATM-equivalent vol.
    lower = (1e-4 * scale, -0.999, 1e-4)
    upper = (5.0 * scale, 0.999, 5.0)
    x0 = tuple(float(np.clip(v, lo, hi)) for v, lo, hi in zip(x0, lower, upper))
    result = least_squares(residuals, x0, bounds=(lower, upper), xtol=1e-12, ftol=1e-12)

    alpha, rho, nu = result.x
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu, "rmse": rmse}
