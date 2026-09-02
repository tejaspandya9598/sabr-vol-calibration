"""
SABR implied volatility — Hagan et al. (2002) lognormal approximation.

SABR models the forward with stochastic vol:  dF = alpha F^beta dW,  d(alpha) = nu alpha dZ,
with corr(dW, dZ) = rho. Rather than solve the SDE, Hagan's closed-form maps the four
parameters to a Black implied vol per strike — fast enough to calibrate in a loop and
the market standard for rates/FX smiles.
"""
from __future__ import annotations

import numpy as np


def hagan_lognormal_vol(F: float, K: float, T: float,
                        alpha: float, beta: float, rho: float, nu: float) -> float:
    F, K = float(F), float(K)
    if not (-1.0 < rho < 1.0):
        # x(z) divides by (1 - rho). The calibrator bounds rho to +/-0.999, but this
        # function is public and gets called directly in tests and notebooks.
        raise ValueError(f"rho must lie strictly inside (-1, 1), got {rho}")
    if alpha <= 0 or F <= 0 or K <= 0:
        raise ValueError("alpha, F and K must all be positive")

    # At-the-money limit (the general formula has a 0/0 there).
    if abs(F - K) < 1e-12:
        Fb = F ** (1 - beta)
        corr = (((1 - beta) ** 2 / 24) * alpha ** 2 / Fb ** 2
                + 0.25 * rho * beta * nu * alpha / Fb
                + (2 - 3 * rho ** 2) / 24 * nu ** 2)
        return alpha / Fb * (1 + corr * T)

    log_fk = np.log(F / K)
    fk_beta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk

    # z/x(z) -> 1 as z -> 0, but computing it directly is 0/0 and the expression loses
    # precision well before that: near the forward, sqrt(1 - 2*rho*z + z^2) + z - rho
    # is a difference of near-equal terms. Below the cutoff use the series
    # x(z) = z * (1 - rho*z/2 + ...), so z/x = 1 + rho*z/2 + (2 - 3*rho^2)*z^2/12.
    # Without this the smile has a numerical notch a few strikes wide around the ATM
    # point, and least_squares fits that notch instead of the market.
    if abs(z) < 1e-7:
        z_over_x = 1.0 + 0.5 * rho * z + (2.0 - 3.0 * rho ** 2) * z ** 2 / 12.0
    else:
        x = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
        z_over_x = z / x

    a = alpha / (fk_beta * (1 + ((1 - beta) ** 2 / 24) * log_fk ** 2
                           + ((1 - beta) ** 4 / 1920) * log_fk ** 4))
    b = (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / fk_beta ** 2
              + 0.25 * rho * beta * nu * alpha / fk_beta
              + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T)
    return a * z_over_x * b


def smile(F: float, strikes, T: float, alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
    return np.array([hagan_lognormal_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
