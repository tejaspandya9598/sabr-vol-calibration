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
    x = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))

    a = alpha / (fk_beta * (1 + ((1 - beta) ** 2 / 24) * log_fk ** 2
                           + ((1 - beta) ** 4 / 1920) * log_fk ** 4))
    b = (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / fk_beta ** 2
              + 0.25 * rho * beta * nu * alpha / fk_beta
              + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T)
    return a * (z / x) * b


def smile(F: float, strikes, T: float, alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
    return np.array([hagan_lognormal_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
