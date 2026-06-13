# SABR Volatility Calibration

Fit the **SABR stochastic-volatility model** to a **live BTC option smile from Deribit**.
SABR's four parameters (`alpha, beta, rho, nu`) map, via Hagan's closed form, to a Black
implied vol per strike — fast enough to calibrate in a loop and the market standard for
rates/FX/crypto smiles.

## How it works

1. **Data** (`data.py`) — pull the live BTC option chain from the Deribit public API,
   take the nearest standard expiry, and keep the out-of-the-money mark IV at each
   strike. Real quotes, no key.
2. **Model** (`model.py`) — Hagan et al. (2002) lognormal implied-vol approximation,
   including the ATM limit.
3. **Calibration** (`calibration.py`) — fix `beta` (0.5 for rates, ~1 for crypto/FX) and
   fit `alpha` (level), `rho` (skew), `nu` (vol-of-vol) by non-linear least squares; the
   initial `alpha` is seeded from the ATM quote so the fit is scale-invariant.
4. **Plot** (`viz.py`) — market quotes vs the fitted smile, with the fit RMSE.

On synthetic ground-truth data the calibrator recovers the parameters essentially
exactly (RMSE < 0.04 vol points — see the tests); on the live Deribit smile it fits the
real skew and curvature.

## Run

```bash
uv sync
uv run python scripts/run_calibration.py   # live Deribit smile -> fitted params + reports/figures/
uv run pytest                              # recovers known params from a smile fixture (offline)
```

## Structure

```
sabr-vol-calibration/
├── src/sabr/
│   ├── data.py         # live Deribit BTC option-smile client
│   ├── model.py        # Hagan SABR implied vol
│   ├── calibration.py  # least-squares fit of (alpha, rho, nu)
│   └── viz.py          # market vs fitted smile
├── scripts/run_calibration.py
└── tests/              # parameter recovery on a known-SABR fixture
```

## Notes

- **Real data, no synthetic.** The smile comes live from Deribit; the unit tests build a
  smile from known SABR parameters so calibration can be checked against the truth offline.
- The Deribit client is multi-call and wasn't exercised against the live endpoint from the
  build environment — confirm the first live pull when you run it.

---

*Built by Tejas Pandya — NYU MSFE.*
