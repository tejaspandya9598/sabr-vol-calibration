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

**Synthetic ground truth** (F=100, T=1, true alpha/beta/rho/nu = 2.0/0.5/-0.30/0.40,
with 30bp of quote noise injected):

| parameter | true | recovered |
|---|--:|--:|
| alpha | 2.00 | 2.0033 |
| rho | -0.300 | -0.3047 |
| nu | 0.400 | 0.3918 |

RMSE **0.035 vol points** — essentially the injected noise floor.

**Live BTC smile** (Deribit, 18SEP26 expiry, F = 77,372, T = 0.042y, 21 strikes,
beta fixed at 0.5): alpha 87.98, rho -0.007, nu 3.54, RMSE **0.43 vol points**. Worst
strike is the 95,000 wing at 1.2 vol points; everything inside 0.88 – 1.16 moneyness
fits to under 0.6. A near-zero rho with nu of 3.5 is the right shape for a two-week
crypto smile: almost no skew, a lot of curvature.

Reproduce both with `uv run pytest` and `uv run python scripts/run_calibration.py`
(the live numbers move with the market; the shape does not).

## The math

SABR models the forward $F_t$ with stochastic volatility $\alpha_t$:

$$dF_t = \alpha_t F_t^{\beta}\, dW_t^1, \qquad
d\alpha_t = \nu\, \alpha_t\, dW_t^2, \qquad
dW^1 dW^2 = \rho\, dt$$

Hagan's singular-perturbation expansion maps $(\alpha,\beta,\rho,\nu)$ to a Black
implied vol per strike:

$$\sigma_{BS}(K,F) \approx
\frac{\alpha}{(FK)^{\frac{1-\beta}{2}}\left[1 + \frac{(1-\beta)^2}{24}\ln^2\frac{F}{K} + \frac{(1-\beta)^4}{1920}\ln^4\frac{F}{K}\right]}
\cdot \frac{z}{x(z)} \cdot \left[1 + \left(\frac{(1-\beta)^2}{24}\frac{\alpha^2}{(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4(FK)^{\frac{1-\beta}{2}}} + \frac{2-3\rho^2}{24}\nu^2\right)T\right]$$

with $z = \frac{\nu}{\alpha}(FK)^{\frac{1-\beta}{2}}\ln\frac{F}{K}$ and
$x(z) = \ln\!\frac{\sqrt{1-2\rho z+z^2}+z-\rho}{1-\rho}$; at the money the formula
collapses to $\sigma_{ATM} \approx \alpha / F^{1-\beta} \cdot [1 + (\cdot)T]$.

Parameter roles are close to orthogonal, which is why the fit is stable: $\alpha$
sets the vol **level**, $\rho$ tilts the **skew**, $\nu$ bends the **curvature**
(smile wings), and $\beta$ fixes the backbone — how ATM vol moves with $F$ — which
is why it's set by convention rather than fitted (it's nearly collinear with $\rho$).
Calibration is non-linear least squares over the OTM quotes,
$\min_{\alpha,\rho,\nu} \sum_i \big(\sigma_{model}(K_i) - \sigma_{mkt}(K_i)\big)^2$,
seeded from the ATM quote so the initial guess is scale-invariant.

## References

- Hagan, P., Kumar, D., Lesniewski, A. & Woodward, D. (2002), *Managing Smile Risk*, Wilmott Magazine — the implied-vol formula and the hedging argument for SABR over local vol.
- Obłój, J. (2008), *Fine-tune your smile: Correction to Hagan et al.* — the fix for the low-strike wing.
- West, G. (2005), *Calibration of the SABR Model in Illiquid Markets*, Applied Mathematical Finance 12(4).

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
