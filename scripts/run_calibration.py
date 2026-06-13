"""
Calibrate SABR to the live BTC option smile from Deribit and plot the fit.

    python scripts/run_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sabr.calibration import calibrate   # noqa: E402
from sabr.data import fetch_btc_smile     # noqa: E402
from sabr.viz import plot_smile           # noqa: E402


def main() -> None:
    print("Fetching live BTC option smile from Deribit...")
    F, strikes, market_vols, T, beta, expiry = fetch_btc_smile()
    print(f"  expiry {expiry}  forward {F:,.0f}  {len(strikes)} strikes  T={T:.3f}y\n")

    fit = calibrate(F, strikes, market_vols, T, beta=beta)
    print(f"SABR fit (beta fixed at {beta}):")
    for k in ("alpha", "rho", "nu"):
        print(f"  {k:<6} {fit[k]:>9.4f}")
    print(f"  RMSE   {fit['rmse'] * 100:>8.4f} vol points")

    out = ROOT / "reports" / "figures" / "sabr_smile.png"
    plot_smile(F, strikes, market_vols, fit, T, out)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
