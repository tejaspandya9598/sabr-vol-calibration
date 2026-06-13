"""Plot the calibrated SABR smile against the market quotes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sabr.model import smile


def plot_smile(F, strikes, market_vols, fit, T, out_path: str | Path) -> None:
    strikes = np.asarray(strikes, dtype=float)
    grid = np.linspace(strikes.min(), strikes.max(), 200)
    fitted = smile(F, grid, T, fit["alpha"], fit["beta"], fit["rho"], fit["nu"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(strikes, np.asarray(market_vols) * 100, color="#c0392b", zorder=3, label="market quotes")
    ax.plot(grid, fitted * 100, color="#3b6ea5", lw=1.9,
            label=f"SABR fit  (RMSE {fit['rmse'] * 100:.3f} vol pts)")
    ax.axvline(F, color="gray", ls=":", lw=1, label="forward")
    ax.set_title("SABR volatility smile fit", fontweight="bold")
    ax.set_xlabel("strike")
    ax.set_ylabel("implied volatility (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
