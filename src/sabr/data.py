"""
Real BTC option smile from the Deribit public API (no key, no synthetic data).

Deribit publishes a mark implied vol per listed option. We take the nearest expiry
that's at least `min_days` out, keep the out-of-the-money quote at each strike (the
liquid side), and return a clean (forward, strikes, vols) smile to calibrate.

NOTE: this is a multi-call live client; it needs a real network round-trip to Deribit
and hasn't been run against the live endpoint from the build sandbox — verify the first
pull when you run it.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import requests

DERIBIT = "https://www.deribit.com/api/v2/public"


def _get(method: str, **params):
    resp = requests.get(f"{DERIBIT}/{method}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["result"]


def _days_to(expiry: str) -> float:
    # Deribit expiry codes look like "27JUN25" (08:00 UTC); day precision is enough.
    return (time.mktime(time.strptime(expiry, "%d%b%y")) - time.time()) / 86_400.0


def fetch_btc_smile(min_days: float = 10.0, beta: float = 0.5):
    """Returns (forward, strikes, market_vols, T_years, beta, expiry_label)."""
    summary = _get("get_book_summary_by_currency", currency="BTC", kind="option")
    forward = float(_get("get_index_price", index_name="btc_usd")["index_price"])

    by_expiry: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
    for s in summary:
        parts = s["instrument_name"].split("-")          # BTC-<expiry>-<strike>-<C|P>
        if len(parts) != 4 or not s.get("mark_iv"):
            continue
        by_expiry[parts[1]].append((float(parts[2]), parts[3], float(s["mark_iv"]) / 100.0))

    expiry = next(e for e in sorted(by_expiry, key=_days_to) if _days_to(e) >= min_days)
    T = _days_to(expiry) / 365.0

    # one quote per strike: the out-of-the-money option (call above forward, put below)
    smile: dict[float, float] = {}
    for strike, cp, iv in by_expiry[expiry]:
        if (strike >= forward and cp == "C") or (strike < forward and cp == "P"):
            smile[strike] = iv

    strikes = np.array(sorted(smile))
    vols = np.array([smile[k] for k in strikes])
    return forward, strikes, vols, T, beta, expiry
