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
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

DERIBIT = "https://www.deribit.com/api/v2/public"

# Deribit expiry codes are "27JUN25". Parsing them with strptime("%d%b%y") breaks twice:
# %b is locale-dependent, so the pull fails outright under a non-English locale, and
# time.mktime interprets the result as LOCAL midnight. Deribit options expire at 08:00
# UTC, so the old code was off by the local UTC offset plus eight hours — up to ~22
# hours of error in T, which on a ten-day expiry is a ~9% error in the variance the
# calibration is fitting. Fixed month table, explicit UTC, explicit expiry hour.
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_EXPIRY_HOUR_UTC = 8


def _get(method: str, **params):
    resp = requests.get(f"{DERIBIT}/{method}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["result"]


def _expiry_datetime(expiry: str) -> datetime:
    """"27JUN25" -> 2025-06-27 08:00 UTC."""
    code = expiry.upper()
    day, mon, yr = int(code[:-5]), code[-5:-2], int(code[-2:])
    if mon not in _MONTHS:
        raise ValueError(f"unrecognised Deribit expiry code: {expiry!r}")
    return datetime(2000 + yr, _MONTHS[mon], day, _EXPIRY_HOUR_UTC, tzinfo=timezone.utc)


def _days_to(expiry: str) -> float:
    delta = _expiry_datetime(expiry) - datetime.now(timezone.utc)
    return delta.total_seconds() / 86_400.0


def fetch_btc_smile(min_days: float = 10.0, beta: float = 0.5):
    """Returns (forward, strikes, market_vols, T_years, beta, expiry_label)."""
    summary = _get("get_book_summary_by_currency", currency="BTC", kind="option")

    by_expiry: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
    forwards: dict[str, list[float]] = defaultdict(list)
    for row in summary:
        parts = row["instrument_name"].split("-")        # BTC-<expiry>-<strike>-<C|P>
        if len(parts) != 4 or not row.get("mark_iv"):
            continue
        by_expiry[parts[1]].append((float(parts[2]), parts[3], float(row["mark_iv"]) / 100.0))
        # SABR is a forward model, so the smile has to be centred on the FORWARD for
        # this expiry, not on spot. The old code used the BTC index (spot) for every
        # expiry, which mis-centres the moneyness axis by the whole cost of carry and
        # pushes that error straight into rho — the skew parameter is exactly the one
        # that is sensitive to getting the centre wrong. Deribit publishes
        # underlying_price per instrument; that IS the expiry forward.
        if row.get("underlying_price"):
            forwards[parts[1]].append(float(row["underlying_price"]))

    candidates = sorted((e for e in by_expiry if _days_to(e) >= min_days), key=_days_to)
    if not candidates:
        raise RuntimeError(
            f"no BTC option expiry at least {min_days} days out; "
            f"saw {sorted(by_expiry)}")
    expiry = candidates[0]
    T = _days_to(expiry) / 365.0

    if forwards.get(expiry):
        forward = float(np.median(forwards[expiry]))
    else:  # pragma: no cover - only if Deribit stops publishing underlying_price
        forward = float(_get("get_index_price", index_name="btc_usd")["index_price"])

    # one quote per strike: the out-of-the-money option (call above forward, put below)
    smile: dict[float, float] = {}
    for strike, cp, iv in by_expiry[expiry]:
        if (strike >= forward and cp == "C") or (strike < forward and cp == "P"):
            smile[strike] = iv

    strikes = np.array(sorted(smile))
    vols = np.array([smile[k] for k in strikes])
    return forward, strikes, vols, T, beta, expiry
