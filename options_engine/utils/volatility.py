"""
Volatility utilities.

  historical_volatility  — annualised realised vol from a price series
  implied_volatility     — IV solver using Brent's method on Black-Scholes
  vol_surface            — enrich a full option chain DataFrame with IV per row
"""

from __future__ import annotations
import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

try:
    import pytz
    _HAS_PYTZ = True
except ImportError:
    _HAS_PYTZ = False


def time_to_expiry_years(
    expiry_date_str: str,
    expiry_time: str = "15:30",
    timezone: str = "Asia/Kolkata",
) -> float:
    """
    Compute T (time to expiry in years) using precise datetime arithmetic
    down to the minute rather than naive days/365 integer arithmetic.

    This matters most for short-dated options (< 10 DTE) where a few hours
    of difference causes a meaningful pricing error.

    Parameters
    ----------
    expiry_date_str : "YYYY-MM-DD" — the expiry date
    expiry_time     : "HH:MM" — time of day the option expires (IST for NSE,
                      e.g. "15:30"; use "16:00" for US options)
    timezone        : IANA timezone string (default "Asia/Kolkata" for NSE)

    Returns
    -------
    T in years (always >= 1e-9)
    """
    if _HAS_PYTZ:
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        expiry_date = datetime.date.fromisoformat(expiry_date_str)
        h, m = map(int, expiry_time.split(":"))
        expiry_dt = tz.localize(
            datetime.datetime(expiry_date.year, expiry_date.month, expiry_date.day, h, m, 0)
        )
        seconds = (expiry_dt - now).total_seconds()
    else:
        # Fallback: naive date arithmetic if pytz not installed
        expiry_date = datetime.date.fromisoformat(expiry_date_str)
        seconds = (expiry_date - datetime.date.today()).days * 86400.0

    return max(seconds / (365.25 * 24 * 3600), 1e-9)


def historical_volatility(
    prices: pd.Series,
    window: int = 21,
    annualise: bool = True,
) -> float:
    """
    Annualised historical (realised) volatility from daily closing prices.

    Uses log returns over the trailing `window` trading days.
    Returns 0.0 if fewer than 2 prices are available.

    Parameters
    ----------
    prices    : pd.Series of daily closing prices (must be positive)
    window    : look-back in trading days (default 21 ≈ 1 month)
    annualise : multiply by sqrt(252) if True
    """
    if len(prices) < 2:
        return 0.0
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) == 0:
        return 0.0
    tail = log_returns.tail(window)
    daily_vol = float(tail.std())
    return daily_vol * np.sqrt(252) if annualise else daily_vol


def _bs_price_fast(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """
    Lightweight BS price for the IV solver — avoids importing BlackScholesModel
    to keep the solver dependency-free and fast.
    """
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Optional[float]:
    """
    Solve for implied volatility using Brent's method.

    Returns None when no solution exists:
      - T <= 0 (expired)
      - market_price <= intrinsic value (no extrinsic value to solve for)
      - market_price is 0 or negative
      - numerical solver fails to bracket a root

    Parameters
    ----------
    market_price : observed market price (use mid = (bid+ask)/2 ideally)
    S            : spot price
    K            : strike
    T            : time to expiry in years
    r            : risk-free rate
    option_type  : "call" or "put"
    tol          : solver tolerance (default 1e-6)
    max_iter     : max Brent iterations (default 200)
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
    if market_price <= intrinsic:
        return None

    def objective(sigma: float) -> float:
        return _bs_price_fast(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 1e-6, 20.0, xtol=tol, maxiter=max_iter)
        return float(iv)
    except ValueError:
        # Brent's method failed to bracket — price may be outside BS range
        return None


def vol_surface(
    chain_df: pd.DataFrame,
    S: float,
    r: float,
    risk_free_rate: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute implied vol for every row in an option chain DataFrame.

    Adds columns:
      iv_calculated  — our own IV (independent of exchange-reported IV)
      moneyness      — S / K  (> 1 = ITM call / OTM put)
      days_to_expiry — integer calendar days until expiry

    Parameters
    ----------
    chain_df        : DataFrame from adapter.get_option_chain()
    S               : current spot price
    r               : risk-free rate (use risk_free_rate kwarg as alias)
    risk_free_rate  : alias for r (takes precedence if provided)
    """
    if risk_free_rate is not None:
        r = risk_free_rate

    df    = chain_df.copy()
    today = datetime.date.today()   # still used for days_to_expiry integer column

    def _row_iv(row: pd.Series) -> Optional[float]:
        try:
            T = time_to_expiry_years(str(row["expiry"]))
            # Use mid-price; fall back to last_price if bid/ask unavailable
            bid, ask = float(row.get("bid", 0)), float(row.get("ask", 0))
            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float(row.get("last_price", 0))
            if mid <= 0:
                return None
            return implied_volatility(mid, S, float(row["strike"]), T, r, str(row["option_type"]))
        except Exception:
            return None

    def _days(expiry_str: str) -> Optional[int]:
        try:
            return (datetime.date.fromisoformat(str(expiry_str)) - today).days
        except Exception:
            return None

    df["iv_calculated"]  = df.apply(_row_iv, axis=1)
    df["moneyness"]      = S / df["strike"]
    df["days_to_expiry"] = df["expiry"].apply(_days)
    # Also store precise T (years) for downstream pricing
    df["T_precise"] = df["expiry"].apply(
        lambda e: time_to_expiry_years(str(e))
    )
    return df
