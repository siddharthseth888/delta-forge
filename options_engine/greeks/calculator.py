"""
Greeks calculator for option chains and multi-leg positions.

  enrich_chain_with_greeks — add BS Greeks to every row of a chain DataFrame
  position_greeks          — aggregate Greeks across a multi-leg position
"""

from __future__ import annotations
import datetime
import pandas as pd
import numpy as np

from options_engine.pricing.black_scholes import BlackScholesModel
from options_engine.utils.volatility import time_to_expiry_years


def enrich_chain_with_greeks(
    chain_df: pd.DataFrame,
    r: float,
    sigma_override: float | None = None,
    use_iv_column: bool = True,
    fallback_sigma: float = 0.25,
) -> pd.DataFrame:
    """
    Add Black-Scholes Greeks to every row of an option chain DataFrame.

    Adds columns: bs_price, delta, gamma, theta, vega, rho, sigma_used.

    Sigma priority per row:
      1. sigma_override  — if provided, used for every row
      2. iv_calculated   — per-row implied vol (if use_iv_column=True and column exists)
      3. fallback_sigma  — neutral fallback (default 0.25 = 25%)

    Rows where Greeks cannot be computed get None values (not NaN/crash).

    Parameters
    ----------
    chain_df       : DataFrame from adapter.get_option_chain(), optionally
                     enriched with vol_surface()
    r              : risk-free rate (annualised)
    sigma_override : force a single vol for all rows
    use_iv_column  : use per-row iv_calculated when available
    fallback_sigma : vol to use when IV is unavailable (default 0.25)
    """
    df    = chain_df.copy()

    GREEK_COLS = ["bs_price", "delta", "gamma", "theta", "vega", "rho", "sigma_used"]

    def _row_greeks(row: pd.Series) -> pd.Series:
        null = pd.Series({c: None for c in GREEK_COLS})
        try:
            # Use precise datetime-based T if available (added by vol_surface),
            # otherwise compute it fresh with the same precise method.
            if "T_precise" in row.index and pd.notna(row["T_precise"]) and row["T_precise"] > 0:
                T = float(row["T_precise"])
            else:
                T = time_to_expiry_years(str(row["expiry"]))

            # Determine sigma
            if sigma_override is not None:
                sigma = float(sigma_override)
            elif (use_iv_column
                  and "iv_calculated" in row.index
                  and row["iv_calculated"] is not None
                  and pd.notna(row["iv_calculated"])
                  and float(row["iv_calculated"]) > 0):
                sigma = float(row["iv_calculated"])
            else:
                sigma = fallback_sigma

            bs = BlackScholesModel(
                S=float(row["underlying_price"]),
                K=float(row["strike"]),
                T=T,
                r=r,
                sigma=sigma,
                option_type=str(row["option_type"]),
            ).evaluate()

            return pd.Series({
                "bs_price":   bs.price,
                "delta":      bs.delta,
                "gamma":      bs.gamma,
                "theta":      bs.theta,
                "vega":       bs.vega,
                "rho":        bs.rho,
                "sigma_used": sigma,
            })
        except Exception:
            return null

    greeks = df.apply(_row_greeks, axis=1)
    return pd.concat([df, greeks], axis=1)


def position_greeks(legs: list[dict], r: float) -> dict:
    """
    Aggregate Greeks across a multi-leg position.

    Each leg is a dict with keys:
        S           : spot price
        K           : strike
        T           : time to expiry (years)
        sigma       : volatility
        option_type : "call" or "put"
        quantity    : number of lots (positive = long, negative = short)
        lot_size    : units per lot (e.g. 50 for NIFTY, 100 for US)

    Returns a dict with keys:
        bs_value, delta, gamma, theta, vega, rho

    Returns all zeros for an empty legs list.
    """
    total: dict[str, float] = {
        "bs_value": 0.0,
        "delta":    0.0,
        "gamma":    0.0,
        "theta":    0.0,
        "vega":     0.0,
        "rho":      0.0,
    }

    for leg in legs:
        qty = float(leg.get("quantity", 1)) * float(leg.get("lot_size", 1))
        try:
            bs = BlackScholesModel(
                S=float(leg["S"]),
                K=float(leg["K"]),
                T=float(leg["T"]),
                r=r,
                sigma=float(leg["sigma"]),
                option_type=str(leg["option_type"]),
            ).evaluate()
            total["bs_value"] += bs.price * qty
            total["delta"]    += bs.delta * qty
            total["gamma"]    += bs.gamma * qty
            total["theta"]    += bs.theta * qty
            total["vega"]     += bs.vega  * qty
            total["rho"]      += bs.rho   * qty
        except Exception:
            # Skip malformed legs rather than crashing the whole position
            continue

    return total
