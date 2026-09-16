"""
DeltaForge Engine — Unified Quantitative Entry Point.

Usage:
    from options_engine import OptionsEngine
    from options_engine.data.yfinance_adapter import YFinanceAdapter

    engine = OptionsEngine(adapter=YFinanceAdapter(), risk_free_rate=0.065)
    chain  = engine.get_enriched_chain("RELIANCE.NS")
    greeks = engine.aggregate_position([...])
"""

from __future__ import annotations
import pandas as pd

from options_engine.data.base import MarketDataAdapter, SpotData
from options_engine.utils.volatility import historical_volatility, vol_surface
from options_engine.greeks.calculator import enrich_chain_with_greeks, position_greeks
from options_engine.pricing.black_scholes import BlackScholesModel, BSResult


class OptionsEngine:
    """
    Facade that wires together data, pricing, and Greeks.
    The Streamlit UI (or any other front-end) should only import this class.
    """

    def __init__(self, adapter: MarketDataAdapter, risk_free_rate: float = 0.065):
        self.adapter = adapter
        self.r = risk_free_rate

    # ── Data helpers ──────────────────────────────────────────────────────────

    def get_spot(self, ticker: str) -> SpotData:
        return self.adapter.get_spot(ticker)

    def get_available_expiries(self, ticker: str) -> list[str]:
        return self.adapter.get_available_expiries(ticker)

    def get_historical_vol(self, ticker: str, window: int = 21, period: str = "1y") -> float:
        """Return annualised historical (realised) vol over the given window."""
        prices = self.adapter.get_historical_prices(ticker, period=period)
        return historical_volatility(prices, window=window)

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def get_enriched_chain(
        self,
        ticker: str,
        expiry: str | None = None,
        sigma_override: float | None = None,
    ) -> pd.DataFrame:
        """
        Full pipeline: fetch chain → compute IV → enrich with Greeks.

        Returns a DataFrame ready for the screener / dashboard UI.
        Columns include: strike, option_type, expiry, bid, ask, last_price,
                         iv_calculated, bs_price, delta, gamma, theta, vega, rho,
                         moneyness, days_to_expiry, sigma_used.
        """
        spot = self.adapter.get_spot(ticker)
        chain = self.adapter.get_option_chain(ticker, expiry=expiry)
        chain = vol_surface(chain, S=spot.price, r=self.r)
        chain = enrich_chain_with_greeks(chain, r=self.r, sigma_override=sigma_override)
        return chain

    # ── Single contract pricing ───────────────────────────────────────────────

    def price_contract(
        self,
        ticker: str,
        strike: float,
        expiry: str,
        option_type: str,
        sigma: float | None = None,
    ) -> BSResult:
        """
        Price a single contract. If sigma is None, use historical vol (21-day).
        """
        import datetime
        spot = self.adapter.get_spot(ticker)
        if sigma is None:
            sigma = self.get_historical_vol(ticker)
        expiry_date = datetime.date.fromisoformat(expiry)
        T = max((expiry_date - datetime.date.today()).days / 365.0, 1e-9)
        return BlackScholesModel(
            S=spot.price, K=strike, T=T, r=self.r,
            sigma=sigma, option_type=option_type,
        ).evaluate()

    # ── Position aggregation ──────────────────────────────────────────────────

    def aggregate_position(self, legs: list[dict]) -> dict:
        """
        Aggregate Greeks across a multi-leg position.
        See greeks/calculator.py::position_greeks for leg format.
        """
        return position_greeks(legs, r=self.r)
