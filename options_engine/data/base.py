"""
Abstract base class for all market data adapters.
Swap yfinance for Zerodha (or anything else) by implementing this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class OptionContract:
    """Represents a single option contract."""
    ticker: str
    expiry: str                  # "YYYY-MM-DD"
    strike: float
    option_type: str             # "call" or "put"
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float    # raw from exchange (may be 0 — we recalculate)
    underlying_price: float


@dataclass
class SpotData:
    """Current spot price + basic market info."""
    ticker: str
    price: float
    previous_close: float
    day_high: float
    day_low: float


class MarketDataAdapter(ABC):
    """
    Abstract adapter. All data sources must implement this contract.
    The pricing engine talks only to this interface — never to yfinance directly.
    """

    @abstractmethod
    def get_spot(self, ticker: str) -> SpotData:
        """Fetch current spot price and basic market data."""
        ...

    @abstractmethod
    def get_option_chain(self, ticker: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """
        Return full option chain as a DataFrame with columns matching OptionContract fields.
        If expiry is None, return the nearest available expiry.
        """
        ...

    @abstractmethod
    def get_available_expiries(self, ticker: str) -> list[str]:
        """List all available option expiry dates for the ticker."""
        ...

    @abstractmethod
    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.Series:
        """
        Return daily closing prices for the given period.
        period: yfinance-style string e.g. "1y", "6mo", "3mo"
        """
        ...
