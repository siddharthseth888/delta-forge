"""
Market configuration — India (NSE/BSE) only.

Single source of truth for:
  - Currency, lot sizes, rate defaults
  - Ticker suffix conventions for yfinance
  - Index symbol mappings
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MarketConfig:
    """Complete market context passed through the entire app."""
    market:           str
    name:             str
    currency:         str
    currency_symbol:  str
    risk_free_rate:   float        # annualised (e.g. 0.065 for 6.5%)
    rate_label:       str
    default_lot_size: int
    lot_sizes:        dict[str, int]   = field(default_factory=dict)
    index_tickers:    dict[str, str]   = field(default_factory=dict)
    ticker_suffix:    str              = ""
    index_name:       str              = ""
    sample_tickers:   list[str]        = field(default_factory=list)

    def format_currency(self, value: float, decimals: int = 2) -> str:
        if abs(value) >= 1e7:
            return f"{self.currency_symbol}{value/1e7:,.2f} Cr"
        elif abs(value) >= 1e5:
            return f"{self.currency_symbol}{value/1e5:,.2f} L"
        return f"{self.currency_symbol}{value:,.{decimals}f}"

    def get_lot_size(self, ticker: str) -> int:
        clean = ticker.replace(self.ticker_suffix, "").replace("^", "").upper()
        # Reverse-map yfinance index symbols → F&O name
        reverse = {
            v.replace(self.ticker_suffix, "").replace("^", "").upper(): k
            for k, v in self.index_tickers.items()
        }
        clean = reverse.get(clean, clean)
        return self.lot_sizes.get(clean, self.default_lot_size)

    def add_suffix(self, ticker: str) -> str:
        """Return correct yfinance symbol for a user-typed ticker."""
        clean = ticker.replace(self.ticker_suffix, "").replace("^", "").upper()
        if clean in self.index_tickers:
            return self.index_tickers[clean]
        if self.ticker_suffix and not ticker.endswith(self.ticker_suffix):
            return ticker + self.ticker_suffix
        return ticker

    def strip_suffix(self, ticker: str) -> str:
        """Human-readable display name."""
        for name, symbol in self.index_tickers.items():
            if ticker == symbol:
                return name
        return ticker.replace(self.ticker_suffix, "").replace("^", "")

    def is_index(self, ticker: str) -> bool:
        return ticker in self.index_tickers.values() or ticker.startswith("^")


# ── NSE/BSE index → yfinance symbol ──────────────────────────────────────────
NSE_INDEX_TICKERS: dict[str, str] = {
    "NIFTY":       "^NSEI",
    "BANKNIFTY":   "^NSEBANK",
    "FINNIFTY":    "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY":  "^NSEMDCP50",
    "SENSEX":      "^BSESN",          # BSE Sensex 30
}

# ── NSE/BSE F&O lot sizes ─────────────────────────────────────────────────────
NSE_LOT_SIZES: dict[str, int] = {
    # Indices
    "NIFTY":       65,    # updated lot size
    "BANKNIFTY":   15,
    "FINNIFTY":    40,
    "MIDCPNIFTY":  75,
    "SENSEX":      10,    # BSE Sensex options lot size

    # Large-cap stocks
    "RELIANCE":    250,
    "TCS":         150,
    "INFY":        300,
    "HDFCBANK":    550,
    "ICICIBANK":   700,
    "SBIN":        1500,
    "WIPRO":       1500,
    "LT":          175,
    "AXISBANK":    625,
    "KOTAKBANK":   400,
    "BAJFINANCE":  125,
    "MARUTI":      100,
    "TATAMOTORS":  1425,
    "TATASTEEL":   5500,
    "HINDALCO":    2150,
    "SUNPHARMA":   700,
    "DRREDDY":     125,
    "CIPLA":       650,
    "ONGC":        1925,
    "NTPC":        2250,
    "POWERGRID":   2300,
    "ADANIENT":    500,
    "ADANIPORTS":  1250,
    "ULTRACEMCO":  100,
    "ASIANPAINT":  200,
    "NESTLEIND":   40,
    "TITAN":       375,
    "BAJAJFINSV":  125,
    "HCLTECH":     700,
    "TECHM":       600,
    "INDUSINDBK":  500,
    "GRASIM":      275,
    "BRITANNIA":   100,
    "PIDILITIND":  250,
    "HAVELLS":     500,
    "DABUR":       1250,
    "ITC":         3200,
    "HINDUNILVR":  300,
    "JSWSTEEL":    1350,
    "SAIL":        5400,
    "VEDL":        2000,
}


def get_config() -> MarketConfig:
    """Return the India NSE/BSE market config."""
    return MarketConfig(
        market="IN",
        name="India (NSE/BSE)",
        currency="INR",
        currency_symbol="₹",
        risk_free_rate=0.0525,    # RBI repo rate as of April 2025 — update at MPC meetings
        rate_label="RBI Repo Rate",
        default_lot_size=500,
        lot_sizes=NSE_LOT_SIZES,
        index_tickers=NSE_INDEX_TICKERS,
        ticker_suffix=".NS",
        index_name="NIFTY 50",
        sample_tickers=[
            "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX",
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "WIPRO", "LT", "AXISBANK", "BAJFINANCE",
            "MARUTI", "TATAMOTORS", "HCLTECH", "TECHM", "KOTAKBANK",
        ],
    )
