"""
yfinance implementation of MarketDataAdapter.
Replace this with a ZerodhaAdapter later — the engine won't need to change.
"""

from typing import Optional
import pandas as pd
import yfinance as yf

from .base import MarketDataAdapter, SpotData


class YFinanceAdapter(MarketDataAdapter):

    def get_spot(self, ticker: str) -> SpotData:
        t = yf.Ticker(ticker)
        info = t.fast_info
        hist = t.history(period="2d")

        price = float(info.last_price)
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price

        return SpotData(
            ticker=ticker,
            price=price,
            previous_close=prev_close,
            day_high=float(info.day_high),
            day_low=float(info.day_low),
        )

    def get_available_expiries(self, ticker: str) -> list[str]:
        return list(yf.Ticker(ticker).options)

    def get_option_chain(self, ticker: str, expiry: Optional[str] = None) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            raise ValueError(f"No options available for {ticker}")

        target = expiry if expiry and expiry in expiries else expiries[0]
        spot = self.get_spot(ticker).price
        chain = t.option_chain(target)

        def _normalise(df: pd.DataFrame, option_type: str) -> pd.DataFrame:
            df = df.copy()
            df["option_type"] = option_type
            df["expiry"] = target
            df["ticker"] = ticker
            df["underlying_price"] = spot
            df = df.rename(columns={
                "strike": "strike",
                "lastPrice": "last_price",
                "bid": "bid",
                "ask": "ask",
                "volume": "volume",
                "openInterest": "open_interest",
                "impliedVolatility": "implied_volatility",
            })
            keep = [
                "ticker", "expiry", "strike", "option_type",
                "last_price", "bid", "ask", "volume",
                "open_interest", "implied_volatility", "underlying_price",
            ]
            return df[[c for c in keep if c in df.columns]]

        calls = _normalise(chain.calls, "call")
        puts = _normalise(chain.puts, "put")
        result = pd.concat([calls, puts], ignore_index=True)
        result["volume"] = result["volume"].fillna(0).astype(int)
        result["open_interest"] = result["open_interest"].fillna(0).astype(int)
        return result

    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.Series:
        hist = yf.Ticker(ticker).history(period=period)
        return hist["Close"].dropna()
