"""
Shared utilities for all Streamlit pages — India (NSE/BSE) only.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import datetime

from options_engine import OptionsEngine
from options_engine.data.yfinance_adapter import YFinanceAdapter
from options_engine.market_config import MarketConfig, get_config


@st.cache_resource
def get_engine(risk_free_rate: float) -> OptionsEngine:
    return OptionsEngine(adapter=YFinanceAdapter(), risk_free_rate=risk_free_rate)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_spot(ticker: str, r: float):
    return get_engine(r).get_spot(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_expiries(ticker: str, r: float):
    return get_engine(r).get_available_expiries(ticker)



@st.cache_data(ttl=3600, show_spinner=False)
def fetch_hist_vol(ticker: str, r: float, window: int = 21) -> float:
    return get_engine(r).get_historical_vol(ticker, window=window)


def days_to_expiry(expiry_str: str) -> int:
    try:
        return (datetime.date.fromisoformat(expiry_str) - datetime.date.today()).days
    except Exception:
        return 0


def fmt(value: float, cfg: MarketConfig, decimals: int = 2) -> str:
    return cfg.format_currency(value, decimals)



def sidebar_rate_input(cfg: MarketConfig) -> float:
    rate_pct = float(round(cfg.risk_free_rate * 100, 2))
    st.sidebar.markdown(
        f"**{cfg.rate_label}**  \n"
        f"Current default: **{rate_pct:.2f}%**  \n"
        f"<small>Update when RBI changes repo rate (MPC meetings ~every 2 months)</small>",
        unsafe_allow_html=True,
    )
    return st.sidebar.number_input(
        "Override rate (%)",
        min_value=0.0,
        max_value=30.0,
        value=rate_pct,
        step=0.05,
        format="%.2f",
        label_visibility="collapsed",
    ) / 100


def ticker_input(cfg: MarketConfig) -> str:
    """Dropdown ticker selector with indices first, then stocks."""
    index_options = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]
    stock_options = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "WIPRO", "LT", "AXISBANK", "BAJFINANCE",
        "MARUTI", "TATAMOTORS", "TATASTEEL", "HCLTECH", "TECHM",
        "KOTAKBANK", "ADANIENT", "SUNPHARMA", "DRREDDY", "ITC",
    ]

    quick_options = (
        ["— type custom ticker below —"]
        + [f"{t} (pricing only — no chain)" for t in index_options]
        + ["── Stocks (full chain via yfinance) ──"]
        + stock_options
    )

    selected = st.sidebar.selectbox(
        "Ticker",
        quick_options,
        index=1,
        help="Indices: single contract pricing works. Stocks: full option chain from yfinance.",
    )

    if " (pricing only" in selected:
        selected = selected.split(" (")[0]

    if selected.startswith("—") or selected.startswith("──"):
        raw = st.sidebar.text_input(
            "Custom ticker", value="", placeholder="e.g. BAJAJFINSV"
        ).upper().strip()
    else:
        raw = selected
        st.sidebar.caption(f"Selected: **{raw}**")

    if not raw:
        raw = "NIFTY"

    raw = raw.replace(".NS", "").replace(".BO", "").replace("^", "")
    return cfg.add_suffix(raw)
