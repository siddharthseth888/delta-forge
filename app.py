"""
DeltaForge — Quantitative Options Valuation & Derivatives Analytics Suite (NSE/BSE)
Run with: python -m streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="DeltaForge | Derivatives Analytics & Risk Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from pages_impl.shared import sidebar_rate_input, ticker_input
from options_engine.market_config import get_config

cfg = get_config()

st.sidebar.title("⚡ DeltaForge")
st.sidebar.caption("Indian Equity & Index Derivatives Engine (NSE/BSE)")
st.sidebar.divider()

ticker = ticker_input(cfg)
r      = sidebar_rate_input(cfg)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Single Contract", "Strategy Builder", "Trade Evaluator", "Backtester"],
)

st.sidebar.divider()
st.sidebar.caption(
    "Data: yfinance · Pricing: Black-Scholes + CRR Binomial Tree\n\n"
    "⚠️ Educational use only. Not financial advice."
)

if page == "Single Contract":
    from pages_impl.single_contract import render
    render(ticker, r, cfg)
elif page == "Strategy Builder":
    from pages_impl.strategy_builder import render
    render(ticker, r, cfg)
elif page == "Trade Evaluator":
    from pages_impl.trade_advisor import render
    render(ticker, r, cfg)
elif page == "Backtester":
    from pages_impl.backtester import render
    render(ticker, r, cfg)
