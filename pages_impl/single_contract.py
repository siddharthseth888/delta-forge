"""
Page 1 — Single Contract Pricer
Market-aware: INR/USD, lot sizes, STT/SEBI costs, RBI/Fed rate defaults.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pages_impl.shared import fetch_spot, fetch_expiries, fetch_hist_vol, days_to_expiry, fmt
from options_engine.utils.volatility import time_to_expiry_years
from options_engine.pricing.black_scholes import BlackScholesModel
from options_engine.pricing.binomial_tree import BinomialTreeModel
from options_engine.costs import CostCalculator
from options_engine.market_config import MarketConfig


def render(ticker: str, r: float, cfg: MarketConfig):
    sym          = cfg.currency_symbol
    market_badge = "🇮🇳 NSE" if cfg.market == "IN" else "🇺🇸 US"

    st.title(f"Single Contract Pricer  {market_badge}")
    st.caption(f"{cfg.name} · {cfg.rate_label}: {r*100:.2f}%")

    # ── Market data ───────────────────────────────────────────────────────────
    with st.spinner("Fetching market data…"):
        try:
            spot_data = fetch_spot(ticker, r)
            expiries  = fetch_expiries(ticker, r)
            hist_vol  = fetch_hist_vol(ticker, r)
            data_ok   = True
        except Exception as e:
            st.warning(f"Could not fetch live data for **{cfg.strip_suffix(ticker)}**: {e}")
            data_ok, spot_data, expiries, hist_vol = False, None, [], 0.25

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if data_ok:
            pct = (spot_data.price - spot_data.previous_close) / spot_data.previous_close * 100
            st.metric("Spot", fmt(spot_data.price, cfg), f"{pct:+.2f}%")
        else:
            st.metric("Spot", "—")
    with c2:
        st.metric("21d hist vol", f"{hist_vol*100:.1f}%")

    st.divider()

    # ── Inputs ────────────────────────────────────────────────────────────────
    st.subheader("Contract parameters")
    default_spot = float(spot_data.price) if data_ok else (20000.0 if cfg.market == "IN" else 100.0)

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        S_input = st.number_input(f"Spot (S) [{sym}]", value=default_spot, min_value=0.01, step=1.0)
    with i2:
        K_input = st.number_input(f"Strike (K) [{sym}]", value=float(round(default_spot)), min_value=0.01, step=1.0)
    with i3:
        option_type = st.selectbox("Type", ["call", "put"])
    with i4:
        exercise_style = st.selectbox("Exercise", ["european", "american"])

    i5, i6, i7, i8 = st.columns(4)
    with i5:
        if expiries:
            expiry_str = st.selectbox("Expiry", expiries)
            T_input    = time_to_expiry_years(expiry_str)
        else:
            dte     = st.number_input("Days to expiry", value=30, min_value=1, max_value=730)
            # For manual DTE, use precise time: assume expiry at 15:30 IST on that date
            import datetime as _dt
            manual_expiry = (_dt.date.today() + _dt.timedelta(days=int(dte))).isoformat()
            T_input = time_to_expiry_years(manual_expiry)
    with i6:
        sigma_input = st.slider("Volatility σ (%)", 5, 150, max(5, int(hist_vol * 100)), step=1) / 100
    with i7:
        lot_size = st.number_input(
            "Lot size", value=int(cfg.get_lot_size(ticker)), min_value=1, step=1,
            help="NSE F&O lot size. Prefilled from known lots table." if cfg.market == "IN" else "US options: 100 shares/contract.",
        )
    with i8:
        q_input = st.number_input("Div yield q (%)", value=0.0, min_value=0.0, max_value=20.0, step=0.1) / 100

    # ── Pricing ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Model outputs")

    bs_res = BlackScholesModel(S=S_input, K=K_input, T=T_input, r=r, sigma=sigma_input, option_type=option_type).evaluate()
    bt     = BinomialTreeModel(S=S_input, K=K_input, T=T_input, r=r, sigma=sigma_input,
                               option_type=option_type, exercise_style=exercise_style, q=q_input, steps=150)
    bt_price   = bt.price()
    bt_delta   = bt.delta()
    bt_gamma   = bt.gamma()
    bt_theta   = bt.theta()
    ee_premium = bt.early_exercise_premium()

    mc = st.columns(2)
    with mc[0]:
        st.markdown("**Black-Scholes** *(European, analytical)*")
        a, b, c = st.columns(3)
        a.metric("Price",       fmt(bs_res.price, cfg))
        b.metric("Δ Delta",     f"{bs_res.delta:.4f}")
        c.metric("Γ Gamma",     f"{bs_res.gamma:.4f}")
        d, e, f = st.columns(3)
        d.metric("Θ Theta/day", f"{bs_res.theta:.4f}")
        e.metric("ν Vega/1%",   f"{bs_res.vega:.4f}")
        f.metric("ρ Rho/1%",    f"{bs_res.rho:.4f}")

    with mc[1]:
        st.markdown(f"**Binomial Tree** *({exercise_style.capitalize()}, CRR 150-step)*")
        a, b, c = st.columns(3)
        diff = bt_price - bs_res.price
        a.metric("Price",       fmt(bt_price, cfg), f"{'+' if diff>=0 else ''}{fmt(diff,cfg,2)} vs BS")
        b.metric("Δ Delta",     f"{bt_delta:.4f}")
        c.metric("Γ Gamma",     f"{bt_gamma:.4f}")
        d, e, f = st.columns(3)
        d.metric("Θ Theta/day", f"{bt_theta:.4f}")
        e.metric("EE Premium",  fmt(ee_premium, cfg, 4))
        ee_pct = ee_premium / bt_price * 100 if bt_price > 0 else 0
        f.metric("EE %",        f"{ee_pct:.2f}%")

    # ── Transaction costs ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Transaction costs")

    calc = CostCalculator(cfg)
    rt   = calc.round_trip(premium=bs_res.price, lot_size=lot_size, quantity=1)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Entry cost / lot",     fmt(rt.entry.total, cfg))
    t2.metric("Exit cost / lot",      fmt(rt.exit.total, cfg))
    t3.metric("Round-trip total",     fmt(rt.total, cfg))
    t4.metric("Cost as % of premium", f"{rt.cost_as_pct_of_premium:.2f}%",
              help="If this is high, the contract is cheap but transaction costs dominate your edge.")

    with st.expander("📋 Full cost breakdown"):
        col_e, col_x = st.columns(2)
        with col_e:
            st.markdown("**Entry (buy)**")
            for k, v in rt.entry.breakdown().items():
                if v > 0:
                    st.markdown(f"- {k}: {fmt(v, cfg, 4)}")
        with col_x:
            st.markdown("**Exit (sell)**")
            for k, v in rt.exit.breakdown().items():
                if v > 0:
                    st.markdown(f"- {k}: {fmt(v, cfg, 4)}")
        if cfg.market == "IN":
            st.caption("STT: 0.0625% on sell premium · Stamp duty: 0.003% on buy · NSE txn: 0.053% · SEBI: ₹10/cr · GST: 18% on brokerage+exchange · Brokerage: ₹20 flat")
        else:
            st.caption("Commission: $0.65/contract · SEC fee: $0.000008 × notional (sell side)")

    # ── Greeks explainer ──────────────────────────────────────────────────────
    with st.expander("📖 What do these Greeks mean for your trade?"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
**Δ Delta = {bs_res.delta:.3f}**
For every {sym}1 the spot moves, this option changes by ~{sym}{abs(bs_res.delta):.2f}.
{"Profits when spot rises." if option_type == "call" else "Profits when spot falls."}

**Γ Gamma = {bs_res.gamma:.4f}**
Rate of change of delta. Highest near ATM and near expiry — delta accelerates fast on big moves.

**Θ Theta = {bs_res.theta:.4f}/day**
You lose ~{sym}{abs(bs_res.theta):.2f} per calendar day. {"Last 2 weeks before expiry are the most punishing for option buyers." if T_input < 0.15 else ""}
""")
        with col_b:
            st.markdown(f"""
**ν Vega = {bs_res.vega:.4f} per 1% vol**
IV up 1% → position gains ~{sym}{bs_res.vega:.2f}. Long options benefit from vol spikes — earnings season, RBI policy, macro events.

**EE Premium = {fmt(ee_premium, cfg, 4)}**
{"Non-trivial — prefer the binomial price for this contract." if ee_premium > 0.001 else "Near zero — early exercise irrelevant here, BS price is reliable."}

**Gamma-Theta trade-off**
Long gamma pays theta. Short gamma collects theta. This tension defines every options trade.
""")

    st.divider()

    # ── Sensitivity charts ────────────────────────────────────────────────────
    st.subheader("Sensitivity analysis")
    tab1, tab2, tab3 = st.tabs(["Price vs Spot", "Greeks vs Spot", "Price vs Volatility"])
    spot_range = np.linspace(S_input * 0.6, S_input * 1.4, 80)

    with tab1:
        prices    = [BlackScholesModel(float(s), K_input, T_input, r, sigma_input, option_type).price() for s in spot_range]
        intrinsic = [max(0.0, float(s)-K_input) if option_type=="call" else max(0.0, K_input-float(s)) for s in spot_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=prices, name="BS Price", line=dict(color="#3498db", width=2.5)))
        fig.add_trace(go.Scatter(x=spot_range, y=intrinsic, name="Intrinsic", line=dict(color="#95a5a6", width=1.5, dash="dash")))
        fig.add_vline(x=S_input, line_dash="dot", line_color="#e74c3c", annotation_text="Spot")
        fig.add_vline(x=K_input, line_dash="dot", line_color="#2ecc71", annotation_text="Strike")
        fig.update_layout(xaxis_title=f"Spot ({sym})", yaxis_title=f"Price ({sym})", height=380, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        deltas = [BlackScholesModel(float(s), K_input, T_input, r, sigma_input, option_type).delta() for s in spot_range]
        gammas = [BlackScholesModel(float(s), K_input, T_input, r, sigma_input, option_type).gamma() for s in spot_range]
        thetas = [BlackScholesModel(float(s), K_input, T_input, r, sigma_input, option_type).theta() for s in spot_range]

        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=spot_range, y=deltas, name="Delta", line=dict(color="#3498db", width=2)))
        fig_d.add_vline(x=S_input, line_dash="dot", line_color="#888")
        fig_d.add_hline(y=0, line_color="#ccc", line_width=1)
        fig_d.update_layout(title="Δ Delta vs Spot", xaxis_title=f"Spot ({sym})", yaxis_title="Delta",
                            height=260, template="plotly_white", showlegend=False,
                            margin=dict(t=40, b=40))
        st.plotly_chart(fig_d, use_container_width=True)

        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=spot_range, y=gammas, name="Gamma", line=dict(color="#e67e22", width=2)))
        fig_g.add_vline(x=S_input, line_dash="dot", line_color="#888")
        fig_g.update_layout(title="Γ Gamma vs Spot", xaxis_title=f"Spot ({sym})", yaxis_title="Gamma",
                            height=260, template="plotly_white", showlegend=False,
                            margin=dict(t=40, b=40))
        st.plotly_chart(fig_g, use_container_width=True)

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=spot_range, y=thetas, name="Theta/day", line=dict(color="#e74c3c", width=2)))
        fig_t.add_vline(x=S_input, line_dash="dot", line_color="#888")
        fig_t.add_hline(y=0, line_color="#ccc", line_width=1)
        fig_t.update_layout(title="Θ Theta/day vs Spot", xaxis_title=f"Spot ({sym})", yaxis_title="Theta (₹/day)",
                            height=260, template="plotly_white", showlegend=False,
                            margin=dict(t=40, b=40))
        st.plotly_chart(fig_t, use_container_width=True)

    with tab3:
        vol_range  = np.linspace(0.05, 1.0, 80)
        vol_prices = [BlackScholesModel(S_input, K_input, T_input, r, float(v), option_type).price() for v in vol_range]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=vol_range*100, y=vol_prices, name="Price", line=dict(color="#9b59b6", width=2.5)))
        fig3.add_vline(x=sigma_input*100, line_dash="dot", line_color="#e74c3c", annotation_text=f"σ={sigma_input*100:.0f}%")
        fig3.update_layout(xaxis_title="Volatility (%)", yaxis_title=f"Price ({sym})", height=380, template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)
