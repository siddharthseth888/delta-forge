"""
Page 3 — Strategy Builder  (market-aware, with transaction cost overlay)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pages_impl.shared import fetch_spot, fetch_expiries, fetch_hist_vol, days_to_expiry, fmt
from options_engine.pricing.black_scholes import BlackScholesModel
from options_engine.greeks.calculator import position_greeks
from options_engine.costs import CostCalculator
from options_engine.market_config import MarketConfig


TEMPLATES = {
    "Custom":            [],
    "Long Call":         [{"option_type":"call","quantity": 1,"strike_offset": 0}],
    "Long Put":          [{"option_type":"put", "quantity": 1,"strike_offset": 0}],
    "Covered Call":      [{"option_type":"call","quantity":-1,"strike_offset": 0.05}],
    "Bull Call Spread":  [{"option_type":"call","quantity": 1,"strike_offset": 0},
                          {"option_type":"call","quantity":-1,"strike_offset": 0.05}],
    "Bear Put Spread":   [{"option_type":"put", "quantity": 1,"strike_offset": 0},
                          {"option_type":"put", "quantity":-1,"strike_offset":-0.05}],
    "Long Straddle":     [{"option_type":"call","quantity": 1,"strike_offset": 0},
                          {"option_type":"put", "quantity": 1,"strike_offset": 0}],
    "Long Strangle":     [{"option_type":"call","quantity": 1,"strike_offset": 0.03},
                          {"option_type":"put", "quantity": 1,"strike_offset":-0.03}],
    "Iron Condor":       [{"option_type":"put", "quantity": 1,"strike_offset":-0.10},
                          {"option_type":"put", "quantity":-1,"strike_offset":-0.05},
                          {"option_type":"call","quantity":-1,"strike_offset": 0.05},
                          {"option_type":"call","quantity": 1,"strike_offset": 0.10}],
}


def payoff_at_expiry(legs: list[dict], S_range: np.ndarray) -> np.ndarray:
    total = np.zeros_like(S_range)
    for leg in legs:
        K, ot = leg["K"], leg["option_type"]
        qty   = leg["quantity"] * leg["lot_size"]
        prem  = leg["premium"]
        intrinsic = np.where(ot=="call", np.maximum(S_range-K, 0), np.maximum(K-S_range, 0))
        total += qty * (intrinsic - prem)
    return total


def render(ticker: str, r: float, cfg: MarketConfig):
    sym          = cfg.currency_symbol
    market_badge = "🇮🇳 NSE" if cfg.market == "IN" else "🇺🇸 US"

    st.title(f"Strategy Builder  {market_badge}")
    st.caption(f"{cfg.name} · P&L at expiry · Aggregate Greeks · Transaction costs")

    # ── Market data ───────────────────────────────────────────────────────────
    with st.spinner("Fetching market data…"):
        try:
            spot_data = fetch_spot(ticker, r)
            expiries  = fetch_expiries(ticker, r)
            hist_vol  = fetch_hist_vol(ticker, r)
            spot      = spot_data.price
            data_ok   = True
        except Exception as e:
            st.warning(f"Could not fetch live data: {e}")
            data_ok, spot, expiries, hist_vol = False, (20000.0 if cfg.market=="IN" else 100.0), [], 0.25

    # ── Global controls ───────────────────────────────────────────────────────
    g1, g2, g3 = st.columns([1, 1, 2])
    with g1:
        template_name = st.selectbox("Load template", list(TEMPLATES.keys()))
    with g2:
        if expiries:
            selected_expiry = st.selectbox("Expiry", expiries)
            T = max(days_to_expiry(selected_expiry) / 365.0, 1e-6)
            dte_days = days_to_expiry(selected_expiry)
        else:
            dte_days = st.number_input("Days to expiry", value=30, min_value=1)
            T = dte_days / 365.0
    with g3:
        sigma = st.slider("Volatility σ (%)", 5, 150, max(5, int(hist_vol*100)), step=1) / 100

    st.caption(f"Spot: {fmt(spot, cfg)} · DTE: {dte_days} days · σ: {sigma*100:.0f}% · r: {r*100:.1f}% · Lot: {cfg.get_lot_size(ticker)}")

    st.divider()

    # ── Leg builder ───────────────────────────────────────────────────────────
    st.subheader("Position legs")

    # Use percentage offsets so templates work across price scales (INR vs USD)
    def template_strike(offset_pct: float) -> float:
        raw = spot * (1 + offset_pct)
        # Round to a "nice" number — nearest 50 for NSE, nearest 5 for US
        step = 50 if cfg.market == "IN" else 5
        return float(round(raw / step) * step)

    template    = TEMPLATES[template_name]
    atm_strike  = template_strike(0)

    if "legs" not in st.session_state or st.session_state.get("last_template") != template_name:
        st.session_state.legs = [
            {"option_type": t["option_type"],
             "K":           template_strike(t.get("strike_offset", 0)),
             "quantity":    t["quantity"],
             "lot_size":    cfg.get_lot_size(ticker)}
            for t in template
        ] if template else [{"option_type":"call","K":atm_strike,"quantity":1,"lot_size":cfg.get_lot_size(ticker)}]
        st.session_state.last_template = template_name

    if st.button("＋ Add leg"):
        st.session_state.legs.append(
            {"option_type":"call","K":atm_strike,"quantity":1,"lot_size":cfg.get_lot_size(ticker)}
        )

    legs_updated = []
    colors = ["#3498db","#e67e22","#9b59b6","#1abc9c","#e74c3c","#f39c12"]

    for i, leg in enumerate(st.session_state.legs):
        cols = st.columns([1.5, 1.5, 1.5, 1.5, 1.8, 0.5])
        with cols[0]:
            ot  = st.selectbox("Type",  ["call","put"], key=f"ot_{i}",
                               index=0 if leg["option_type"]=="call" else 1)
        with cols[1]:
            K   = st.number_input(f"Strike ({sym})", value=float(leg["K"]), step=float(50 if cfg.market=="IN" else 5), key=f"K_{i}")
        with cols[2]:
            qty = st.number_input("Qty (+long / -short)", value=int(leg["quantity"]),
                                  min_value=-100, max_value=100, step=1, key=f"qty_{i}")
        with cols[3]:
            lot = st.number_input("Lot size", value=int(leg.get("lot_size", cfg.default_lot_size)),
                                  min_value=1, step=1, key=f"lot_{i}")
        with cols[4]:
            bs_p = BlackScholesModel(S=spot, K=K, T=T, r=r, sigma=sigma, option_type=ot).price()
            st.metric("BS premium", fmt(bs_p, cfg, 2))
        with cols[5]:
            remove = st.button("✕", key=f"rm_{i}")

        if not remove:
            legs_updated.append({"option_type":ot,"K":K,"quantity":qty,"lot_size":lot,"premium":bs_p})

    st.session_state.legs = [
        {"option_type":l["option_type"],"K":l["K"],"quantity":l["quantity"],"lot_size":l["lot_size"]}
        for l in legs_updated
    ]

    if not legs_updated:
        st.info("Add at least one leg.")
        return

    st.divider()

    # ── Payoff chart ──────────────────────────────────────────────────────────
    st.subheader("P&L at expiry")

    strikes   = [l["K"] for l in legs_updated]
    S_min     = min(min(strikes)*0.75, spot*0.75)
    S_max     = max(max(strikes)*1.25, spot*1.25)
    S_range   = np.linspace(S_min, S_max, 400)
    pnl       = payoff_at_expiry(legs_updated, S_range)

    # Transaction costs (total round-trip, all legs)
    calc       = CostCalculator(cfg)
    cost_info  = calc.multi_leg_cost(legs_updated)
    total_cost = cost_info["grand_total"]

    # P&L net of costs
    pnl_net = pnl - total_cost

    show_costs = st.toggle("Show P&L net of transaction costs", value=True)
    pnl_display = pnl_net if show_costs else pnl

    fig = go.Figure()
    # Profit/loss shading
    fig.add_trace(go.Scatter(
        x=np.concatenate([S_range, S_range[::-1]]),
        y=np.concatenate([np.where(pnl_display>0, pnl_display, 0), np.zeros(len(S_range))]),
        fill="toself", fillcolor="rgba(46,204,113,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Profit zone",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([S_range, S_range[::-1]]),
        y=np.concatenate([np.where(pnl_display<0, pnl_display, 0), np.zeros(len(S_range))]),
        fill="toself", fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Loss zone",
    ))
    # Net P&L line
    fig.add_trace(go.Scatter(x=S_range, y=pnl_display, name="Net P&L", mode="lines",
                             line=dict(color="#2c3e50", width=2.5)))
    # Gross P&L (if showing net)
    if show_costs:
        fig.add_trace(go.Scatter(x=S_range, y=pnl, name="Gross P&L (before costs)",
                                 mode="lines", line=dict(color="#95a5a6", width=1.2, dash="dot")))
    # Individual legs
    for j, leg in enumerate(legs_updated):
        leg_pnl = payoff_at_expiry([leg], S_range)
        label   = f"Leg {j+1}: {'+' if leg['quantity']>0 else ''}{leg['quantity']} {leg['option_type'].capitalize()} {sym}{leg['K']:.0f}"
        fig.add_trace(go.Scatter(x=S_range, y=leg_pnl, name=label, mode="lines",
                                 line=dict(color=colors[j%len(colors)], width=1.2, dash="dot"), opacity=0.6))

    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.add_vline(x=spot, line_dash="dot", line_color="#e74c3c",
                  annotation_text=f"Spot {sym}{spot:,.0f}")
    for K in strikes:
        fig.add_vline(x=K, line_dash="dot", line_color="#bdc3c7", line_width=1)

    fig.update_layout(
        title=f"P&L at expiry — {cfg.strip_suffix(ticker)}",
        xaxis_title=f"Spot at expiry ({sym})",
        yaxis_title=f"P&L per position ({sym})",
        height=460, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Strategy summary ──────────────────────────────────────────────────────
    st.subheader("Strategy summary")
    s1, s2, s3, s4, s5 = st.columns(5)

    net_premium = sum(l["quantity"]*l["lot_size"]*l["premium"] for l in legs_updated)
    breakevens  = S_range[np.where(np.diff(np.sign(pnl_display)))[0]]
    be_str      = "  |  ".join([f"{sym}{b:,.0f}" for b in breakevens]) if len(breakevens) else "—"
    max_profit  = pnl_display.max()
    max_loss    = pnl_display.min()

    s1.metric("Net premium",     fmt(net_premium, cfg),
              help="Positive = net debit (you paid). Negative = net credit (you received).")
    s2.metric("Max profit",      fmt(max_profit, cfg) if max_profit < 1e9 else "Unlimited")
    s3.metric("Max loss",        fmt(max_loss, cfg)   if max_loss > -1e9 else "Unlimited")
    s4.metric("Break-even(s)",   be_str)
    s5.metric("Total txn costs", fmt(total_cost, cfg),
              help="Round-trip costs across all legs (entry + exit).")

    # ── Transaction cost detail ───────────────────────────────────────────────
    with st.expander("📋 Transaction cost breakdown by leg"):
        for i, (leg, rt) in enumerate(zip(legs_updated, cost_info["itemised"])):
            direction = "Long" if leg["quantity"] > 0 else "Short"
            st.markdown(f"**Leg {i+1}: {direction} {leg['option_type'].capitalize()} {sym}{leg['K']:.0f} × {abs(leg['quantity'])} lot(s)**")
            col_e, col_x = st.columns(2)
            with col_e:
                st.markdown("*Entry*")
                for k, v in rt.entry.breakdown().items():
                    if v > 0:
                        st.markdown(f"- {k}: {fmt(v, cfg, 4)}")
            with col_x:
                st.markdown("*Exit*")
                for k, v in rt.exit.breakdown().items():
                    if v > 0:
                        st.markdown(f"- {k}: {fmt(v, cfg, 4)}")
            st.markdown(f"*Leg round-trip: {fmt(rt.total, cfg, 2)} ({rt.cost_as_pct_of_premium:.2f}% of premium)*")
            st.divider()

    # ── Aggregate Greeks ──────────────────────────────────────────────────────
    st.subheader("Position Greeks")

    pos_legs = [{"S":spot,"K":l["K"],"T":T,"sigma":sigma,"option_type":l["option_type"],
                 "quantity":l["quantity"],"lot_size":l["lot_size"]} for l in legs_updated]
    pos = position_greeks(pos_legs, r=r)

    pg1, pg2, pg3, pg4, pg5 = st.columns(5)
    pg1.metric("Net Δ Delta",    f"{pos['delta']:.3f}")
    pg2.metric("Net Γ Gamma",    f"{pos['gamma']:.4f}")
    pg3.metric("Net Θ Theta/d",  f"{pos['theta']:.3f}")
    pg4.metric("Net ν Vega/1%",  f"{pos['vega']:.3f}")
    pg5.metric("Total BS value", fmt(pos["bs_value"], cfg))

    with st.expander("📖 Reading the position Greeks"):
        st.markdown(f"""
**Delta = {pos['delta']:.3f}** — {"Near delta-neutral." if abs(pos['delta']) < 0.05 else f"{'Bullish' if pos['delta'] > 0 else 'Bearish'} directional exposure. {sym}1 move in {cfg.strip_suffix(ticker)} → ~{sym}{abs(pos['delta']):.2f} P&L change."}

**Gamma = {pos['gamma']:.4f}** — {"Positive gamma: large spot moves in either direction benefit you (long vol)." if pos['gamma'] > 0 else "Negative gamma: large moves hurt you. Characteristic of premium-selling strategies."}

**Theta = {pos['theta']:.3f}/day** — {"Positive theta: time decay earns you money (short premium)." if pos['theta'] > 0 else f"Negative theta: you lose ~{fmt(abs(pos['theta']), cfg, 2)}/day to time decay. Over a week: ~{fmt(abs(pos['theta'])*7, cfg, 2)}."}

**Vega = {pos['vega']:.3f} per 1% vol** — {"Positive vega: you profit if IV rises (pre-earnings, uncertainty)." if pos['vega'] > 0 else "Negative vega: rising IV hurts. Be cautious entering before known vol events."}

**Gamma-Theta trade-off**: Long gamma always pays theta. Short gamma always collects theta. This is the core tension in every options position — there is no free lunch.
""")

    # ── NSE-specific guidance ─────────────────────────────────────────────────
    if cfg.market == "IN":
        st.divider()
        with st.expander("ℹ️ Indian market context for this strategy"):
            total_margin_est = abs(pos["delta"]) * spot * cfg.get_lot_size(ticker) * 0.15
            st.markdown(f"""
**STT on expiry warning**: If you hold ITM options to expiry, STT is levied on the full intrinsic value — not just the premium. This can be 10–20× the normal trading STT. Always square off before expiry unless you intend delivery.

**Lot size**: {cfg.get_lot_size(ticker)} units for {cfg.strip_suffix(ticker)}. Multiply all Greeks and P&L figures by this to get per-contract exposure.

**Margin estimate**: For naked short legs, SPAN + exposure margin applies. A rough directional margin estimate for this position is ~{fmt(total_margin_est, cfg, 0)} (15% of delta-notional). Use your broker's actual margin calculator for precise figures.

**Transaction cost impact**: Total round-trip cost for this strategy is {fmt(total_cost, cfg, 2)}, which is {total_cost/abs(net_premium)*100 if net_premium != 0 else 0:.1f}% of the net premium. {"This is manageable." if total_cost/abs(net_premium) < 0.05 else "This is significant — ensure your edge exceeds transaction costs before entering."}
""")
