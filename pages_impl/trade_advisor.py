"""
Page 4 — Trade Evaluator
You describe the trade you're considering (strike, market price, expiry,
holding period, your market prediction). The tool rates how much sense
it makes and tells you exactly why — no strike recommendation, just a
clear verdict on the trade you've already identified.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pages_impl.shared import fetch_spot, fetch_hist_vol, fetch_expiries, fmt
from options_engine.pricing.black_scholes import BlackScholesModel
from options_engine.utils.volatility import time_to_expiry_years, implied_volatility
from options_engine.costs import CostCalculator
from options_engine.market_config import MarketConfig


# ── Rating system ─────────────────────────────────────────────────────────────

def _rate_trade(
    spot, K, T, r, sigma_input, option_type, long_short,
    market_price, expected_move_pct, holding_days, T_days, cfg
) -> dict:
    """
    Evaluate a specific trade and return a structured rating.

    Components scored 0–100, then combined into a verdict.
    """
    sym      = cfg.currency_symbol
    intraday = holding_days < 1

    bs    = BlackScholesModel(S=spot, K=K, T=T, r=r, sigma=sigma_input, option_type=option_type)
    bs_res = bs.evaluate()

    price    = market_price
    delta    = bs_res.delta
    gamma    = bs_res.gamma
    theta    = bs_res.theta
    vega     = bs_res.vega
    bs_price = bs_res.price

    abs_delta = abs(delta)
    theta_pct_day = abs(theta) / price * 100 if price > 0 else 0

    # Breakeven
    breakeven    = (K - price) if option_type == "put" else (K + price)
    be_move_pct  = abs(spot - breakeven) / spot * 100

    # IV from market price
    iv_market = implied_volatility(price, spot, K, T, r, option_type)
    iv_market_pct = iv_market * 100 if iv_market else None

    # Theoretical edge: BS price vs market price
    # For long: negative edge means you're overpaying
    if long_short == "long":
        edge_pct = (bs_price - price) / price * 100   # positive = cheap vs model
    else:
        edge_pct = (price - bs_price) / price * 100   # positive = collecting more than model

    # ── Score each component ──────────────────────────────────────────────────

    # 1. Direction alignment: does option type match view?
    is_bearish = (option_type == "put" and long_short == "long") or \
                 (option_type == "call" and long_short == "short")
    is_bullish = (option_type == "call" and long_short == "long") or \
                 (option_type == "put" and long_short == "short")
    direction_aligned = True   # always true since we don't ask direction separately now
    direction_score = 100

    # 2. Breakeven reachability vs expected move
    ratio = expected_move_pct / be_move_pct if be_move_pct > 0 else 0
    if long_short == "long":
        if ratio >= 2.0:
            be_score, be_label = 100, "Excellent — expected move is 2× the break-even distance"
        elif ratio >= 1.5:
            be_score, be_label = 85, "Good — expected move comfortably exceeds break-even"
        elif ratio >= 1.0:
            be_score, be_label = 65, "Borderline — expected move just reaches break-even"
        elif ratio >= 0.7:
            be_score, be_label = 35, "Weak — expected move falls short of break-even"
        else:
            be_score, be_label = 10, "Poor — expected move is far below break-even"
    else:
        # Short: want expected move to NOT reach break-even
        if ratio <= 0.5:
            be_score, be_label = 100, "Excellent — market is unlikely to reach your short strike"
        elif ratio <= 0.8:
            be_score, be_label = 75, "Good — market needs a larger move than expected to hurt you"
        elif ratio <= 1.0:
            be_score, be_label = 45, "Borderline — market is close to your break-even"
        else:
            be_score, be_label = 10, "Risky — market expected to move past your break-even"

    # 3. Theta sustainability
    if intraday:
        theta_cost_total = theta_pct_day * 0.4   # ~40% of a day
    else:
        theta_cost_total = theta_pct_day * holding_days

    if long_short == "long":
        if theta_cost_total < 5:
            theta_score, theta_label = 100, f"Very low total decay ({theta_cost_total:.1f}% of premium over hold period)"
        elif theta_cost_total < 15:
            theta_score, theta_label = 75, f"Manageable decay ({theta_cost_total:.1f}% of premium over hold period)"
        elif theta_cost_total < 30:
            theta_score, theta_label = 45, f"High decay ({theta_cost_total:.1f}% of premium) — move must come quickly"
        elif theta_cost_total < 50:
            theta_score, theta_label = 20, f"Very high decay ({theta_cost_total:.1f}%) — theta is a major headwind"
        else:
            theta_score, theta_label = 5, f"Extreme decay ({theta_cost_total:.1f}%) — time is heavily against you"
    else:
        # Short: theta works for you
        if theta_cost_total > 20:
            theta_score, theta_label = 100, f"Fast decay ({theta_cost_total:.1f}%) — time decay earns you premium quickly"
        elif theta_cost_total > 10:
            theta_score, theta_label = 80, f"Good decay ({theta_cost_total:.1f}%) — steady theta income"
        else:
            theta_score, theta_label = 60, f"Slow decay ({theta_cost_total:.1f}%) — moderate theta income"

    # 4. Pricing: is market price fair vs BS model?
    if long_short == "long":
        if edge_pct >= 5:
            price_score, price_label = 100, f"Cheap vs model by {edge_pct:.1f}% — positive theoretical edge"
        elif edge_pct >= 0:
            price_score, price_label = 70, f"Fairly priced — within {abs(edge_pct):.1f}% of BS fair value"
        elif edge_pct >= -5:
            price_score, price_label = 45, f"Slightly expensive — {abs(edge_pct):.1f}% above BS fair value"
        elif edge_pct >= -15:
            price_score, price_label = 20, f"Expensive — {abs(edge_pct):.1f}% above BS fair value"
        else:
            price_score, price_label = 5, f"Very expensive — {abs(edge_pct):.1f}% above BS fair value"
    else:
        if edge_pct >= 5:
            price_score, price_label = 100, f"Rich premium — collecting {edge_pct:.1f}% above BS fair value"
        elif edge_pct >= 0:
            price_score, price_label = 70, f"Fair premium at roughly BS fair value"
        elif edge_pct >= -5:
            price_score, price_label = 40, f"Cheap premium — collecting {abs(edge_pct):.1f}% below model"
        else:
            price_score, price_label = 15, f"Very cheap — selling significantly below model value"

    # 5. Delta quality: is the delta appropriate for the trade?
    if long_short == "long":
        # For directional long: delta 0.3–0.6 is sweet spot
        if 0.30 <= abs_delta <= 0.60:
            delta_score, delta_label = 100, f"Ideal delta ({abs_delta:.2f}) — good directional exposure with reasonable cost"
        elif 0.20 <= abs_delta < 0.30:
            delta_score, delta_label = 60, f"Low delta ({abs_delta:.2f}) — cheap but needs large move; lottery-ticket profile"
        elif 0.60 < abs_delta <= 0.80:
            delta_score, delta_label = 75, f"High delta ({abs_delta:.2f}) — expensive but high probability; behaves more like stock"
        elif abs_delta > 0.80:
            delta_score, delta_label = 50, f"Very high delta ({abs_delta:.2f}) — deep ITM, low leverage, consider the underlying instead"
        else:
            delta_score, delta_label = 40, f"Very low delta ({abs_delta:.2f}) — deep OTM, very unlikely to profit"
    else:
        # Short: want delta close to 0 (far OTM is safer)
        if abs_delta <= 0.20:
            delta_score, delta_label = 100, f"Low delta ({abs_delta:.2f}) — far OTM, high probability of full profit"
        elif abs_delta <= 0.35:
            delta_score, delta_label = 75, f"Moderate delta ({abs_delta:.2f}) — reasonable probability, watch gamma near expiry"
        elif abs_delta <= 0.50:
            delta_score, delta_label = 45, f"ATM delta ({abs_delta:.2f}) — risky short; consider further OTM"
        else:
            delta_score, delta_label = 15, f"High delta ({abs_delta:.2f}) — dangerous short; heavily exposed to move"

    # 6. DTE risk
    if T_days >= 15:
        dte_score, dte_label = 90, f"{T_days}d DTE — adequate time for the trade to develop"
    elif T_days >= 7:
        dte_score, dte_label = 70, f"{T_days}d DTE — short window; move must come soon"
    elif T_days >= 3:
        dte_score, dte_label = 40, f"{T_days}d DTE — very short; gamma risk elevated, theta accelerating"
    else:
        dte_score, dte_label = 15, f"{T_days}d DTE — extreme gamma risk; only for very high-conviction intraday"

    # ── Weighted composite ────────────────────────────────────────────────────
    if intraday:
        weights = {"be": 0.35, "theta": 0.10, "price": 0.25, "delta": 0.20, "dte": 0.10}
    else:
        weights = {"be": 0.30, "theta": 0.25, "price": 0.20, "delta": 0.15, "dte": 0.10}

    composite = (
        be_score    * weights["be"]    +
        theta_score * weights["theta"] +
        price_score * weights["price"] +
        delta_score * weights["delta"] +
        dte_score   * weights["dte"]
    )

    # ── Verdict ───────────────────────────────────────────────────────────────
    if composite >= 75:
        verdict      = "Strong trade"
        verdict_sub  = "The setup aligns well with your prediction. Risk/reward is favourable."
        verdict_color = "#27ae60"
        verdict_icon  = "✅"
    elif composite >= 60:
        verdict      = "Reasonable trade"
        verdict_sub  = "Acceptable setup. One or two factors are working against you — keep position size moderate."
        verdict_color = "#2980b9"
        verdict_icon  = "🟡"
    elif composite >= 45:
        verdict      = "Marginal trade"
        verdict_sub  = "Several factors are unfavourable. Only enter with high conviction and small size."
        verdict_color = "#f39c12"
        verdict_icon  = "⚠️"
    else:
        verdict      = "Poor trade"
        verdict_sub  = "The odds are stacked against this trade. Reconsider the strike, expiry, or your prediction."
        verdict_color = "#e74c3c"
        verdict_icon  = "🔴"

    return {
        "composite":       round(composite, 1),
        "verdict":         verdict,
        "verdict_sub":     verdict_sub,
        "verdict_color":   verdict_color,
        "verdict_icon":    verdict_icon,
        "bs_price":        round(bs_price, 2),
        "edge_pct":        round(edge_pct, 1),
        "iv_market_pct":   iv_market_pct,
        "delta":           delta,
        "gamma":           gamma,
        "theta":           theta,
        "vega":            vega,
        "breakeven":       round(breakeven, 1),
        "be_move_pct":     round(be_move_pct, 2),
        "theta_pct_day":   round(theta_pct_day, 1),
        "theta_cost_total":round(theta_cost_total, 1),
        "scores": {
            "Breakeven reachability": (be_score,    be_label),
            "Theta sustainability":   (theta_score, theta_label),
            "Pricing vs model":       (price_score, price_label),
            "Delta quality":          (delta_score, delta_label),
            "DTE risk":               (dte_score,   dte_label),
        },
    }


# ── Main render ───────────────────────────────────────────────────────────────

def render(ticker: str, r: float, cfg: MarketConfig):
    sym          = cfg.currency_symbol
    market_badge = "🇮🇳 NSE" if cfg.market == "IN" else "🇺🇸 US"

    st.title(f"Trade Evaluator  {market_badge}")
    st.caption(
        "Describe the trade you're considering from Zerodha → "
        "get a rating and breakdown of whether it makes sense"
    )

    # ── Auto-fetch ────────────────────────────────────────────────────────────
    spot_fetched = None
    vol_fetched  = None
    expiries     = []

    with st.spinner(f"Fetching {cfg.strip_suffix(ticker)} market data…"):
        try:
            spot_data    = fetch_spot(ticker, r)
            spot_fetched = spot_data.price
            pct_chg      = (spot_data.price - spot_data.previous_close) / spot_data.previous_close * 100
        except Exception:
            pct_chg = 0.0

        try:
            vol_fetched = fetch_hist_vol(ticker, r) * 100
        except Exception:
            vol_fetched = 20.0

        try:
            expiries = fetch_expiries(ticker, r)
        except Exception:
            expiries = []

    h1, h2, h3 = st.columns(3)
    if spot_fetched:
        h1.metric(f"{cfg.strip_suffix(ticker)} Spot", fmt(spot_fetched, cfg), f"{pct_chg:+.2f}%")
    else:
        h1.metric("Spot", "—")
    h2.metric("21d hist vol", f"{vol_fetched:.1f}%")
    h3.metric("Expiries available", f"{len(expiries)}" if expiries else "Enter manually")

    st.divider()

    # ── Trade inputs ──────────────────────────────────────────────────────────
    st.subheader("The trade you're considering")

    col1, col2, col3 = st.columns(3)

    with col1:
        spot_input = st.number_input(
            f"Current spot ({sym})",
            value=float(round(spot_fetched)) if spot_fetched else (20000.0 if cfg.market == "IN" else 500.0),
            min_value=1.0, step=1.0,
            help="Auto-filled. Override if needed."
        )
        option_type = st.selectbox("Option type", ["call", "put"])
        long_short  = st.selectbox("Long or short", ["long", "short"],
                                   help="Long = buying. Short = selling (requires margin).")

    with col2:
        K_input = st.number_input(
            f"Strike price ({sym})",
            value=float(round(spot_fetched / 50) * 50) if spot_fetched else (20000.0 if cfg.market == "IN" else 500.0),
            min_value=1.0, step=1.0,
            help="The strike you're looking at on Zerodha."
        )
        market_price = st.number_input(
            f"Market price ({sym}) — from Zerodha",
            value=100.0, min_value=0.01, step=0.5,
            help="The LTP or price you expect to pay/receive. This drives all calculations."
        )

    with col3:
        if expiries:
            expiry_str  = st.selectbox("Expiry", expiries)
            expiry_date = datetime.date.fromisoformat(expiry_str)
        else:
            expiry_date = st.date_input(
                "Expiry date",
                value=datetime.date.today() + datetime.timedelta(days=7),
                min_value=datetime.date.today() + datetime.timedelta(days=1),
            )
            expiry_str = expiry_date.isoformat()

        lot_size = st.number_input(
            "Lot size", value=50 if cfg.market == "IN" else 100, min_value=1, step=1,
            help="NIFTY=50, BANKNIFTY=15, US=100"
        )

    st.divider()

    # ── Your prediction ───────────────────────────────────────────────────────
    st.subheader("Your prediction")

    col4, col5, col6 = st.columns(3)

    with col4:
        expected_move_pct = st.number_input(
            "Expected move (%)", value=1.5, min_value=0.1, max_value=20.0, step=0.1,
            help="How much % do you expect the underlying to move in your favour?"
        )

    with col5:
        sigma_pct = st.number_input(
            "Implied vol (%)",
            value=float(round(vol_fetched, 1)) if vol_fetched else 20.0,
            min_value=1.0, max_value=200.0, step=0.5,
            help="Auto-filled from 21d historical vol. Override with ATM IV from Zerodha option chain."
        )

    with col6:
        hold_type = st.radio(
            "Holding period",
            ["Intraday", "Multi-day"],
            horizontal=True,
        )
        T_days_total = (expiry_date - datetime.date.today()).days
        if hold_type == "Intraday":
            holding_days = 0.5
        else:
            holding_days = float(st.slider(
                "Hold for how many days?",
                min_value=1,
                max_value=max(1, T_days_total - 1),
                value=min(3, max(1, T_days_total - 1)),
            ))

    st.divider()

    if not st.button("📊  Evaluate this trade", type="primary", use_container_width=True):
        st.stop()

    # ── Run evaluation ────────────────────────────────────────────────────────
    T       = time_to_expiry_years(expiry_str)
    sigma   = sigma_pct / 100
    intraday = holding_days < 1

    result = _rate_trade(
        spot_input, K_input, T, r, sigma, option_type, long_short,
        market_price, expected_move_pct, holding_days, T_days_total, cfg
    )

    # ── Verdict banner ────────────────────────────────────────────────────────
    st.subheader("Verdict")

    vc = result["verdict_color"]
    st.markdown(f"""
<div style="border:2px solid {vc};border-radius:12px;padding:22px 28px;margin-bottom:16px">
    <div style="font-size:13px;color:#888;margin-bottom:6px">
        {long_short.upper()} {option_type.upper()} · Strike {sym}{K_input:,.0f} ·
        {'INTRADAY' if intraday else f'{holding_days:.0f}-DAY HOLD'} ·
        Overall score {result['composite']:.0f}/100
    </div>
    <div style="font-size:34px;font-weight:700;color:{vc};margin-bottom:8px">
        {result['verdict_icon']}  {result['verdict']}
    </div>
    <div style="font-size:15px;color:#444">{result['verdict_sub']}</div>
</div>
""", unsafe_allow_html=True)

    # IV note
    if result["iv_market_pct"]:
        iv_diff = result["iv_market_pct"] - sigma_pct
        if abs(iv_diff) > 1:
            st.info(
                f"At market price {sym}{market_price:.2f}, this option implies **IV = {result['iv_market_pct']:.1f}%** "
                f"vs your input of {sigma_pct:.1f}%. "
                + (f"The market is pricing **{iv_diff:.1f}% more vol** than your assumption — you're paying for that extra uncertainty."
                   if iv_diff > 0
                   else f"The market is pricing **{abs(iv_diff):.1f}% less vol** than your assumption — option looks cheap relative to your vol view."),
                icon="📊",
            )

    # Edge note
    if long_short == "long":
        edge_note = (
            f"BS fair value: {sym}{result['bs_price']:.2f} · "
            f"Market price: {sym}{market_price:.2f} · "
            + (f"You're paying **{sym}{market_price - result['bs_price']:.2f} above** BS model."
               if market_price > result["bs_price"]
               else f"You're buying **{sym}{result['bs_price'] - market_price:.2f} below** BS model — positive edge.")
        )
    else:
        edge_note = (
            f"BS fair value: {sym}{result['bs_price']:.2f} · "
            f"Market price: {sym}{market_price:.2f} · "
            + (f"Collecting **{sym}{market_price - result['bs_price']:.2f} above** BS model — rich premium."
               if market_price > result["bs_price"]
               else f"Collecting **{sym}{result['bs_price'] - market_price:.2f} below** BS model — lean premium.")
        )
    st.caption(edge_note)

    # ── Score breakdown ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Score breakdown")

    scores_df = pd.DataFrame([
        {"Component": k, "Score": v[0], "What this means": v[1]}
        for k, v in result["scores"].items()
    ])

    fig = go.Figure(go.Bar(
        x=scores_df["Score"],
        y=scores_df["Component"],
        orientation="h",
        marker_color=[
            "#27ae60" if s >= 75 else "#f39c12" if s >= 45 else "#e74c3c"
            for s in scores_df["Score"]
        ],
        text=[f"{s:.0f}" for s in scores_df["Score"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=260, xaxis_range=[0, 115],
        xaxis_title="Score (0–100)",
        margin=dict(l=0, r=50, t=10, b=10),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    for comp, (score, label) in result["scores"].items():
        icon = "✅" if score >= 75 else "🟡" if score >= 45 else "🔴"
        st.markdown(f"{icon} **{comp}** ({score:.0f}/100) — {label}")

    # ── Key numbers ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Key numbers")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Break-even",     f"{sym}{result['breakeven']:,.0f}")
    m2.metric("Move needed",    f"{result['be_move_pct']:.2f}%",
              delta=f"{expected_move_pct - result['be_move_pct']:+.1f}% vs forecast",
              delta_color="normal")
    m3.metric("Delta",          f"{result['delta']:.3f}")
    m4.metric("Theta / day",    f"{sym}{abs(result['theta']):.2f}  ({result['theta_pct_day']:.1f}%/day)")
    m5.metric(f"Theta over {'session' if intraday else f'{holding_days:.0f}d'}",
              f"{result['theta_cost_total']:.1f}% of premium")

    calc = CostCalculator(cfg)
    rt   = calc.round_trip(premium=market_price, lot_size=lot_size, quantity=1)

    m6, m7, m8, m9, m10 = st.columns(5)
    m6.metric("Vega / 1% vol",  f"{sym}{result['vega']:.3f}")
    m7.metric("DTE",            f"{T_days_total}d")
    m8.metric("Cost per lot",   fmt(market_price * lot_size, cfg, 0))
    m9.metric("Round-trip txn", fmt(rt.total, cfg, 0))
    m10.metric("Txn as % of premium", f"{rt.cost_as_pct_of_premium:.1f}%",
               help="If this is >3%, transaction costs are eating a meaningful chunk of your edge.")

    # ── P&L chart ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("P&L at expiry vs at exit")

    S_range  = np.linspace(spot_input * 0.85, spot_input * 1.15, 300)
    intrinsic = np.maximum(K_input - S_range, 0) if option_type == "put" else np.maximum(S_range - K_input, 0)
    sign     = 1 if long_short == "long" else -1
    pnl_exp  = sign * (intrinsic - market_price) * lot_size

    # Mid-hold P&L
    T_hold = max(T - (holding_days / 365.0 if not intraday else T * 0.5), 1e-6)
    bs_hold = np.array([
        BlackScholesModel(float(s), K_input, T_hold, r, sigma, option_type).price()
        for s in S_range[::5]
    ])
    idx_full   = np.linspace(0, len(S_range)-1, len(S_range))
    idx_sample = np.linspace(0, len(S_range)-1, len(bs_hold))
    bs_full    = np.interp(idx_full, idx_sample, bs_hold)
    pnl_hold   = sign * (bs_full - market_price) * lot_size

    target = spot_input * (1 + expected_move_pct/100) if (
        (option_type == "call" and long_short == "long") or
        (option_type == "put"  and long_short == "short")
    ) else spot_input * (1 - expected_move_pct/100)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=np.concatenate([S_range, S_range[::-1]]),
        y=np.concatenate([np.where(pnl_exp > 0, pnl_exp, 0), np.zeros(len(S_range))]),
        fill="toself", fillcolor="rgba(39,174,96,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    fig2.add_trace(go.Scatter(
        x=np.concatenate([S_range, S_range[::-1]]),
        y=np.concatenate([np.where(pnl_exp < 0, pnl_exp, 0), np.zeros(len(S_range))]),
        fill="toself", fillcolor="rgba(231,76,60,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    fig2.add_trace(go.Scatter(x=S_range, y=pnl_exp,  name="At expiry",
                              line=dict(color="#2c3e50", width=2.5)))
    fig2.add_trace(go.Scatter(x=S_range, y=pnl_hold,
                              name=f"At exit ({'intraday est.' if intraday else f'{holding_days:.0f}d, BS'})",
                              line=dict(color="#3498db", width=1.8, dash="dot")))

    fig2.add_vline(x=spot_input, line_dash="dot", line_color="#e74c3c",
                   annotation_text=f"Spot {sym}{spot_input:,.0f}")
    fig2.add_vline(x=K_input, line_dash="dot", line_color="#3498db",
                   annotation_text=f"Strike {sym}{K_input:,.0f}")
    fig2.add_vline(x=result["breakeven"], line_dash="dot", line_color="#27ae60",
                   annotation_text="Break-even")
    fig2.add_vline(x=target, line_dash="dash", line_color="#9b59b6",
                   annotation_text=f"Your target {sym}{target:,.0f}")
    fig2.add_hline(y=0, line_color="#888", line_width=1)
    fig2.update_layout(
        xaxis_title=f"Market price ({sym})",
        yaxis_title=f"P&L per lot ({sym})",
        height=420, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Exit plan ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Exit plan")

    if intraday:
        tgt_price  = market_price * 1.5
        stop_price = market_price * 0.6
        hard_exit  = "Square off before 3:20 PM IST (NSE) — no overnight risk on intraday."
        rule       = f"If underlying hasn't moved {expected_move_pct:.1f}% by 1 PM, exit regardless."
    else:
        tgt_price  = market_price * 2.0
        stop_price = market_price * 0.5
        hard_exit  = f"Exit by DTE = {max(1, T_days_total - 2)} — never hold to expiry."
        rule       = (
            f"If underlying hasn't moved {result['be_move_pct']:.1f}% "
            f"within {max(1, int(holding_days // 2))} days, reassess — "
            f"remaining theta ({sym}{abs(result['theta']):.2f}/day) will dominate."
        )

    e1, e2, e3 = st.columns(3)
    e1.metric("Take profit at option price",  f"{sym}{tgt_price:.2f}")
    e2.metric("Stop loss at option price",    f"{sym}{stop_price:.2f}")
    e3.metric("Hard exit rule",               hard_exit)
    st.info(rule, icon="💡")
