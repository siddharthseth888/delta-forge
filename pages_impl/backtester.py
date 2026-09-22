"""
Page 5 — Backtester
Model-based options backtesting using reconstructed BS prices.
Tests the logic of a trade setup over historical data.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pages_impl.shared import get_engine, fmt
from options_engine.backtest import BacktestEngine, BacktestConfig
from options_engine.market_config import MarketConfig


# ── Cached data fetch ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_prices(ticker: str, period: str, r: float) -> pd.Series:
    engine = get_engine(r)
    return engine.adapter.get_historical_prices(ticker, period=period)


# ── Main render ───────────────────────────────────────────────────────────────

def render(ticker: str, r: float, cfg: MarketConfig):
    sym          = cfg.currency_symbol
    market_badge = "🇮🇳 NSE" if cfg.market == "IN" else "🇺🇸 US"

    st.title(f"Backtester  {market_badge}")
    st.caption(
        "Model-based backtesting using reconstructed Black-Scholes prices. "
        "Tests trade decision logic over historical data — not actual option fills."
    )

    with st.expander("ℹ️ How this works and what it can / cannot tell you"):
        st.markdown("""
**What it does**
For each entry date in the selected period, the engine:
1. Takes the historical closing price as spot
2. Computes 21-day rolling realised vol as a proxy for implied vol
3. Prices the option using Black-Scholes at those inputs
4. Simulates holding and checks exit conditions daily (profit target, stop loss, max hold days)
5. Records P&L for each trade and aggregates results

**What it can tell you**
- Whether the *decision logic* of a trade setup tends to produce positive expectation
- How sensitive outcomes are to profit target vs stop loss ratios
- How often the move needed to profit actually materialised historically
- Distribution of hold times and exit reasons

**What it cannot tell you**
- Actual fills — BS reconstructed prices are theoretical, not market prices
- Slippage, bid-ask spread, or liquidity effects
- The impact of actual implied vol skew (this uses realised vol as proxy)
- Whether past patterns will repeat

**Bottom line**: use this to stress-test logic and build intuition, not to predict future returns.
""")

    st.divider()

    # ── Inputs ────────────────────────────────────────────────────────────────
    st.subheader("Backtest setup")

    col1, col2, col3 = st.columns(3)

    with col1:
        period = st.selectbox(
            "Historical period",
            ["6mo", "1y", "2y"],
            index=1,
            help="How far back to test. More data = more trades = more reliable stats.",
        )
        option_type = st.selectbox("Option type", ["put", "call"])
        long_short  = st.selectbox("Long or short", ["long", "short"])

    with col2:
        strike_offset = st.number_input(
            "Strike offset from spot (%)",
            value=-2.0 if option_type == "put" else 2.0,
            min_value=-20.0, max_value=20.0, step=0.5,
            help=(
                "How far OTM to enter. "
                "-2% = put strike 2% below spot. "
                "+2% = call strike 2% above spot. "
                "0% = ATM."
            ),
        )
        dte_entry = st.number_input(
            "DTE at entry (days)",
            value=14, min_value=3, max_value=60, step=1,
            help="How many days to expiry when you enter the trade.",
        )
        entry_freq = st.radio(
            "Entry frequency",
            ["weekly", "daily"],
            horizontal=True,
            help="Weekly = one trade per week max. Daily = enter every eligible day.",
        )

    with col3:
        profit_target = st.number_input(
            "Profit target (%)",
            value=100.0, min_value=10.0, max_value=500.0, step=10.0,
            help="Exit when option gains this % (long) or loses this % (short).",
        )
        stop_loss = st.number_input(
            "Stop loss (%)",
            value=50.0, min_value=5.0, max_value=100.0, step=5.0,
            help="Exit when option loses this % (long) or gains this % (short).",
        )
        max_hold = st.number_input(
            "Max hold days",
            value=7, min_value=1, max_value=30, step=1,
            help="Force exit after this many calendar days regardless of P&L.",
        )
        min_dte = st.number_input(
            "Min DTE at entry",
            value=3, min_value=1, max_value=14, step=1,
            help=(
                "Skip entry if fewer than this many days remain to next Tuesday expiry. "
                "Prevents entering on Monday/Tuesday when the option has almost no time value."
            ),
        )

    # Vol filter
    with st.expander("Vol filter (optional)"):
        vc1, vc2 = st.columns(2)
        with vc1:
            min_vol = st.number_input("Min vol to enter (%)", value=0.0, step=1.0,
                                      help="Skip entry if 21d vol is below this.")
        with vc2:
            max_vol = st.number_input("Max vol to enter (%)", value=100.0, step=1.0,
                                      help="Skip entry if 21d vol is above this.")

    lot_size = st.number_input(
        "Lot size (for P&L scaling)",
        value=cfg.get_lot_size(ticker), min_value=1, step=1,
        help="Scales P&L to reflect actual lot exposure. Use 1 for per-unit analysis.",
    )

    st.divider()

    if not st.button("▶  Run backtest", type="primary", use_container_width=True):
        st.stop()

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {cfg.strip_suffix(ticker)} price history ({period})…"):
        try:
            prices = _fetch_prices(ticker, period, r)
            if len(prices) < 30:
                st.error("Not enough price history returned. Try a longer period or different ticker.")
                st.stop()
        except Exception as e:
            st.error(f"Could not fetch price history: {e}")
            st.stop()

    # ── Run backtest ──────────────────────────────────────────────────────────
    config = BacktestConfig(
        option_type=option_type,
        long_short=long_short,
        strike_offset_pct=strike_offset,
        dte_at_entry=dte_entry,
        profit_target_pct=profit_target,
        stop_loss_pct=stop_loss,
        max_hold_days=max_hold,
        min_dte_at_entry=min_dte,
        min_vol_pct=min_vol,
        max_vol_pct=max_vol,
        entry_frequency=entry_freq,
        risk_free_rate=r,
        lot_size=lot_size,
    )

    with st.spinner("Running backtest…"):
        try:
            engine = BacktestEngine(prices, config, ticker=cfg.strip_suffix(ticker), period=period)
            result = engine.run()
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    if not result.trades:
        st.warning("No trades were generated. Try a longer period, different vol filter, or daily entry frequency.")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("Results summary")

    verdict_color = "#27ae60" if result.avg_pnl > 0 and result.win_rate >= 50 else \
                    "#f39c12" if result.avg_pnl > 0 else "#e74c3c"
    verdict_text  = "Positive expectation" if result.avg_pnl > 0 else "Negative expectation"

    st.markdown(f"""
<div style="border:2px solid {verdict_color};border-radius:12px;padding:16px 22px;margin-bottom:12px">
    <div style="font-size:12px;color:#888;margin-bottom:4px">
        {result.total_trades} trades · {period} · {long_short.upper()} {option_type.upper()} ·
        Strike {strike_offset:+.1f}% · DTE {dte_entry}
    </div>
    <div style="font-size:26px;font-weight:600;color:{verdict_color}">{verdict_text}</div>
</div>
""", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total trades",    result.total_trades)
    m2.metric("Win rate",        f"{result.win_rate:.1f}%")
    m3.metric("Avg P&L / unit",  f"{sym}{result.avg_pnl:+.2f}")
    m4.metric("Avg P&L %",       f"{result.avg_pnl_pct:+.1f}%")
    m5.metric("Profit factor",   f"{result.profit_factor:.2f}",
              help="Sum of wins / sum of losses. >1 = profitable overall.")
    m6.metric("Max drawdown",    f"{sym}{result.max_drawdown:.2f}",
              help="Largest peak-to-trough drop in cumulative P&L.")

    m7, m8, m9, m10 = st.columns(4)
    m7.metric("Best trade",      f"{sym}{result.best_trade:+.2f}")
    m8.metric("Worst trade",     f"{sym}{result.worst_trade:+.2f}")
    m9.metric("Avg hold days",   f"{result.avg_hold_days:.1f}")
    m10.metric("Total P&L",      f"{sym}{result.total_pnl:+.2f}",
               help="Sum of all P&L per unit. Multiply by lot size for actual exposure.")

    # Exit reason breakdown
    st.divider()
    er_col1, er_col2 = st.columns([1, 2])
    with er_col1:
        st.markdown("**Exit reason breakdown**")
        for reason, count in result.exit_reasons.items():
            pct = count / result.total_trades * 100
            icon = {"profit_target": "✅", "stop_loss": "🔴",
                    "max_hold": "⏱️", "expiry": "📅"}.get(reason, "•")
            st.markdown(f"{icon} **{reason.replace('_',' ').title()}**: {count} ({pct:.0f}%)")

    with er_col2:
        # Pie chart
        reasons = list(result.exit_reasons.keys())
        counts  = list(result.exit_reasons.values())
        colors  = {"profit_target": "#27ae60", "stop_loss": "#e74c3c",
                   "max_hold": "#f39c12", "expiry": "#3498db"}
        fig_pie = go.Figure(go.Pie(
            labels=[r.replace("_", " ").title() for r in reasons],
            values=counts,
            marker_colors=[colors.get(r, "#95a5a6") for r in reasons],
            hole=0.4,
        ))
        fig_pie.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                              showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Cumulative P&L chart ──────────────────────────────────────────────────
    st.subheader("Cumulative P&L over time")

    df = result.to_dataframe()
    df["Entry date"]  = pd.to_datetime(df["Entry date"])
    df["Cumulative P&L"] = df["P&L / unit"].cumsum()

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=df["Entry date"], y=df["Cumulative P&L"],
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(
            color=["#27ae60" if p > 0 else "#e74c3c" for p in df["P&L / unit"]],
            size=7,
        ),
        name="Cumulative P&L",
    ))
    fig_cum.add_hline(y=0, line_color="#888", line_width=1)
    fig_cum.update_layout(
        xaxis_title="Entry date",
        yaxis_title=f"Cumulative P&L per unit ({sym})",
        height=360, template="plotly_white",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── P&L distribution ──────────────────────────────────────────────────────
    st.subheader("P&L distribution")

    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["P&L / unit"],
            nbinsx=20,
            marker_color=[
                "#27ae60" if v > 0 else "#e74c3c"
                for v in df["P&L / unit"]
            ],
            marker_line_width=0.5,
            marker_line_color="white",
        ))
        fig_hist.add_vline(x=0, line_color="#888", line_width=1)
        fig_hist.add_vline(x=float(df["P&L / unit"].mean()), line_dash="dot",
                           line_color="#3498db", annotation_text=f"Mean {sym}{df['P&L / unit'].mean():.2f}")
        fig_hist.update_layout(
            title="P&L per trade",
            xaxis_title=f"P&L per unit ({sym})",
            yaxis_title="Count",
            height=300, template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with dist_col2:
        # Win/loss by vol regime
        df["Vol bucket"] = pd.cut(
            df["Sigma %"],
            bins=[0, 15, 25, 35, 100],
            labels=["<15%", "15-25%", "25-35%", ">35%"],
        )
        vol_summary = df.groupby("Vol bucket", observed=False)["P&L / unit"].agg(
            ["mean", "count"]
        ).reset_index()
        vol_summary.columns = ["Vol regime", "Avg P&L", "Trades"]

        fig_vol = go.Figure(go.Bar(
            x=vol_summary["Vol regime"],
            y=vol_summary["Avg P&L"],
            marker_color=["#27ae60" if v > 0 else "#e74c3c" for v in vol_summary["Avg P&L"]],
            text=[f"{sym}{v:.2f}\n({n} trades)" for v, n in zip(vol_summary["Avg P&L"], vol_summary["Trades"])],
            textposition="outside",
        ))
        fig_vol.add_hline(y=0, line_color="#888", line_width=1)
        fig_vol.update_layout(
            title="Avg P&L by vol regime",
            xaxis_title="21d vol at entry",
            yaxis_title=f"Avg P&L ({sym})",
            height=300, template="plotly_white",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── Underlying price + trade markers ─────────────────────────────────────
    st.subheader("Trades on price history")

    prices_df = pd.DataFrame({"Date": prices.index, "Close": prices.values})
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=prices_df["Date"], y=prices_df["Close"],
        mode="lines", name="Spot price",
        line=dict(color="#bdc3c7", width=1.5),
    ))

    wins  = df[df["P&L / unit"] > 0]
    losses= df[df["P&L / unit"] <= 0]

    # Strip tz from prices index for safe lookup
    prices_tz = prices.copy()
    if hasattr(prices_tz.index, 'tz') and prices_tz.index.tz is not None:
        prices_tz.index = prices_tz.index.tz_localize(None)
    prices_tz.index = pd.to_datetime(prices_tz.index)

    fig_price.add_trace(go.Scatter(
        x=wins["Entry date"], y=[float(prices_tz.asof(pd.Timestamp(d))) for d in wins["Entry date"]],
        mode="markers", name="Winning entry",
        marker=dict(color="#27ae60", size=9, symbol="triangle-up"),
    ))
    fig_price.add_trace(go.Scatter(
        x=losses["Entry date"], y=[float(prices_tz.asof(pd.Timestamp(d))) for d in losses["Entry date"]],
        mode="markers", name="Losing entry",
        marker=dict(color="#e74c3c", size=9, symbol="triangle-down"),
    ))
    fig_price.update_layout(
        xaxis_title="Date",
        yaxis_title=f"Price ({sym})",
        height=360, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Full trade log")

    display_df = df.copy()
    display_df["P&L / unit"] = display_df["P&L / unit"].apply(
        lambda x: f"{sym}{x:+.2f}"
    )
    display_df["P&L %"] = display_df["P&L %"].apply(lambda x: f"{x:+.1f}%")

    def highlight_pnl(row):
        val = row.get("P&L %", "")
        if isinstance(val, str) and val.startswith("+"):
            return ["background-color: #f0fff4"] * len(row)
        elif isinstance(val, str) and val.startswith("-"):
            return ["background-color: #fff5f5"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(highlight_pnl, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ Model-based backtest using Black-Scholes reconstructed prices. "
        "Option prices are theoretical — actual fills would differ due to bid-ask spread, "
        "IV skew, and liquidity. Past performance of this simulation does not predict future results."
    )
