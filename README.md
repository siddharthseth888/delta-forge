# ⚡ DeltaForge

**High-Precision Options Valuation, Greeks Analytics & Systematic Backtesting Suite for Indian Derivatives (NSE/BSE)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

DeltaForge is a quantitative derivatives valuation and risk management engine designed specifically for the Indian equity derivatives ecosystem. Built with production-grade numerical methods, it bridges institutional option pricing theory with real-world Indian exchange mechanics (STT taxation, lot sizes, RBI repo rates, and weekly expiry cycles).

---

## 🎯 Key Modules

### 1. Dual-Model Quantitative Pricer
- **Closed-Form Black-Scholes**: Fast analytical valuation for European-style contracts with continuous dividend yields.
- **Cox-Ross-Rubinstein (CRR) Binomial Lattice**: 150-step numerical tree capturing American early exercise premiums and discrete dividend timings.
- **Surface Sensitivity Analysis**: Interactive 2D/3D visualizations of price, delta, gamma, and theta across varying spot trajectories and volatility levels.

### 2. Multi-Leg Strategy Studio & Payoff Modeler
- **Structured Strategy Templates**: Instant construction of Long/Short Straddles, Strangles, Bull/Bear Spreads, and Iron Condors.
- **Position Greeks Aggregation**: Net exposure calculation across heterogeneous strikes, maturities, and directional biases.
- **Indian Statutory Cost Engine**: Real-world P&L accounting factoring in Securities Transaction Tax (STT), NSE transaction levies, SEBI charges, Stamp Duty, and 18% GST.

### 3. Five-Factor Trade Diagnostic Engine
- Evaluates individual trade propositions against theoretical fair value:
  - **Breakeven Reachability**: Probabilistic assessment of required underlying movement.
  - **Theta Decay Resistance**: Quantifying time decay drag relative to expected holding windows.
  - **Mispricing Alpha**: Divergence between quoted market premium and theoretical model pricing.
  - **Delta Sensitivity Quality**: Directional efficiency and leverage ratio.
  - **DTE Risk Assessment**: Convexity behavior approaching expiration.

### 4. Regime-Aware Historical Backtester
- Model-based historical simulation framework evaluating systematic entry/exit criteria over time-series data.
- Accounts for standard Indian weekly expiry schedules (e.g. NIFTY Tuesday/Thursday expiries).
- Performance attribution partitioned by volatility regimes (Low, Medium, High Volatility).
- Comprehensive metrics: Win Rate, Profit Factor, Max Drawdown, and Exit Reason distribution.

---

## 📐 Mathematical Foundations

### Black-Scholes Pricing & Analytical Greeks
For European calls ($C$) and puts ($P$):

$$d_1 = \frac{\ln(S / K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}$$

$$d_2 = d_1 - \sigma \sqrt{T}$$

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

### Numerical Implied Volatility Inversion
Implied volatility is extracted via a robust **Brent's method root-finder** minimizing $|V_{\text{market}} - V_{\text{model}}(\sigma)|$ down to a convergence tolerance of $10^{-6}$.

### IST Precise Maturity Calculation
Maturity ($T$) is computed using microsecond-precision Indian Standard Time (IST) arithmetic anchored to market close (15:30 IST) on expiry day, avoiding single-day rounding distortions common in naive date math.

---

## 🏗️ System Architecture

```
deltaforge/
├── app.py                        # Streamlit quantitative dashboard entry
├── broker_config.json            # Brokerage & statutory tax schedule (Zerodha/NSE)
├── requirements.txt              # Core mathematical and UI dependencies
├── README.md                     # Technical documentation & usage guide
├── options_engine/               # Core quantitative engine
│   ├── __init__.py               # Engine facade & pipeline orchestration
│   ├── market_config.py          # Indian exchange specifications (lots, symbols, rates)
│   ├── costs.py                  # Statutory fee & execution friction calculator
│   ├── backtest.py               # Systematic backtesting & simulation engine
│   ├── data/
│   │   ├── base.py               # Abstract market data provider interface
│   │   └── yfinance_adapter.py   # Historical and live option chain pipeline
│   ├── pricing/
│   │   ├── black_scholes.py      # Closed-form pricing & closed-form Greeks
│   │   └── binomial_tree.py      # CRR 150-step lattice with early exercise premium
│   ├── greeks/
│   │   └── calculator.py         # Option chain Greeks enrichment & portfolio aggregation
│   └── utils/
│       └── volatility.py         # Realised volatility, IV solver & skew surface
└── pages_impl/                   # Presentation and analytics UI layer
    ├── shared.py                 # Common charting widgets & caching utilities
    ├── single_contract.py        # Single instrument pricer & sensitivity curves
    ├── strategy_builder.py       # Multi-leg payoff matrix & transaction fee analysis
    ├── trade_advisor.py          # Five-pillar trade evaluator & diagnostic scores
    └── backtester.py             # Historical performance engine & regime analytics
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/siddharthseth888/deltaforge.git
cd deltaforge
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Engine
```bash
python -m streamlit run app.py
```

---

## ⚙️ Configuration & Market Customization

- **Execution Fee Parameters**: Edit `broker_config.json` to reflect your broker's specific fee schedules (defaults set to Zerodha equity derivative rates).
- **Benchmark Risk-Free Rate**: Configured in `options_engine/market_config.py` (aligned with the latest Reserve Bank of India Repo Rate).

---

## 🛡️ License & Disclaimer

Distributed under the MIT License. See `LICENSE` for details.

*Disclaimer: DeltaForge is developed strictly for quantitative research, backtesting, and educational exploration. It does not constitute financial or investment advice.*
