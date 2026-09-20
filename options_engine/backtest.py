"""
Options Backtesting Engine
===========================
Model-based backtesting using reconstructed option prices.

Since historical option prices are not available from yfinance, we reconstruct
them using Black-Scholes with:
  - Historical closing prices (from yfinance)
  - Rolling 21-day historical volatility computed from those prices

This tests the DECISION LOGIC of a trade setup — not exact fills.
It answers: "If I had applied this exact trade rule over the past N months,
what would the distribution of outcomes have looked like?"

How it works:
  1. Fetch N months of daily closing prices for the ticker
  2. Compute 21d rolling vol at each date (used as the sigma proxy)
  3. On each eligible trading date, simulate entry:
     - Compute BS price for the specified option (type, strike offset, DTE)
     - Record entry: date, spot, strike, sigma, BS price, delta, theta
  4. Simulate exit based on rules:
     - Profit target hit (option price doubles / reaches X%)
     - Stop loss hit (option loses X% of value)
     - Hold period expires
  5. Record P&L for each trade
  6. Aggregate: win rate, avg P&L, max drawdown, Sharpe-like ratio

Limitations (stated honestly):
  - Uses BS-reconstructed prices, not actual historical option prices
  - No slippage, bid-ask spread, or liquidity constraints modelled
  - Vol used at entry = trailing 21d realised vol, not actual implied vol
  - Tests logic validity, not exact execution quality
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import datetime
import numpy as np
import pandas as pd

from options_engine.pricing.black_scholes import BlackScholesModel


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """All parameters that define a backtest run."""
    # Trade definition
    option_type:       str    = "put"      # "call" or "put"
    long_short:        str    = "long"     # "long" or "short"
    strike_offset_pct: float  = -2.0       # % OTM from spot (+ve = OTM call, -ve = OTM put)
    dte_at_entry:      int    = 14         # days to expiry when entering

    # Exit rules
    profit_target_pct: float  = 100.0      # exit when option gains X% (long) / loses X% (short)
    stop_loss_pct:     float  = 50.0       # exit when option loses X% (long) / gains X% (short)
    max_hold_days:     int    = 7          # max calendar days to hold before forced exit

    # Entry filter
    min_vol_pct:       float  = 0.0        # only enter if vol >= this (e.g. 15%)
    max_vol_pct:       float  = 100.0      # only enter if vol <= this (e.g. 60%)
    entry_frequency:   str    = "weekly"   # "daily" or "weekly" (avoid overlapping trades)
    min_dte_at_entry:  int    = 3          # skip entry if DTE to next expiry < this

    # Market params
    risk_free_rate:    float  = 0.065
    lot_size:          int    = 1          # units per trade (set to NSE lot for realistic sizing)


@dataclass
class TradeRecord:
    """Result of a single simulated trade."""
    entry_date:     datetime.date
    exit_date:      datetime.date
    spot_at_entry:  float
    spot_at_exit:   float
    strike:         float
    option_type:    str
    long_short:     str
    sigma_at_entry: float
    entry_price:    float
    exit_price:     float
    delta_at_entry: float
    theta_at_entry: float
    hold_days:      int
    exit_reason:    str           # "profit_target" | "stop_loss" | "expiry" | "max_hold"
    pnl_per_unit:   float
    pnl_pct:        float         # % of entry price


@dataclass
class BacktestResult:
    """Aggregated results from a full backtest run."""
    trades:           list[TradeRecord]
    ticker:           str
    period:           str
    config:           BacktestConfig

    # Summary stats
    total_trades:     int    = 0
    winning_trades:   int    = 0
    losing_trades:    int    = 0
    win_rate:         float  = 0.0
    avg_pnl:          float  = 0.0
    avg_pnl_pct:      float  = 0.0
    total_pnl:        float  = 0.0
    best_trade:       float  = 0.0
    worst_trade:      float  = 0.0
    avg_hold_days:    float  = 0.0
    profit_factor:    float  = 0.0   # sum of wins / sum of losses
    max_drawdown:     float  = 0.0   # max peak-to-trough in cumulative P&L

    # Exit reason breakdown
    exit_reasons:     dict   = field(default_factory=dict)

    def compute_stats(self):
        """Compute all summary stats from the trades list."""
        if not self.trades:
            return self

        self.total_trades  = len(self.trades)
        pnls               = [t.pnl_per_unit for t in self.trades]
        pnl_pcts           = [t.pnl_pct for t in self.trades]

        self.winning_trades = sum(1 for p in pnls if p > 0)
        self.losing_trades  = sum(1 for p in pnls if p <= 0)
        self.win_rate       = self.winning_trades / self.total_trades * 100
        self.avg_pnl        = float(np.mean(pnls))
        self.avg_pnl_pct    = float(np.mean(pnl_pcts))
        self.total_pnl      = float(np.sum(pnls))
        self.best_trade     = float(np.max(pnls))
        self.worst_trade    = float(np.min(pnls))
        self.avg_hold_days  = float(np.mean([t.hold_days for t in self.trades]))

        wins  = sum(p for p in pnls if p > 0)
        losses= abs(sum(p for p in pnls if p < 0))
        self.profit_factor  = wins / losses if losses > 0 else float('inf')

        # Max drawdown on cumulative P&L curve
        cum_pnl = np.cumsum(pnls)
        peak    = np.maximum.accumulate(cum_pnl)
        dd      = cum_pnl - peak
        self.max_drawdown   = float(dd.min())

        # Exit reason breakdown
        self.exit_reasons = {}
        for t in self.trades:
            self.exit_reasons[t.exit_reason] = self.exit_reasons.get(t.exit_reason, 0) + 1

        return self

    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "Entry date":    t.entry_date.isoformat(),
                "Exit date":     t.exit_date.isoformat(),
                "Hold days":     t.hold_days,
                "Spot entry":    round(t.spot_at_entry, 2),
                "Strike":        round(t.strike, 2),
                "Sigma %":       round(t.sigma_at_entry * 100, 1),
                "Entry price":   round(t.entry_price, 2),
                "Exit price":    round(t.exit_price, 2),
                "P&L / unit":    round(t.pnl_per_unit, 2),
                "P&L %":         round(t.pnl_pct, 1),
                "Delta":         round(t.delta_at_entry, 3),
                "Exit reason":   t.exit_reason,
            })
        return pd.DataFrame(rows)


# ── Core engine ───────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Runs a model-based options backtest on historical price data.

    Usage:
        engine = BacktestEngine(prices, config, risk_free_rate=0.065)
        result = engine.run()
    """

    def __init__(
        self,
        prices: pd.Series,
        config: BacktestConfig,
        ticker: str = "",
        period: str = "",
    ):
        # Strip timezone from index — yfinance returns tz-aware index
        # (Asia/Kolkata for NSE, UTC for US) which causes comparison errors.
        prices_clean = prices.dropna().sort_index()
        if hasattr(prices_clean.index, 'tz') and prices_clean.index.tz is not None:
            prices_clean.index = prices_clean.index.tz_localize(None)
        self.prices = prices_clean
        self.config = config
        self.ticker = ticker
        self.period = period

        if len(self.prices) < 30:
            raise ValueError("Need at least 30 days of price history to backtest.")

        # Pre-compute rolling 21d vol at every date
        log_ret   = np.log(self.prices / self.prices.shift(1)).dropna()
        self.vols = (log_ret.rolling(21).std() * np.sqrt(252)).dropna()

    def _snap_strike(self, spot: float) -> float:
        """
        Apply strike offset and snap to exchange-appropriate step.
        strike_offset_pct: negative = OTM put, positive = OTM call
        """
        raw = spot * (1 + self.config.strike_offset_pct / 100)
        # Snap to nearest 50 for typical NSE range, 5 for US
        step = 50 if spot > 1000 else 5
        return round(raw / step) * step

    def _bs_price_and_greeks(
        self, spot: float, strike: float, T: float, sigma: float
    ) -> tuple[float, float, float]:
        """Return (price, delta, theta) from BS."""
        bs = BlackScholesModel(
            S=spot, K=strike, T=T,
            r=self.config.risk_free_rate,
            sigma=sigma,
            option_type=self.config.option_type,
        )
        return bs.price(), bs.delta(), bs.theta()

    def _simulate_exit(
        self,
        entry_date: datetime.date,
        entry_price: float,
        strike: float,
        T_entry: float,
        sigma_entry: float,
    ) -> tuple[datetime.date, float, float, str]:
        """
        Simulate daily mark-to-market from entry until exit condition hit.
        Returns (exit_date, exit_spot, exit_price, exit_reason).
        """
        config  = self.config
        sign    = 1 if config.long_short == "long" else -1
        max_gain_pct = config.profit_target_pct
        max_loss_pct = config.stop_loss_pct

        # Get all price dates after entry
        future_dates = self.prices.index[self.prices.index > pd.Timestamp(entry_date)]

        for i, date in enumerate(future_dates):
            hold_days = i + 1
            T_remain  = max(T_entry - hold_days / 365.0, 1e-6)
            spot_now  = float(self.prices[date])

            # Use rolling vol at this date if available, else carry entry vol
            sigma_now = (
                float(self.vols[date])
                if date in self.vols.index and not np.isnan(self.vols[date])
                else sigma_entry
            )

            current_price, _, _ = self._bs_price_and_greeks(
                spot_now, strike, T_remain, sigma_now
            )

            # P&L % from entry
            pnl_pct = (current_price - entry_price) / entry_price * 100 * sign

            # Check exit conditions
            if sign == 1:   # long
                if pnl_pct >= max_gain_pct:
                    return date.date(), spot_now, current_price, "profit_target"
                if pnl_pct <= -max_loss_pct:
                    return date.date(), spot_now, current_price, "stop_loss"
            else:           # short
                if pnl_pct <= -max_gain_pct:   # short profits when option loses value
                    return date.date(), spot_now, current_price, "profit_target"
                if pnl_pct >= max_loss_pct:
                    return date.date(), spot_now, current_price, "stop_loss"

            if hold_days >= config.max_hold_days:
                return date.date(), spot_now, current_price, "max_hold"

            # Expiry check
            if T_remain <= 1 / 365:
                intrinsic = max(0.0, (spot_now - strike) if config.option_type == "call"
                               else (strike - spot_now))
                return date.date(), spot_now, intrinsic, "expiry"

        # No exit triggered — force close on last available date
        last_date  = future_dates[-1]
        last_spot  = float(self.prices[future_dates[-1]])
        T_last     = max(T_entry - len(future_dates) / 365.0, 1e-6)
        sigma_last = (float(self.vols[future_dates[-1]])
                      if future_dates[-1] in self.vols.index else sigma_entry)
        last_price, _, _ = self._bs_price_and_greeks(last_spot, strike, T_last, sigma_last)
        return last_date.date(), last_spot, last_price, "max_hold"

    def run(self) -> BacktestResult:
        """Execute the full backtest and return results."""
        config = self.config
        trades: list[TradeRecord] = []

        # Entry dates: skip first 21 days (vol warmup)
        eligible = self.vols.index.sort_values()

        # Weekly or daily entry frequency
        if config.entry_frequency == "weekly":
            # Enter on the first eligible trading day of each week
            weeks_seen: set = set()
            entry_dates = []
            for d in eligible:
                week_key = (d.year, d.isocalendar()[1])
                if week_key not in weeks_seen:
                    weeks_seen.add(week_key)
                    entry_dates.append(d)
        else:
            entry_dates = list(eligible)

        last_exit_date = None   # prevent overlapping trades

        for entry_ts in entry_dates:
            entry_date = entry_ts.date()

            # Skip if previous trade hasn't exited yet
            if last_exit_date and entry_date <= last_exit_date:
                continue

            spot  = float(self.prices[entry_ts])
            sigma = float(self.vols[entry_ts])

            # Vol filter
            if not (config.min_vol_pct / 100 <= sigma <= config.max_vol_pct / 100):
                continue

            # Strike
            strike = self._snap_strike(spot)

            # ── NIFTY Tuesday expiry logic ────────────────────────────────────
            # Find the next Tuesday from entry_date (NIFTY weekly options expire Tuesday)
            days_ahead = (1 - entry_date.weekday()) % 7   # 1 = Tuesday
            if days_ahead == 0:
                days_ahead = 7   # already Tuesday — use NEXT Tuesday
            next_expiry = entry_date + __import__('datetime').timedelta(days=days_ahead)
            dte_to_expiry = (next_expiry - entry_date).days

            # Skip if DTE is below minimum threshold
            if dte_to_expiry < config.min_dte_at_entry:
                continue

            T_entry = dte_to_expiry / 365.0

            # Entry price
            entry_price, delta, theta = self._bs_price_and_greeks(
                spot, strike, T_entry, sigma
            )

            if entry_price <= 0.01:
                continue   # skip worthless options

            # Simulate exit
            exit_date, exit_spot, exit_price, exit_reason = self._simulate_exit(
                entry_date, entry_price, strike, T_entry, sigma
            )

            hold_days = (exit_date - entry_date).days
            sign      = 1 if config.long_short == "long" else -1
            pnl       = sign * (exit_price - entry_price)
            pnl_pct   = pnl / entry_price * 100

            trades.append(TradeRecord(
                entry_date=entry_date,
                exit_date=exit_date,
                spot_at_entry=spot,
                spot_at_exit=exit_spot,
                strike=strike,
                option_type=config.option_type,
                long_short=config.long_short,
                sigma_at_entry=sigma,
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                delta_at_entry=round(delta, 3),
                theta_at_entry=round(theta, 3),
                hold_days=hold_days,
                exit_reason=exit_reason,
                pnl_per_unit=round(pnl, 2),
                pnl_pct=round(pnl_pct, 1),
            ))

            last_exit_date = exit_date

        result = BacktestResult(
            trades=trades,
            ticker=self.ticker,
            period=self.period,
            config=config,
        )
        return result.compute_stats()
