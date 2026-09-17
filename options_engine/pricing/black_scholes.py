"""
Black-Scholes option pricing model.

Supports European calls and puts with analytical Greeks.
For American options or dividend-paying stocks, use BinomialTreeModel.

All inputs are validated at construction time with clear error messages.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass
class BSResult:
    """All outputs from a single Black-Scholes evaluation."""
    price:       float
    delta:       float
    gamma:       float
    theta:       float       # per calendar day
    vega:        float       # per 1% move in vol
    rho:         float       # per 1% move in rates
    S:           float
    K:           float
    T:           float
    r:           float
    sigma:       float
    option_type: str

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class BlackScholesModel:
    """
    Closed-form Black-Scholes pricer + analytical Greeks.

    Parameters
    ----------
    S           : float  — current spot price  (must be > 0)
    K           : float  — strike price         (must be > 0)
    T           : float  — time to expiry in years (>= 0; 0 → intrinsic value)
    r           : float  — annualised risk-free rate (e.g. 0.065 for 6.5%)
    sigma       : float  — annualised volatility (>= 0; 0 → forward price model)
    option_type : str    — "call" or "put"
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
    ):
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if S <= 0:
            raise ValueError(f"Spot price S must be > 0, got {S}")
        if K <= 0:
            raise ValueError(f"Strike K must be > 0, got {K}")
        if T < 0:
            raise ValueError(f"Time T must be >= 0, got {T}")
        if sigma < 0:
            raise ValueError(f"Volatility sigma must be >= 0, got {sigma}")

        self.S           = float(S)
        self.K           = float(K)
        self.T           = float(T)
        self.r           = float(r)
        self.sigma       = float(sigma)
        self.option_type = option_type

        self._expired   = T == 0
        self._zero_vol  = sigma == 0
        self._d1, self._d2 = self._compute_d1_d2()

    def _compute_d1_d2(self) -> tuple[float, float]:
        if self._expired or self._zero_vol:
            return 0.0, 0.0
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        return d1, d1 - self.sigma * np.sqrt(self.T)

    # ── Intrinsic value ───────────────────────────────────────────────────────

    def intrinsic(self) -> float:
        if self.option_type == "call":
            return max(0.0, self.S - self.K)
        return max(0.0, self.K - self.S)

    def forward_value(self) -> float:
        """Discounted forward price: max(0, S - K·e^-rT) for calls."""
        if self.option_type == "call":
            return max(0.0, self.S - self.K * np.exp(-self.r * self.T))
        return max(0.0, self.K * np.exp(-self.r * self.T) - self.S)

    # ── Price ─────────────────────────────────────────────────────────────────

    def price(self) -> float:
        """Theoretical option price."""
        if self._expired:
            return self.intrinsic()
        if self._zero_vol:
            # With zero vol, option is worth its discounted forward payoff
            return self.forward_value()
        S, K, T, r = self.S, self.K, self.T, self.r
        d1, d2     = self._d1, self._d2
        if self.option_type == "call":
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    # ── Greeks ────────────────────────────────────────────────────────────────

    def delta(self) -> float:
        """
        Rate of change of price w.r.t. spot.
        Call delta ∈ (0, 1). Put delta ∈ (-1, 0).
        """
        if self._expired:
            if self.option_type == "call":
                return 1.0 if self.S > self.K else (0.5 if self.S == self.K else 0.0)
            return -1.0 if self.S < self.K else (-0.5 if self.S == self.K else 0.0)
        if self._zero_vol:
            if self.option_type == "call":
                return 1.0 if self.S > self.K * np.exp(-self.r * self.T) else 0.0
            return -1.0 if self.S < self.K * np.exp(-self.r * self.T) else 0.0
        if self.option_type == "call":
            return float(norm.cdf(self._d1))
        return float(norm.cdf(self._d1) - 1)

    def gamma(self) -> float:
        """
        Rate of change of delta w.r.t. spot. Always >= 0 for long options.
        Highest near ATM and near expiry.
        """
        if self._expired or self._zero_vol:
            return 0.0
        return float(norm.pdf(self._d1) / (self.S * self.sigma * np.sqrt(self.T)))

    def theta(self) -> float:
        """
        Time decay per calendar day. Almost always negative for long options.
        """
        if self._expired or self._zero_vol:
            return 0.0
        S, K, T, r, sigma = self.S, self.K, self.T, self.r, self.sigma
        d1, d2 = self._d1, self._d2
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if self.option_type == "call":
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        return float((term1 + term2) / 365)

    def vega(self) -> float:
        """
        Sensitivity to a 1% absolute change in implied volatility.
        Always >= 0 for long options.
        """
        if self._expired or self._zero_vol:
            return 0.0
        return float(self.S * norm.pdf(self._d1) * np.sqrt(self.T)) / 100

    def rho(self) -> float:
        """
        Sensitivity to a 1% absolute change in risk-free rate.
        """
        if self._expired:
            return 0.0
        K, T, r, d2 = self.K, self.T, self.r, self._d2
        if self.option_type == "call":
            return float(K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
        return float(-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

    def evaluate(self) -> BSResult:
        """Return all prices and Greeks in one call."""
        return BSResult(
            price=self.price(),
            delta=self.delta(),
            gamma=self.gamma(),
            theta=self.theta(),
            vega=self.vega(),
            rho=self.rho(),
            S=self.S, K=self.K, T=self.T,
            r=self.r, sigma=self.sigma,
            option_type=self.option_type,
        )
