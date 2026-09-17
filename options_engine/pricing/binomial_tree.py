"""
Cox-Ross-Rubinstein (CRR) Binomial Tree pricer.

Supports:
  - European and American options (calls and puts)
  - Continuous dividend yield (q)
  - Early exercise detection at every node

Why use this instead of Black-Scholes:
  - American puts can be optimally exercised early, especially with dividends
  - Black-Scholes has no closed form for American options
  - The tree naturally handles early exercise by checking intrinsic value at each node

Numerical Greeks are computed via finite differences (bumping inputs), which is
slower than analytical Greeks but works correctly for both European and American
exercise styles.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BinomialResult:
    price:                  float
    delta:                  float
    gamma:                  float
    theta:                  float          # per calendar day
    early_exercise_premium: float          # American price − European price (≥ 0)
    steps:                  int
    option_type:            str
    exercise_style:         str


class BinomialTreeModel:
    """
    CRR binomial tree pricer.

    Parameters
    ----------
    S              : spot price              (must be > 0)
    K              : strike                  (must be > 0)
    T              : time to expiry (years)  (>= 0)
    r              : annualised risk-free rate
    sigma          : annualised volatility   (must be > 0 for meaningful results)
    option_type    : "call" or "put"
    exercise_style : "european" or "american"
    q              : continuous dividend yield (default 0)
    steps          : tree depth — 200 is a good balance of speed/accuracy
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        exercise_style: str = "american",
        q: float = 0.0,
        steps: int = 200,
    ):
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if exercise_style not in ("european", "american"):
            raise ValueError(f"exercise_style must be 'european' or 'american', got '{exercise_style}'")
        if S <= 0:
            raise ValueError(f"Spot S must be > 0, got {S}")
        if K <= 0:
            raise ValueError(f"Strike K must be > 0, got {K}")
        if T < 0:
            raise ValueError(f"T must be >= 0, got {T}")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")

        self.S              = float(S)
        self.K              = float(K)
        self.T              = float(T)
        self.r              = float(r)
        self.sigma          = float(max(sigma, 1e-8))   # guard against exact zero
        self.option_type    = option_type
        self.exercise_style = exercise_style
        self.q              = float(q)
        self.steps          = int(steps)

    def _intrinsic(self, S_node: float) -> float:
        if self.option_type == "call":
            return max(0.0, S_node - self.K)
        return max(0.0, self.K - S_node)

    def _run_tree(self, S: float, T: float, steps: int) -> float:
        """Core backward induction. Separated so Greeks can bump cleanly."""
        if T <= 0:
            return self._intrinsic(S)

        N        = steps
        dt       = T / N
        u        = np.exp(self.sigma * np.sqrt(dt))
        d        = 1.0 / u
        discount = np.exp(-self.r * dt)
        p        = np.clip(
            (np.exp((self.r - self.q) * dt) - d) / (u - d), 0.0, 1.0
        )

        # Terminal payoffs
        j    = np.arange(N + 1)
        S_T  = S * (u ** (N - j)) * (d ** j)
        V    = np.maximum(0.0, (S_T - self.K) if self.option_type == "call" else (self.K - S_T))

        # Backward induction
        for i in range(N - 1, -1, -1):
            V = discount * (p * V[:-1] + (1 - p) * V[1:])
            if self.exercise_style == "american":
                j_i    = np.arange(i + 1)
                S_i    = S * (u ** (i - j_i)) * (d ** j_i)
                intrinsic = np.maximum(
                    0.0,
                    (S_i - self.K) if self.option_type == "call" else (self.K - S_i),
                )
                V = np.maximum(V, intrinsic)

        return float(V[0])

    def price(self) -> float:
        return self._run_tree(self.S, self.T, self.steps)

    def delta(self) -> float:
        """Central finite difference on spot (1% bump)."""
        h = max(self.S * 0.01, 0.01)
        return (self._run_tree(self.S + h, self.T, self.steps) -
                self._run_tree(self.S - h, self.T, self.steps)) / (2 * h)

    def gamma(self) -> float:
        """Second-order finite difference on spot."""
        h = max(self.S * 0.01, 0.01)
        return (self._run_tree(self.S + h, self.T, self.steps)
                - 2 * self.price()
                + self._run_tree(self.S - h, self.T, self.steps)) / (h ** 2)

    def theta(self) -> float:
        """
        Per-calendar-day theta via time bump.
        Returns 0 if less than 1 day remains.
        """
        dt_day = 1.0 / 365.0
        if self.T <= dt_day:
            return 0.0
        return self._run_tree(self.S, self.T - dt_day, self.steps) - self.price()

    def early_exercise_premium(self) -> float:
        """
        American price − European price.
        Always >= 0 by no-arbitrage.
        Returns 0 for European style (by definition).
        """
        if self.exercise_style == "european":
            return 0.0
        european_price = BinomialTreeModel(
            S=self.S, K=self.K, T=self.T, r=self.r,
            sigma=self.sigma, option_type=self.option_type,
            exercise_style="european", q=self.q, steps=self.steps,
        ).price()
        return max(0.0, self.price() - european_price)

    def evaluate(self) -> BinomialResult:
        return BinomialResult(
            price=self.price(),
            delta=self.delta(),
            gamma=self.gamma(),
            theta=self.theta(),
            early_exercise_premium=self.early_exercise_premium(),
            steps=self.steps,
            option_type=self.option_type,
            exercise_style=self.exercise_style,
        )
