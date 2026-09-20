"""
Transaction cost calculator for Indian F&O.

Reads broker configuration from broker_config.json in the project root.
Falls back to Zerodha defaults if the file is missing or a field is absent.
All missing fields default to 0 — not to Zerodha rates — so the user always
sees the impact of each cost component they've explicitly configured.

To customise: edit broker_config.json in the project root.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from options_engine.market_config import MarketConfig


# ── Load broker config ────────────────────────────────────────────────────────

def _load_broker_config() -> dict:
    """
    Load broker_config.json from the project root.
    Walks up from this file's location to find it.
    """
    # Search up to 3 levels from this file
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for search_dir in [base, os.path.dirname(base)]:
        path = os.path.join(search_dir, "broker_config.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def _get(cfg: dict, key: str, default: float) -> float:
    """Get a value from config, returning default if missing or null."""
    val = cfg.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── Cost model ────────────────────────────────────────────────────────────────

@dataclass
class TransactionCosts:
    """
    Resolved transaction cost rates for one broker.
    Loaded from broker_config.json with 0 as the fallback for missing fields.
    Zerodha defaults are shown in broker_config.json as the reference.
    """
    broker_name:          str   = "Unknown"
    brokerage_flat_inr:   float = 0.0
    stt_rate_sell:        float = 0.0
    nse_txn_rate:         float = 0.0
    sebi_rate:            float = 0.0
    gst_rate:             float = 0.0
    stamp_duty_rate:      float = 0.0

    @classmethod
    def from_config(cls) -> "TransactionCosts":
        cfg = _load_broker_config()
        return cls(
            broker_name        = cfg.get("broker_name", "Custom"),
            brokerage_flat_inr = _get(cfg, "brokerage_flat_inr", 0.0),
            stt_rate_sell      = _get(cfg, "stt_rate_sell",       0.0),
            nse_txn_rate       = _get(cfg, "nse_txn_rate",        0.0),
            sebi_rate          = _get(cfg, "sebi_rate",           0.0),
            gst_rate           = _get(cfg, "gst_rate",            0.0),
            stamp_duty_rate    = _get(cfg, "stamp_duty_rate",     0.0),
        )


@dataclass
class LegCost:
    brokerage:    float = 0.0
    stt:          float = 0.0
    exchange_txn: float = 0.0
    sebi:         float = 0.0
    gst:          float = 0.0
    stamp_duty:   float = 0.0
    total:        float = 0.0
    currency_symbol: str = "₹"

    def breakdown(self) -> dict[str, float]:
        return {
            "Brokerage":        self.brokerage,
            "STT":              self.stt,
            "Exchange charges": self.exchange_txn,
            "SEBI charges":     self.sebi,
            "GST":              self.gst,
            "Stamp duty":       self.stamp_duty,
        }


@dataclass
class RoundTripCost:
    entry: LegCost
    exit:  LegCost
    total: float
    cost_as_pct_of_premium: float
    currency_symbol: str = "₹"


class CostCalculator:
    """
    Compute transaction costs for Indian F&O options.
    Reads from broker_config.json automatically.
    """

    def __init__(self, cfg: MarketConfig):
        self.cfg   = cfg
        self.costs = TransactionCosts.from_config()
        self.sym   = cfg.currency_symbol

    def _indian_leg(
        self, premium: float, lot_size: int, quantity: int, is_sell: bool
    ) -> LegCost:
        c        = self.costs
        turnover = premium * lot_size * quantity

        brokerage    = c.brokerage_flat_inr
        stt          = turnover * c.stt_rate_sell if is_sell else 0.0
        exchange_txn = turnover * c.nse_txn_rate
        sebi         = turnover * c.sebi_rate
        stamp        = turnover * c.stamp_duty_rate if not is_sell else 0.0
        gst          = (brokerage + exchange_txn + sebi) * c.gst_rate
        total        = brokerage + stt + exchange_txn + sebi + stamp + gst

        return LegCost(
            brokerage=brokerage, stt=stt, exchange_txn=exchange_txn,
            sebi=sebi, gst=gst, stamp_duty=stamp,
            total=total, currency_symbol=self.sym,
        )

    def leg_cost(
        self, premium: float, lot_size: int, quantity: int, is_sell: bool
    ) -> LegCost:
        return self._indian_leg(premium, lot_size, quantity, is_sell)

    def round_trip(
        self, premium: float, lot_size: int, quantity: int = 1
    ) -> RoundTripCost:
        entry = self.leg_cost(premium, lot_size, quantity, is_sell=False)
        exit_ = self.leg_cost(premium, lot_size, quantity, is_sell=True)
        total = entry.total + exit_.total
        pct   = (total / (premium * lot_size * quantity) * 100) if premium > 0 else 0.0
        return RoundTripCost(
            entry=entry, exit=exit_,
            total=total,
            cost_as_pct_of_premium=pct,
            currency_symbol=self.sym,
        )

    def multi_leg_cost(self, legs: list[dict]) -> dict:
        entry_total, exit_total, itemised = 0.0, 0.0, []
        for leg in legs:
            rt = self.round_trip(
                premium=leg["premium"],
                lot_size=leg.get("lot_size", self.cfg.default_lot_size),
                quantity=abs(leg.get("quantity", 1)),
            )
            entry_total += rt.entry.total
            exit_total  += rt.exit.total
            itemised.append(rt)
        return {
            "entry_total": entry_total,
            "exit_total":  exit_total,
            "grand_total": entry_total + exit_total,
            "itemised":    itemised,
        }

    @property
    def broker_name(self) -> str:
        return self.costs.broker_name
