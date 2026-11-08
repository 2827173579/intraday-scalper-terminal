"""Deterministic phase-1 strategies for SOXL / SOXS (long-only ETF trades).

Public API used by the signal engine::

    from intraday_scalper.strategies import build_strategies
    strategies = build_strategies(config.strategies)   # ordered list[Strategy]

Each strategy exposes ``name: StrategyName`` and
``evaluate(StrategyContext) -> SignalCandidate | None``. The context is defined
in :mod:`intraday_scalper.strategies.base` and is built by the signal engine
from confirmed 1m bars only.
"""

from .base import Strategy, StrategyContext
from .ema_pullback import EmaTrendPullback
from .orb import OpeningRangeBreakout
from .registry import STRATEGY_ORDER, build_strategies
from .vwap_reclaim import VwapReclaim

__all__ = [
    "STRATEGY_ORDER",
    "EmaTrendPullback",
    "OpeningRangeBreakout",
    "Strategy",
    "StrategyContext",
    "VwapReclaim",
    "build_strategies",
]
