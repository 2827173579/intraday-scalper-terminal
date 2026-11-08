"""Strategy registry: build the enabled deterministic strategies.

The signal engine only needs :func:`build_strategies`; the build order is
stable and documented so the "first valid candidate wins" rule is
reproducible:

1. ``ORB_5M`` (if ``strategies.orb.enabled_5m``)
2. ``ORB_15M`` (if ``strategies.orb.enabled_15m``)
3. ``VWAP_RECLAIM`` (if ``strategies.vwap_reclaim.enabled``)
4. ``EMA_PULLBACK`` (if ``strategies.ema_pullback.enabled``)

All tuning comes from ``core.config`` (``config/default.yaml``); nothing is
hard-coded here.
"""

from __future__ import annotations

from intraday_scalper.core import StrategyConfig, StrategyName

from .base import Strategy
from .ema_pullback import EmaTrendPullback
from .orb import OpeningRangeBreakout
from .vwap_reclaim import VwapReclaim

__all__ = ["STRATEGY_ORDER", "build_strategies"]

#: Canonical evaluation order of the phase-1 strategies.
STRATEGY_ORDER: tuple[StrategyName, ...] = (
    StrategyName.ORB_5M,
    StrategyName.ORB_15M,
    StrategyName.VWAP_RECLAIM,
    StrategyName.EMA_PULLBACK,
)


def build_strategies(strategy_config: StrategyConfig) -> list[Strategy]:
    """Instantiate every enabled strategy in the canonical order."""
    if not isinstance(strategy_config, StrategyConfig):
        raise TypeError("build_strategies expects a StrategyConfig instance")

    strategies: list[Strategy] = []
    orb = strategy_config.orb
    if orb.enabled_5m:
        strategies.append(OpeningRangeBreakout(orb, timeframe_minutes=5))
    if orb.enabled_15m:
        strategies.append(OpeningRangeBreakout(orb, timeframe_minutes=15))
    if strategy_config.vwap_reclaim.enabled:
        strategies.append(VwapReclaim(strategy_config.vwap_reclaim))
    if strategy_config.ema_pullback.enabled:
        strategies.append(EmaTrendPullback(strategy_config.ema_pullback))
    return strategies
