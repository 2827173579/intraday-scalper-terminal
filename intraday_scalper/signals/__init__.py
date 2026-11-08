"""Signal engine and dashboard state contracts.

Public API used by ``main.py``::

    from intraday_scalper.signals import SignalEngine, EngineSnapshot

The engine is dependency injected and never places orders: every alert tells the
operator to trade manually.
"""

from __future__ import annotations

from .engine import SignalEngine
from .state import EngineSnapshot, PositionSnapshot, SymbolSnapshot

__all__ = [
    "SignalEngine",
    "EngineSnapshot",
    "PositionSnapshot",
    "SymbolSnapshot",
]
