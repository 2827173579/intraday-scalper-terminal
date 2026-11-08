"""Risk control: position sizing and the manual-trade state machine.

Public API::

    from intraday_scalper.risk import size_position, describe_plan
    from intraday_scalper.risk import RiskManager, RiskState, EntryDecision, DailyStats

Nothing in this package places orders or talks to a broker.
"""

from .manager import DailyStats, EntryDecision, RiskManager, RiskState
from .sizing import describe_plan, size_position

__all__ = [
    "DailyStats",
    "EntryDecision",
    "RiskManager",
    "RiskState",
    "describe_plan",
    "size_position",
]
