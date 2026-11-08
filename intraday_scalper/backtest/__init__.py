"""Event-driven backtesting, performance metrics, walk-forward validation.

Public API
----------
* :class:`~intraday_scalper.backtest.engine.EventDrivenBacktester` — replays a
  time-sorted stream of completed 1m bars; signals confirmed on bar *i* fill at
  the open of bar *i+1*; same-bar TP+SL collisions fill the stop
  (``pessimism="stop_first"``); slippage and commissions are charged.
* :class:`~intraday_scalper.backtest.engine.BacktestResult` — trades, equity
  curve, :class:`BacktestMetrics`, notes.
* :func:`~intraday_scalper.backtest.metrics.compute_metrics` — pure metric
  calculation from closed trades + equity curve.
* :func:`~intraday_scalper.backtest.walkforward.walk_forward_splits` /
  :func:`~intraday_scalper.backtest.walkforward.run_walkforward` — chronological,
  leakage-checked out-of-sample folds.
* :mod:`intraday_scalper.backtest.synthetic` — deterministic **SYNTHETIC DATA
  ONLY** bar generators.

IMPORTANT DATA LIMITATION
-------------------------
The current Finnhub account has **no historical candle permission**
(``/stock/candle`` -> HTTP 403) and **no real-time US trade feed**, so real
historical data is unavailable.  Everything here runs on caller-supplied bars
(synthetic in the current phase).  Numbers produced by this package are logic
checks, **never** real-market backtest performance, and must not be presented
as such.  The Lead's report script must keep the "当前无真实历史数据" statement.
"""

from .engine import (
    DATA_CAVEAT,
    BacktestResult,
    EventDrivenBacktester,
    run_backtest,
    run_synthetic_backtest,
    validate_bars_1m,
)
from .metrics import BacktestMetrics, MetricGroup, compute_metrics, to_jsonable
from .risk_adapter import (
    DailyStats,
    EntryDecision,
    RiskState,
    SimulatedRiskManager,
    resolve_risk_manager,
    resolve_sizing_fn,
    size_position_equivalent,
)
from .synthetic import (
    SYNTHETIC_DATA_ONLY,
    generate_session,
    generate_sessions,
    make_choppy_session,
    make_trending_session,
    next_trading_dates,
)
from .walkforward import (
    Fold,
    FoldResult,
    WalkForwardResult,
    assert_no_leakage,
    run_walkforward,
    walk_forward_splits,
)

__all__ = [
    "DATA_CAVEAT",
    "SYNTHETIC_DATA_ONLY",
    "BacktestMetrics",
    "BacktestResult",
    "DailyStats",
    "EntryDecision",
    "EventDrivenBacktester",
    "Fold",
    "FoldResult",
    "MetricGroup",
    "RiskState",
    "SimulatedRiskManager",
    "WalkForwardResult",
    "assert_no_leakage",
    "compute_metrics",
    "generate_session",
    "generate_sessions",
    "make_choppy_session",
    "make_trending_session",
    "next_trading_dates",
    "resolve_risk_manager",
    "resolve_sizing_fn",
    "run_backtest",
    "run_synthetic_backtest",
    "run_walkforward",
    "size_position_equivalent",
    "to_jsonable",
    "validate_bars_1m",
    "walk_forward_splits",
]
