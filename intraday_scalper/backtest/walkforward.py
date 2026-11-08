"""Walk-forward validation for the event-driven backtester.

The purpose of walk-forward analysis is to make it much harder to fool
yourself: parameters are chosen on a *training* window and then measured on an
immediately following, never-before-seen *test* window.  The split logic here is
deliberately strict:

* dates are sorted and de-duplicated;
* every training window is entirely **before** its test window;
* test windows of consecutive folds never overlap (``step_days >= test_days``);
* :func:`assert_no_leakage` re-checks the actual bar timestamps of every fold
  and raises ``AssertionError`` if a training bar is not strictly older than
  every test bar.

Deterministic strategies (phase 1) have no fitted parameters, so
``strategy_factory`` is normally a zero-argument factory returning fresh,
stateless strategy instances.  If the factory accepts one positional argument
it is called with the fold's training bars, which is the hook a future
parameter-optimising version would use ("fit on train, evaluate on test").

Current data limitation: Finnhub free-tier candles are unavailable (HTTP 403)
and there is no real-time US trade feed, so walk-forward runs can only use
synthetic or locally recorded bars.  Small samples therefore return
``WALK-FORWARD INCOMPLETE`` notes instead of pretending to be statistically
meaningful.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping, Sequence

from intraday_scalper.core import AppConfig, Bar, et_date

from .engine import (
    DATA_CAVEAT,
    BacktestResult,
    EventDrivenBacktester,
    validate_bars_1m,
)
from .metrics import BacktestMetrics, compute_metrics, to_jsonable

__all__ = [
    "Fold",
    "FoldResult",
    "WalkForwardResult",
    "assert_no_leakage",
    "run_walkforward",
    "walk_forward_splits",
]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward fold: a train window immediately followed by a test window."""

    index: int
    train_dates: tuple[date, ...]
    test_dates: tuple[date, ...]

    @property
    def train_start(self) -> date | None:
        return self.train_dates[0] if self.train_dates else None

    @property
    def train_end(self) -> date | None:
        return self.train_dates[-1] if self.train_dates else None

    @property
    def test_start(self) -> date | None:
        return self.test_dates[0] if self.test_dates else None

    @property
    def test_end(self) -> date | None:
        return self.test_dates[-1] if self.test_dates else None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def walk_forward_splits(
    dates: Sequence[date],
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[Fold]:
    """Build chronological, non-overlapping train/test folds.

    Parameters
    ----------
    dates:
        Session dates (any order; duplicates are removed, result is sorted).
    train_days:
        Number of dates per training window (must be positive).
    test_days:
        Number of dates per test window (must be positive).
    step_days:
        How many dates to advance between folds.  Defaults to ``test_days``.
        Must be ``>= test_days`` so test windows never overlap.

    Returns
    -------
    list[Fold]
        Empty when there are not enough dates for even one full fold.
    """

    train_days = _require_positive_int(train_days, "train_days")
    test_days = _require_positive_int(test_days, "test_days")
    step = test_days if step_days is None else _require_positive_int(step_days, "step_days")
    if step < test_days:
        raise ValueError(
            "step_days must be >= test_days so test windows do not overlap "
            f"(step_days={step}, test_days={test_days})"
        )
    unique_dates: list[date] = []
    seen: set[date] = set()
    for item in dates:
        if not isinstance(item, date) or isinstance(item, type(None)):
            raise TypeError(f"dates must contain datetime.date, got {type(item)!r}")
        if item in seen:
            continue
        seen.add(item)
        unique_dates.append(item)
    unique_dates.sort()

    window = train_days + test_days
    if len(unique_dates) < window:
        return []
    folds: list[Fold] = []
    for start in range(0, len(unique_dates) - window + 1, step):
        train = tuple(unique_dates[start : start + train_days])
        test = tuple(unique_dates[start + train_days : start + window])
        folds.append(Fold(index=len(folds), train_dates=train, test_dates=test))
    return folds


# --------------------------------------------------------------------------- #
# Leakage check
# --------------------------------------------------------------------------- #
def assert_no_leakage(train_bars: Sequence[Bar], test_bars: Sequence[Bar]) -> None:
    """Raise ``AssertionError`` if any training bar is not strictly older.

    The check is on the real bar timestamps (``start_utc``) as well as on the
    exchange-local session dates, so a fold whose "training" data accidentally
    includes test-window bars is rejected instead of producing a flattering
    out-of-sample curve.
    """

    if train_bars and test_bars:
        train_max = max(bar.start_utc for bar in train_bars)
        test_min = min(bar.start_utc for bar in test_bars)
        assert train_max < test_min, (
            "LEAKAGE: training data overlaps the test window "
            f"(train_max={train_max.isoformat()} >= test_min={test_min.isoformat()})"
        )
    train_dates = {et_date(bar.start_utc) for bar in train_bars}
    test_dates = {et_date(bar.start_utc) for bar in test_bars}
    overlap = train_dates & test_dates
    assert not overlap, f"LEAKAGE: training and test share session dates {sorted(overlap)}"


# --------------------------------------------------------------------------- #
# Fold execution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FoldResult:
    """Results of one fold: train metrics (in-sample) + test metrics (out-of-sample)."""

    fold: Fold
    train_result: BacktestResult
    test_result: BacktestResult
    leakage_checked: bool = True

    @property
    def train_metrics(self) -> BacktestMetrics:
        return self.train_result.metrics

    @property
    def test_metrics(self) -> BacktestMetrics:
        return self.test_result.metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold.to_dict(),
            "train_metrics": self.train_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "train_trades": self.train_result.total_trades,
            "test_trades": self.test_result.total_trades,
            "leakage_checked": self.leakage_checked,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """All folds plus an aggregate of the out-of-sample (test) performance."""

    folds: tuple[FoldResult, ...]
    summary: Mapping[str, Any]
    aggregate_test_metrics: BacktestMetrics
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "summary": to_jsonable(self.summary),
            "aggregate_test_metrics": self.aggregate_test_metrics.to_dict(),
            "notes": list(self.notes),
        }


def _call_strategy_factory(factory: Any, train_bars: Sequence[Bar]) -> list[Any]:
    """Call a strategy factory, supporting zero-arg and one-arg (train bars) forms."""

    if not callable(factory):
        strategies = list(factory)
        if not strategies:
            raise ValueError("strategy_factory produced no strategies")
        return strategies

    try:
        signature = inspect.signature(factory)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):  # pragma: no cover - builtins / exotic callables
        positional = []
    produced = factory(train_bars) if positional else factory()
    strategies = list(produced) if produced is not None else []
    if not strategies:
        raise ValueError("strategy_factory produced no strategies")
    return strategies


def _bars_for_dates(
    grouped: Mapping[date, list[Bar]], target_dates: Sequence[date]
) -> list[Bar]:
    bars: list[Bar] = []
    for target in target_dates:
        bars.extend(grouped.get(target, []))
    return bars


def run_walkforward(
    bars_1m: Sequence[Bar],
    config: AppConfig,
    strategy_factory: Any,
    *,
    train_days: int = 10,
    test_days: int = 5,
    step_days: int | None = None,
    initial_equity: float | None = None,
    risk_factory: Callable[[], Any] | None = None,
    backtester_factory: Callable[[AppConfig, list[Any]], EventDrivenBacktester] | None = None,
    engine_kwargs: Mapping[str, Any] | None = None,
) -> WalkForwardResult:
    """Run a strict walk-forward evaluation over ``bars_1m``.

    Each fold runs the engine twice with **fresh instances**: once on the
    training window (in-sample, diagnostic only) and once on the following test
    window (out-of-sample).  Every fold starts from the same
    ``initial_equity``; P&L is *not* compounded across folds.

    ``risk_factory`` should be a zero-argument callable returning a *fresh*
    risk manager per run; when ``None`` the engine resolves one itself (also
    fresh per run).  Passing a stateful risk manager instance directly is not
    supported on purpose, because state must never leak between folds.
    """

    bars = validate_bars_1m(bars_1m)
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    equity = float(config.risk.account_equity if initial_equity is None else initial_equity)
    if equity <= 0:
        raise ValueError("initial_equity must be > 0")

    grouped: dict[date, list[Bar]] = {}
    for bar in bars:
        grouped.setdefault(et_date(bar.start_utc), []).append(bar)
    dates = sorted(grouped)
    splits = walk_forward_splits(dates, train_days, test_days, step_days)

    notes: list[str] = []
    if not splits:
        notes.append(
            "WALK-FORWARD INCOMPLETE: not enough session dates for one full fold "
            f"(have {len(dates)}, need train_days={train_days} + test_days={test_days}). "
            "Import/provide more history (there is currently no real historical data "
            "for this account) before drawing any out-of-sample conclusion."
        )
        empty = compute_metrics([], [], initial_equity=equity)
        return WalkForwardResult(
            folds=(),
            summary={
                "folds": 0,
                "train_days": int(train_days),
                "test_days": int(test_days),
                "step_days": int(test_days if step_days is None else step_days),
                "session_dates": len(dates),
                "test_trades": 0,
            },
            aggregate_test_metrics=empty,
            notes=tuple(notes),
        )

    engine_options = dict(engine_kwargs or {})
    if "risk" in engine_options:
        raise ValueError(
            "pass risk_factory instead of engine_kwargs['risk'] so each fold gets fresh state"
        )

    fold_results: list[FoldResult] = []
    all_test_trades = []
    for fold in splits:
        train_bars = _bars_for_dates(grouped, fold.train_dates)
        test_bars = _bars_for_dates(grouped, fold.test_dates)
        assert_no_leakage(train_bars, test_bars)
        if not train_bars or not test_bars:
            notes.append(
                f"fold {fold.index}: missing bars (train={len(train_bars)}, test={len(test_bars)}); "
                "result is incomplete"
            )

        def _make_engine(strategies: list[Any]) -> EventDrivenBacktester:
            if backtester_factory is not None:
                return backtester_factory(config, strategies)
            risk = risk_factory() if risk_factory is not None else None
            return EventDrivenBacktester(
                config,
                strategies,
                initial_equity=equity,
                risk=risk,
                **engine_options,
            )

        train_strategies = _call_strategy_factory(strategy_factory, train_bars)
        test_strategies = _call_strategy_factory(strategy_factory, train_bars)
        train_result = _make_engine(train_strategies).run(train_bars)
        test_result = _make_engine(test_strategies).run(test_bars)
        all_test_trades.extend(test_result.trades)
        fold_results.append(
            FoldResult(fold=fold, train_result=train_result, test_result=test_result)
        )

    aggregate = compute_metrics(all_test_trades, [], initial_equity=equity)
    test_trades = [trade for fold in fold_results for trade in fold.test_result.trades]
    test_wins = sum(1 for trade in test_trades if trade.pnl > 0.0)
    test_net = sum(trade.pnl for trade in test_trades)
    drawdowns = [fold.test_metrics.max_drawdown_pct for fold in fold_results]
    summary = {
        "folds": len(fold_results),
        "train_days": int(train_days),
        "test_days": int(test_days),
        "step_days": int(test_days if step_days is None else step_days),
        "session_dates": len(dates),
        "total_train_trades": sum(fold.train_result.total_trades for fold in fold_results),
        "total_test_trades": len(test_trades),
        "test_wins": test_wins,
        "test_win_rate": (test_wins / len(test_trades)) if test_trades else 0.0,
        "test_net_pnl_dollars": test_net,
        "test_expectancy_dollars": (test_net / len(test_trades)) if test_trades else 0.0,
        "avg_test_max_drawdown_pct": (
            sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
        ),
        "test_days_total": sum(len(fold.fold.test_dates) for fold in fold_results),
    }
    if not test_trades:
        notes.append(
            "WALK-FORWARD INCOMPLETE: the test folds produced no trades; the result is "
            "descriptive only and cannot support any statistical claim"
        )
    notes.append(
        "each fold starts from the same initial equity (no compounding across folds); "
        "train metrics are in-sample diagnostics, only test metrics are out-of-sample"
    )
    notes.append(DATA_CAVEAT)
    return WalkForwardResult(
        folds=tuple(fold_results),
        summary=summary,
        aggregate_test_metrics=aggregate,
        notes=tuple(notes),
    )
