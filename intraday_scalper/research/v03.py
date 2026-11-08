"""V0.3 structural ablation research (built on top of the frozen V0.2 module).

V0.2 (``research/v02.py``) reduced the 1m noise problem but still lost money:
the full period was -$385.40 and the 29-fold walk-forward OOS was -$852.52.
V0.3 implements **four a-priori structural changes** (no parameter tuning) and
measures them as a *cumulative* ablation:

===========  ==================================================================
Step 0       V0.2 baseline reproduced exactly (sanity anchor)
Step 1       + fractional shares (fix the SOXS ZERO_SHARES failure)
Step 2       + remove ORB_15M_5M (only VWAP_RECLAIM_5M / EMA_PULLBACK_5M)
Step 3       + no new entries at/after 14:00 ET (structural cutoff)
Step 4       + rule-based +1.2R breakeven shield  == V0.3 final
===========  ==================================================================

Design contract
---------------
* **Do not modify** ``research/v02.py`` (or any production module): V0.3 only
  *imports* the V0.2 data preparation, feature series, strategy candidate
  functions and mirror filter, and owns its own replay engine
  (:class:`_V03Replay`) with the three extra switches.
* **No tuning**: every numeric threshold below is a one-shot a-priori value
  (``# a-priori, not tuned``).  The 29-fold walk-forward grid (120/20/20), the
  fold construction (``walk_forward_splits``) and the OOS aggregation口径 are
  identical to V0.2, and every fold/run gets a fresh ``RiskManager``.
* **Step 0 is the self-consistency anchor**: its engine path must reproduce the
  V0.2 numbers exactly (1,206 trades / -$385.40 / PF 0.967 / WF 1,035 trades /
  -$852.52 / 11 positive folds) within the report's stated tolerance.
* **Data caveat**: all numbers are Alpaca **IEX-only** bars (single venue, small
  share of US consolidated volume).  This is a structural ablation, not proof of
  profitability and not consolidated-tape performance.

V0.3 rule details
-----------------
1. *Fractional shares*: ``config`` is overridden in memory with
   ``risk.allow_fractional_shares=True`` and ``risk.fractional_precision=2``
   (Firstrade 0.01-share granularity); ``risk.sizing.size_position`` already
   truncates downward, so the $ risk / notional caps are never exceeded.
   ``config/default.yaml`` is untouched.
2. *Strategy set*: only ``VWAP_RECLAIM_5M`` then ``EMA_PULLBACK_5M``
   (strategy-major order across symbols, same as V0.2).
3. *14:00 ET entry cutoff*: a completed 5m bar whose ``end_utc`` (the next 5m
   bar's open = fill instant) is at/after 14:00 ET creates no signal; a pending
   order whose fill instant is at/after 14:00 ET is cancelled.  Open positions
   are unaffected (managed to stop/target/15:59 ET EOD flatten).
4. *+1.2R breakeven shield*: track the highest price after entry.  Once
   ``high >= entry_fill + 1.2 * (entry_fill - initial_stop)`` the shield arms and
   the stop moves to ``max(initial_stop, entry_fill + 0.05 * risk_per_share)``
   **starting with the next 1m bar** (no same-bar look-ahead).  Same-bar TP/SL
   collisions remain conservative (stop fill).  Both entry and exit slippage are
   charged, so the 0.05R offset covers a tiny positive cost-adjusted breakeven.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Mapping, Sequence

from intraday_scalper.backtest.metrics import BacktestMetrics, compute_metrics
from intraday_scalper.backtest.walkforward import (
    Fold,
    assert_no_leakage,
    walk_forward_splits,
)
from intraday_scalper.core import (
    AppConfig,
    Bar,
    RiskPlan,
    SignalCandidate,
    TradeRecord,
    et_date,
    load_config,
    to_et,
)
from intraday_scalper.risk.manager import RiskManager
from intraday_scalper.risk.sizing import size_position

from intraday_scalper.research.v02 import (
    DATA_CAVEAT,
    EMA_PULLBACK_5M,
    ENUM_TO_LABEL,
    ORB_15M_5M,
    STRATEGY_LABELS,
    StrategyView,
    V02Data,
    VWAP_RECLAIM_5M,
    build_strategy_view,
    ema_pullback_5m_candidate,
    orb_15m_5m_candidate,
    prepare_v02_data,
    vwap_reclaim_5m_candidate,
)

__all__ = [
    "ABLATION_VARIANTS",
    "BREAKEVEN_OFFSET_R",
    "BREAKEVEN_TRIGGER_R",
    "DATA_CAVEAT",
    "ENTRY_CUTOFF_ET",
    "FRACTIONAL_PRECISION",
    "STEP0_VARIANT",
    "STEP1_VARIANT",
    "STEP2_VARIANT",
    "STEP3_VARIANT",
    "STEP4_VARIANT",
    "V03_SIGNAL_PREFIX",
    "V03FoldResult",
    "V03Result",
    "V03Variant",
    "V03WalkForwardResult",
    "build_effective_config",
    "run_v03_backtest",
    "run_v03_walkforward",
]

# --------------------------------------------------------------------------- #
# V0.3 constants — a-priori, not tuned
# --------------------------------------------------------------------------- #
#: Structural entry cut-off: no new signal/fill at or after 14:00 ET.
ENTRY_CUTOFF_ET = time(14, 0)

#: Breakeven shield trigger: arm once unrealised profit reaches +1.2R.
BREAKEVEN_TRIGGER_R = 1.2

#: Breakeven shield stop offset: entry + 0.05R (covers two-sided slippage).
BREAKEVEN_OFFSET_R = 0.05

#: Firstrade fractional-share granularity (0.01 share).
FRACTIONAL_PRECISION = 2

#: Signal id prefix for the V0.3 ledger.
V03_SIGNAL_PREFIX = "V03-"

_BUCKET_MINUTES = 5
_PRICE_EPS = 1e-9

_RULES: Mapping[str, Callable[[StrategyView], SignalCandidate | None]] = {
    ORB_15M_5M: orb_15m_5m_candidate,
    VWAP_RECLAIM_5M: vwap_reclaim_5m_candidate,
    EMA_PULLBACK_5M: ema_pullback_5m_candidate,
}


# --------------------------------------------------------------------------- #
# Variant description
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class V03Variant:
    """One ablation step: the structural switches applied on top of V0.2."""

    step: int
    name: str
    enabled_strategies: tuple[str, ...] = STRATEGY_LABELS
    allow_fractional_shares: bool = False
    entry_cutoff_et: time | None = None
    breakeven_shield: bool = False

    def __post_init__(self) -> None:
        unknown = [label for label in self.enabled_strategies if label not in _RULES]
        if unknown:
            raise ValueError(f"unknown V0.3 strategy label(s): {unknown}")
        if not self.enabled_strategies:
            raise ValueError("a V0.3 variant must enable at least one strategy")

    def flags(self) -> dict[str, Any]:
        """JSON-safe switch summary for the report."""
        return {
            "step": int(self.step),
            "name": self.name,
            "strategies": list(self.enabled_strategies),
            "allow_fractional_shares": bool(self.allow_fractional_shares),
            "fractional_precision": (
                FRACTIONAL_PRECISION if self.allow_fractional_shares else None
            ),
            "entry_cutoff_et": (
                self.entry_cutoff_et.strftime("%H:%M")
                if self.entry_cutoff_et is not None
                else None
            ),
            "breakeven_trigger_r": (
                BREAKEVEN_TRIGGER_R if self.breakeven_shield else None
            ),
            "breakeven_offset_r": (
                BREAKEVEN_OFFSET_R if self.breakeven_shield else None
            ),
        }


_TWO_STRATEGIES: tuple[str, ...] = (VWAP_RECLAIM_5M, EMA_PULLBACK_5M)

STEP0_VARIANT = V03Variant(step=0, name="Step 0 · V0.2 Baseline")
STEP1_VARIANT = V03Variant(
    step=1, name="Step 1 · + 碎股", allow_fractional_shares=True
)
STEP2_VARIANT = V03Variant(
    step=2,
    name="Step 2 · + 移除 ORB_15M_5M",
    enabled_strategies=_TWO_STRATEGIES,
    allow_fractional_shares=True,
)
STEP3_VARIANT = V03Variant(
    step=3,
    name="Step 3 · + 14:00 ET 截止",
    enabled_strategies=_TWO_STRATEGIES,
    allow_fractional_shares=True,
    entry_cutoff_et=ENTRY_CUTOFF_ET,
)
STEP4_VARIANT = V03Variant(
    step=4,
    name="Step 4 · + 1.2R 保本盾 = V0.3 最终",
    enabled_strategies=_TWO_STRATEGIES,
    allow_fractional_shares=True,
    entry_cutoff_et=ENTRY_CUTOFF_ET,
    breakeven_shield=True,
)

#: Cumulative ablation ladder (each step keeps every earlier change).
ABLATION_VARIANTS: tuple[V03Variant, ...] = (
    STEP0_VARIANT,
    STEP1_VARIANT,
    STEP2_VARIANT,
    STEP3_VARIANT,
    STEP4_VARIANT,
)


def build_effective_config(config: AppConfig, variant: V03Variant) -> AppConfig:
    """Return the in-memory config override for *variant* (frozen, never saved).

    Only the fractional-share switches are overridden; ``config/default.yaml``
    and every other risk parameter stay untouched (a-priori, not tuned).
    """

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not variant.allow_fractional_shares:
        return config
    return replace(
        config,
        risk=replace(
            config.risk,
            allow_fractional_shares=True,
            fractional_precision=FRACTIONAL_PRECISION,
        ),
    )


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class V03Result:
    """One V0.3 run (full period, a segment, or a single walk-forward fold)."""

    variant: V03Variant
    trades: list[TradeRecord]
    equity_curve: list[tuple[datetime, float]]
    metrics: BacktestMetrics
    trade_details: dict[str, dict[str, Any]]
    initial_equity: float
    final_equity: float
    sessions_run: int
    session_dates: list[date]
    buckets_evaluated: int
    skipped_buckets: int
    signals: int
    pending_cancelled: int
    ambiguous_bars: int
    breaker_days: int
    risk_block_counts: dict[str, int]
    notes: list[str]
    cutoff_skipped_evaluations: int = 0
    breakeven_armed_trades: int = 0
    shield_stop_exits: int = 0
    shield_saved_trades: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def trades_per_session(self) -> float:
        return (len(self.trades) / self.sessions_run) if self.sessions_run else 0.0

    def extra_metrics(self) -> dict[str, Any]:
        """V0.3-only statistics that :class:`BacktestMetrics` does not carry."""
        return {
            "step": int(self.variant.step),
            "variant": self.variant.name,
            "ambiguous_bars": int(self.ambiguous_bars),
            "breaker_days": int(self.breaker_days),
            "pending_cancelled": int(self.pending_cancelled),
            "cutoff_skipped_evaluations": int(self.cutoff_skipped_evaluations),
            "signals": int(self.signals),
            "buckets_evaluated": int(self.buckets_evaluated),
            "skipped_buckets": int(self.skipped_buckets),
            "sessions_run": int(self.sessions_run),
            "trades_per_session": round(self.trades_per_session, 4),
            "breakeven_armed_trades": int(self.breakeven_armed_trades),
            "shield_stop_exits": int(self.shield_stop_exits),
            "shield_saved_trades": int(self.shield_saved_trades),
            "risk_block_counts": dict(self.risk_block_counts),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class V03FoldResult:
    """One walk-forward fold: fresh engine/risk state for train *and* test."""

    fold: Fold
    train_result: V03Result
    test_result: V03Result
    leakage_checked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": {
                "index": self.fold.index,
                "train_start": self.fold.train_start.isoformat()
                if self.fold.train_start
                else None,
                "train_end": self.fold.train_end.isoformat()
                if self.fold.train_end
                else None,
                "test_start": self.fold.test_start.isoformat()
                if self.fold.test_start
                else None,
                "test_end": self.fold.test_end.isoformat()
                if self.fold.test_end
                else None,
                "train_days": len(self.fold.train_dates),
                "test_days": len(self.fold.test_dates),
            },
            "train_trades": self.train_result.total_trades,
            "test_trades": self.test_result.total_trades,
            "train_metrics": self.train_result.metrics.to_dict(),
            "test_metrics": self.test_result.metrics.to_dict(),
            "test_extra": self.test_result.extra_metrics(),
            "leakage_checked": self.leakage_checked,
        }


@dataclass(frozen=True, slots=True)
class V03WalkForwardResult:
    """All walk-forward folds plus the aggregated out-of-sample summary."""

    variant: V03Variant
    folds: tuple[V03FoldResult, ...]
    summary: Mapping[str, Any]
    aggregate_test_metrics: BacktestMetrics
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant.flags(),
            "folds": [fold.to_dict() for fold in self.folds],
            "summary": dict(self.summary),
            "aggregate_test_metrics": self.aggregate_test_metrics.to_dict(),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Replay engine (adapted from v02._V02Replay with the V0.3 switches)
# --------------------------------------------------------------------------- #
@dataclass
class _PendingEntry:
    candidate: SignalCandidate
    plan: RiskPlan
    signal_id: str
    expected_bucket_start_utc: datetime


@dataclass
class _OpenTrade:
    signal_id: str
    candidate: SignalCandidate
    plan: RiskPlan
    shares: float
    entry_fill: float
    entry_time_utc: datetime
    stop: float
    target: float
    mae_dollars: float = 0.0
    mfe_dollars: float = 0.0
    ambiguous: bool = False
    breakeven_armed: bool = False
    shield_stop: float | None = None
    shield_armed_bar_start_utc: datetime | None = None


class _V03Replay:
    """Deterministic 5m-driven replay with the V0.3 structural switches."""

    def __init__(
        self,
        data: V02Data,
        config: AppConfig,
        variant: V03Variant,
        *,
        risk: Any | None = None,
        initial_equity: float | None = None,
    ) -> None:
        if not isinstance(data, V02Data):
            raise TypeError("data must be a V02Data prepared by prepare_v02_data")
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        if not isinstance(variant, V03Variant):
            raise TypeError("variant must be a V03Variant")
        self.data = data
        self.variant = variant
        self.config = build_effective_config(config, variant)
        self.risk = risk if risk is not None else RiskManager(self.config)
        equity = float(
            self.config.risk.account_equity if initial_equity is None else initial_equity
        )
        if not math.isfinite(equity) or equity <= 0.0:
            raise ValueError(
                f"initial_equity must be finite and > 0, got {initial_equity!r}"
            )
        self.initial_equity = equity
        self.symbols = tuple(
            symbol for symbol in self.config.data.symbols if symbol in data.features
        )
        if not self.symbols:
            raise ValueError(
                "config.data.symbols does not intersect the prepared dataset symbols "
                f"({data.symbols})"
            )
        self._rules = tuple(_RULES[label] for label in variant.enabled_strategies)
        self._cutoff: time | None = variant.entry_cutoff_et
        self._shield_enabled = bool(variant.breakeven_shield)
        self._slippage_rate = float(self.config.risk.est_slippage_bps) / 10_000.0
        self._commission_per_share = float(self.config.risk.commission_per_share)

        # per-run state
        self._pending: _PendingEntry | None = None
        self._open: _OpenTrade | None = None
        self._trades: list[TradeRecord] = []
        self._trade_details: dict[str, dict[str, Any]] = {}
        self._equity_curve: list[tuple[datetime, float]] = []
        self._equity = self.initial_equity
        self._session = None  # type: Any
        self._buckets_evaluated = 0
        self._skipped_buckets = 0
        self._signals = 0
        self._pending_cancelled = 0
        self._ambiguous_bars = 0
        self._cutoff_skipped = 0
        self._risk_blocks: dict[str, int] = {}
        self._notes: list[str] = []
        self._signal_seq = 0

    # -- public --------------------------------------------------------- #
    def run(self, session_dates: Sequence[date] | None = None) -> V03Result:
        sessions = self.data.sessions_for(session_dates)
        for session in sessions:
            self._run_session(session)
        self._finalize()
        metrics = compute_metrics(
            self._trades,
            self._equity_curve,
            initial_equity=self.initial_equity,
            trade_details=self._trade_details,
        )
        (
            armed_trades,
            shield_exits,
            shield_saved,
        ) = self._shield_statistics()
        breaker_days = self._count_breaker_days()
        notes = list(self._notes)
        notes.append(DATA_CAVEAT)
        if self._risk_blocks:
            summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(self._risk_blocks.items())
            )
            notes.append(f"risk blocks: {summary}")
        return V03Result(
            variant=self.variant,
            trades=list(self._trades),
            equity_curve=list(self._equity_curve),
            metrics=metrics,
            trade_details=dict(self._trade_details),
            initial_equity=self.initial_equity,
            final_equity=self._equity,
            sessions_run=len(sessions),
            session_dates=[session.et_date for session in sessions],
            buckets_evaluated=self._buckets_evaluated,
            skipped_buckets=self._skipped_buckets,
            signals=self._signals,
            pending_cancelled=self._pending_cancelled,
            ambiguous_bars=self._ambiguous_bars,
            breaker_days=breaker_days,
            risk_block_counts=dict(self._risk_blocks),
            notes=notes,
            cutoff_skipped_evaluations=self._cutoff_skipped,
            breakeven_armed_trades=armed_trades,
            shield_stop_exits=shield_exits,
            shield_saved_trades=shield_saved,
        )

    # -- session loop (same sequencing as V0.2) ------------------------- #
    def _run_session(self, session: Any) -> None:
        self._session = session
        # Hard contract inherited from V0.2: sync the risk clock onto the
        # replay session (``roll`` refuses to move backwards and would latch the
        # wall-clock day -> the V0.1 circuit-breaker lock-up).
        begin_session = getattr(self.risk, "begin_session", None)
        if not callable(begin_session):
            raise TypeError(
                "risk manager must expose begin_session(session_open_utc); "
                "roll() alone is not enough for historical replay"
            )
        begin_session(session.open_utc)
        self._pending = None
        self._open = None
        for session_symbol in session.symbols.values():
            self._skipped_buckets += session_symbol.skipped_slots
        for bucket_start in session.bucket_starts:
            self._buckets_evaluated += 1
            if self._pending is not None:
                expected = self._pending.expected_bucket_start_utc
                if bucket_start != expected:
                    self._cancel_pending(
                        "missing next 5m bucket (gap); order cancelled, no stale fill"
                    )
                else:
                    self._try_fill_pending(session, bucket_start)
            if self._open is not None:
                self._manage_open_trade(session, bucket_start)
            if self._open is None and self._pending is None:
                self._evaluate_bucket(session, bucket_start)
        self._end_session(session)

    def _end_session(self, session: Any) -> None:
        if self._pending is not None:
            self._cancel_pending("new session started before the pending fill")
        if self._open is not None:
            open_trade = self._open
            symbol_session = session.symbols.get(open_trade.candidate.symbol)
            last_bar = symbol_session.bars_1m[-1] if symbol_session is not None else None
            if last_bar is not None:
                self._notes.append(
                    "open position force-closed at session's last observed 1m bar "
                    f"({open_trade.candidate.symbol}, reason=EOD_EXIT)"
                )
                self._close_trade(
                    reference_price=last_bar.close,
                    fill_price=last_bar.close * (1.0 - self._slippage_rate),
                    ts=last_bar.end_utc,
                    reason="EOD_EXIT",
                    stop_kind=None,
                )
            else:  # pragma: no cover - a position can only exist after a bar
                self._open = None

    def _finalize(self) -> None:
        if self._pending is not None:
            self._cancel_pending("end of data")
        if self._open is not None:  # pragma: no cover - _end_session closes first
            trade = self._open
            self._close_trade(
                reference_price=trade.entry_fill,
                fill_price=trade.entry_fill,
                ts=trade.entry_time_utc,
                reason="END_OF_DATA",
                stop_kind=None,
            )

    # -- 14:00 ET entry cutoff helpers ---------------------------------- #
    def _evaluation_allowed(self, bucket_start_utc: datetime) -> bool:
        """A completed 5m bar creates no signal if its close (= fill instant) >= cutoff."""
        if self._cutoff is None:
            return True
        bucket_end = bucket_start_utc + timedelta(minutes=_BUCKET_MINUTES)
        return to_et(bucket_end).time() < self._cutoff

    def _fill_time_allowed(self, fill_utc: datetime) -> bool:
        if self._cutoff is None:
            return True
        return to_et(fill_utc).time() < self._cutoff

    # -- pending entries / fills --------------------------------------- #
    def _try_fill_pending(self, session: Any, bucket_start: datetime) -> None:
        pending = self._pending
        if pending is None:  # pragma: no cover - caller guarantees
            return
        symbol = pending.candidate.symbol
        symbol_session = session.symbols.get(symbol)
        bucket = (
            None
            if symbol_session is None
            else symbol_session.bucket_by_start.get(bucket_start)
        )
        if bucket is None or not bucket.members:
            self._cancel_pending(
                f"next 5m bucket has no 1m bars for {symbol}; order cancelled"
            )
            return
        first_bar = bucket.members[0]
        if not self._fill_time_allowed(first_bar.start_utc):
            # Structural 14:00 ET cutoff: never fill at/after the cutoff.
            self._pending = None
            self._pending_cancelled += 1
            self._notes.append(
                "pending signal cancelled: fill time is at/after the 14:00 ET cutoff "
                f"({first_bar.start_utc.isoformat()})"
            )
            self._cancel_risk_pending()
            return
        self._pending = None
        entry_open = float(first_bar.open)
        candidate = pending.candidate
        if entry_open <= candidate.stop_loss:
            self._pending_cancelled += 1
            self._notes.append(
                "entry cancelled: next 5m open gapped through the stop "
                f"({symbol} open={entry_open:.4f} stop={candidate.stop_loss:.4f})"
            )
            self._cancel_risk_pending()
            return
        if entry_open >= candidate.take_profit_price:
            self._pending_cancelled += 1
            self._notes.append(
                "entry cancelled: next 5m open gapped through the target "
                f"({symbol} open={entry_open:.4f} target={candidate.take_profit_price:.4f})"
            )
            self._cancel_risk_pending()
            return
        entry_fill = entry_open * (1.0 + self._slippage_rate)
        try:
            self.risk.confirm_entry(
                pending.plan,
                fill_price=entry_fill,
                fill_utc=first_bar.start_utc,
                signal_id=pending.signal_id,
                mode=_mode_value(self.risk.mode),
            )
        except TypeError:  # pragma: no cover - simplified fake risk managers
            self.risk.confirm_entry(
                pending.plan,
                fill_price=entry_fill,
                fill_utc=first_bar.start_utc,
                signal_id=pending.signal_id,
            )
        self._open = _OpenTrade(
            signal_id=pending.signal_id,
            candidate=candidate,
            plan=pending.plan,
            shares=float(pending.plan.shares),
            entry_fill=entry_fill,
            entry_time_utc=first_bar.start_utc,
            stop=float(candidate.stop_loss),
            target=float(candidate.take_profit_price),
        )

    def _cancel_pending(self, reason: str) -> None:
        if self._pending is None:
            return
        self._pending = None
        self._pending_cancelled += 1
        self._notes.append(f"pending signal cancelled: {reason}")
        self._cancel_risk_pending()

    def _cancel_risk_pending(self) -> None:
        record_cancel = getattr(self.risk, "record_cancel", None)
        if callable(record_cancel):
            try:
                record_cancel()
            except Exception:  # pragma: no cover - optional integration
                pass

    # -- exits + breakeven shield -------------------------------------- #
    def _effective_stop(self, trade: _OpenTrade) -> float:
        """Stop used for the *current* bar (shield arms only for the next bar)."""

        if self._shield_enabled and trade.breakeven_armed and trade.shield_stop is not None:
            return max(trade.stop, trade.shield_stop)
        return trade.stop

    def _manage_open_trade(self, session: Any, bucket_start: datetime) -> None:
        trade = self._open
        if trade is None:  # pragma: no cover - caller guarantees
            return
        symbol = trade.candidate.symbol
        symbol_session = session.symbols.get(symbol)
        if symbol_session is None:  # pragma: no cover - position implies bars
            return
        bucket = symbol_session.bucket_by_start.get(bucket_start)
        if bucket is None:
            return
        for bar in bucket.members:
            if bar.start_utc < trade.entry_time_utc:
                continue
            # 1) Exit checks use only the stop state established *before* this
            #    bar (a shield armed later in this bar applies from the next bar).
            effective_stop = self._effective_stop(trade)
            stop_hit = bar.low <= effective_stop + _PRICE_EPS
            target_hit = bar.high >= trade.target - _PRICE_EPS
            if stop_hit or target_hit:
                if stop_hit and target_hit:
                    # Same-bar TP/SL collision -> conservative stop fill.
                    self._ambiguous_bars += 1
                    trade.ambiguous = True
                    prefer_stop = True
                else:
                    prefer_stop = stop_hit
                if prefer_stop:
                    reference = bar.open if bar.open <= effective_stop else effective_stop
                    stop_kind = (
                        "SHIELD"
                        if effective_stop > trade.stop + _PRICE_EPS
                        else "INITIAL"
                    )
                    self._close_trade(
                        reference_price=reference,
                        fill_price=reference * (1.0 - self._slippage_rate),
                        ts=bar.end_utc,
                        reason="STOP",
                        stop_kind=stop_kind,
                    )
                else:
                    reference = trade.target
                    self._close_trade(
                        reference_price=reference,
                        fill_price=reference * (1.0 - self._slippage_rate),
                        ts=bar.end_utc,
                        reason="TARGET",
                        stop_kind=None,
                    )
                return
            # 2) Excursions on every observed 1m bar (dollar terms, like V0.1).
            trade.mae_dollars = max(
                trade.mae_dollars, max(0.0, (trade.entry_fill - bar.low) * trade.shares)
            )
            trade.mfe_dollars = max(
                trade.mfe_dollars, max(0.0, (bar.high - trade.entry_fill) * trade.shares)
            )
            # 3) Arm the breakeven shield from this bar's high; effective next bar.
            risk_per_share = trade.entry_fill - trade.stop
            if (
                self._shield_enabled
                and not trade.breakeven_armed
                and risk_per_share > 0.0
                and bar.high
                >= trade.entry_fill + BREAKEVEN_TRIGGER_R * risk_per_share - _PRICE_EPS
            ):
                trade.breakeven_armed = True
                trade.shield_stop = max(
                    trade.stop,
                    trade.entry_fill + BREAKEVEN_OFFSET_R * risk_per_share,
                )
                trade.shield_armed_bar_start_utc = bar.start_utc
            # 4) Intraday system: flatten on the session's last 1m bar.
            if bar.start_utc == symbol_session.last_bar_start_utc:
                self._close_trade(
                    reference_price=bar.close,
                    fill_price=bar.close * (1.0 - self._slippage_rate),
                    ts=bar.end_utc,
                    reason="EOD_EXIT",
                    stop_kind=None,
                )
                return

    def _close_trade(
        self,
        *,
        reference_price: float,
        fill_price: float,
        ts: datetime,
        reason: str,
        stop_kind: str | None,
    ) -> None:
        trade = self._open
        if trade is None:  # pragma: no cover - caller guarantees
            return
        record = self.risk.confirm_exit(fill_price=fill_price, fill_utc=ts, reason=reason)
        self._open = None
        self._trades.append(record)
        self._equity += record.pnl
        self._equity_curve.append((ts, self._equity))
        entry_bucket_start = trade.entry_time_utc
        if self._session is not None:
            minutes = (trade.entry_time_utc - self._session.open_utc).total_seconds() / 60.0
            index = int(math.floor(minutes / _BUCKET_MINUTES))
            entry_bucket_start = self._session.open_utc + timedelta(
                minutes=_BUCKET_MINUTES * index
            )
        risk_dollars = max(trade.entry_fill - trade.stop, 0.0) * trade.shares
        self._trade_details[record.signal_id] = {
            "mae_dollars": trade.mae_dollars,
            "mfe_dollars": trade.mfe_dollars,
            "mae_r": (trade.mae_dollars / risk_dollars) if risk_dollars > 0 else 0.0,
            "mfe_r": (trade.mfe_dollars / risk_dollars) if risk_dollars > 0 else 0.0,
            "entry_slippage_dollars": (trade.entry_fill - trade.candidate.entry_ref)
            * trade.shares,
            "exit_slippage_dollars": (reference_price - fill_price) * trade.shares,
            "ambiguous_bar": bool(trade.ambiguous),
            "exit_reason": reason,
            "exit_stop_kind": stop_kind,
            "holding_minutes": record.holding_minutes,
            "fees_dollars": record.fees,
            "strategy_label": ENUM_TO_LABEL.get(record.strategy, record.strategy.value),
            "signal_bar_start_utc": trade.candidate.confirmation_bar_start_utc.isoformat(),
            "entry_bucket_start_utc": entry_bucket_start.isoformat(),
            "breakeven_armed": bool(trade.breakeven_armed),
            "shield_stop": trade.shield_stop,
            "shield_armed_bar_start_utc": (
                trade.shield_armed_bar_start_utc.isoformat()
                if trade.shield_armed_bar_start_utc is not None
                else None
            ),
        }

    # -- strategy evaluation ------------------------------------------- #
    def _evaluate_bucket(self, session: Any, bucket_start: datetime) -> None:
        # Structural 14:00 ET cutoff: a bar closing at/after the cutoff creates
        # no signal at all (applies to every strategy, checked before lookups).
        if not self._evaluation_allowed(bucket_start):
            self._cutoff_skipped += 1
            return
        views: list[StrategyView] = []
        for symbol in self.symbols:
            symbol_session = session.symbols.get(symbol)
            if symbol_session is None:
                continue
            if bucket_start not in symbol_session.bucket_by_start:
                continue
            view = build_strategy_view(self.data, session, symbol, bucket_start)
            if view is not None:
                views.append(view)
        if not views:
            return
        candidate: SignalCandidate | None = None
        for rule in self._rules:
            for view in views:
                candidate = rule(view)
                if candidate is not None:
                    break
            if candidate is not None:
                break
        if candidate is None:
            return
        self._signals += 1
        plan = size_position(candidate, self.config.risk, equity=self._equity)
        decision = self.risk.evaluate_entry(
            candidate, plan, candidate.created_utc, data_ready=True
        )
        if not bool(getattr(decision, "allowed", False)):
            reason = getattr(decision, "reason", None)
            key = getattr(reason, "value", str(reason))
            self._risk_blocks[key] = self._risk_blocks.get(key, 0) + 1
            return
        allowed_plan = getattr(decision, "plan", None) or plan
        self._signal_seq += 1
        self._pending = _PendingEntry(
            candidate=candidate,
            plan=allowed_plan,
            signal_id=f"{V03_SIGNAL_PREFIX}{self._signal_seq:06d}",
            expected_bucket_start_utc=bucket_start + timedelta(minutes=_BUCKET_MINUTES),
        )

    # -- statistics ----------------------------------------------------- #
    def _shield_statistics(self) -> tuple[int, int, int]:
        armed = 0
        shield_exits = 0
        saved = 0
        pnl_by_id = {trade.signal_id: trade.pnl for trade in self._trades}
        for signal_id, detail in self._trade_details.items():
            if not detail.get("breakeven_armed"):
                continue
            armed += 1
            if detail.get("exit_reason") == "STOP" and detail.get("exit_stop_kind") == "SHIELD":
                shield_exits += 1
            if pnl_by_id.get(signal_id, 0.0) > 0.0:
                saved += 1
        return armed, shield_exits, saved

    def _count_breaker_days(self) -> int:
        threshold = -abs(float(self.config.risk.max_daily_net_loss))
        daily: dict[date, float] = {}
        for trade in self._trades:
            if trade.exit_time_utc is None:
                continue
            day = et_date(trade.exit_time_utc)
            daily[day] = daily.get(day, 0.0) + trade.pnl
        return sum(1 for pnl in daily.values() if pnl <= threshold + 1e-9)


def _mode_value(mode: Any) -> Any:
    return getattr(mode, "value", mode)


# --------------------------------------------------------------------------- #
# Public run / walk-forward entry points
# --------------------------------------------------------------------------- #
def run_v03_backtest(
    data: V02Data | Sequence[Bar],
    config: AppConfig | None = None,
    *,
    variant: V03Variant,
    risk: Any | None = None,
    initial_equity: float | None = None,
    session_dates: Sequence[date] | None = None,
) -> V03Result:
    """Run one V0.3 ablation variant over *data*.

    ``variant`` is mandatory: every run must state which structural switches it
    uses, so the ablation table cannot silently drift.  ``data`` may be a
    prepared :class:`~intraday_scalper.research.v02.V02Data` (preferred: built
    once and reused by all 5 steps + every fold) or a sorted sequence of
    completed regular-session 1m bars.
    """

    app_config = load_config() if config is None else config
    prepared = data if isinstance(data, V02Data) else prepare_v02_data(data)
    replay = _V03Replay(
        prepared,
        app_config,
        variant,
        risk=risk,
        initial_equity=initial_equity,
    )
    return replay.run(session_dates)


def _collect_5m_bars(data: V02Data, dates: Sequence[date]) -> list[Bar]:
    wanted = set(dates)
    collected: list[Bar] = []
    for session in data.sessions:
        if session.et_date not in wanted:
            continue
        for symbol_session in session.symbols.values():
            collected.extend(bucket.bar for bucket in symbol_session.buckets)
    collected.sort(key=lambda bar: bar.start_utc)
    return collected


def run_v03_walkforward(
    data: V02Data | Sequence[Bar],
    config: AppConfig | None = None,
    *,
    variant: V03Variant,
    train_days: int = 120,
    test_days: int = 20,
    step_days: int = 20,
    initial_equity: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> V03WalkForwardResult:
    """Strict chronological walk-forward with the exact V0.2 fold discipline.

    Same split grid as V0.2 (``walk_forward_splits(dates, 120, 20, 20)``, 29
    folds), :func:`assert_no_leakage` re-checked on the real 5m bar timestamps,
    a **fresh** ``RiskManager`` and replay state for each train/test run, and
    no P&L compounding across folds.
    """

    app_config = load_config() if config is None else config
    prepared = data if isinstance(data, V02Data) else prepare_v02_data(data)
    effective = build_effective_config(app_config, variant)
    equity = float(
        effective.risk.account_equity if initial_equity is None else initial_equity
    )
    if not math.isfinite(equity) or equity <= 0.0:
        raise ValueError("initial_equity must be finite and > 0")

    dates = list(prepared.dates)
    splits = walk_forward_splits(dates, train_days, test_days, step_days)
    notes: list[str] = []
    if not splits:
        empty = compute_metrics([], [], initial_equity=equity)
        notes.append(
            "WALK-FORWARD INCOMPLETE: not enough session dates for one full fold "
            f"(have {len(dates)}, need {train_days}+{test_days})"
        )
        return V03WalkForwardResult(
            variant=variant,
            folds=(),
            summary={
                "folds": 0,
                "train_days": int(train_days),
                "test_days": int(test_days),
                "step_days": int(step_days),
                "session_dates": len(dates),
                "total_test_trades": 0,
            },
            aggregate_test_metrics=empty,
            notes=tuple(notes),
        )

    fold_results: list[V03FoldResult] = []
    all_test_trades: list[TradeRecord] = []
    all_test_details: dict[str, dict[str, Any]] = {}
    all_train_trades: list[TradeRecord] = []
    breakers = 0
    ambiguous = 0
    for position, fold in enumerate(splits, start=1):
        train_bars = _collect_5m_bars(prepared, fold.train_dates)
        test_bars = _collect_5m_bars(prepared, fold.test_dates)
        assert_no_leakage(train_bars, test_bars)
        train_result = run_v03_backtest(
            prepared,
            app_config,
            variant=variant,
            risk=RiskManager(effective),  # fresh state for the fold
            initial_equity=equity,
            session_dates=list(fold.train_dates),
        )
        test_result = run_v03_backtest(
            prepared,
            app_config,
            variant=variant,
            risk=RiskManager(effective),
            initial_equity=equity,
            session_dates=list(fold.test_dates),
        )
        all_train_trades.extend(train_result.trades)
        all_test_trades.extend(test_result.trades)
        all_test_details.update(test_result.trade_details)
        breakers += test_result.breaker_days
        ambiguous += test_result.ambiguous_bars
        fold_results.append(
            V03FoldResult(
                fold=fold,
                train_result=train_result,
                test_result=test_result,
                leakage_checked=True,
            )
        )
        if progress is not None:
            progress(position, len(splits))

    aggregate = compute_metrics(all_test_trades, [], initial_equity=equity)
    test_net = sum(trade.pnl for trade in all_test_trades)
    fold_pnls = [fold.test_result.metrics.net_pnl_dollars for fold in fold_results]
    fold_drawdowns = [
        fold.test_result.metrics.max_drawdown_dollars for fold in fold_results
    ]
    fold_wins = sum(1 for pnl in fold_pnls if pnl > 0.0)
    summary = {
        "folds": len(fold_results),
        "train_days": int(train_days),
        "test_days": int(test_days),
        "step_days": int(step_days),
        "session_dates": len(dates),
        "total_train_trades": len(all_train_trades),
        "total_test_trades": aggregate.total_trades,
        "test_wins": aggregate.wins,
        "test_win_rate": aggregate.win_rate,
        "test_profit_factor": aggregate.profit_factor,
        "test_net_pnl_dollars": test_net,
        "test_avg_r": aggregate.avg_r,
        "test_max_drawdown_dollars_max_fold": max(fold_drawdowns) if fold_drawdowns else 0.0,
        "test_max_drawdown_pct_max_fold": max(
            (fold.test_result.metrics.max_drawdown_pct for fold in fold_results),
            default=0.0,
        ),
        "positive_folds": fold_wins,
        "best_fold_pnl": max(fold_pnls) if fold_pnls else 0.0,
        "worst_fold_pnl": min(fold_pnls) if fold_pnls else 0.0,
        "test_breaker_days_total": breakers,
        "test_ambiguous_bars_total": ambiguous,
        "test_trades_per_session": (
            len(all_test_trades) / sum(len(fold.fold.test_dates) for fold in fold_results)
            if fold_results
            else 0.0
        ),
    }
    notes.append(
        "each fold starts from the same initial equity (no compounding across folds); "
        "train metrics are in-sample diagnostics, only test metrics are out-of-sample"
    )
    notes.append(DATA_CAVEAT)
    return V03WalkForwardResult(
        variant=variant,
        folds=tuple(fold_results),
        summary=summary,
        aggregate_test_metrics=aggregate,
        notes=tuple(notes),
    )
