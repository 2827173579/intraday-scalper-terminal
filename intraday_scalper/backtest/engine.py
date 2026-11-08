"""Event-driven backtester for the SOXL / SOXS intraday scalper.

Design guarantees
-----------------
* **Event driven, bar by bar.**  The engine replays a time-sorted stream of
  *completed* regular-session 1m :class:`~intraday_scalper.core.models.Bar`
  objects.  At each bar it first manages any open trade, then updates the
  rolling session state, and only then evaluates the strategies on that
  confirmed bar.
* **No look-ahead.**  A strategy sees only bars up to and including the bar it
  is evaluated on.  A signal confirmed on bar *i* can only be filled at the
  **open of bar i+1** (never at the signal bar's close).  A candidate whose
  ``confirmation_bar_start_utc`` does not match the bar being evaluated is
  rejected.  If the next bar is missing (data gap), the pending signal is
  cancelled rather than filled later on stale information.
* **Pessimistic intra-bar handling.**  Stops and targets are checked against
  the bar's high/low.  When a single bar touches both, the default
  ``pessimism="stop_first"`` fills the **stop** and flags the bar as ambiguous.
  Stops that gap through the level fill at the (worse) open.
* **Costs are charged.**  One-way slippage in basis points plus a per-share
  commission are applied to both entry and exit fills.  Trade P&L in
  :class:`~intraday_scalper.core.models.TradeRecord` is net of fees.
* **Risk rules live in the risk manager.**  Daily loss circuit breaker, max
  entries per day, consecutive-stop cooldown, SOXL/SOXS mutual exclusion, the
  15:30 ET entry cut-off and the reward/risk floor are all delegated to an
  injectable ``RiskManager``-compatible object (see
  :mod:`intraday_scalper.backtest.risk_adapter` for the built-in equivalent
  used while ``risk.manager`` is still in flight).
* **No overnight carry.**  Positions are independently force-closed at the
  last regular-session bar; a carried position is force-closed at the first bar
  of the next session as a safety net.
* **Strict input validation.**  Bars must be sorted by ``start_utc``,
  non-duplicated and 1m.  Shuffled input raises ``ValueError`` instead of
  silently producing a fake excellent equity curve.

Data limitation (important)
---------------------------
The current Finnhub account has no historical candle permission and no live US
trade feed, so **real historical data is unavailable**.  Every result produced
here is a deterministic logic check on caller-supplied bars (synthetic in the
current phase) and must never be presented as real-market backtest
performance.  Every :class:`BacktestResult` carries this caveat in ``notes``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Sequence

from intraday_scalper.core import (
    AppConfig,
    Bar,
    DataStatus,
    Mode,
    RiskBlockReason,
    RiskPlan,
    SignalCandidate,
    TradeRecord,
    TradeStatus,
    et_date,
    is_regular_session,
    market_phase,
    session_close_utc,
    session_open_utc,
    to_et,
)
from intraday_scalper.core.timeutil import phase_allows_new_entries

from ._compat import (
    FallbackStrategyContext,
    make_context,
    resolve_context_class,
    resolve_indicators,
)
from .metrics import BacktestMetrics, compute_metrics, to_jsonable
from .risk_adapter import (
    resolve_risk_manager,
    resolve_sizing_fn,
    risk_manager_is_simulated,
)

__all__ = [
    "DATA_CAVEAT",
    "BacktestResult",
    "EventDrivenBacktester",
    "run_backtest",
    "run_synthetic_backtest",
    "validate_bars_1m",
]

REGULAR_SESSION_MINUTES = 390

DATA_CAVEAT = (
    "DATA LIMITATION: 回测结果只反映调用方提供的数据口径。若数据为合成 bar，"
    "则仅验证引擎逻辑；若为 Alpaca IEX 1m bar，则为单一交易所（IEX）口径，"
    "成交量只含 IEX 份额、VWAP 为 IEX-trade VWAP，**不是全市场 consolidated "
    "tape，也不代表任何真实市场回测成绩或盈利预期**。"
)

DEFAULT_PESSIMISM = "stop_first"
_VALID_PESSIMISM = ("stop_first", "target_first")


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_bars_1m(bars: Sequence[Bar]) -> list[Bar]:
    """Validate the bar stream and return it as a list.

    Raises
    ------
    TypeError
        If an element is not a :class:`~intraday_scalper.core.models.Bar`.
    ValueError
        If a bar is not 1-minute, the stream is not sorted by ``start_utc``, or
        the same symbol has two bars with the same start timestamp.  Silently
        accepting shuffled input would fabricate look-ahead profits, so it is a
        hard error.
    """

    if isinstance(bars, (str, bytes)):
        raise TypeError("bars_1m must be a sequence of Bar, not a string")
    try:
        items = list(bars)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError("bars_1m must be an iterable of Bar") from exc

    previous_start: datetime | None = None
    seen: set[tuple[str, datetime]] = set()
    for bar in items:
        if not isinstance(bar, Bar):
            raise TypeError(f"bars_1m must contain Bar objects, got {type(bar)!r}")
        if bar.timeframe_minutes != 1:
            raise ValueError(
                f"bars_1m must contain 1-minute bars, got {bar.timeframe_minutes}m "
                f"({bar.symbol} {bar.start_utc.isoformat()})"
            )
        if previous_start is not None and bar.start_utc < previous_start:
            raise ValueError(
                "bars_1m must be sorted by start_utc ascending; "
                f"got {bar.start_utc.isoformat()} after {previous_start.isoformat()}"
            )
        key = (bar.symbol, bar.start_utc)
        if key in seen:
            raise ValueError(f"duplicate bar for {bar.symbol} at {bar.start_utc.isoformat()}")
        seen.add(key)
        previous_start = bar.start_utc
    return items


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    """Everything a report script needs from one backtest run."""

    trades: list[TradeRecord]
    equity_curve: list[tuple[datetime, float]]
    metrics: BacktestMetrics
    bars_processed: int
    skipped_bars: int
    notes: list[str]
    initial_equity: float
    final_equity: float
    ambiguous_bars: int = 0
    risk_block_counts: dict[str, int] = field(default_factory=dict)
    trade_details: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary for the Lead's report script / CLI output."""

        return {
            "trades": [to_jsonable(trade) for trade in self.trades],
            "equity_curve": [
                {"ts_utc": ts.isoformat(), "equity": float(value)}
                for ts, value in self.equity_curve
            ],
            "metrics": self.metrics.to_dict(),
            "bars_processed": self.bars_processed,
            "skipped_bars": self.skipped_bars,
            "ambiguous_bars": self.ambiguous_bars,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "risk_block_counts": dict(self.risk_block_counts),
            "trade_details": to_jsonable(self.trade_details),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Internal replay state
# --------------------------------------------------------------------------- #
@dataclass
class _SymbolState:
    symbol: str
    bars: list[Bar] = field(default_factory=list)
    five_min: list[Bar] = field(default_factory=list)
    session_pv: float = 0.0
    session_volume: float = 0.0
    session_vwap: float | None = None


@dataclass
class _PendingEntry:
    candidate: SignalCandidate
    plan: RiskPlan | None
    signal_id: str
    expected_bar_start_utc: datetime
    mode: Mode


@dataclass
class _OpenTrade:
    signal_id: str
    candidate: SignalCandidate
    plan: RiskPlan
    mode: Mode
    shares: float
    entry_fill: float
    entry_time_utc: datetime
    stop: float
    target: float
    mae_dollars: float = 0.0
    mfe_dollars: float = 0.0
    ambiguous: bool = False


# --------------------------------------------------------------------------- #
# The backtester
# --------------------------------------------------------------------------- #
class EventDrivenBacktester:
    """Deterministic, offline, event-driven backtester.

    Parameters
    ----------
    config:
        Validated :class:`~intraday_scalper.core.config.AppConfig`.
    strategies:
        Sequence of objects exposing ``evaluate(ctx) -> SignalCandidate | None``
        (the frozen :mod:`intraday_scalper.strategies` contract).  Order is
        fixed; the first valid, risk-approved candidate wins.
    risk:
        Optional ``RiskManager``-compatible instance.  Must expose ``roll``,
        ``evaluate_entry``, ``confirm_entry`` and ``confirm_exit``.  ``None``
        (default) resolves the production manager when importable and falls
        back to the built-in equivalent (deterministic, offline).
    data_quality:
        Optional gate.  Either ``callable(now) -> bool`` or an object with
        ``report(now) -> QualityReport`` (``.aggregate`` status) / ``.status``.
        A non-``OK`` :class:`~intraday_scalper.core.models.DataStatus` disables
        new evaluations (no signals), exactly like the live system's
        ``DATA_UNAVAILABLE`` behaviour.
    mode:
        Override the configured mode (``long_soxl`` / ``long_soxs`` /
        ``dual_watch``).
    initial_equity:
        Starting equity; defaults to ``config.risk.account_equity``.
    slippage_bps:
        One-way slippage override; defaults to ``config.risk.est_slippage_bps``.
    commission_per_share:
        Per-share commission per side; defaults to
        ``config.risk.commission_per_share`` (Firstrade: 0).
    pessimism:
        ``"stop_first"`` (default) or ``"target_first"`` for the ambiguous
        same-bar TP+SL case.  ``stop_first`` is the conservative choice.
    close_at_session_end:
        Force-close any open position at the last regular-session bar
        (default ``True``; the system is an intraday system and never holds
        overnight by design).
    sizing_fn / indicators / context_class:
        Injection points for tests and for swapping implementations; ``None``
        resolves the frozen production interfaces.
    """

    def __init__(
        self,
        config: AppConfig,
        strategies: Sequence[Any],
        *,
        risk: Any | None = None,
        data_quality: Any | None = None,
        mode: Mode | str | None = None,
        initial_equity: float | None = None,
        slippage_bps: float | None = None,
        commission_per_share: float | None = None,
        pessimism: str = DEFAULT_PESSIMISM,
        close_at_session_end: bool = True,
        sizing_fn: Callable[..., RiskPlan] | None = None,
        indicators: Any | None = None,
        context_class: Any | None = None,
    ) -> None:
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        self.config = config
        self._strategies = list(strategies)
        if not self._strategies:
            raise ValueError("at least one strategy is required")
        for strategy in self._strategies:
            if not callable(getattr(strategy, "evaluate", None)):
                raise TypeError(f"strategy {strategy!r} has no callable evaluate(ctx)")

        self.mode = Mode(mode) if mode is not None else Mode(config.mode)
        self._symbols = frozenset(config.data.symbols)

        self._risk_factory: Callable[[], Any] | None = None
        if risk is None:
            self._risk_factory = lambda: resolve_risk_manager(config)
            self.risk = self._risk_factory()
        else:
            self.risk = risk
        self._risk_is_simulated = risk_manager_is_simulated(self.risk)
        if not all(
            callable(getattr(self.risk, name, None))
            for name in ("roll", "evaluate_entry", "confirm_entry", "confirm_exit")
        ):
            raise TypeError("risk manager must expose roll/evaluate_entry/confirm_entry/confirm_exit")

        equity = float(config.risk.account_equity if initial_equity is None else initial_equity)
        if not math.isfinite(equity) or equity <= 0.0:
            raise ValueError(f"initial_equity must be finite and > 0, got {initial_equity!r}")
        self.initial_equity = equity

        self.slippage_bps = float(
            config.risk.est_slippage_bps if slippage_bps is None else slippage_bps
        )
        if not math.isfinite(self.slippage_bps) or self.slippage_bps < 0.0:
            raise ValueError(f"slippage_bps must be finite and >= 0, got {slippage_bps!r}")
        self.commission_per_share = float(
            config.risk.commission_per_share
            if commission_per_share is None
            else commission_per_share
        )
        if not math.isfinite(self.commission_per_share) or self.commission_per_share < 0.0:
            raise ValueError("commission_per_share must be finite and >= 0")

        if pessimism not in _VALID_PESSIMISM:
            raise ValueError(f"pessimism must be one of {_VALID_PESSIMISM}, got {pessimism!r}")
        self.pessimism = pessimism
        self.close_at_session_end = bool(close_at_session_end)

        self._sizing_fn = sizing_fn if sizing_fn is not None else resolve_sizing_fn()
        self._indicators = indicators if indicators is not None else resolve_indicators()
        self._context_class = context_class
        self._data_quality = data_quality
        if data_quality is not None:
            supported = (
                callable(data_quality)
                or callable(getattr(data_quality, "report", None))
                or hasattr(data_quality, "status")
                or hasattr(data_quality, "aggregate")
            )
            if not supported:
                raise TypeError(
                    "data_quality must be callable or expose report(now)/status/aggregate"
                )

        self._reset_run()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, bars_1m: Sequence[Bar]) -> BacktestResult:
        """Replay ``bars_1m`` and return the closed-trade ledger plus metrics."""

        bars = validate_bars_1m(bars_1m)
        if self._risk_factory is not None:
            self.risk = self._risk_factory()
            self._risk_is_simulated = risk_manager_is_simulated(self.risk)
        self._reset_run()

        if not bars:
            self._notes.append("no bars provided: nothing to simulate")

        for bar in bars:
            self._on_bar(bar)
        self._finalize()

        metrics = compute_metrics(
            self._trades,
            self._equity_curve,
            initial_equity=self.initial_equity,
            trade_details=self._trade_details,
        )
        notes = self._build_notes(metrics)
        return BacktestResult(
            trades=list(self._trades),
            equity_curve=list(self._equity_curve),
            metrics=metrics,
            bars_processed=self._bars_processed,
            skipped_bars=self._skipped_bars,
            notes=notes,
            initial_equity=self.initial_equity,
            final_equity=self._equity,
            ambiguous_bars=self._ambiguous_bars,
            risk_block_counts=dict(self._risk_blocks),
            trade_details=dict(self._trade_details),
        )

    # ------------------------------------------------------------------ #
    # Run-state lifecycle
    # ------------------------------------------------------------------ #
    def _reset_run(self) -> None:
        self._symbol_states: dict[str, _SymbolState] = {}
        self._et_session_date: date | None = None
        self._pending: _PendingEntry | None = None
        self._open: _OpenTrade | None = None
        self._trades: list[TradeRecord] = []
        self._trade_details: dict[str, dict[str, Any]] = {}
        self._equity_curve: list[tuple[datetime, float]] = []
        self._equity = self.initial_equity
        self._bars_processed = 0
        self._skipped_bars = 0
        self._ambiguous_bars = 0
        self._risk_blocks: Counter[str] = Counter()
        self._data_blocks = 0
        self._notes: list[str] = []
        self._last_bar_by_symbol: dict[str, Bar] = {}
        self._signal_seq = 0
        self._unknown_symbols: set[str] = set()
        self._context_class_resolved = self._context_class or resolve_context_class()
        self._context_build_failures = 0

    def _build_notes(self, metrics: BacktestMetrics) -> list[str]:
        notes = list(self._notes)
        notes.extend(metrics.notes)
        notes.append(DATA_CAVEAT)
        notes.append(
            f"bars_processed={self._bars_processed} skipped_bars={self._skipped_bars} "
            f"closed_trades={len(self._trades)} ambiguous_bars={self._ambiguous_bars}"
        )
        if self._risk_blocks:
            summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(self._risk_blocks.items())
            )
            notes.append(f"risk blocks: {summary}")
        if self._data_blocks:
            notes.append(f"data-quality gate blocked {self._data_blocks} strategy evaluation(s)")
        if self._unknown_symbols:
            notes.append(
                "skipped bars for symbols outside config.data.symbols: "
                + ", ".join(sorted(self._unknown_symbols))
            )
        if self._open is not None:  # pragma: no cover - finalize closes first
            notes.append("WARNING: a position remained open at the end of the run")
        if self._risk_is_simulated:
            notes.append(
                "risk: built-in equivalent risk manager (production risk.manager unavailable "
                "or not injected); rules mirror the frozen RiskManager contract"
            )
        return notes

    # ------------------------------------------------------------------ #
    # Bar dispatch
    # ------------------------------------------------------------------ #
    def _on_bar(self, bar: Bar) -> None:
        if bar.symbol not in self._symbols:
            self._unknown_symbols.add(bar.symbol)
            self._skipped_bars += 1
            return

        bar_session_date = et_date(bar.start_utc)
        if bar_session_date != self._et_session_date:
            self._start_session(bar_session_date, bar)

        if not is_regular_session(bar.start_utc):
            self._skipped_bars += 1
            return
        if not bar.is_complete:
            self._skipped_bars += 1
            return

        self._bars_processed += 1
        self._last_bar_by_symbol[bar.symbol] = bar
        state = self._symbol_state(bar.symbol)

        # 1) A pending signal may only fill at the *open* of its next bar.
        if self._pending is not None and self._pending.candidate.symbol == bar.symbol:
            self._try_fill_pending(bar)

        # 2) Manage the open trade on this bar with its high/low (pessimistic).
        if self._open is not None and self._open.candidate.symbol == bar.symbol:
            self._manage_open_trade(bar)

        # 3) Intraday system: no overnight carry; flatten on the last bar.
        if (
            self.close_at_session_end
            and self._open is not None
            and self._open.candidate.symbol == bar.symbol
            and bar.start_utc >= self._last_regular_bar_start(bar.start_utc)
        ):
            self._close_trade(
                reference_price=bar.close,
                fill_price=bar.close * (1.0 - self._slippage_rate),
                ts=bar.end_utc,
                reason="EOD_EXIT",
            )

        # 4) Update the rolling session windows (only completed bars).
        self._update_state(state, bar)

        # 5) Expire a pending order that missed its exact next-bar-open fill.
        if self._pending is not None and bar.start_utc > self._pending.expected_bar_start_utc:
            self._cancel_pending("no bar at the expected next-minute open (data gap)")

        # 6) Evaluate strategies on this confirmed bar.
        if self._pending is None and self._phase_allows_entries(bar.end_utc):
            self._evaluate(bar, state)

    def _start_session(self, session_date: date, bar: Bar) -> None:
        if self._open is not None:
            last = self._last_bar_by_symbol.get(self._open.candidate.symbol)
            if last is not None:
                self._notes.append(
                    "carried position force-closed at the first bar of the next session "
                    f"({self._open.candidate.symbol}, reason=EOD_EXIT)"
                )
                self._close_trade(
                    reference_price=last.close,
                    fill_price=last.close * (1.0 - self._slippage_rate),
                    ts=last.end_utc,
                    reason="EOD_EXIT",
                )
            else:  # pragma: no cover - no bar to price against
                self._open = None
        if self._pending is not None:
            self._cancel_pending("new session started before the pending fill")
        self._et_session_date = session_date
        self._symbol_states = {}
        # ``begin_session`` lets a risk manager that was constructed with the
        # real wall clock (e.g. the production RiskManager) sync onto the
        # replay date, including when that date is in the past.  Plain
        # ``roll`` refuses to move backwards by design, which would otherwise
        # latch the daily state across the whole replay.
        begin = getattr(self.risk, "begin_session", None)
        if callable(begin):
            begin(bar.start_utc)
        else:  # pragma: no cover - simplified fake risk managers
            try:
                self.risk.roll(bar.start_utc)
            except TypeError:  # pragma: no cover
                self.risk.roll()

    # ------------------------------------------------------------------ #
    # Pending entry
    # ------------------------------------------------------------------ #
    def _try_fill_pending(self, bar: Bar) -> None:
        pending = self._pending
        if pending is None:  # pragma: no cover - caller guarantees
            return
        if bar.start_utc != pending.expected_bar_start_utc:
            self._cancel_pending("pending fill is not at the next bar open")
            return
        self._pending = None

        if self._open is not None:
            self._risk_blocks["PENDING_BLOCKED_BY_POSITION"] += 1
            self._cancel_pending("another position is already open")
            return
        if not self._phase_allows_entries(bar.start_utc):
            self._risk_blocks[RiskBlockReason.NO_NEW_ENTRIES_TIME.value] += 1
            self._cancel_pending("entry would occur after the no-new-entries cut-off")
            return
        if pending.plan is None or not pending.plan.valid:
            self._risk_blocks[RiskBlockReason.ZERO_SHARES.value] += 1
            self._cancel_pending("plan is not tradable (0 shares)")
            return

        candidate = pending.candidate
        entry_open = float(bar.open)
        # Never enter through the stop (or already past the target): the setup
        # the strategy confirmed no longer exists at a tradable price.
        if entry_open <= candidate.stop_loss:
            self._notes.append(
                "entry cancelled: next-bar open gapped through the stop "
                f"({candidate.symbol} open={entry_open:.4f} stop={candidate.stop_loss:.4f})"
            )
            self._cancel_pending("gap through stop")
            return
        if entry_open >= candidate.take_profit_price:
            self._notes.append(
                "entry cancelled: next-bar open gapped through the target "
                f"({candidate.symbol} open={entry_open:.4f} target={candidate.take_profit_price:.4f})"
            )
            self._cancel_pending("gap through target")
            return

        shares = float(pending.plan.shares)
        entry_fill = entry_open * (1.0 + self._slippage_rate)
        try:
            self.risk.confirm_entry(
                pending.plan,
                fill_price=entry_fill,
                fill_utc=bar.start_utc,
                signal_id=pending.signal_id,
                mode=pending.mode,
            )
        except TypeError:  # pragma: no cover - simplified fake risk managers
            self.risk.confirm_entry(
                pending.plan,
                fill_price=entry_fill,
                fill_utc=bar.start_utc,
                signal_id=pending.signal_id,
            )
        self._open = _OpenTrade(
            signal_id=pending.signal_id,
            candidate=candidate,
            plan=pending.plan,
            mode=pending.mode,
            shares=shares,
            entry_fill=entry_fill,
            entry_time_utc=bar.start_utc,
            stop=float(candidate.stop_loss),
            target=float(candidate.take_profit_price),
        )

    def _cancel_pending(self, reason: str) -> None:
        if self._pending is None:
            return
        self._notes.append(f"pending signal cancelled: {reason}")
        self._pending = None
        record_cancel = getattr(self.risk, "record_cancel", None)
        if callable(record_cancel):
            try:
                record_cancel()
            except Exception:  # pragma: no cover - optional integration
                pass

    # ------------------------------------------------------------------ #
    # Exit management
    # ------------------------------------------------------------------ #
    def _manage_open_trade(self, bar: Bar) -> None:
        trade = self._open
        if trade is None:  # pragma: no cover - caller guarantees
            return
        trade.mae_dollars = max(
            trade.mae_dollars, max(0.0, (trade.entry_fill - bar.low) * trade.shares)
        )
        trade.mfe_dollars = max(
            trade.mfe_dollars, max(0.0, (bar.high - trade.entry_fill) * trade.shares)
        )
        stop_hit = bar.low <= trade.stop
        target_hit = bar.high >= trade.target
        if not stop_hit and not target_hit:
            return
        if stop_hit and target_hit:
            self._ambiguous_bars += 1
            trade.ambiguous = True
            prefer_stop = self.pessimism == "stop_first"
        else:
            prefer_stop = stop_hit

        if prefer_stop:
            # A gap below the stop fills at the open (worse); otherwise at the stop.
            reference = bar.open if bar.open <= trade.stop else trade.stop
            self._close_trade(
                reference_price=reference,
                fill_price=reference * (1.0 - self._slippage_rate),
                ts=bar.end_utc,
                reason="STOP",
            )
        else:
            # Sell limit at the target: pessimistic fill at exactly the target
            # even when the bar opened above it.
            reference = trade.target
            self._close_trade(
                reference_price=reference,
                fill_price=reference * (1.0 - self._slippage_rate),
                ts=bar.end_utc,
                reason="TARGET",
            )

    def _close_trade(
        self,
        *,
        reference_price: float,
        fill_price: float,
        ts: datetime,
        reason: str,
    ) -> None:
        trade = self._open
        if trade is None:  # pragma: no cover - caller guarantees
            return
        shares = float(trade.shares)
        fees = shares * self.commission_per_share * 2.0
        record = TradeRecord(
            signal_id=trade.signal_id,
            symbol=trade.candidate.symbol,
            strategy=trade.candidate.strategy,
            mode=trade.mode,
            shares=shares,
            entry_time_utc=trade.entry_time_utc,
            entry_price_signal=trade.candidate.entry_ref,
            entry_price_fill=trade.entry_fill,
            stop_price=trade.stop,
            target_price=trade.target,
            exit_time_utc=ts,
            exit_price_fill=fill_price,
            exit_reason=reason,
            fees=fees,
            status=TradeStatus.STOPPED if reason == "STOP" else TradeStatus.CLOSED,
            trade_id=len(self._trades) + 1,
        )
        self._trades.append(record)
        self._equity += record.pnl
        self._equity_curve.append((ts, self._equity))
        self._trade_details[record.signal_id] = {
            "mae_dollars": trade.mae_dollars,
            "mfe_dollars": trade.mfe_dollars,
            "entry_slippage_dollars": (trade.entry_fill - trade.candidate.entry_ref) * shares,
            "exit_slippage_dollars": (reference_price - fill_price) * shares,
            "ambiguous_bar": trade.ambiguous,
            "exit_reason": reason,
            "holding_minutes": record.holding_minutes,
            "fees_dollars": fees,
        }
        try:
            self.risk.confirm_exit(fill_price=fill_price, fill_utc=ts, reason=reason, fees=fees)
        except TypeError:  # pragma: no cover - production RiskManager has no fees kwarg
            self.risk.confirm_exit(fill_price=fill_price, fill_utc=ts, reason=reason)
        self._open = None

    def _finalize(self) -> None:
        if self._pending is not None:
            self._cancel_pending("end of data")
        if self._open is not None:
            symbol = self._open.candidate.symbol
            last = self._last_bar_by_symbol.get(symbol)
            if last is not None:
                self._notes.append(
                    f"open position on {symbol} force-closed at the last provided bar "
                    f"({last.start_utc.isoformat()}, reason=END_OF_DATA)"
                )
                self._close_trade(
                    reference_price=last.close,
                    fill_price=last.close * (1.0 - self._slippage_rate),
                    ts=last.end_utc,
                    reason="END_OF_DATA",
                )

    # ------------------------------------------------------------------ #
    # Strategy evaluation
    # ------------------------------------------------------------------ #
    def _evaluate(self, bar: Bar, state: _SymbolState) -> None:
        if not self._data_ready(bar.end_utc):
            self._data_blocks += 1
            return
        context = self._build_context(bar, state)
        if context is None:  # pragma: no cover - defensive
            return
        for strategy in self._strategies:
            candidate = strategy.evaluate(context)
            if candidate is None:
                continue
            if not isinstance(candidate, SignalCandidate):
                self._notes.append(
                    f"strategy {type(strategy).__name__} returned a non-SignalCandidate; ignored"
                )
                continue
            if candidate.symbol != bar.symbol:
                self._notes.append(
                    f"candidate skipped: symbol {candidate.symbol} does not match evaluated "
                    f"bar {bar.symbol}"
                )
                continue
            # Anti look-ahead: a candidate must be confirmed by the bar the
            # engine is currently evaluating (not an earlier or later bar).
            if candidate.confirmation_bar_start_utc != bar.start_utc:
                self._notes.append(
                    "candidate skipped: confirmation_bar_start_utc "
                    f"{candidate.confirmation_bar_start_utc.isoformat()} != evaluated bar "
                    f"{bar.start_utc.isoformat()}"
                )
                continue
            if not self._mode_allows(candidate.symbol):
                self._risk_blocks[RiskBlockReason.DUAL_CONFLICT.value] += 1
                continue

            plan = self._size(candidate)
            try:
                decision = self.risk.evaluate_entry(
                    candidate, plan, bar.end_utc, data_ready=True
                )
            except TypeError:  # pragma: no cover - simplified fake risk managers
                decision = self.risk.evaluate_entry(candidate, plan, bar.end_utc)
            if not bool(getattr(decision, "allowed", False)):
                reason = getattr(decision, "reason", RiskBlockReason.NONE)
                key = getattr(reason, "value", reason)
                self._risk_blocks[str(key)] += 1
                continue

            self._signal_seq += 1
            self._pending = _PendingEntry(
                candidate=candidate,
                plan=plan,
                signal_id=f"BT-{self._signal_seq:06d}",
                expected_bar_start_utc=bar.end_utc,
                mode=self.mode,
            )
            break

    def _size(self, candidate: SignalCandidate) -> RiskPlan | None:
        try:
            return self._sizing_fn(candidate, self.config.risk, equity=self._equity)
        except TypeError:  # pragma: no cover - two-argument sizing implementations
            return self._sizing_fn(candidate, self.config.risk)

    def _build_context(self, bar: Bar, state: _SymbolState) -> Any | None:
        session_bars = state.bars
        closes = [item.close for item in session_bars]
        strategies = self.config.strategies
        indicators = self._indicators
        try:
            ema_fast = _last_value(indicators.ema(closes, strategies.ema_fast))
            ema_slow = _last_value(indicators.ema(closes, strategies.ema_slow))
            rsi_value = _last_value(indicators.rsi(closes, strategies.rsi_period))
            atr_value = _last_value(indicators.atr(session_bars, strategies.atr_period))
            session_open = session_open_utc(bar.start_utc)
            opening_5 = indicators.opening_range(session_bars, session_open, 5)
            opening_15 = indicators.opening_range(session_bars, session_open, 15)
        except Exception as exc:  # pragma: no cover - defensive integration guard
            self._context_build_failures += 1
            if self._context_build_failures == 1:
                self._notes.append(f"context build failure (indicators): {exc!r}")
            return None
        if len(session_bars) >= 21:
            prior = session_bars[-21:-1]
            avg_volume_20 = sum(item.volume for item in prior) / 20.0
        else:
            avg_volume_20 = None
        kwargs = dict(
            symbol=bar.symbol,
            now_utc=bar.end_utc,
            session_phase=market_phase(bar.end_utc),
            mode=self.mode,
            bars_1m=tuple(session_bars),
            bars_5m=tuple(state.five_min),
            vwap=state.session_vwap,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi=rsi_value,
            atr=atr_value,
            opening_range_5m=opening_5,
            opening_range_15m=opening_15,
            avg_volume_20=avg_volume_20,
            last_price=bar.close,
        )
        try:
            return make_context(self._context_class_resolved, **kwargs)
        except TypeError as exc:  # pragma: no cover - defensive integration guard
            self._context_build_failures += 1
            if self._context_build_failures == 1:
                self._notes.append(f"strategy context fallback: {exc!r}")
            return FallbackStrategyContext(**kwargs)

    # ------------------------------------------------------------------ #
    # Session state helpers
    # ------------------------------------------------------------------ #
    def _symbol_state(self, symbol: str) -> _SymbolState:
        state = self._symbol_states.get(symbol)
        if state is None:
            state = _SymbolState(symbol=symbol)
            self._symbol_states[symbol] = state
        return state

    def _update_state(self, state: _SymbolState, bar: Bar) -> None:
        state.bars.append(bar)
        state.session_pv += bar.vwap * bar.volume
        state.session_volume += bar.volume
        state.session_vwap = (
            state.session_pv / state.session_volume if state.session_volume > 0.0 else None
        )
        state.five_min = _aggregate_five_min(state.bars)

    def _data_ready(self, now: datetime) -> bool:
        quality = self._data_quality
        if quality is None:
            return True
        if callable(quality):
            return bool(quality(now))
        report = getattr(quality, "report", None)
        status: Any
        if callable(report):
            result = report(now)
            status = getattr(result, "aggregate", None)
            if status is None:
                status = getattr(result, "status", None)
        else:
            status = getattr(quality, "status", None)
            if status is None:
                status = getattr(quality, "aggregate", None)
        if status is None:
            return True
        if isinstance(status, DataStatus):
            return status == DataStatus.OK
        return getattr(status, "value", status) == DataStatus.OK.value

    def _phase_allows_entries(self, ts: datetime) -> bool:
        if not self._symbols:
            return False
        return phase_allows_new_entries(market_phase(ts)) and (
            to_et(ts).time() < self.config.session.no_new_entries_et
        )

    def _mode_allows(self, symbol: str) -> bool:
        if self.mode == Mode.LONG_SOXL:
            return symbol == "SOXL"
        if self.mode == Mode.LONG_SOXS:
            return symbol == "SOXS"
        return symbol in self._symbols

    @property
    def _slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    @staticmethod
    def _last_regular_bar_start(ts: datetime) -> datetime:
        return session_close_utc(ts) - timedelta(minutes=1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _last_value(series: Sequence[float | None]) -> float | None:
    return series[-1] if series else None


def _aggregate_five_min(bars: Sequence[Bar]) -> list[Bar]:
    """Aggregate completed 1m bars into 5m bars anchored at the 09:30 ET open.

    Only buckets whose full 5-minute window has ended are emitted, and only
    from *completed* 1m bars.  A bucket with missing minutes is still emitted
    from the minutes that exist (never a fabricated bar).
    """

    if not bars:
        return []
    session_open = session_open_utc(bars[0].start_utc)
    now = bars[-1].end_utc
    buckets: dict[datetime, list[Bar]] = {}
    for bar in bars:
        index = int(math.floor((bar.start_utc - session_open).total_seconds() / 60.0 / 5.0))
        bucket_start = session_open + timedelta(minutes=5 * index)
        buckets.setdefault(bucket_start, []).append(bar)

    result: list[Bar] = []
    for bucket_start in sorted(buckets):
        bucket_end = bucket_start + timedelta(minutes=5)
        if bucket_end > now:
            continue
        members = buckets[bucket_start]
        volume = sum(item.volume for item in members)
        low = min(item.low for item in members)
        high = max(item.high for item in members)
        if volume > 0.0:
            vwap = sum(item.vwap * item.volume for item in members) / volume
        else:
            vwap = members[-1].close
        vwap = min(max(vwap, low), high)
        result.append(
            Bar(
                symbol=members[0].symbol,
                timeframe_minutes=5,
                start_utc=bucket_start,
                end_utc=bucket_end,
                open=members[0].open,
                high=high,
                low=low,
                close=members[-1].close,
                volume=volume,
                vwap=vwap,
                trade_count=sum(item.trade_count for item in members),
                is_complete=True,
            )
        )
    return result


# --------------------------------------------------------------------------- #
# Convenience entry points (used by the Lead's report / CLI script)
# --------------------------------------------------------------------------- #
def run_backtest(
    bars_1m: Sequence[Bar],
    config: AppConfig,
    strategies: Sequence[Any],
    **kwargs: Any,
) -> BacktestResult:
    """One-call wrapper: ``EventDrivenBacktester(config, strategies, **kw).run(bars)``."""

    return EventDrivenBacktester(config, strategies, **kwargs).run(bars_1m)


def run_synthetic_backtest(
    config: AppConfig,
    strategies: Sequence[Any],
    *,
    days: int = 5,
    seed: int = 7,
    symbols: Sequence[str] | None = None,
    start_date: date | None = None,
    trend: float = 0.0,
    vol: float = 0.0015,
    n_minutes: int = REGULAR_SESSION_MINUTES,
    initial_equity: float | None = None,
    **kwargs: Any,
) -> BacktestResult:
    """Run the engine on deterministic synthetic sessions (logic check only).

    This is what ``python -m intraday_scalper.main backtest --synthetic`` is
    expected to call.  It always injects the explicit caveat that the result is
    **not** real-market performance.
    """

    from .synthetic import SYNTHETIC_DATA_ONLY, generate_sessions, next_trading_dates

    symbol_list = tuple(symbols) if symbols is not None else tuple(config.data.symbols)
    dates = next_trading_dates(start_date or date(2026, 11, 2), days)
    bars = generate_sessions(
        symbol_list, dates, seed=seed, trend=trend, vol=vol, n_minutes=n_minutes
    )
    result = EventDrivenBacktester(
        config, strategies, initial_equity=initial_equity, **kwargs
    ).run(bars)
    result.notes.insert(0, SYNTHETIC_DATA_ONLY)
    result.notes.append(
        f"synthetic run: {len(dates)} session(s) x {len(symbol_list)} symbol(s), seed={seed}"
    )
    return result
