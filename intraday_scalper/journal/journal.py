"""SQLite journal for manually executed SOXL / SOXS trades.

The journal is the system's only durable record of what happened.  It never
sends an order: every row is written because a human typed a command
(``ENTRY`` / ``EXIT`` / ``STOP`` / ``CANCEL`` / ``FLAT``) after trading manually
at the broker.

Schema (all timestamps are ISO-8601 UTC strings, WAL journal mode)::

    signals(signal_id PK, symbol, strategy, mode, entry_ref, stop_loss,
            take_profit, target_r, risk_per_share, shares, notional,
            created_utc, rationale, status, trade_id)
    trades(trade_id PK AUTOINCREMENT, signal_id, symbol, strategy, mode,
           shares, entry_time_utc, entry_price_signal, entry_price_fill,
           stop_price, target_price, exit_time_utc, exit_price_fill,
           exit_reason, fees, status)
    events(id PK AUTOINCREMENT, ts_utc, kind, level, message, payload_json)

Both the signal reference price and the real fill price are stored per trade so
execution slippage is measured from *actual* fills, not assumed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from ..core import (
    AlertEvent,
    AppConfig,
    Mode,
    PathsConfig,
    Position,
    RiskPlan,
    SignalCandidate,
    StrategyName,
    TradeRecord,
    TradeStatus,
    UTC,
    new_signal_id,
    now_utc,
    to_utc,
)
from ..core.timeutil import et_date as _et_date_of

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    mode TEXT,
    entry_ref REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    target_r REAL NOT NULL,
    risk_per_share REAL NOT NULL,
    shares REAL NOT NULL,
    notional REAL NOT NULL,
    created_utc TEXT NOT NULL,
    rationale TEXT,
    status TEXT NOT NULL,
    trade_id INTEGER
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    mode TEXT,
    shares REAL NOT NULL,
    entry_time_utc TEXT NOT NULL,
    entry_price_signal REAL NOT NULL,
    entry_price_fill REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    exit_time_utc TEXT,
    exit_price_fill REAL,
    exit_reason TEXT,
    fees REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    kind TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_signal ON trades(signal_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_utc);
"""

#: Exit reasons whose planned exit price is known, used for exit-slippage stats.
_STOP_REASONS = {"STOP", "STOPPED"}
_TARGET_REASONS = {"TARGET", "TAKE_PROFIT", "TP"}


class Journal:
    """Thread-safe SQLite journal.  Deterministic (all times are injected)."""

    def __init__(
        self,
        db_path: str | Path | AppConfig | PathsConfig | None = None,
        *,
        config: AppConfig | None = None,
        clock: Callable[[], datetime] = now_utc,
    ) -> None:
        if isinstance(db_path, AppConfig):  # convenience: Journal(config)
            config = config if config is not None else db_path
            db_path = config.paths.journal_db
        elif isinstance(db_path, PathsConfig):  # convenience: Journal(config.paths)
            db_path = db_path.journal_db
        if db_path is None:
            if config is None:
                raise ValueError(
                    "Journal requires db_path, or an AppConfig exposing paths.journal_db"
                )
            db_path = config.paths.journal_db
        self._config = config
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._default_mode = config.mode if config is not None else Mode.DUAL_WATCH
        self._commission_per_share = (
            float(config.risk.commission_per_share) if config is not None else 0.0
        )
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Lifecycle / helpers
    # ------------------------------------------------------------------ #
    @property
    def db_path(self) -> Path:
        return self._path

    @property
    def default_mode(self) -> Mode:
        return self._default_mode

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.commit()
                self._conn.close()
                self._closed = True

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @staticmethod
    def _iso(ts: datetime) -> str:
        return to_utc(ts).isoformat()

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return to_utc(parsed)

    def now(self) -> datetime:
        """Current time according to the injected clock (UTC aware)."""
        return to_utc(self._clock())

    # ------------------------------------------------------------------ #
    # Signals
    # ------------------------------------------------------------------ #
    def record_signal(
        self,
        candidate: SignalCandidate,
        plan: RiskPlan,
        *,
        mode: Mode | str | None = None,
    ) -> str:
        """Persist one strategy candidate (status PENDING) and return its id."""
        if not isinstance(candidate, SignalCandidate):
            raise TypeError(f"candidate must be a SignalCandidate, got {type(candidate)!r}")
        if not isinstance(plan, RiskPlan):
            raise TypeError(f"plan must be a RiskPlan, got {type(plan)!r}")
        resolved_mode = Mode(mode) if mode is not None else self._default_mode
        signal_id = new_signal_id(candidate.created_utc)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO signals (
                    signal_id, symbol, strategy, mode, entry_ref, stop_loss,
                    take_profit, target_r, risk_per_share, shares, notional,
                    created_utc, rationale, status, trade_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    signal_id,
                    candidate.symbol,
                    candidate.strategy.value,
                    resolved_mode.value,
                    candidate.entry_ref,
                    candidate.stop_loss,
                    candidate.take_profit_price,
                    candidate.target_r,
                    candidate.risk_per_share,
                    plan.shares,
                    plan.notional,
                    self._iso(candidate.created_utc),
                    candidate.rationale,
                    TradeStatus.PENDING.value,
                ),
            )
        return signal_id

    def record_entry(
        self,
        signal_id: str,
        fill_price: float,
        shares: float,
        ts: datetime,
        *,
        mode: Mode | str | None = None,
    ) -> int:
        """Persist a human-confirmed entry fill.  Returns the ``trade_id``."""
        moment = to_utc(ts)
        fill = float(fill_price)
        qty = float(shares)
        if fill <= 0:
            raise ValueError(f"fill_price must be > 0, got {fill_price!r}")
        if qty <= 0:
            raise ValueError(f"shares must be > 0, got {shares!r}")
        with self._lock, self._conn:
            signal_row = self._conn.execute(
                "SELECT * FROM signals WHERE signal_id = ?", (str(signal_id),)
            ).fetchone()
            if signal_row is None:
                raise ValueError(f"未知信号 ID：{signal_id}")
            signal_status = TradeStatus(signal_row["status"])
            if signal_status is not TradeStatus.PENDING:
                raise ValueError(
                    f"信号 {signal_id} 当前状态为 {signal_status.value}，"
                    f"不是待确认（PENDING）状态，无法记录 ENTRY"
                )
            resolved_mode = (
                Mode(mode)
                if mode is not None
                else (
                    Mode(signal_row["mode"])
                    if signal_row["mode"]
                    else self._default_mode
                )
            )
            cursor = self._conn.execute(
                """
                INSERT INTO trades (
                    signal_id, symbol, strategy, mode, shares, entry_time_utc,
                    entry_price_signal, entry_price_fill, stop_price, target_price,
                    exit_time_utc, exit_price_fill, exit_reason, fees, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 0, ?)
                """,
                (
                    str(signal_id),
                    signal_row["symbol"],
                    signal_row["strategy"],
                    resolved_mode.value,
                    qty,
                    self._iso(moment),
                    float(signal_row["entry_ref"]),
                    fill,
                    float(signal_row["stop_loss"]),
                    float(signal_row["take_profit"]),
                    TradeStatus.OPEN.value,
                ),
            )
            trade_id = int(cursor.lastrowid)
            self._conn.execute(
                "UPDATE signals SET status = ?, trade_id = ?, mode = COALESCE(mode, ?) "
                "WHERE signal_id = ?",
                (TradeStatus.OPEN.value, trade_id, resolved_mode.value, str(signal_id)),
            )
        return trade_id

    def cancel_signal(self, signal_id: str, reason: str = "MANUAL") -> bool:
        """Cancel a PENDING signal.  Returns True when the status changed."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT status FROM signals WHERE signal_id = ?", (str(signal_id),)
            ).fetchone()
            if row is None:
                raise ValueError(f"未知信号 ID：{signal_id}")
            if TradeStatus(row["status"]) is not TradeStatus.PENDING:
                return False
            self._conn.execute(
                "UPDATE signals SET status = ? WHERE signal_id = ?",
                (TradeStatus.CANCELLED.value, str(signal_id)),
            )
        return True

    # ------------------------------------------------------------------ #
    # Trades / exits
    # ------------------------------------------------------------------ #
    def record_exit(
        self,
        signal_id: str,
        fill_price: float,
        ts: datetime,
        reason: str,
    ) -> TradeRecord:
        """Persist a human-confirmed exit (or stop) and return the trade record."""
        moment = to_utc(ts)
        fill = float(fill_price)
        if fill <= 0:
            raise ValueError(f"fill_price must be > 0, got {fill_price!r}")
        reason_text = str(reason or "MANUAL").strip().upper() or "MANUAL"
        status = (
            TradeStatus.STOPPED if reason_text in _STOP_REASONS else TradeStatus.CLOSED
        )
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM trades WHERE signal_id = ? AND status = ? "
                "ORDER BY trade_id DESC LIMIT 1",
                (str(signal_id), TradeStatus.OPEN.value),
            ).fetchone()
            if row is None:
                raise ValueError(f"没有找到信号 {signal_id} 的未平仓交易记录")
            fees = round(float(row["shares"]) * self._commission_per_share, 10)
            self._conn.execute(
                """
                UPDATE trades
                   SET exit_time_utc = ?, exit_price_fill = ?, exit_reason = ?,
                       fees = ?, status = ?
                 WHERE trade_id = ?
                """,
                (
                    self._iso(moment),
                    fill,
                    reason_text,
                    fees,
                    status.value,
                    int(row["trade_id"]),
                ),
            )
            self._conn.execute(
                "UPDATE signals SET status = ? WHERE signal_id = ?",
                (status.value, str(signal_id)),
            )
            updated = self._conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (int(row["trade_id"]),)
            ).fetchone()
        return self._row_to_trade(updated)

    def record_stop(self, signal_id: str, fill_price: float, ts: datetime) -> TradeRecord:
        """Convenience wrapper: ``record_exit(..., reason="STOP")``."""
        return self.record_exit(signal_id, fill_price, ts, "STOP")

    def confirm_flat(self) -> int:
        """Mark every pending signal and open trade as CANCELLED.

        Administrative reset after the user confirms the broker account is
        flat; it never invents fills or PnL.  Returns the number of rows changed.
        """
        with self._lock, self._conn:
            changed = 0
            for old_status in (TradeStatus.PENDING, TradeStatus.OPEN):
                cursor = self._conn.execute(
                    "UPDATE signals SET status = ? WHERE status = ?",
                    (TradeStatus.CANCELLED.value, old_status.value),
                )
                changed += max(int(cursor.rowcount), 0)
            cursor = self._conn.execute(
                "UPDATE trades SET status = ?, exit_reason = 'FLAT_CONFIRMED' "
                "WHERE status = ?",
                (TradeStatus.CANCELLED.value, TradeStatus.OPEN.value),
            )
            changed += max(int(cursor.rowcount), 0)
        return changed

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def open_positions(self) -> list[Position]:
        """All OPEN trades as :class:`Position` objects (oldest first)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE status = ? ORDER BY trade_id",
                (TradeStatus.OPEN.value,),
            ).fetchall()
        return [self._row_to_position(row) for row in rows]

    def closed_trades(self, et_date: date | None = None) -> list[TradeRecord]:
        """CLOSED / STOPPED trades, optionally filtered by ET trading date."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trades WHERE status IN (?, ?) ORDER BY trade_id",
                (TradeStatus.CLOSED.value, TradeStatus.STOPPED.value),
            ).fetchall()
        trades = [self._row_to_trade(row) for row in rows]
        if et_date is not None:
            trades = [
                trade
                for trade in trades
                if trade.exit_time_utc is not None
                and _et_date_of(trade.exit_time_utc) == et_date
            ]
        return trades

    def daily_stats(self, et_date: date | None = None) -> dict[str, Any]:
        """Aggregated stats for one ET session (defaults to the clock's date)."""
        target = et_date if et_date is not None else _et_date_of(self.now())
        trades = self.closed_trades(target)
        count = len(trades)
        wins = sum(1 for trade in trades if trade.pnl > 0)
        losses = sum(1 for trade in trades if trade.pnl < 0)
        stops = sum(
            1 for trade in trades if trade.status is TradeStatus.STOPPED
        )
        pnl = sum(trade.pnl for trade in trades)
        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = -sum(trade.pnl for trade in trades if trade.pnl < 0)
        return {
            "et_date": target.isoformat(),
            "trades": count,
            "wins": wins,
            "losses": losses,
            "stops": stops,
            "pnl": pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "win_rate": (wins / count) if count else 0.0,
            "expectancy": (pnl / count) if count else 0.0,
            "avg_r": (
                sum(trade.r_multiple for trade in trades) / count if count else 0.0
            ),
        }

    def slippage_stats(self) -> dict[str, Any]:
        """Realised execution slippage (bps) from stored signal/fill prices.

        Entry slippage = ``(fill - signal) / signal * 10000`` (buy side, so a
        positive number is worse than planned).  Exit slippage is only defined
        when the planned exit price is known from the exit reason (STOP ->
        stop price, TAKE_PROFIT -> target price); other exits are counted as
        ``exit_unknown_reference`` instead of being invented.
        """
        trades = self.closed_trades()
        entry_values: list[float] = []
        exit_values: list[float] = []
        unknown_exit_reference = 0
        by_symbol: dict[str, dict[str, Any]] = {}
        by_strategy: dict[str, dict[str, Any]] = {}

        def _bucket(mapping: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
            return mapping.setdefault(
                key,
                {"entry_values": [], "exit_values": [], "trades": 0},
            )

        for trade in trades:
            entry_bps = trade.entry_slippage_bps
            entry_values.append(entry_bps)
            reference = self._exit_reference(trade)
            exit_bps: float | None = None
            if reference is not None and trade.exit_price_fill is not None:
                exit_bps = (reference - trade.exit_price_fill) / reference * 10_000.0
                exit_values.append(exit_bps)
            else:
                unknown_exit_reference += 1
            for mapping, key in (
                (by_symbol, trade.symbol),
                (by_strategy, trade.strategy.value),
            ):
                bucket = _bucket(mapping, key)
                bucket["entry_values"].append(entry_bps)
                bucket["trades"] += 1
                if exit_bps is not None:
                    bucket["exit_values"].append(exit_bps)

        def _avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def _finalise(mapping: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
            result: dict[str, dict[str, Any]] = {}
            for key in sorted(mapping):
                bucket = mapping[key]
                result[key] = {
                    "trades": bucket["trades"],
                    "entry_avg_bps": _avg(bucket["entry_values"]),
                    "exit_avg_bps": _avg(bucket["exit_values"]),
                }
            return result

        return {
            "trades": len(trades),
            "entry_avg_bps": _avg(entry_values),
            "entry_count": len(entry_values),
            "exit_avg_bps": _avg(exit_values),
            "exit_count": len(exit_values),
            "exit_unknown_reference": unknown_exit_reference,
            "by_symbol": _finalise(by_symbol),
            "by_strategy": _finalise(by_strategy),
        }

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def record_event(self, event: AlertEvent) -> int:
        """Persist one :class:`AlertEvent`; returns the event row id."""
        if not isinstance(event, AlertEvent):
            raise TypeError(f"event must be an AlertEvent, got {type(event)!r}")
        payload = {
            "symbol": event.symbol,
            "signal_id": event.signal_id,
            "details": dict(event.details),
        }
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO events (ts_utc, kind, level, message, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    self._iso(event.ts_utc),
                    event.kind.value,
                    event.level.value,
                    event.message,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                ),
            )
            event_id = int(cursor.lastrowid)
        return event_id

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Newest-first list of event rows (JSON-friendly dicts)."""
        if limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            raw_payload = row["payload_json"]
            try:
                payload = json.loads(raw_payload) if raw_payload else {}
            except (TypeError, ValueError):  # pragma: no cover - defensive
                payload = {}
            events.append(
                {
                    "id": int(row["id"]),
                    "ts_utc": row["ts_utc"],
                    "kind": row["kind"],
                    "level": row["level"],
                    "message": row["message"],
                    "payload": payload,
                }
            )
        return events

    # ------------------------------------------------------------------ #
    # Row mapping
    # ------------------------------------------------------------------ #
    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        return TradeRecord(
            signal_id=row["signal_id"],
            symbol=row["symbol"],
            strategy=StrategyName(row["strategy"]),
            mode=Mode(row["mode"]) if row["mode"] else self._default_mode,
            shares=float(row["shares"]),
            entry_time_utc=self._parse_dt(row["entry_time_utc"]),
            entry_price_signal=float(row["entry_price_signal"]),
            entry_price_fill=float(row["entry_price_fill"]),
            stop_price=float(row["stop_price"]),
            target_price=float(row["target_price"]),
            exit_time_utc=self._parse_dt(row["exit_time_utc"]),
            exit_price_fill=(
                float(row["exit_price_fill"])
                if row["exit_price_fill"] is not None
                else None
            ),
            exit_reason=row["exit_reason"],
            fees=float(row["fees"] or 0.0),
            status=TradeStatus(row["status"]),
            trade_id=int(row["trade_id"]) if row["trade_id"] is not None else None,
        )

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        return Position(
            signal_id=row["signal_id"],
            symbol=row["symbol"],
            strategy=StrategyName(row["strategy"]),
            mode=Mode(row["mode"]) if row["mode"] else self._default_mode,
            shares=float(row["shares"]),
            entry_price=float(row["entry_price_fill"]),
            entry_utc=self._parse_dt(row["entry_time_utc"]),
            stop_loss=float(row["stop_price"]),
            take_profit=float(row["target_price"]),
            entry_price_signal=float(row["entry_price_signal"]),
        )

    @staticmethod
    def _exit_reference(trade: TradeRecord) -> float | None:
        reason = (trade.exit_reason or "").upper()
        if reason in _STOP_REASONS:
            return trade.stop_price
        if reason in _TARGET_REASONS:
            return trade.target_price
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Journal(db_path='{self._path}', default_mode={self._default_mode.value})"
