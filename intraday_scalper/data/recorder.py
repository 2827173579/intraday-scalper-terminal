"""Local persistence for raw ticks and aggregated bars.

Layout (under ``PathsConfig``, i.e. ``intraday_scalper/var/`` which is ignored by
git):

``var/raw_ticks/YYYY-MM-DD.jsonl``
    One JSON object per websocket tick, keyed by the *ET* trading date
    (``Tick.to_dict()``), appended line-buffered so a crash loses at most the
    last line.
``var/bars/SYMBOL_{tf}m/part-*.parquet``
    Completed OHLCV bars, written as immutable pyarrow shards. Bars are written
    **through** (one small shard per bar) so nothing is lost if the process dies
    before :meth:`DataRecorder.close`; the data-layer contract explicitly allows
    sharding. :meth:`DataRecorder.load_bars` reassembles every shard, sorted by
    time, for the backtester.

A future optimisation (not needed for phase 1: ~2 symbols x ~470 bars/day) could
compact the per-bar shards into one daily file; the read path already tolerates
any mix of ``*.parquet`` shards in the directory.
"""

from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, TextIO

import pyarrow as pa
import pyarrow.parquet as pq

from intraday_scalper.core import UTC, Bar, PathsConfig, Tick, et_date, now_utc

__all__ = ["DataRecorder"]

_BAR_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("timeframe_minutes", pa.int32()),
        ("start_utc", pa.timestamp("us", tz="UTC")),
        ("end_utc", pa.timestamp("us", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("vwap", pa.float64()),
        ("trade_count", pa.int32()),
        ("is_complete", pa.bool_()),
    ]
)


class DataRecorder:
    """Thread-safe writer/reader for raw ticks and completed bars."""

    def __init__(self, paths: PathsConfig, clock: Any = now_utc) -> None:
        self._paths = paths
        self._clock = clock
        self._lock = threading.RLock()
        self._tick_files: dict[date, TextIO] = {}
        self._shard_index = 0
        self._tick_count = 0
        self._bar_count = 0
        self._closed = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def append_tick(self, tick: Tick) -> Path:
        """Append one tick to ``var/raw_ticks/<ET-date>.jsonl``; returns the path."""
        if not isinstance(tick, Tick):
            raise TypeError(f"append_tick expects a Tick, got {type(tick)!r}")
        day = et_date(tick.ts_utc)
        path = self.tick_path(day)
        line = json.dumps(tick.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._ensure_open_locked()
            handle = self._tick_files.get(day)
            if handle is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a", encoding="utf-8", newline="\n", buffering=1)
                self._tick_files[day] = handle
            handle.write(line + "\n")
            self._tick_count += 1
        return path

    def append_bar(self, bar: Bar) -> Path:
        """Persist one bar immediately as a pyarrow parquet shard; returns the path."""
        if not isinstance(bar, Bar):
            raise TypeError(f"append_bar expects a Bar, got {type(bar)!r}")
        with self._lock:
            self._ensure_open_locked()
        return self._write_shard(bar.symbol, bar.timeframe_minutes, [bar])

    def flush(self) -> None:
        """Flush buffered raw-tick file handles to disk (bars are already written)."""
        with self._lock:
            self._ensure_open_locked()
            for handle in self._tick_files.values():
                handle.flush()

    def close(self) -> None:
        """Flush and close the raw-tick handles (idempotent)."""
        with self._lock:
            if self._closed:
                return
        try:
            self.flush()
        finally:
            with self._lock:
                self._closed = True
                handles = list(self._tick_files.values())
                self._tick_files.clear()
            for handle in handles:
                try:
                    handle.close()
                except OSError:  # pragma: no cover - defensive
                    pass

    # ------------------------------------------------------------------ #
    # Loading (backtest / verification)
    # ------------------------------------------------------------------ #
    def load_bars(self, symbol: str, timeframe_minutes: int = 1) -> list[Bar]:
        """Read every recorded shard for ``symbol``/``timeframe``, time-sorted."""
        name = str(symbol).strip().upper()
        timeframe = int(timeframe_minutes)
        sources: list[Path] = []
        with self._lock:
            directory = self.bar_directory(name, timeframe)
            if directory.is_dir():
                sources.extend(sorted(directory.glob("*.parquet")))
            single = self._single_bar_path(name, timeframe)
            if single.is_file():
                sources.append(single)

        bars: list[Bar] = []
        for path in sources:
            table = pq.read_table(path, schema=_BAR_SCHEMA)
            for row in table.to_pylist():
                row["start_utc"] = _as_utc(row["start_utc"])
                row["end_utc"] = _as_utc(row["end_utc"])
                row["timeframe_minutes"] = int(row["timeframe_minutes"])
                row["trade_count"] = int(row["trade_count"])
                row["is_complete"] = bool(row["is_complete"])
                bars.append(Bar(**row))
        bars.sort(key=lambda bar: (bar.start_utc, bar.end_utc))
        return bars

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    def tick_path(self, et_day: date) -> Path:
        """``var/raw_ticks/YYYY-MM-DD.jsonl`` for an exchange trading date."""
        return self._paths.raw_tick_dir / f"{et_day.isoformat()}.jsonl"

    def bar_directory(self, symbol: str, timeframe_minutes: int = 1) -> Path:
        """Directory holding the parquet shards for one symbol/timeframe."""
        return self._paths.bar_dir / f"{str(symbol).strip().upper()}_{int(timeframe_minutes)}m"

    def _single_bar_path(self, symbol: str, timeframe_minutes: int) -> Path:
        """Legacy single-file layout; read for backwards compatibility only."""
        return self._paths.bar_dir / f"{str(symbol).strip().upper()}_{int(timeframe_minutes)}m.parquet"

    # ------------------------------------------------------------------ #
    # Counters / state
    # ------------------------------------------------------------------ #
    @property
    def tick_count(self) -> int:
        with self._lock:
            return self._tick_count

    @property
    def bar_count(self) -> int:
        with self._lock:
            return self._bar_count

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("DataRecorder is closed; create a new recorder")

    def _write_shard(self, symbol: str, timeframe_minutes: int, bars: list[Bar]) -> Path:
        directory = self.bar_directory(symbol, timeframe_minutes)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._shard_index += 1
            index = self._shard_index
            stamp = self._clock().astimezone(UTC).strftime("%Y%m%dT%H%M%S%f")
        path = directory / f"part-{index:06d}-{stamp}.parquet"
        table = pa.table(
            {
                "symbol": pa.array([bar.symbol for bar in bars], type=pa.string()),
                "timeframe_minutes": pa.array(
                    [int(bar.timeframe_minutes) for bar in bars], type=pa.int32()
                ),
                "start_utc": pa.array([bar.start_utc for bar in bars], type=pa.timestamp("us", tz="UTC")),
                "end_utc": pa.array([bar.end_utc for bar in bars], type=pa.timestamp("us", tz="UTC")),
                "open": pa.array([bar.open for bar in bars], type=pa.float64()),
                "high": pa.array([bar.high for bar in bars], type=pa.float64()),
                "low": pa.array([bar.low for bar in bars], type=pa.float64()),
                "close": pa.array([bar.close for bar in bars], type=pa.float64()),
                "volume": pa.array([bar.volume for bar in bars], type=pa.float64()),
                "vwap": pa.array([bar.vwap for bar in bars], type=pa.float64()),
                "trade_count": pa.array([int(bar.trade_count) for bar in bars], type=pa.int32()),
                "is_complete": pa.array([bool(bar.is_complete) for bar in bars], type=pa.bool_()),
            },
            schema=_BAR_SCHEMA,
        )
        pq.write_table(table, path)
        with self._lock:
            self._bar_count += len(bars)
        return path


def _as_utc(value: datetime) -> datetime:
    """Normalise a pyarrow-decoded datetime to timezone-aware UTC."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
