"""Live signal watcher: run the production wiring without the Rich dashboard.

Purpose: intraday **observation only** (no orders). It reuses the *production*
``Runtime`` from ``intraday_scalper.main`` (same Alpaca IEX provider, data
quality gate, 1m/5m aggregation, VWAP, strategies, risk manager and signal
engine), but instead of the full-screen Rich dashboard it:

* prints every engine alert (ENTRY_SIGNAL / hints / RISK_BLOCK / INFO / ...),
* appends each alert as one JSON line to ``var/live_signals/signals_<ET-date>.jsonl``,
* refreshes ``var/live_signals/watch_status.json`` periodically (market phase,
  data status, per-symbol bars/last price/session VWAP, pending candidates),
* stops automatically after the US regular session ends.

It never places orders and never modifies any production file.

Run from the repository root::

    python -m intraday_scalper.tools.live_watch
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from zoneinfo import ZoneInfo

from intraday_scalper.core.config import PACKAGE_ROOT, load_config
from intraday_scalper.core.timeutil import market_phase
from intraday_scalper.main import Runtime

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _et_minutes(moment: datetime) -> int:
    et = moment.astimezone(ET)
    return et.hour * 60 + et.minute


def _describe(event: Any) -> Dict[str, Any]:
    details = dict(event.details or {})
    candidate = details.get("candidate") if isinstance(details.get("candidate"), dict) else None
    record: Dict[str, Any] = {
        "ts_utc": event.ts_utc.isoformat(),
        "ts_et": event.ts_utc.astimezone(ET).isoformat(),
        "kind": event.kind.value,
        "level": event.level.value,
        "symbol": event.symbol,
        "signal_id": event.signal_id,
        "message": event.message,
    }
    if candidate:
        record["strategy"] = candidate.get("strategy")
        record["direction"] = candidate.get("direction")
        record["entry_ref"] = candidate.get("entry_ref")
        record["stop_loss"] = candidate.get("stop_loss")
        record["take_profit"] = candidate.get("take_profit")
    if details:
        record["details"] = {k: v for k, v in details.items() if k != "candidate"}
    return record


def _print_event(record: Dict[str, Any]) -> None:
    et = datetime.fromisoformat(record["ts_et"]).strftime("%H:%M:%S")
    parts = [f"[{et} ET] {record['kind']:<15s} {record.get('symbol') or '-':<5s}"]
    if record.get("strategy"):
        parts.append(f"strategy={record['strategy']}")
    if record.get("entry_ref") is not None:
        parts.append(f"entry~{record['entry_ref']}")
    if record.get("stop_loss") is not None:
        parts.append(f"stop={record['stop_loss']}")
    if record.get("take_profit") is not None:
        parts.append(f"target={record['take_profit']}")
    parts.append(record["message"])
    print(" ".join(parts), flush=True)


def run_watch(duration_seconds: Optional[float] = None, status_seconds: float = 5.0) -> int:
    config = load_config()
    log_dir = PACKAGE_ROOT / "var" / "live_signals"
    log_dir.mkdir(parents=True, exist_ok=True)
    now = _now_utc()
    signal_path = log_dir / f"signals_{now.astimezone(ET).date().isoformat()}.jsonl"
    status_path = log_dir / "watch_status.json"

    print(
        f"[watch] start {now.astimezone(KST).isoformat()} | phase={market_phase(now).value} | "
        f"signals -> {signal_path}",
        flush=True,
    )
    runtime = Runtime(config, token="", provider="alpaca", audio=False)
    startup = runtime.startup_validation()
    print(f"[watch] startup validation: {_compact_json(startup)}", flush=True)
    if runtime.alpaca is None:
        raise RuntimeError("Alpaca provider was not constructed")
    runtime.alpaca.start()
    connected = runtime.alpaca.wait_until_connected(timeout=20.0)
    print(f"[watch] provider connected={connected}", flush=True)

    status_thread = threading.Thread(target=runtime._status_loop, name="watch-status", daemon=True)
    status_thread.start()

    started = time.time()
    last_status = 0.0
    processed_events = 0
    try:
        while True:
            if duration_seconds is not None and time.time() - started > duration_seconds:
                print("[watch] duration reached, stopping", flush=True)
                break
            moment = _now_utc()
            if _et_minutes(moment) >= 16 * 60 + 5:
                print("[watch] US regular session finished, stopping", flush=True)
                break

            for event in runtime.drain_events():
                record = _describe(event)
                with signal_path.open("a", encoding="utf-8") as handle:
                    handle.write(_compact_json(record) + "\n")
                processed_events += 1
                _print_event(record)

            if time.time() - last_status >= status_seconds:
                last_status = time.time()
                snapshot = runtime.snapshot()
                try:
                    snap_dict = snapshot.to_dict()
                except Exception:  # noqa: BLE001
                    snap_dict = {}
                status = {
                    "updated_utc": _now_utc().isoformat(),
                    "phase": snap_dict.get("session_phase", market_phase(_now_utc()).value),
                    "mode": snap_dict.get("mode"),
                    "data_status": snap_dict.get("data_status"),
                    "connection_state": snap_dict.get("connection_state"),
                    "entries_today": snap_dict.get("entries_today"),
                    "daily_pnl": snap_dict.get("daily_pnl"),
                    "events_logged": processed_events,
                    "symbols": {
                        key: {
                            "last_price": value.get("last_price"),
                            "vwap": value.get("vwap"),
                            "bars_1m_count": value.get("bars_1m_count"),
                            "bars_5m_count": value.get("bars_5m_count"),
                            "data_age_seconds": value.get("data_age_seconds"),
                            "trend_1m": value.get("trend_1m"),
                            "trend_5m": value.get("trend_5m"),
                        }
                        for key, value in (snap_dict.get("symbols") or {}).items()
                    },
                    "latest_signal": snap_dict.get("latest_signal"),
                    "latest_plan": snap_dict.get("latest_plan"),
                }
                status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                age = ", ".join(
                    f"{sym}={val.get('last_price')} vwap={val.get('vwap')} bars5m={val.get('bars_5m_count')}"
                    for sym, val in status["symbols"].items()
                )
                print(f"[watch] {status['phase']} data={status['data_status']} {age}", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("[watch] interrupted", flush=True)
    finally:
        runtime.shutdown()
        print(f"[watch] stopped after {processed_events} events", flush=True)
    return 0


def _snapshot_bar_count(snapshot: Any) -> str:
    if isinstance(snapshot, dict):
        for key in ("bars", "bar_count", "counts", "symbols"):
            value = snapshot.get(key)
            if isinstance(value, dict):
                return str(len(value))
            if isinstance(value, int):
                return str(value)
    return "?"


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(description="live intraday signal watcher (observation only, no orders)")
    parser.add_argument("--duration", type=float, default=None, help="stop after N seconds (default: until US close)")
    parser.add_argument("--status-seconds", type=float, default=5.0, help="status json refresh interval")
    args = parser.parse_args(argv)
    return run_watch(duration_seconds=args.duration, status_seconds=args.status_seconds)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
