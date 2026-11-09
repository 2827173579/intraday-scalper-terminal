"""Export the full interactive V0.5(A+F) Web Terminal to GitHub Pages (``docs/``).

Generates:
* ``docs/state_bundle.json`` — unified JSON snapshot containing ``state``,
  ``replay``, ``extended``, and multi-timeframe ``candles`` (``1m``, ``5m``,
  ``15m``, ``1d``) for ``SOXL`` and ``SOXS``.
* ``docs/index.html`` — full interactive terminal derived from
  ``intraday_scalper/web/terminal.html`` with ``window.__STATIC_BUNDLE__``
  embedded for zero-latency startup and automatic polling of
  ``./state_bundle.json``.

Works both locally (using ``data/*.parquet`` + Alpaca IEX REST) and in GitHub
Actions (automatically falling back to ``yfinance`` when local parquet or
Alpaca keys are absent).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.core.models import Bar
from intraday_scalper.research import v05
from intraday_scalper.web import terminal_server as ts

REPO_ROOT = PACKAGE_ROOT.parent
DOCS_DIR = REPO_ROOT / "docs"


def _yfinance_fallback_bars(symbols: tuple[str, ...] = ts.SYMBOLS) -> tuple[Dict[str, float], Dict[str, List[Any]], Dict[str, List[Any]], List[Bar]]:
    """Fetch recent 1m and 1d bars via yfinance when Alpaca/local parquet are absent."""
    prev_closes: Dict[str, float] = {}
    today_bars: Dict[str, List[Any]] = {sym: [] for sym in symbols}
    daily_bars: Dict[str, List[Any]] = {sym: [] for sym in symbols}
    warmup_bars: List[Bar] = []
    try:
        import pandas as pd
        import yfinance as yf

        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            try:
                hist_1d = ticker.history(period="6mo", interval="1d", auto_adjust=False)
                if hist_1d is not None and not hist_1d.empty:
                    idx = pd.to_datetime(hist_1d.index, utc=True)
                    records_1d = []
                    for ts_val, row in zip(idx, hist_1d.itertuples(index=False)):
                        c = float(getattr(row, "Close", 0.0) or 0.0)
                        if c <= 0:
                            continue
                        records_1d.append(
                            SimpleNamespace(
                                timestamp=ts_val.to_pydatetime(),
                                open=float(getattr(row, "Open", c) or c),
                                high=float(getattr(row, "High", c) or c),
                                low=float(getattr(row, "Low", c) or c),
                                close=c,
                                volume=float(getattr(row, "Volume", 0.0) or 0.0),
                            )
                        )
                    daily_bars[symbol] = records_1d[-160:]
                    if len(records_1d) >= 2:
                        prev_closes[symbol] = float(records_1d[-2].close)
                    elif records_1d:
                        prev_closes[symbol] = float(records_1d[-1].close)
            except Exception:  # noqa: BLE001
                pass

            try:
                hist_1m = ticker.history(period="5d", interval="1m", prepost=False, auto_adjust=False)
                if hist_1m is not None and not hist_1m.empty:
                    idx = pd.to_datetime(hist_1m.index, utc=True)
                    records_1m = []
                    for ts_val, row in zip(idx, hist_1m.itertuples(index=False)):
                        dt_utc = ts_val.to_pydatetime()
                        et_dt = dt_utc.astimezone(ts.ET)
                        if not (dt_time(9, 30) <= et_dt.time() < dt_time(16, 0)):
                            continue
                        c = float(getattr(row, "Close", 0.0) or 0.0)
                        if c <= 0:
                            continue
                        o = float(getattr(row, "Open", c) or c)
                        h = float(getattr(row, "High", c) or c)
                        l = float(getattr(row, "Low", c) or c)
                        v = float(getattr(row, "Volume", 0.0) or 0.0)
                        vwap = (h + l + c) / 3.0
                        records_1m.append(
                            SimpleNamespace(
                                timestamp=dt_utc,
                                open=o,
                                high=h,
                                low=l,
                                close=c,
                                volume=v,
                                vwap=vwap,
                                trade_count=0,
                            )
                        )
                        warmup_bars.append(
                            Bar(
                                symbol=symbol,
                                timeframe_minutes=1,
                                start_utc=dt_utc,
                                end_utc=dt_utc + timedelta(minutes=1),
                                open=o,
                                high=h,
                                low=l,
                                close=c,
                                volume=v,
                                vwap=vwap,
                                trade_count=0,
                            )
                        )
                    today_bars[symbol] = records_1m
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    warmup_bars.sort(key=lambda b: (b.start_utc, b.symbol))
    return prev_closes, today_bars, daily_bars, warmup_bars


def build_static_bundle() -> Dict[str, Any]:
    """Build the complete static bundle for GitHub Pages."""
    config = ts.b1_config()
    tracker = ts.SessionTracker()
    local = ts.load_local_session()
    local_session = None
    if local.get("records"):
        tracker.backfill(local["records"])
        local_session = tracker.current_day

    prev_closes, today_bars, daily_bars = ts.fetch_startup_market_data()
    rest_dates = [
        bar.timestamp.astimezone(ts.ET).date()
        for bars in (today_bars or {}).values()
        for bar in bars
        if getattr(bar, "timestamp", None) is not None
    ]
    rest_latest = max(rest_dates) if rest_dates else None
    if rest_latest is not None and (local_session is None or rest_latest > local_session):
        tracker.backfill(today_bars)
    if not prev_closes:
        prev_closes = ts.derive_prev_closes_local()
    if daily_bars:
        tracker.backfill_daily(daily_bars)

    history_bars = ts.load_local_history_bars(30)
    if not tracker.books["SOXL"].bars_1m and not tracker.books["SOXS"].bars_1m:
        yf_prev, yf_1m, yf_1d, yf_warmup = _yfinance_fallback_bars()
        if yf_1m.get("SOXL") or yf_1m.get("SOXS"):
            tracker.backfill(yf_1m)
        if yf_1d.get("SOXL") or yf_1d.get("SOXS"):
            tracker.backfill_daily(yf_1d)
        if not prev_closes:
            prev_closes = yf_prev
        if not history_bars:
            history_bars = yf_warmup

    # Build daily bars from local parquet if REST daily_bars wasn't populated
    if not tracker.daily_bars.get("SOXL") and local.get("session_date"):
        try:
            import pandas as pd

            for symbol in ts.SYMBOLS:
                path = ts.LOCAL_DATA_DIR / f"{symbol}_1m_raw.parquet"
                if not path.exists():
                    continue
                df = pd.read_parquet(path, columns=["start_utc", "open", "high", "low", "close", "volume"])
                df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
                df["et_day"] = df["start_utc"].dt.tz_convert(ts.ET).dt.date
                grouped = df.groupby("et_day").agg(
                    start_utc=("start_utc", "first"),
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                ).tail(160)
                tracker.daily_bars[symbol] = [
                    {
                        "start_utc": row.start_utc.to_pydatetime(),
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        "volume": float(row.volume),
                    }
                    for row in grouped.itertuples(index=False)
                ]
        except Exception:  # noqa: BLE001
            pass

    current = tracker.current_day
    session_bars: List[Bar] = []
    for symbol in ts.SYMBOLS:
        for record in tracker.bars_for(symbol, "1m"):
            session_bars.append(
                Bar(
                    symbol=symbol,
                    timeframe_minutes=1,
                    start_utc=record["start_utc"],
                    end_utc=record["end_utc"],
                    open=float(record["open"]),
                    high=float(record["high"]),
                    low=float(record["low"]),
                    close=float(record["close"]),
                    volume=float(record["volume"]),
                    vwap=float(record["vwap"]),
                    trade_count=0,
                )
            )
    prior_bars = [b for b in history_bars if b.start_utc.astimezone(ts.ET).date() < current]
    all_bars = prior_bars + session_bars

    trades: List[Dict[str, Any]] = []
    if all_bars and current is not None:
        try:
            trades = v05.evaluate_v05_session(all_bars, current, variant=v05.V05_PRODUCTION_VARIANT)
        except Exception:  # noqa: BLE001
            trades = []

    replay = ts.replay_payload()
    if not trades and replay.get("recap"):
        v05_recap = ((replay["recap"].get("versions") or {}).get("V0.5") or {})
        trades = list(v05_recap.get("trades") or [])

    markers: List[Dict[str, Any]] = []
    latest_trade = None
    for trade in trades:
        latest_trade = trade
        for marker in trade.get("markers", []) or []:
            markers.append(marker)
    if not markers and replay.get("markers"):
        markers = list(replay.get("markers") or [])

    wins = sum(1 for t in trades if float(t.get("pnl", 0.0)) > 0)
    stops = sum(1 for t in trades if float(t.get("pnl", 0.0)) < 0)
    net_pnl = round(sum(float(t.get("pnl", 0.0)) for t in trades), 2)

    v04_state = {
        "session_date": current.isoformat() if current else replay.get("session_date"),
        "trades": trades[-20:],
        "markers": markers,
        "summary": {
            "trades": len(trades),
            "wins": wins,
            "net": net_pnl,
        },
        "latest_trade": latest_trade,
        "variant": "V0.5 A+F",
    }

    now_et = ts.et_now()
    if now_et.weekday() >= 5:
        phase = "WEEKEND"
    elif dt_time(9, 30) <= now_et.time() < dt_time(16, 0):
        phase = "REGULAR"
    elif dt_time(4, 0) <= now_et.time() < dt_time(9, 30):
        phase = "PRE_MARKET"
    elif dt_time(16, 0) <= now_et.time() < dt_time(20, 0):
        phase = "AFTER_HOURS"
    else:
        phase = "CLOSED"

    snapshot_dict = {
        "mode": "DUAL_WATCH",
        "session_phase": phase,
        "data_status": "OK",
        "connection_state": "CONNECTED",
        "strategy_names": ["V0.5_A+F", "EMA_PULLBACK", "SMC_SWEEP", "VWAP_RECLAIM"],
        "entries_today": len(trades),
        "max_entries_per_day": 15,
        "trades_today": len(trades),
        "wins_today": wins,
        "stops_today": stops,
        "daily_pnl": net_pnl,
        "daily_stopped": net_pnl <= -320.0,
        "risk": {
            "consecutive_losses": 0,
            "cooldown_until_utc": None,
            "daily_realized_pnl": net_pnl,
            "daily_stopped": net_pnl <= -320.0,
        },
        "latest_signal": None,
        "latest_plan": None,
        "position": None,
    }

    events: List[Dict[str, Any]] = []
    for marker in markers[-8:]:
        mtype = str(marker.get("type", "")).upper()
        kind = "ENTRY_SIGNAL" if mtype == "ENTRY" else ("TAKE_PROFIT_HINT" if mtype in {"TARGET", "LADDER_STOP"} else "INFO")
        events.append(
            {
                "ts_utc": marker.get("time_utc"),
                "ts_et": marker.get("time_utc"),
                "kind": kind,
                "level": "CRITICAL" if kind == "ENTRY_SIGNAL" else "INFO",
                "symbol": marker.get("symbol"),
                "signal_id": marker.get("trade_id"),
                "message": f"V0.5(A+F) {mtype} {marker.get('symbol')} @ {marker.get('price')} · {marker.get('strategy', '')}",
            }
        )

    tracker_state = tracker.snapshot()
    ui_state = ts.build_ui_state(
        snapshot_dict,
        tracker_state,
        config,
        prev_closes=prev_closes,
        events=events,
        provider_name="github_pages",
    )
    ui_state["v04"] = v04_state

    extended = ts.extended_payload(force=True)
    candles: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for symbol in ts.SYMBOLS:
        candles[symbol] = {
            "1m": tracker.candles(symbol, "1m"),
            "5m": tracker.candles(symbol, "5m"),
            "15m": tracker.candles(symbol, "15m"),
            "1d": tracker.candles(symbol, "1d"),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": ui_state,
        "replay": replay,
        "extended": extended,
        "candles": candles,
    }


def export_pages(output_dir: Path = DOCS_DIR) -> tuple[Path, Path]:
    """Write ``state_bundle.json`` and ``index.html`` into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "state_bundle.json"

    # Preserve prior candles if running in a restricted environment where both local and network bars failed
    prior_bundle: Dict[str, Any] = {}
    if bundle_path.exists():
        try:
            prior_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prior_bundle = {}

    bundle = build_static_bundle()
    for sym in ts.SYMBOLS:
        if not (bundle["candles"].get(sym, {}).get("1m")) and (prior_bundle.get("candles", {}).get(sym, {}).get("1m")):
            bundle["candles"][sym] = prior_bundle["candles"][sym]
    if not (bundle["extended"].get("symbols")) and (prior_bundle.get("extended", {}).get("symbols")):
        bundle["extended"] = prior_bundle["extended"]

    bundle_json = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), default=str)
    bundle_path.write_text(bundle_json, encoding="utf-8")

    template_html = ts.HTML_PATH.read_text(encoding="utf-8")
    inject_script = f"<script>window.__STATIC_BUNDLE__ = {bundle_json};</script>\n<script>"
    assert "<script>" in template_html, "Expected <script> tag in terminal.html"
    index_html = template_html.replace("<script>", inject_script, 1)
    index_path = output_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    return bundle_path, index_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export V0.5(A+F) web terminal to docs/ for GitHub Pages")
    parser.add_argument("--out", type=Path, default=DOCS_DIR, help="Output directory (default: docs/)")
    args = parser.parse_args(argv)
    bundle_path, index_path = export_pages(args.out)
    print(
        f"[export_github_pages] wrote {bundle_path.name} ({bundle_path.stat().st_size:,} bytes) "
        f"and {index_path.name} ({index_path.stat().st_size:,} bytes)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
