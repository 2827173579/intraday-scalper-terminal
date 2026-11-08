"""Extended-hours (pre/post market) data + indicators + intraday signal anchors.

Why this exists
---------------
Alpaca's free IEX stream has no pre-market trades, so the regular-session engine
starts at 09:30 ET with cold 5m EMA/ATR series and ignores pre-market structure.
SOXL / SOXS often price most of the move before 09:30 (gap up/down >8%).

This module uses ``yfinance`` 5-minute bars with ``prepost=True`` (04:00-20:00 ET)
to build a **display / signal-observation** extended-hours context:

* continuous 5m EMA9 / EMA21 / ATR(14) across pre + regular + post bars;
* pre-market anchors ``PM_High`` / ``PM_Low`` / ``PM_VWAP`` (04:00-09:30 ET);
* regular-session VWAP for right-side confirmation;
* deterministic signal candidates:
  - ``SMC_PM_LOW_SWEEP``  : regular bar sweeps PM_Low then closes back above it,
    with close >= EMA21 and close >= session VWAP (high-priority buy, next bar open);
  - ``PM_HIGH_BREAKOUT``  : breakout above PM_High, then a pullback that holds
    EMA9 / VWAP (trend continuation);
  - ``EMA_PULLBACK`` / ``SMC_SWEEP`` inside the pre window 07:30-09:30 ET and
    post window 16:00-17:30 ET (labelled 盘前/盘后 limit-order candidates).

All levels are causal (only completed 5m bars).  yfinance extended bars have
``volume == 0`` in most cases, so VWAP uses volume weighting when volume exists
otherwise an arithmetic mean of bar typical prices (documented in the payload).

No broker orders are placed; this is observation/display only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

from intraday_scalper.core.models import Bar
from intraday_scalper.core.config import PACKAGE_ROOT

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
SYMBOLS = ("SOXL", "SOXS")

PRE_START = time(4, 0)
PRE_END = time(9, 30)
REGULAR_END = time(16, 0)
POST_END = time(20, 0)
PRE_SIGNAL_START = time(7, 30)
PRE_SIGNAL_END = time(9, 30)
POST_SIGNAL_START = time(16, 0)
POST_SIGNAL_END = time(17, 30)

EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14
TARGET_R = 2.0
STOP_ATR_BUFFER = 0.25
SWEEP_LOOKBACK = 20


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable offline)
# --------------------------------------------------------------------------- #
def ema_series(values: Sequence[float], span: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def atr_series(bars: Sequence[Bar], period: int = ATR_PERIOD) -> List[Optional[float]]:
    """Simple rolling true-range ATR on completed bars (causal)."""
    out: List[Optional[float]] = []
    ranges: List[float] = []
    for index, bar in enumerate(bars):
        prev_close = float(bars[index - 1].close) if index > 0 else float(bar.close)
        tr = max(
            float(bar.high) - float(bar.low),
            abs(float(bar.high) - prev_close),
            abs(float(bar.low) - prev_close),
        )
        ranges.append(tr)
        if len(ranges) < period:
            out.append(None)
        else:
            out.append(sum(ranges[-period:]) / period)
    return out


def phase_of(ts_et: datetime) -> str:
    clock = ts_et.time()
    if PRE_START <= clock < PRE_END:
        return "pre"
    if PRE_END <= clock < REGULAR_END:
        return "regular"
    if REGULAR_END <= clock < POST_END:
        return "post"
    return "closed"


def session_vwap(bars: Sequence[Bar]) -> Optional[float]:
    total_value = 0.0
    total_volume = 0.0
    for bar in bars:
        volume = float(bar.volume or 0.0)
        if volume <= 0:
            continue
        typical = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        total_value += typical * volume
        total_volume += volume
    if total_volume <= 0:
        return None
    return total_value / total_volume


def mean_price(bars: Sequence[Bar]) -> Optional[float]:
    if not bars:
        return None
    return sum((float(bar.high) + float(bar.low) + float(bar.close)) / 3.0 for bar in bars) / len(bars)


def build_pre_market_anchors(bars: Sequence[Bar], session_date: Optional[date] = None) -> Dict[str, Any]:
    pre = [
        bar
        for bar in bars
        if phase_of(bar.start_utc.astimezone(ET)) == "pre"
        and (session_date is None or bar.start_utc.astimezone(ET).date() == session_date)
    ]
    if not pre:
        return {"ready": False, "pm_high": None, "pm_low": None, "pm_vwap": None, "bars": 0}
    pm_vwap = session_vwap(pre)
    if pm_vwap is None:
        pm_vwap = mean_price(pre)
    return {
        "ready": True,
        "pm_high": round(max(float(bar.high) for bar in pre), 4),
        "pm_low": round(min(float(bar.low) for bar in pre), 4),
        "pm_vwap": round(pm_vwap, 4) if pm_vwap is not None else None,
        "bars": len(pre),
    }


# --------------------------------------------------------------------------- #
# Signal candidates
# --------------------------------------------------------------------------- #
def _make_signal(
    *,
    kind: str,
    symbol: str,
    bar: Bar,
    entry: float,
    stop: float,
    target: float,
    window: str,
    note: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "type": kind,
        "symbol": symbol,
        "window": window,
        "time_utc": bar.start_utc.isoformat(),
        "t": int(bar.start_utc.timestamp() * 1000),
        "price": round(float(bar.close), 4),
        "entry": round(float(entry), 4),
        "stop": round(float(stop), 4),
        "target": round(float(target), 4),
        "note": note,
    }
    if extra:
        payload.update(extra)
    return payload


def evaluate_extended_signals(
    symbol: str,
    bars: Sequence[Bar],
    *,
    session_date: Optional[date] = None,
    require_pm_sweep_confirm: bool = True,
    min_signal_spacing_minutes: int = 20,
) -> List[Dict[str, Any]]:
    """Build causal, display-only signal candidates for **one session**.

    ``bars`` may include previous sessions for 5m EMA/ATR warm-up; only bars whose
    ET date equals ``session_date`` (default: latest date in ``bars``) can emit
    signals.  EMA_PULLBACK / SMC_SWEEP are de-duplicated with a 20-minute spacing.
    """
    ordered = sorted(bars, key=lambda bar: bar.start_utc)
    if not ordered:
        return []
    target_date = session_date or ordered[-1].start_utc.astimezone(ET).date()
    closes = [float(bar.close) for bar in ordered]
    ema9 = ema_series(closes, EMA_FAST)
    ema21 = ema_series(closes, EMA_SLOW)
    atrs = atr_series(ordered)
    anchors = build_pre_market_anchors(ordered, target_date)
    signals: List[Dict[str, Any]] = []
    last_window_signal: Dict[str, datetime] = {}

    regular_bars = [bar for bar in ordered if phase_of(bar.start_utc.astimezone(ET)) == "regular"]
    running_regular: List[Bar] = []
    for index, bar in enumerate(ordered):
        ts_et = bar.start_utc.astimezone(ET)
        if ts_et.date() != target_date:
            continue
        phase = phase_of(ts_et)
        atr = atrs[index]
        if phase == "regular":
            running_regular.append(bar)
        vwap = session_vwap(running_regular) if running_regular else None
        if vwap is None:
            vwap = session_vwap(ordered[: index + 1])
        e9, e21 = ema9[index], ema21[index]
        if atr is None or not is_finite(atr) or atr <= 0:
            continue

        # -- 1) SMC_PM_LOW_SWEEP: sweep pre-market low then reclaim, with right-side confirm
        regular_pm_key = "regular:sweep"
        regular_recent = last_window_signal.get(regular_pm_key)
        regular_spacing = regular_recent is None or (ts_et - regular_recent).total_seconds() >= 30 * 60
        if (
            phase == "regular"
            and anchors.get("ready")
            and len(running_regular) >= 2
            and regular_spacing
        ):
            pm_low = float(anchors["pm_low"])
            hit_low = float(bar.low) < pm_low
            reclaim = float(bar.close) > pm_low
            if hit_low and reclaim:
                confirm = (float(bar.close) >= e21) and (vwap is None or float(bar.close) >= vwap)
                if confirm or not require_pm_sweep_confirm:
                    stop = min(pm_low, float(bar.low)) - STOP_ATR_BUFFER * atr
                    entry = float(bar.close)
                    risk = max(entry - stop, 1e-6)
                    last_window_signal[regular_pm_key] = ts_et
                    signals.append(
                        _make_signal(
                            kind="SMC_PM_LOW_SWEEP",
                            symbol=symbol,
                            bar=bar,
                            entry=entry,
                            stop=stop,
                            target=entry + TARGET_R * risk,
                            window="regular",
                            note="盘前低点刺穿收回 + 站上 EMA21/VWAP（高优先级买点）",
                            extra={"pm_low": pm_low},
                        )
                    )
        # -- 2) PM_HIGH_BREAKOUT: break pre-market high, then pullback holds EMA9/VWAP
        regular_break_key = "regular:breakout"
        break_recent = last_window_signal.get(regular_break_key)
        break_spacing = break_recent is None or (ts_et - break_recent).total_seconds() >= 30 * 60
        if phase == "regular" and anchors.get("ready") and break_spacing:
            pm_high = float(anchors["pm_high"])
            recent = ordered[max(0, index - 8) : index]
            breakout_recently = any(
                phase_of(item.start_utc.astimezone(ET)) == "regular" and float(item.close) > pm_high
                for item in recent
            )
            pullback_touch = float(bar.low) <= e9 and float(bar.close) >= e9
            holds_vwap = vwap is None or float(bar.close) >= vwap
            holds_ema21 = float(bar.close) >= e21
            if breakout_recently and pullback_touch and holds_vwap and holds_ema21:
                stop = min(float(bar.low), e21) - STOP_ATR_BUFFER * atr
                entry = float(bar.close)
                risk = max(entry - stop, 1e-6)
                last_window_signal[regular_break_key] = ts_et
                signals.append(
                    _make_signal(
                        kind="PM_HIGH_BREAKOUT",
                        symbol=symbol,
                        bar=bar,
                        entry=entry,
                        stop=stop,
                        target=entry + TARGET_R * risk,
                        window="regular",
                        note="盘前高点突破后回踩 EMA9/VWAP 不破（顺势跟进）",
                        extra={"pm_high": pm_high},
                    )
                )

        # -- 3) pre/post limit-order window: EMA pullback + generic liquidity sweep
        window = None
        if PRE_SIGNAL_START <= ts_et.time() < PRE_SIGNAL_END:
            window = "pre"
        elif POST_SIGNAL_START <= ts_et.time() < POST_SIGNAL_END:
            window = "post"
        if window is not None:
            trend_ok = e9 > e21
            pullback_ok = float(bar.low) <= e9 and float(bar.close) >= e9 and trend_ok
            last_key = f"{window}:pullback"
            recent = last_window_signal.get(last_key)
            spacing_ok = recent is None or (ts_et - recent).total_seconds() >= min_signal_spacing_minutes * 60
            if pullback_ok and spacing_ok:
                last_window_signal[last_key] = ts_et
                stop = min(float(bar.low), e21) - STOP_ATR_BUFFER * atr
                entry = float(bar.close)
                risk = max(entry - stop, 1e-6)
                signals.append(
                    _make_signal(
                        kind="EMA_PULLBACK",
                        symbol=symbol,
                        bar=bar,
                        entry=entry,
                        stop=stop,
                        target=entry + TARGET_R * risk,
                        window=window,
                        note="盘前信号·限价单（Firstrade Day+Ext）" if window == "pre" else "盘后信号·限价单（Firstrade Day+Ext）",
                    )
                )
            if index >= SWEEP_LOOKBACK + 1:
                prior = ordered[index - SWEEP_LOOKBACK : index]
                swing_low = min(float(item.low) for item in prior)
                sweep_key = f"{window}:sweep"
                sweep_recent = last_window_signal.get(sweep_key)
                sweep_spacing = sweep_recent is None or (ts_et - sweep_recent).total_seconds() >= min_signal_spacing_minutes * 60
                if (
                    float(bar.low) < swing_low
                    and float(bar.close) > swing_low
                    and float(bar.close) >= e21
                    and sweep_spacing
                    and int(bar.start_utc.timestamp() * 1000) not in {sig.get("t") for sig in signals}
                ):
                    last_window_signal[sweep_key] = ts_et
                    stop = min(swing_low, float(bar.low)) - STOP_ATR_BUFFER * atr
                    entry = float(bar.close)
                    risk = max(entry - stop, 1e-6)
                    signals.append(
                        _make_signal(
                            kind="SMC_SWEEP",
                            symbol=symbol,
                            bar=bar,
                            entry=entry,
                            stop=stop,
                            target=entry + TARGET_R * risk,
                            window=window,
                            note="盘前刺穿收回·限价单" if window == "pre" else "盘后刺穿收回·限价单",
                        )
                    )
    return signals


def is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


# --------------------------------------------------------------------------- #
# Payload builder used by the web terminal
# --------------------------------------------------------------------------- #
def build_extended_context(symbol: str, bars: Sequence[Bar], session_date: Optional[date] = None) -> Dict[str, Any]:
    ordered = sorted(bars, key=lambda bar: bar.start_utc)
    if not ordered:
        return {"symbol": symbol, "bars": [], "pm": {"ready": False}, "signals": [], "phase_counts": {}}
    target_date = session_date or ordered[-1].start_utc.astimezone(ET).date()
    anchors = build_pre_market_anchors(ordered, target_date)
    closes = [float(bar.close) for bar in ordered]
    ema9 = ema_series(closes, EMA_FAST)
    ema21 = ema_series(closes, EMA_SLOW)
    atrs = atr_series(ordered)
    regular: List[Bar] = []
    payload_bars = []
    for index, bar in enumerate(ordered):
        ts_et = bar.start_utc.astimezone(ET)
        phase = phase_of(ts_et)
        if phase == "regular" and ts_et.date() == target_date:
            regular.append(bar)
        if ts_et.date() != target_date:
            continue
        payload_bars.append(
            {
                "t": int(bar.start_utc.timestamp() * 1000),
                "o": round(float(bar.open), 4),
                "h": round(float(bar.high), 4),
                "l": round(float(bar.low), 4),
                "c": round(float(bar.close), 4),
                "v": int(round(float(bar.volume or 0.0))),
                "phase": phase,
                "ema9": round(ema9[index], 4) if len(ema9) > index else None,
                "ema21": round(ema21[index], 4) if len(ema21) > index else None,
                "atr": round(atrs[index], 4) if atrs[index] is not None else None,
            }
        )
    signals = evaluate_extended_signals(symbol, ordered, session_date=target_date)
    phases = [phase_of(bar.start_utc.astimezone(ET)) for bar in ordered if bar.start_utc.astimezone(ET).date() == target_date]
    return {
        "symbol": symbol,
        "bars": payload_bars,
        "pm": anchors,
        "signals": signals,
        "first_t": payload_bars[0]["t"] if payload_bars else None,
        "last_t": payload_bars[-1]["t"] if payload_bars else None,
        "phase_counts": {name: phases.count(name) for name in ("pre", "regular", "post")},
        "session_date": target_date.isoformat(),
        "note": "yfinance 5m prepost；盘前/盘后成交量多为 0，PM_VWAP 无成交量时用典型价均值。",
    }


def fetch_extended_bars(symbols: Sequence[str] = SYMBOLS, period: str = "5d") -> Dict[str, List[Bar]]:
    """Fetch 5m bars 04:00-20:00 ET via yfinance (network; call outside request loop)."""
    import yfinance as yf

    out: Dict[str, List[Bar]] = {}
    for symbol in symbols:
        frame = yf.download(symbol, period=period, interval="5m", prepost=True, auto_adjust=True, progress=False)
        if frame is None or frame.empty:
            continue
        if hasattr(frame.columns, "levels"):
            frame.columns = frame.columns.get_level_values(0)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize(ET)
        bars: List[Bar] = []
        for stamp, row in frame.iterrows():
            start = stamp.astimezone(UTC).to_pydatetime()
            end = start + timedelta(minutes=5)
            try:
                bar = Bar(
                    symbol=symbol,
                    timeframe_minutes=5,
                    start_utc=start,
                    end_utc=end,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"] or 0.0),
                    vwap=(float(row["High"]) + float(row["Low"]) + float(row["Close"])) / 3.0,
                    trade_count=0,
                )
            except Exception:  # noqa: BLE001 - skip malformed extended rows
                continue
            bars.append(bar)
        out[symbol] = bars
    return out


def load_cached_extended(path: Path) -> Optional[Dict[str, Any]]:
    import json

    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "Bar",
    "ExtendedHoursProvider",
    "build_extended_context",
    "fetch_alpaca_snapshot_quotes",
    "merge_quote_into_extended_bars",
    "build_pre_market_anchors",
    "evaluate_extended_signals",
    "fetch_extended_bars",
    "phase_of",
    "session_vwap",
]


def fetch_alpaca_snapshot_quotes(symbols: Sequence[str] = SYMBOLS) -> Dict[str, Dict[str, Any]]:
    """Latest Alpaca IEX trade/quote snapshot (0-delay when the market is open).

    Uses the REST snapshot endpoint only every ~20 s from the web server, never for
    signal polling loops; data vintage is returned so the UI can show data age.
    """
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest

    from intraday_scalper.providers.alpaca_provider import load_alpaca_credentials

    credentials = load_alpaca_credentials()
    client = StockHistoricalDataClient(credentials.api_key, credentials.secret_key)
    request = StockSnapshotRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX)
    snapshots = client.get_stock_snapshot(request)
    now = datetime.now(UTC)
    out: Dict[str, Dict[str, Any]] = {}
    for symbol, snapshot in snapshots.items():
        trade = getattr(snapshot, "latest_trade", None)
        quote = getattr(snapshot, "latest_quote", None)
        price = None
        timestamp = None
        source = "alpaca_iex_snapshot"
        if trade is not None and getattr(trade, "price", None):
            price = float(trade.price)
            timestamp = getattr(trade, "timestamp", None)
        if price is None and quote is not None:
            bid = getattr(quote, "bid_price", None)
            ask = getattr(quote, "ask_price", None)
            if bid and ask:
                price = (float(bid) + float(ask)) / 2.0
                timestamp = getattr(quote, "timestamp", None)
                source = "alpaca_iex_quote_mid"
        if price is None or timestamp is None:
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        age = max(0.0, (now - timestamp).total_seconds())
        out[symbol] = {
            "price": round(price, 4),
            "timestamp_utc": timestamp.astimezone(UTC).isoformat(),
            "age_seconds": round(age, 1),
            "source": source,
        }
    return out


def merge_quote_into_extended_bars(
    symbol: str,
    bars: Sequence[Bar],
    price: float,
    now: Optional[datetime] = None,
) -> List[Bar]:
    """Merge a live last price into the current 5m extended-hours bucket.

    Only the most recent bucket is updated (high/low/close); no future bars are
    created.  The caller decides whether the quote is fresh enough to merge.
    """
    if price is None or price <= 0:
        return list(bars)
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    bucket = moment.astimezone(UTC).replace(second=0, microsecond=0) - timedelta(minutes=moment.astimezone(UTC).minute % 5)
    merged = list(bars)
    if merged and merged[-1].start_utc >= bucket:
        last = merged[-1]
        if last.start_utc == bucket:
            high = max(float(last.high), float(price))
            low = min(float(last.low), float(price))
            merged[-1] = Bar(
                symbol=symbol,
                timeframe_minutes=5,
                start_utc=last.start_utc,
                end_utc=last.end_utc,
                open=float(last.open),
                high=high,
                low=low,
                close=float(price),
                volume=float(last.volume or 0.0),
                vwap=(high + low + float(price)) / 3.0,
                trade_count=int(last.trade_count or 0),
            )
        return merged
    merged.append(
        Bar(
            symbol=symbol,
            timeframe_minutes=5,
            start_utc=bucket,
            end_utc=bucket + timedelta(minutes=5),
            open=float(price),
            high=float(price),
            low=float(price),
            close=float(price),
            volume=0.0,
            vwap=float(price),
            trade_count=0,
        )
    )
    return merged


class ExtendedHoursProvider:
    """Tiny refresh/caching wrapper used by the web server and live loop."""

    def __init__(self, symbols: Sequence[str] = SYMBOLS, cache_dir: Optional[Path] = None, period: str = "5d") -> None:
        self.symbols = tuple(symbols)
        self.period = period
        self.cache_dir = cache_dir or (PACKAGE_ROOT / "var" / "extended")
        self._lock = None
        self._contexts: Dict[str, Any] = {}
        self._raw_bars: Dict[str, List[Bar]] = {}
        self._quotes: Dict[str, Dict[str, Any]] = {}
        self._last_quote_poll: Optional[datetime] = None
        self._last_full_refresh: Optional[datetime] = None

    def refresh(self) -> Dict[str, Any]:
        bars_by_symbol = fetch_extended_bars(self.symbols, self.period)
        self._raw_bars = {symbol: list(bars) for symbol, bars in bars_by_symbol.items()}
        for symbol, bars in bars_by_symbol.items():
            self._contexts[symbol] = build_extended_context(symbol, bars)
        self._last_full_refresh = datetime.now(UTC)
        return self._contexts

    def refresh_quotes(self) -> Dict[str, Dict[str, Any]]:
        """Poll Alpaca IEX snapshot (<=1/20s) and merge into the current 5m bucket."""
        now = datetime.now(UTC)
        self._last_quote_poll = now
        try:
            quotes = fetch_alpaca_snapshot_quotes(self.symbols)
        except Exception:  # noqa: BLE001 - keep last known quotes
            return self._quotes
        self._quotes = quotes
        current_et = now.astimezone(ET)
        merge_window = time(8, 0) <= current_et.time() < time(17, 0)
        for symbol, quote in quotes.items():
            bars = self._raw_bars.get(symbol)
            if bars is None:
                continue
            if merge_window and float(quote.get("age_seconds", 9999.0)) <= 600.0:
                merged = merge_quote_into_extended_bars(symbol, bars, float(quote["price"]), now)
                if merged is not bars and len(merged) != len(bars):
                    self._raw_bars[symbol] = merged
                else:
                    self._raw_bars[symbol] = merged
            self._contexts[symbol] = build_extended_context(symbol, self._raw_bars.get(symbol, bars))
        return quotes

    def latest(self) -> Dict[str, Any]:
        return dict(self._contexts)

    def payload(self) -> Dict[str, Any]:
        if not self._contexts:
            self.refresh()
        latest_date = None
        bars = {}
        signals: List[Dict[str, Any]] = []
        pm = {}
        bar_ages: Dict[str, float] = {}
        now = datetime.now(UTC)
        for symbol, context in self._contexts.items():
            bars[symbol] = context.get("bars", [])
            pm[symbol] = context.get("pm", {})
            signals.extend(context.get("signals", []))
            if context.get("bars"):
                last_dt = datetime.fromtimestamp(context["bars"][-1]["t"] / 1000, tz=UTC)
                latest_date = last_dt.astimezone(ET).date()
                bar_ages[symbol] = round(max(0.0, (now - last_dt).total_seconds()), 1)
        ages = [float(q.get("age_seconds", 0.0)) for q in self._quotes.values() if q.get("age_seconds") is not None]
        if not ages and bar_ages:
            ages = list(bar_ages.values())
        return {
            "session_date": latest_date.isoformat() if latest_date else None,
            "symbols": bars,
            "pm": pm,
            "signals": signals,
            "quotes": dict(self._quotes),
            "data_age_seconds": round(max(ages), 1) if ages else None,
            "bar_ages_seconds": bar_ages,
            "quote_source": "alpaca_iex_snapshot" if self._quotes else "yfinance_prepost",
            "last_quote_poll_utc": self._last_quote_poll.isoformat() if self._last_quote_poll else None,
            "last_full_refresh_utc": self._last_full_refresh.isoformat() if self._last_full_refresh else None,
            "note": (
                "Extended hours 04:00-20:00 ET: yfinance 5m prepost base + Alpaca IEX snapshot last price "
                "merged into the current 5m bucket every ~20 s (08:00-17:00 ET); data_age_seconds is the "
                "latest trade age. Pre/post volume is sparse; PM_VWAP falls back to typical-price mean."
            ),
        }
