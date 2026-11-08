"""Download Alpaca IEX 1-minute bars into a local parquet cache.

This tool only reads market data (no trading endpoints, no orders). It pulls
regular-session (09:30-16:00 ET) 1-minute bars for SOXL / SOXS from
``2024-01-01`` to now and stores one parquet per symbol:

    intraday_scalper/data/{symbol}_1m_2024_2026[_<adjustment>].parquet

Design notes
------------
* Data source: Alpaca Markets, ``feed=iex`` (single venue: volume is only the
  IEX share of the consolidated tape; see ``docs/alpaca_probe_result.md``).
* ``adjustment=all`` keeps the series consistent across splits/dividends.
* Downloads are chunked by month; the Alpaca SDK paginates internally, so a
  chunk request returns every page before moving on (free tier ~200 req/min).
* Re-running without ``--force`` resumes: existing rows are kept and only the
  missing tail is fetched.
* The parquet files are covered by the project's ``*.parquet`` gitignore rule
  and must never be committed.

Usage::

    python -m intraday_scalper.tools.download_history
    python -m intraday_scalper.tools.download_history --start 2024-01-01 --force
    python -m intraday_scalper.tools.download_history --symbols SOXL,SOXS --end 2026-11-06
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from intraday_scalper.core import is_regular_session, to_utc
from intraday_scalper.core.config import PACKAGE_ROOT
from intraday_scalper.providers.alpaca_provider import load_alpaca_credentials

DATA_DIR = PACKAGE_ROOT / "data"
MANIFEST_DIR = PACKAGE_ROOT / "var" / "backtest"
DEFAULT_SYMBOLS = ("SOXL", "SOXS")
DEFAULT_START = date(2024, 1, 1)
FEED = "iex"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def month_chunks(start: date, end: date) -> List[tuple[date, date]]:
    """Split [start, end] into calendar-month chunks (UTC boundaries)."""
    chunks: List[tuple[date, date]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        chunk_start = max(cursor, start)
        chunk_end = min(next_month - timedelta(days=1), end)
        if chunk_start <= chunk_end:
            chunks.append((chunk_start, chunk_end))
        cursor = next_month
    return chunks


def _utc_day_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def _utc_day_end(day: date) -> datetime:
    return _utc_day_start(day) + timedelta(days=1)


def output_path(symbol: str, data_dir: Path = DATA_DIR, adjustment: str = "all") -> Path:
    suffix = "" if adjustment == "all" else f"_{adjustment}"
    return data_dir / f"{symbol}_1m_2024_2026{suffix}.parquet"


def manifest_path(data_dir: Path = DATA_DIR) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return MANIFEST_DIR / "history_download_manifest.json"


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def _make_client(credentials):
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(credentials.api_key, credentials.secret_key)


def fetch_chunk(
    client, symbol: str, start_day: date, end_day: date, adjustment: str = "all"
) -> List[Dict]:
    """Fetch one calendar-month chunk for one symbol (regular session only)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=_utc_day_start(start_day),
        end=_utc_day_end(end_day),
        feed=DataFeed.IEX,
        adjustment={
            "all": Adjustment.ALL,
            "raw": Adjustment.RAW,
            "split": Adjustment.SPLIT,
        }[adjustment],
    )
    response = client.get_stock_bars(request)
    data = getattr(response, "data", None) or {}
    rows: List[Dict] = []
    for bar in data.get(symbol, []):
        ts = bar.timestamp
        if ts.tzinfo is None:  # alpaca usually returns tz-aware UTC; be defensive
            ts = ts.replace(tzinfo=timezone.utc)
        ts = to_utc(ts)
        if not is_regular_session(ts):
            continue
        vwap = getattr(bar, "vwap", None)
        volume = float(getattr(bar, "volume", 0.0) or 0.0)
        if vwap is None or volume <= 0:
            # Never fabricate VWAP: bars without a real volume-weighted price are skipped.
            continue
        rows.append(
            {
                "symbol": symbol,
                "start_utc": ts,
                "end_utc": ts + timedelta(minutes=1),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": volume,
                "vwap": float(vwap),
                "trade_count": int(getattr(bar, "trade_count", 0) or 0),
            }
        )
    return rows


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "start_utc" in frame.columns:
        frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
    return frame


def _resume_start(frame: pd.DataFrame, fallback: date) -> date:
    if frame.empty or "start_utc" not in frame.columns:
        return fallback
    last = frame["start_utc"].max()
    if pd.isna(last):
        return fallback
    last_day = pd.Timestamp(last).date()
    return max(fallback, last_day - timedelta(days=1))


def download_symbol(
    client,
    symbol: str,
    start: date,
    end: date,
    *,
    force: bool,
    adjustment: str = "all",
    data_dir: Path = DATA_DIR,
) -> Dict:
    path = output_path(symbol, data_dir, adjustment)
    existing = pd.DataFrame() if force else _load_existing(path)
    resume = _resume_start(existing, start)
    found: Dict[tuple, Dict] = {}

    chunks = month_chunks(resume, end)
    print(f"[{symbol}] 请求 {len(chunks)} 个月块: {resume} .. {end}")
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        rows = fetch_chunk(client, symbol, chunk_start, chunk_end, adjustment)
        for row in rows:
            found[(row["symbol"], row["start_utc"])] = row
        print(
            f"  [{symbol}] {index}/{len(chunks)} {chunk_start:%Y-%m}..{chunk_end:%Y-%m} "
            f"+{len(rows)} 条 (累计 {len(found)})"
        )

    if existing.empty:
        new_frame = pd.DataFrame(found.values())
    else:
        keep = existing[existing["start_utc"] < pd.Timestamp(_utc_day_start(resume))]
        new_frame = pd.concat([keep, pd.DataFrame(found.values())], ignore_index=True)
    if new_frame.empty:
        print(f"[{symbol}] 无数据，跳过写入")
        return {"symbol": symbol, "rows": 0, "path": str(path)}

    new_frame["start_utc"] = pd.to_datetime(new_frame["start_utc"], utc=True)
    new_frame["end_utc"] = pd.to_datetime(new_frame["end_utc"], utc=True)
    new_frame = (
        new_frame.drop_duplicates(subset=["symbol", "start_utc"])
        .sort_values("start_utc")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    new_frame.to_parquet(path, index=False, compression="snappy")
    print(f"[{symbol}] 写入 {path} · {len(new_frame):,} 根 1m bar")
    return {
        "symbol": symbol,
        "rows": int(len(new_frame)),
        "path": str(path),
        "start": str(new_frame["start_utc"].min()),
        "end": str(new_frame["start_utc"].max()),
        "size_mb": round(path.stat().st_size / 1_048_576, 2),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    try:  # Windows console: make Chinese/symbols render instead of crashing on GBK
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="Download Alpaca IEX 1m bars to local parquet.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma separated symbols")
    parser.add_argument("--start", type=_parse_date, default=DEFAULT_START, help="YYYY-MM-DD (default 2024-01-01)")
    parser.add_argument("--end", type=_parse_date, default=date.today(), help="YYYY-MM-DD (default today)")
    parser.add_argument("--force", action="store_true", help="ignore existing parquet and re-download")
    parser.add_argument(
        "--adjustment",
        choices=["all", "raw", "split"],
        default="all",
        help="corporate-action adjustment for the price series (default: all)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if not symbols:
        print("至少需要一个 symbol", file=sys.stderr)
        return 2

    credentials = load_alpaca_credentials()
    client = _make_client(credentials)
    print(
        f"Alpaca IEX 1m bars · symbols={','.join(symbols)} · period={args.start}..{args.end} "
        f"· adjustment={args.adjustment} · feed={FEED}"
    )
    print("口径提醒：IEX 是单一交易所，成交量仅含 IEX 份额；本数据不是全市场 consolidated tape。")
    if args.adjustment != "raw":
        print(
            "⚠️ 复权口径提醒：Alpaca 的 adjusted bars 会把成交量按合股比例缩小，低成交分钟会被"
            " round 成 v=0，本工具会丢弃 volume<=0 的 bar（避免伪造 VWAP），可能导致数据覆盖缺失；"
            "日内研究请优先使用 --adjustment raw。"
        )


    report = {
        "feed": FEED,
        "adjustment": args.adjustment,
        "start": str(args.start),
        "end": str(args.end),
        "symbols": {},
    }
    for symbol in symbols:
        report["symbols"][symbol] = download_symbol(
            client,
            symbol,
            args.start,
            args.end,
            force=args.force,
            adjustment=args.adjustment,
            data_dir=args.data_dir,
        )

    report["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest = manifest_path(args.data_dir)
    if manifest.exists():
        try:
            history = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = {"runs": []}
    else:
        history = {"runs": []}
    history.setdefault("runs", []).append(report)
    manifest.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"清单写入 {manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
