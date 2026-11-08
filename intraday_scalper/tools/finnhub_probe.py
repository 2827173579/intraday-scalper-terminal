#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finnhub 数据权限探测脚本（只读、无副作用）。

用途
----
用真实 API key 实测当前 Finnhub 账户的数据权限，回答三个核心问题：
  1. REST /quote 是否可用？返回的 t 是实时价还是上一交易日收盘？
  2. REST /stock/candle 历史 K 线（1/5/15/60/D/W）是否有权限？
  3. WebSocket 实时成交（美股 SOXL/SOXS/AAPL vs 加密 BINANCE:BTCUSDT）能否收到数据？

用法（工作目录 intraday_scalper/）：
  .venv\\Scripts\\python.exe tools\\finnhub_probe.py --check quote
  .venv\\Scripts\\python.exe tools\\finnhub_probe.py --check candle
  .venv\\Scripts\\python.exe tools\\finnhub_probe.py --check tier
  .venv\\Scripts\\python.exe tools\\finnhub_probe.py --check ws --ws-seconds 60
  .venv\\Scripts\\python.exe tools\\finnhub_probe.py --check all --json-out docs/evidence/probe.json

可选参数：--timeout 10（REST 超时秒）、--ws-seconds 60（每个 WS 场景时长）、
--symbols SOXL,SOXS（覆盖默认标的）、--json-out 路径（相对 intraday_scalper/，脱敏后落盘）。

安全约束
--------
* 密钥解析顺序：环境变量 FINNHUB_API_KEY → intraday_scalper/.env（本脚本上一级目录）。
* 密钥只存在于内存；所有打印、异常信息、JSON 落盘都经过 scrub() 脱敏，绝不出现完整密钥。
* WebSocket 的 URL 需要带 token（Finnhub 协议要求），但脚本从不打印 URL。
* 只做只读 GET / 订阅，不写远端状态；除 --json-out 指定的本地文件外无副作用。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

REST_BASE = "https://finnhub.io/api/v1"
WS_URL = "wss://ws.finnhub.io"
REPO_DIR = Path(__file__).resolve().parents[1]  # intraday_scalper/
ENV_FILE = REPO_DIR / ".env"
PACE_SECONDS = 0.8  # REST 调用最小间隔，规避免费档 60 次/分钟限流

_SECRET = ""  # 仅在内存中使用

try:  # pragma: no cover - 依赖平台
    from zoneinfo import ZoneInfo

    _ET = ZoneInfo("America/New_York")
except Exception:  # Windows 下可能缺 IANA tzdata，退回手工 EST/EDT 规则
    _ET = None


# --------------------------------------------------------------------------- #
# 脱敏 / 文本工具
# --------------------------------------------------------------------------- #
def scrub(text) -> str:
    """把文本中的密钥替换为 ***，所有输出与落盘前必须经过本函数。"""
    s = str(text)
    if _SECRET:
        s = s.replace(_SECRET, "***REDACTED***")
    s = re.sub(r"(token=)[A-Za-z0-9_\-]+", r"\1***REDACTED***", s)
    return s


def truncate(text, n: int = 200) -> str:
    """截断并压成单行，用于证据样本。"""
    s = scrub(text).replace("\r", " ").replace("\n", " ")
    if len(s) <= n:
        return s
    return s[:n] + "...(truncated, total %d chars)" % len(s)


def pad(text, width: int) -> str:
    """按显示宽度补空格（中文按 2 计）。"""
    s = str(text)
    w = sum(2 if ord(c) > 0x2E7F else 1 for c in s)
    return s + " " * max(1, width - w)


# --------------------------------------------------------------------------- #
# 时间工具（美东时间：Nov-Mar 为 UTC-5，Mar-Nov 为 UTC-4）
# --------------------------------------------------------------------------- #
def _nth_sunday(year: int, month: int, n: int) -> datetime:
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    days_ahead = (6 - first.weekday()) % 7
    return datetime(year, month, 1 + days_ahead + 7 * (n - 1), tzinfo=timezone.utc)


def _et_offset_hours(dt_utc: datetime) -> int:
    dst_start = _nth_sunday(dt_utc.year, 3, 2) + timedelta(hours=7)  # 02:00 EST
    dst_end = _nth_sunday(dt_utc.year, 11, 1) + timedelta(hours=6)  # 02:00 EDT
    return -4 if dst_start <= dt_utc < dst_end else -5


def iso_utc(ts) -> str:
    if ts in (None, ""):
        return "-"
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    except (OSError, OverflowError, ValueError, TypeError):
        return f"bad_ts({ts})"


def iso_et(ts) -> str:
    if ts in (None, ""):
        return "-"
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except (OSError, OverflowError, ValueError, TypeError):
        return f"bad_ts({ts})"
    if _ET is not None:
        return dt.astimezone(_ET).strftime("%Y-%m-%d %H:%M:%S %Z")
    off = _et_offset_hours(dt)
    return (dt + timedelta(hours=off)).strftime("%Y-%m-%d %H:%M:%S") + f" UTC{off:+d}"


def now_et() -> str:
    return iso_et(int(time.time()))


def ms_iso(ms) -> str:
    try:
        return iso_et(int(ms) / 1000.0)
    except (TypeError, ValueError):
        return "-"


# --------------------------------------------------------------------------- #
# 密钥解析
# --------------------------------------------------------------------------- #
def load_api_key():
    """返回 (key, source)。缺失时返回 ("", "missing")。"""
    val = os.environ.get("FINNHUB_API_KEY", "").strip()
    if val:
        return val, "env:FINNHUB_API_KEY"
    if ENV_FILE.is_file():
        try:
            text = ENV_FILE.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for raw in text.splitlines():
            line = raw.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "FINNHUB_API_KEY":
                v = v.strip().strip('"').strip("'")
                if v:
                    return v, "file:intraday_scalper/.env"
    return "", "missing"


# --------------------------------------------------------------------------- #
# REST 探测基础设施
# --------------------------------------------------------------------------- #
@dataclass
class RestResult:
    check: str
    label: str
    path: str
    params: dict
    status: int | None
    elapsed_ms: float
    body: str
    json_obj: object = None
    error: str | None = None
    ratelimit: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        body = self.json_obj if self.json_obj is not None else {"_raw": truncate(self.body, 300)}
        return {
            "check": self.check,
            "label": self.label,
            "path": self.path,
            "params": {k: scrub(v) for k, v in self.params.items() if k != "token"},
            "http_status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": scrub(self.error) if self.error else None,
            "ratelimit": {k: scrub(v) for k, v in self.ratelimit.items()},
            "evidence": truncate(json.dumps(body, ensure_ascii=False, default=str), 400)
            if isinstance(body, (dict, list))
            else truncate(self.body, 300),
        }


_last_call = 0.0


def _throttle() -> None:
    global _last_call
    wait = PACE_SECONDS - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def paced_get(session, check: str, path: str, params: dict, timeout: float,
              label: str, retry_429: bool = True) -> RestResult:
    """带节流与 429 退避的只读 GET；params 里可含 token，但落盘前会被脱敏。"""
    full_params = dict(params)
    full_params["token"] = _SECRET  # 仅内存；RestResult.params 记录时会剔除
    attempts = 2 if retry_429 else 1
    for attempt in range(1, attempts + 1):
        _throttle()
        t0 = time.perf_counter()
        try:
            resp = session.get(REST_BASE + path, params=full_params, timeout=timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            body = resp.text if resp.text is not None else ""
            obj = None
            try:
                obj = resp.json()
            except ValueError:
                obj = None
            ratelimit = {
                k.lower(): v for k, v in resp.headers.items() if k.lower().startswith("x-ratelimit")
            }
            result = RestResult(
                check=check, label=label, path=path,
                params={k: v for k, v in full_params.items() if k != "token"},
                status=resp.status_code, elapsed_ms=elapsed_ms, body=body,
                json_obj=obj, ratelimit=ratelimit,
            )
            if resp.status_code == 429 and attempt < attempts:
                print(f"  [!] {label} 触发 429 限流，等待 61 秒后重试一次 ...")
                time.sleep(61)
                continue
            return result
        except Exception as exc:  # requests 异常统一记录，不外抛
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return RestResult(
                check=check, label=label, path=path,
                params={k: v for k, v in full_params.items() if k != "token"},
                status=None, elapsed_ms=elapsed_ms, body="",
                error=f"{type(exc).__name__}: {exc}",
            )
    return RestResult(check=check, label=label, path=path,
                      params={k: v for k, v in full_params.items() if k != "token"},
                      status=None, elapsed_ms=0.0, body="", error="unreachable")


def _make_session(timeout: float):
    import requests  # 延迟导入，缺依赖时给出清晰提示

    session = requests.Session()
    session.headers.update({
        "User-Agent": "intraday-scalper-finnhub-probe/1.0 (read-only diagnostic)",
        "Accept": "application/json",
    })
    return session


# --------------------------------------------------------------------------- #
# 检查项 1：/quote
# --------------------------------------------------------------------------- #
QUOTE_SYMBOLS = ["SOXL", "SOXS", "SMH", "SOXX", "AAPL", "SPY"]


def check_quote(session, args, records) -> list:
    symbols = _symbols_arg(args) or QUOTE_SYMBOLS
    symbols = list(dict.fromkeys(symbols + ["BINANCE:BTCUSDT"]))  # 加密做时钟/新鲜度对照
    print("\n=== [/quote] 报价探测（含 BTCUSDT 对照，验证 t 时间戳含义） ===")
    print("美东当前时间:", now_et(), "| 分类:",
          "盘前" if _is_premarket() else "非盘前(脚本未细分)")
    rows = []
    for sym in symbols:
        res = paced_get(session, "quote", "/quote", {"symbol": sym}, args.timeout, f"GET /quote?symbol={sym}")
        records.append(res.as_record())
        q = res.json_obj if isinstance(res.json_obj, dict) else {}
        status_txt = "OK" if res.status == 200 else f"HTTP {res.status}" if res.status else "ERROR"
        c = q.get("c")
        if res.status == 200 and (c in (0, None)):
            status_txt = "OK(无有效报价)"
        t = q.get("t")
        age = "-"
        if isinstance(t, (int, float)) and t > 1_000_000_000:
            age_s = int(time.time() - t)
            age = f"{age_s}s 前"
        row = {
            "symbol": sym, "status": status_txt, "ms": f"{res.elapsed_ms:.0f}",
            "c": q.get("c"), "d": q.get("d"), "dp": q.get("dp"),
            "o/h/l": f"{q.get('o')}/{q.get('h')}/{q.get('l')}",
            "pc": q.get("pc"), "t": t, "t_et": iso_et(t), "t_age": age,
            "err_body": truncate(res.body, 160) if res.status != 200 else "",
        }
        rows.append(row)
        print(f"  {pad(sym,14)}{pad(status_txt,16)}{pad(str(q.get('c')),10)}"
              f"d={pad(q.get('d'),8)} dp={pad(q.get('dp'),8)} pc={pad(q.get('pc'),10)} "
              f"t={pad(t,12)} ET={pad(iso_et(t),24)} age={age}")
        if res.status != 200:
            print(f"      响应体: {truncate(res.body, 160)}")
        if res.ratelimit:
            print(f"      ratelimit headers: {res.ratelimit}")
    # 结论速览
    stock_rows = [r for r in rows if r["symbol"] in symbols and r["symbol"] != "BINANCE:BTCUSDT"]
    stock_ok = [r for r in stock_rows if r["status"] == "OK"]
    print(f"\n  [quote 小结] 美股标的 {len(stock_ok)}/{len(stock_rows)} 返回有效报价；"
          f"t 多数为 {stock_ok[0]['t_et'] if stock_ok else '-'}（上一交易日收盘时间戳）"
          if stock_ok else "\n  [quote 小结] 无可用的美股报价")
    return rows


def _is_premarket() -> bool:
    try:
        if _ET is not None:
            now = datetime.now(_ET)
        else:
            now = datetime.now(timezone.utc) + timedelta(hours=_et_offset_hours(datetime.now(timezone.utc)))
        if now.weekday() >= 5:
            return False
        mins = now.hour * 60 + now.minute
        return 4 * 60 <= mins < 9 * 60 + 30
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 检查项 2：/stock/candle（+ 加密 /crypto/candle 正向对照）
# --------------------------------------------------------------------------- #
RESOLUTIONS = ["1", "5", "15", "60", "D", "W"]
_RANGES = [
    ("近2日", int(time.time()) - 2 * 86400, int(time.time())),
    ("近30日", int(time.time()) - 30 * 86400, int(time.time())),
    ("2024-01历史",
     int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()),
     int(datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())),
]


def check_candle(session, args, records) -> list:
    symbols = _symbols_arg(args) or ["SOXL", "SOXS"]
    # 额外对 AAPL/SPY 做一次抽样，排除"只有杠杆 ETF 无权限"的可能
    extra = ["AAPL", "SPY"]
    print("\n=== [/stock/candle] 历史 K 线权限探测 ===")
    print(f"标的: {symbols} (+抽样 {extra})，分辨率 {RESOLUTIONS}，区间 {[r[0] for r in _RANGES]}")
    rows = []
    for sym in symbols:
        for reso in RESOLUTIONS:
            for range_name, frm, to in _RANGES:
                label = f"GET /stock/candle {sym} {reso} {range_name}"
                res = paced_get(session, "candle", "/stock/candle",
                                {"symbol": sym, "resolution": reso, "from": frm, "to": to},
                                args.timeout, label)
                records.append(res.as_record())
                n_bars, first_c, errflag = _candle_evidence(res.json_obj)
                status_txt = "OK" if res.status == 200 and n_bars else (
                    "OK(空)" if res.status == 200 else (f"HTTP {res.status}" if res.status else "ERROR"))
                row = {"symbol": sym, "resolution": reso, "range": range_name,
                       "status": status_txt, "ms": f"{res.elapsed_ms:.0f}",
                       "n_bars": n_bars, "first_close": first_c,
                       "err_json": truncate(res.body, 200) if res.status != 200 else "",
                       "error": scrub(res.error) if res.error else "",
                       "ratelimit": res.ratelimit}
                rows.append(row)
                print(f"  {pad(sym,6)}{pad('res=' + str(reso),8)}{pad(range_name,12)}"
                      f"{pad(status_txt,12)}{pad('ms=' + str(round(res.elapsed_ms)),10)}"
                      f"bars={pad(n_bars,6)}{truncate(res.body, 120) if res.status != 200 else ''}")
    # AAPL/SPY 抽样：仅 res=5，近30日
    for sym in extra:
        res = paced_get(session, "candle", "/stock/candle",
                        {"symbol": sym, "resolution": "5", "from": _RANGES[1][1], "to": _RANGES[1][2]},
                        args.timeout, f"GET /stock/candle {sym} 5 近30日(抽样)")
        records.append(res.as_record())
        status_txt = "OK" if res.status == 200 else (f"HTTP {res.status}" if res.status else "ERROR")
        rows.append({"symbol": sym, "resolution": "5", "range": "近30日", "status": status_txt,
                     "ms": f"{res.elapsed_ms:.0f}", "n_bars": _candle_evidence(res.json_obj)[0],
                     "err_json": truncate(res.body, 200) if res.status != 200 else ""})
        print(f"  {pad(sym,6)}{pad('res=5',8)}{pad('近30日(抽样)',14)}{pad(status_txt,12)}"
              f"{truncate(res.body, 120) if res.status != 200 else ''}")
    # 正向对照：加密 1 分钟 K 线（免费档通常可用，用来证明 candle 端点本身正常）
    btc_from = int(time.time()) - 3600 * 3
    btc_to = int(time.time())
    res = paced_get(session, "candle", "/crypto/candle",
                    {"symbol": "BINANCE:BTCUSDT", "resolution": "1", "from": btc_from, "to": btc_to},
                    args.timeout, "GET /crypto/candle BINANCE:BTCUSDT 1 近3小时(正向对照)")
    records.append(res.as_record())
    n_bars, first_c, _ = _candle_evidence(res.json_obj)
    btc_status = "OK" if res.status == 200 and n_bars else (f"HTTP {res.status}" if res.status else "ERROR")
    print(f"  {pad('BTCUSDT',8)}{pad('crypto/1',10)}{pad('近3小时',10)}{pad(btc_status,12)}"
          f"bars={n_bars}  first_close={first_c}  "
          f"{truncate(res.body, 120) if res.status != 200 else ''}")
    rows.append({"symbol": "BINANCE:BTCUSDT", "resolution": "1crypto", "range": "近3小时(对照)",
                 "status": btc_status, "ms": f"{res.elapsed_ms:.0f}", "n_bars": n_bars,
                 "err_json": truncate(res.body, 200) if res.status != 200 else ""})
    # 小结
    stock_candle_ok = [r for r in rows if r.get("resolution") not in ("1crypto",) and r["status"].startswith("OK")]
    stock_403 = [r for r in rows if "403" in str(r["status"])]
    print(f"\n  [candle 小结] 美股历史 K 线可用组合: {len(stock_candle_ok)}/{len(rows) - 1}；"
          f"HTTP 403 次数: {len(stock_403)}；加密对照: {btc_status} bars={n_bars}")
    return rows


def _candle_evidence(obj) -> tuple:
    """返回 (bar 数, 首个 close, 错误标记)。"""
    if not isinstance(obj, dict):
        return 0, None, None
    if obj.get("s") == "ok" and isinstance(obj.get("t"), list):
        closes = obj.get("c") or []
        return len(obj["t"]), (closes[0] if closes else None), None
    err = obj.get("error")
    return 0, None, err


# --------------------------------------------------------------------------- #
# 检查项 3：账户 tier 边界探测
# --------------------------------------------------------------------------- #
def check_tier(session, args, records) -> list:
    print("\n=== [tier] 免费/付费边界探测 ===")
    probes = []
    # 免费档常见可用项（用 AAPL 做对照组，排除"ETF 无档案"导致的空对象误判）
    probes.append(("profile2", "/stock/profile2", {"symbol": "SOXL"}))
    probes.append(("profile2_AAPL", "/stock/profile2", {"symbol": "AAPL"}))
    probes.append(("metric", "/stock/metric", {"symbol": "SOXL", "metric": "all"}))
    probes.append(("metric_AAPL", "/stock/metric", {"symbol": "AAPL", "metric": "all"}))
    probes.append(("symbol_us", "/stock/symbol", {"exchange": "US"}))
    probes.append(("market_status", "/stock/market-status", {"exchange": "US"}))
    probes.append(("news_general", "/news", {"category": "general"}))
    probes.append(("company_news", "/company-news",
                   {"symbol": "SOXL", "from": "2026-10-01", "to": "2026-11-06"}))
    # 付费档专属项（用于判断账户 tier；预期 403）
    probes.append(("recommendation", "/stock/recommendation", {"symbol": "SOXL"}))
    probes.append(("price_target", "/stock/price-target", {"symbol": "SOXL"}))
    probes.append(("insider_tx", "/stock/insider-transactions", {"symbol": "SOXL"}))
    probes.append(("bidask", "/stock/bidask", {"symbol": "SOXL"}))
    rows = []
    for name, path, params in probes:
        res = paced_get(session, "tier", path, params, args.timeout, f"GET {path}")
        records.append(res.as_record())
        summary, detail = _summarize_tier_payload(name, res)
        status_txt = "OK" if res.status == 200 else (f"HTTP {res.status}" if res.status else "ERROR")
        row = {"name": name, "path": path, "status": status_txt,
               "ms": f"{res.elapsed_ms:.0f}", "summary": summary,
               "detail": truncate(detail, 220), "ratelimit": res.ratelimit}
        rows.append(row)
        print(f"  {pad(name,14)}{pad(status_txt,10)}{pad('ms=' + str(round(res.elapsed_ms)),10)}{summary}")
        if res.status != 200:
            print(f"      响应体: {truncate(res.body, 200)}")
        if res.ratelimit:
            print(f"      ratelimit headers: {res.ratelimit}")
    print("\n  [tier 小结] 用上述端点命中/403 组合判断账户档位（见报告结论）。")
    return rows


def _summarize_tier_payload(name: str, res: RestResult):
    obj = res.json_obj
    if res.status != 200:
        return f"拒绝/错误: {truncate(res.body, 120)}", res.body
    try:
        if name == "profile2" and isinstance(obj, dict):
            return (f"name={obj.get('name')} exchange={obj.get('exchange')} "
                    f"ticker={obj.get('ticker')} industry={obj.get('finnhubIndustry')}", json.dumps(obj, ensure_ascii=False))
        if name == "metric" and isinstance(obj, dict):
            metric = obj.get("metric") or {}
            sample = list(metric.keys())[:6]
            return f"metric 数={len(metric)} 样例字段={sample}", json.dumps(metric, ensure_ascii=False)
        if name == "symbol_us" and isinstance(obj, list):
            first = obj[0] if obj else {}
            return (f"US 标的数={len(obj)} 首项 keys={list(first.keys()) if isinstance(first, dict) else '-'}", json.dumps(first, ensure_ascii=False))
        if name == "market_status" and isinstance(obj, dict):
            return f"isOpen={obj.get('isOpen')} session={obj.get('session')} t={obj.get('t')}", json.dumps(obj, ensure_ascii=False)
        if name == "news_general" and isinstance(obj, list):
            head = obj[0] if obj else {}
            return (f"条数={len(obj)} 首条={truncate(str(head.get('headline')), 80)}", json.dumps(obj[:3], ensure_ascii=False))
        if name == "company_news" and isinstance(obj, list):
            return f"条数={len(obj)}", json.dumps(obj[:3], ensure_ascii=False)
    except Exception as exc:  # 容错：字段缺失不影响探测
        return f"解析异常: {type(exc).__name__}", str(obj)
    return truncate(json.dumps(obj, ensure_ascii=False), 160), json.dumps(obj, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 检查项 4：WebSocket 实时成交
# --------------------------------------------------------------------------- #
def check_ws(token: str, args, records) -> list:
    try:
        import websocket  # noqa: F401
    except ImportError:
        print("\n[!] 缺少 websocket-client，无法执行 WS 探测。"
              "请先安装： .venv\\Scripts\\pip.exe install websocket-client")
        return []
    scenarios = [
        {"name": "组合场景(美股+加密对照)", "symbols": ["SOXL", "SOXS", "AAPL", "BINANCE:BTCUSDT"],
         "seconds": max(60.0, args.ws_seconds)},
        {"name": "SOXL/SOXS 单独订阅", "symbols": ["SOXL", "SOXS"],
         "seconds": max(60.0, args.ws_seconds)},
    ]
    results = []
    for idx, scenario in enumerate(scenarios):
        print(f"\n=== [ws/{idx + 1}] {scenario['name']} "
              f"({', '.join(scenario['symbols'])}, {scenario['seconds']:.0f}s, 美东 {now_et()}) ===")
        run = _run_ws_scenario(token, scenario["symbols"], scenario["seconds"], args.timeout)
        records.append(run)
        results.append(run)
        _print_ws_run(run)
        # 若组合场景里加密也没有数据，说明是网络/端点问题而非权限，再补一个 BTC 单独 20s 对照
        if idx == 0 and run["trade_counts"].get("BINANCE:BTCUSDT", 0) == 0:
            print("\n  [ws 补充] 组合场景中 BINANCE:BTCUSDT 也无成交，追加 BTCUSDT 单独 20s 对照 ...")
            run2 = _run_ws_scenario(token, ["BINANCE:BTCUSDT"], 20.0, args.timeout)
            records.append(run2)
            _print_ws_run(run2)
    return results


def _run_ws_scenario(token: str, symbols: list, seconds: float, timeout: float) -> dict:
    """单个 WebSocket 场景；返回脱敏后的统计结果。"""
    import websocket  # 已在 check_ws 中确认可用

    url = f"{WS_URL}?token={token}"  # 绝不打印
    run = {
        "symbols": symbols, "requested_seconds": seconds,
        "connect_ok": False, "connect_ms": None,
        "open_at": now_et(),
        "total_messages": 0, "msg_types": {},
        "trade_counts": {s: 0 for s in symbols},
        "first_trade": {}, "last_trade": {},
        "server_ping_frames": 0, "server_pong_frames": 0,
        "json_pings_sent": 0, "json_pongs_received": 0,
        "error_msgs": [], "on_error": [], "close_code": None, "close_reason": None,
        "unknown_frames": [],
    }
    conn = None
    started = time.time()
    try:
        t0 = time.perf_counter()
        conn = websocket.create_connection(url, timeout=timeout, enable_multithread=True)
        run["connect_ok"] = True
        run["connect_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        for sym in symbols:
            conn.send(json.dumps({"type": "subscribe", "symbol": sym}))
        deadline = time.time() + seconds
        next_ping = time.time() + 15
        while time.time() < deadline:
            if time.time() >= next_ping:
                try:
                    conn.send(json.dumps({"type": "ping"}))
                    run["json_pings_sent"] += 1
                except Exception as exc:
                    run["on_error"].append(scrub(f"ping_send: {type(exc).__name__}: {exc}"))
                    break
                next_ping = time.time() + 15
            remain = max(0.2, min(1.0, deadline - time.time()))
            conn.settimeout(remain)
            try:
                opcode, data = conn.recv_data(control_frame=True)
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:
                run["on_error"].append(scrub(f"recv: {type(exc).__name__}: {exc}"))
                break
            if opcode == websocket.ABNF.OPCODE_PING:
                run["server_ping_frames"] += 1
                try:
                    conn.pong(data)
                except Exception:
                    pass
                continue
            if opcode == websocket.ABNF.OPCODE_PONG:
                run["server_pong_frames"] += 1
                continue
            if opcode == websocket.ABNF.OPCODE_CLOSE:
                try:
                    if isinstance(data, (bytes, bytearray)) and len(data) >= 2:
                        run["close_code"] = int.from_bytes(data[:2], "big")
                        run["close_reason"] = scrub(bytes(data[2:]).decode("utf-8", "replace"))
                    else:
                        run["close_reason"] = scrub(str(data))
                except Exception:
                    pass
                break
            if opcode != websocket.ABNF.OPCODE_TEXT:
                run["unknown_frames"].append(f"opcode={opcode} len={len(data) if data else 0}")
                continue
            run["total_messages"] += 1
            try:
                msg = json.loads(data)
            except (ValueError, TypeError):
                run["msg_types"]["unparsable"] = run["msg_types"].get("unparsable", 0) + 1
                continue
            mtype = str(msg.get("type", "unknown"))
            run["msg_types"][mtype] = run["msg_types"].get(mtype, 0) + 1
            if mtype == "trade" and isinstance(msg.get("data"), list):
                for trade in msg["data"]:
                    sym = trade.get("s", "?")
                    run["trade_counts"][sym] = run["trade_counts"].get(sym, 0) + 1
                    sample = json.dumps(trade, ensure_ascii=False, separators=(",", ":"))
                    if sym not in run["first_trade"]:
                        run["first_trade"][sym] = {
                            "at": now_et(), "sample": truncate(sample, 300), "ts_et": ms_iso(trade.get("t"))}
                    run["last_trade"][sym] = {
                        "at": now_et(), "sample": truncate(sample, 300), "ts_et": ms_iso(trade.get("t"))}
            elif mtype == "ping":
                run["server_ping_frames"] += 1
            elif mtype == "pong":
                run["json_pongs_received"] += 1
            elif mtype == "error":
                run["error_msgs"].append(truncate(json.dumps(msg, ensure_ascii=False), 300))
            elif mtype in ("subscribed", "subscribe"):
                pass
            else:
                if len(run["unknown_frames"]) < 5:
                    run["unknown_frames"].append(truncate(data, 200))
    except Exception as exc:
        run["on_error"].append(scrub(f"{type(exc).__name__}: {exc}"))
    finally:
        run["elapsed_seconds"] = round(time.time() - started, 1)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    # 记录结束时刻再取一次 close_code 可能拿不到（取决于服务器），已尽力
    return run


def _print_ws_run(run: dict) -> None:
    print(f"  connect_ok={run['connect_ok']} connect_ms={run['connect_ms']} "
          f"elapsed={run['elapsed_seconds']}s close_code={run['close_code']} "
          f"reason={run['close_reason']}")
    print(f"  total_messages={run['total_messages']} msg_types={run['msg_types']} "
          f"server_ping_frames={run['server_ping_frames']} server_pong_frames={run['server_pong_frames']} "
          f"json_pings_sent={run['json_pings_sent']} json_pongs_received={run['json_pongs_received']}")
    for sym in run["symbols"]:
        print(f"    - {pad(sym,16)} trades={pad(run['trade_counts'].get(sym, 0),6)} "
              f"first_ts={run['first_trade'].get(sym, {}).get('ts_et', '-')}")
        if sym in run["first_trade"]:
            print(f"        first: {run['first_trade'][sym]['sample']}")
            print(f"        last : {run['last_trade'][sym]['sample']}")
    if run["error_msgs"]:
        print(f"  error 消息: {run['error_msgs']}")
    if run["on_error"]:
        print(f"  异常: {run['on_error']}")
    if run["unknown_frames"]:
        print(f"  其他帧/消息: {run['unknown_frames']}")


# --------------------------------------------------------------------------- #
# 汇总与入口
# --------------------------------------------------------------------------- #
def _symbols_arg(args):
    raw = (args.symbols or "").strip()
    if not raw:
        return None
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _write_json(path_str: str, payload: dict) -> None:
    if not path_str:
        return
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_DIR / p
    target = p.resolve()
    try:
        target.relative_to(REPO_DIR.resolve())
    except ValueError:
        print(f"[!] --json-out 路径必须在 intraday_scalper/ 内，忽略: {path_str}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    target.write_text(scrub(text), encoding="utf-8")
    print(f"\n[evidence] 脱敏后的探测结果已写入: {target}")


def main(argv=None) -> int:
    global _SECRET
    parser = argparse.ArgumentParser(
        description="Finnhub 数据权限探测（只读、脱敏）。密钥来自环境变量 FINNHUB_API_KEY 或 intraday_scalper/.env。")
    parser.add_argument("--check", required=True, choices=["quote", "candle", "ws", "tier", "all"],
                        help="探测项：quote / candle / ws / tier / all")
    parser.add_argument("--timeout", type=float, default=10.0, help="REST 请求超时秒数（默认 10）")
    parser.add_argument("--ws-seconds", type=float, default=60.0, help="每个 WS 场景运行秒数（默认 60，最少 60）")
    parser.add_argument("--symbols", default="", help="逗号分隔标的覆盖（quote/candle 用）")
    parser.add_argument("--json-out", default="", help="可选：探测结果 JSON 落盘路径（相对 intraday_scalper/，已脱敏）")
    args = parser.parse_args(argv)

    key, source = load_api_key()
    if not key:
        print("[X] 未找到 FINNHUB_API_KEY：请设置环境变量，或在 intraday_scalper/.env 中配置 "
              "FINNHUB_API_KEY=...（该文件已被 .gitignore 忽略）。")
        return 2
    _SECRET = key
    print("=" * 78)
    print("Finnhub 数据权限探测（只读）")
    print(f"密钥来源: {source} | 长度: {len(key)} 字符 | 启动时间: {now_et()}（美东）")
    print("=" * 78)

    records = []
    check_results = {}
    session = None
    try:
        if args.check in ("quote", "candle", "tier", "all"):
            session = _make_session(args.timeout)
        if args.check in ("quote", "all"):
            check_results["quote"] = check_quote(session, args, records)
        if args.check in ("candle", "all"):
            check_results["candle"] = check_candle(session, args, records)
        if args.check in ("tier", "all"):
            check_results["tier"] = check_tier(session, args, records)
        if args.check in ("ws", "all"):
            check_results["ws"] = check_ws(key, args, records)
    finally:
        if session is not None:
            session.close()

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_at_et": now_et(),
        "key_source": source,
        "key_length": len(key),  # 只记长度，不记内容
        "check": args.check,
        "results": records,
        "summary": _final_summary(check_results),
    }
    print("\n" + "=" * 78)
    print("总览:")
    for line in payload["summary"]:
        print("  " + line)
    print("=" * 78)
    _write_json(args.json_out, payload)
    return 0


def _final_summary(check_results: dict) -> list:
    lines = []
    if "quote" in check_results:
        ok = [r for r in check_results["quote"] if r["status"] == "OK"]
        lines.append(f"/quote: 有效报价 {len(ok)} 个标的"
                     + (f"（t_et 多为 {ok[0]['t_et']}，即上一交易日收盘时间戳）" if ok else ""))
    if "candle" in check_results:
        rows = check_results["candle"]
        stock_ok = [r for r in rows if r.get("resolution") != "1crypto" and r["status"].startswith("OK")]
        n403 = len([r for r in rows if "403" in str(r["status"])])
        btc = [r for r in rows if r.get("resolution") == "1crypto"]
        lines.append(f"/stock/candle: 美股可用组合 {len(stock_ok)}；HTTP403 {n403} 次；"
                     f"加密对照 {btc[0]['status'] if btc else '-'} bars={btc[0]['n_bars'] if btc else '-'}")
    if "tier" in check_results:
        statuses = {r["name"]: r["status"] for r in check_results["tier"]}
        lines.append("tier: " + ", ".join(f"{k}={v}" for k, v in statuses.items()))
    if "ws" in check_results:
        for run in check_results["ws"]:
            counts = run["trade_counts"]
            lines.append(f"ws[{','.join(run['symbols'])}]: " + ", ".join(f"{k}={v}" for k, v in counts.items())
                         + f" | total_msg={run['total_messages']} errors={len(run['error_msgs'])}")
    return lines


if __name__ == "__main__":
    sys.exit(main())
