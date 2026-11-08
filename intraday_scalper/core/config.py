"""Typed configuration for the intraday scalper.

All tunables live in ``intraday_scalper/config/default.yaml``; this module only
parses/validates that YAML into frozen dataclasses. Secrets never live here:
the Finnhub key is read from ``FINNHUB_API_KEY`` (or the git-ignored
``intraday_scalper/.env``) by the provider layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import time
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import Mode

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "default.yaml"


# --------------------------------------------------------------------------- #
# Value-object helpers
# --------------------------------------------------------------------------- #
def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _as_int(value: Any, name: str, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {number}")
    return number


def _as_float(
    value: Any,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_zero: bool = True,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not allow_zero and number == 0:
        raise ValueError(f"{name} must be non-zero")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {number}")
    return number


def _as_time(value: Any, name: str) -> time:
    if isinstance(value, time):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an 'HH:MM' string, got {value!r}")
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"{name} must be 'HH:MM', got {value!r}")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{name} has an invalid time: {value!r}")
    return time(hour, minute)


# --------------------------------------------------------------------------- #
# Config sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Risk budget and hard limits. Defaults match the requested $3,000 account."""

    account_equity: float = 3_000.0
    max_risk_per_trade: float = 20.0
    max_notional_per_trade: float = 1_200.0
    max_daily_net_loss: float = 60.0
    max_entries_per_day: int = 15
    consecutive_stop_cooldown_minutes: int = 20
    consecutive_stop_threshold: int = 2
    switch_cooldown_minutes: int = 5
    allow_fractional_shares: bool = False
    fractional_precision: int = 5
    est_slippage_bps: float = 1.5
    commission_per_share: float = 0.0
    min_reward_risk: float = 1.8
    breakeven_hint_r: float = 1.2
    take_profit_hint_r: tuple[float, ...] = (2.0, 2.5)


@dataclass(frozen=True, slots=True)
class OrbParams:
    enabled_5m: bool = True
    enabled_15m: bool = True
    min_body_ratio: float = 0.55
    min_volume_multiple: float = 1.5
    trend_ema_period: int = 21
    atr_period: int = 14
    stop_atr_multiple: float = 1.0
    min_stop_pct: float = 0.0025
    max_stop_pct: float = 0.02
    target_r: float = 2.0
    min_breakout_buffer_pct: float = 0.0005


@dataclass(frozen=True, slots=True)
class VwapReclaimParams:
    enabled: bool = True
    min_volume_multiple: float = 1.2
    ema_fast: int = 9
    ema_slow: int = 21
    atr_period: int = 14
    stop_atr_multiple: float = 1.2
    min_stop_pct: float = 0.003
    max_stop_pct: float = 0.02
    target_r: float = 2.0
    reclaim_buffer_pct: float = 0.0005


@dataclass(frozen=True, slots=True)
class EmaPullbackParams:
    enabled: bool = True
    ema_fast: int = 9
    ema_slow: int = 21
    vwap_trend_bars: int = 3
    atr_period: int = 14
    stop_atr_multiple: float = 1.0
    min_stop_pct: float = 0.0025
    max_stop_pct: float = 0.02
    target_r: float = 2.0
    confirmation_body_ratio: float = 0.35


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    orb: OrbParams = field(default_factory=OrbParams)
    vwap_reclaim: VwapReclaimParams = field(default_factory=VwapReclaimParams)
    ema_pullback: EmaPullbackParams = field(default_factory=EmaPullbackParams)


@dataclass(frozen=True, slots=True)
class DataConfig:
    symbols: tuple[str, ...] = ("SOXL", "SOXS")
    ws_url: str = "wss://ws.finnhub.io"
    staleness_seconds: float = 90.0
    max_reconnect_attempts: int = 50
    reconnect_backoff_initial_seconds: float = 1.0
    reconnect_backoff_max_seconds: float = 60.0
    reconnect_backoff_factor: float = 2.0
    ws_open_timeout_seconds: float = 15.0
    ws_ping_interval_seconds: float = 20.0
    startup_quote_checks_max: int = 2
    rest_timeout_seconds: float = 10.0
    rest_min_interval_seconds: float = 1.1
    rest_max_retries_429: int = 3
    max_tick_age_seconds: float = 300.0
    require_true_vwap: bool = True


@dataclass(frozen=True, slots=True)
class SessionConfig:
    no_new_entries_et: time = time(15, 30)
    eod_check_et: time = time(15, 50)
    eod_urgent_et: time = time(15, 55)
    market_close_et: time = time(16, 0)
    opening_range_5m_et: time = time(9, 35)
    opening_range_15m_et: time = time(9, 45)


@dataclass(frozen=True, slots=True)
class PathsConfig:
    package_root: Path = PACKAGE_ROOT
    var_dir: Path = PACKAGE_ROOT / "var"
    raw_tick_dir: Path = PACKAGE_ROOT / "var" / "raw_ticks"
    bar_dir: Path = PACKAGE_ROOT / "var" / "bars"
    log_dir: Path = PACKAGE_ROOT / "var" / "logs"
    journal_db: Path = PACKAGE_ROOT / "var" / "journal.sqlite3"
    backtest_dir: Path = PACKAGE_ROOT / "var" / "backtest"

    def ensure_dirs(self) -> None:
        for directory in (self.var_dir, self.raw_tick_dir, self.bar_dir, self.log_dir, self.backtest_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class AppConfig:
    mode: Mode = Mode.DUAL_WATCH
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # -- convenience ---------------------------------------------------- #
    def with_mode(self, mode: Mode | str) -> "AppConfig":
        return replace(self, mode=Mode(mode))

    def validate(self) -> "AppConfig":
        if self.risk.max_risk_per_trade > self.risk.account_equity:
            raise ValueError("max_risk_per_trade cannot exceed account_equity")
        if self.risk.max_notional_per_trade > self.risk.account_equity:
            raise ValueError("max_notional_per_trade cannot exceed account_equity")
        if self.risk.max_daily_net_loss > self.risk.account_equity:
            raise ValueError("max_daily_net_loss cannot exceed account_equity")
        if self.risk.max_entries_per_day <= 0:
            raise ValueError("max_entries_per_day must be positive")
        if not self.data.symbols:
            raise ValueError("data.symbols must not be empty")
        if self.data.staleness_seconds <= 0:
            raise ValueError("data.staleness_seconds must be positive")
        return self


# --------------------------------------------------------------------------- #
# YAML -> dataclass
# --------------------------------------------------------------------------- #
def _section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = mapping.get(key) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"config section '{key}' must be a mapping")
    return section


def _mode(value: Any) -> Mode:
    try:
        return Mode(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"mode must be one of {[m.value for m in Mode]}, got {value!r}") from exc


def config_from_mapping(mapping: Mapping[str, Any]) -> AppConfig:
    """Build a validated AppConfig from a parsed YAML mapping."""
    if not isinstance(mapping, Mapping):
        raise ValueError("config root must be a mapping")

    risk_raw = _section(mapping, "risk")
    orb_raw = _section(mapping, "strategies").get("orb") or {}
    vwap_raw = _section(mapping, "strategies").get("vwap_reclaim") or {}
    pull_raw = _section(mapping, "strategies").get("ema_pullback") or {}
    str_raw = _section(mapping, "strategies")
    data_raw = _section(mapping, "data")
    sess_raw = _section(mapping, "session")
    paths_raw = _section(mapping, "paths")

    tp_raw = risk_raw.get("take_profit_hint_r", (2.0, 2.5))
    if isinstance(tp_raw, (int, float)):
        tp_values: tuple[float, ...] = (float(tp_raw),)
    else:
        tp_values = tuple(sorted(_as_float(v, "take_profit_hint_r", minimum=0.1) for v in tp_raw))

    risk = RiskConfig(
        account_equity=_as_float(risk_raw.get("account_equity", 3000.0), "risk.account_equity", minimum=1.0),
        max_risk_per_trade=_as_float(
            risk_raw.get("max_risk_per_trade", 20.0), "risk.max_risk_per_trade", minimum=0.01
        ),
        max_notional_per_trade=_as_float(
            risk_raw.get("max_notional_per_trade", 1200.0), "risk.max_notional_per_trade", minimum=1.0
        ),
        max_daily_net_loss=_as_float(
            risk_raw.get("max_daily_net_loss", 60.0), "risk.max_daily_net_loss", minimum=0.01
        ),
        max_entries_per_day=_as_int(
            risk_raw.get("max_entries_per_day", 15), "risk.max_entries_per_day", minimum=1, maximum=100
        ),
        consecutive_stop_cooldown_minutes=_as_int(
            risk_raw.get("consecutive_stop_cooldown_minutes", 20),
            "risk.consecutive_stop_cooldown_minutes",
            minimum=0,
        ),
        consecutive_stop_threshold=_as_int(
            risk_raw.get("consecutive_stop_threshold", 2), "risk.consecutive_stop_threshold", minimum=1
        ),
        switch_cooldown_minutes=_as_int(
            risk_raw.get("switch_cooldown_minutes", 5), "risk.switch_cooldown_minutes", minimum=0
        ),
        allow_fractional_shares=_as_bool(
            risk_raw.get("allow_fractional_shares", False), "risk.allow_fractional_shares"
        ),
        fractional_precision=_as_int(
            risk_raw.get("fractional_precision", 5), "risk.fractional_precision", minimum=0, maximum=8
        ),
        est_slippage_bps=_as_float(
            risk_raw.get("est_slippage_bps", 1.5), "risk.est_slippage_bps", minimum=0.0
        ),
        commission_per_share=_as_float(
            risk_raw.get("commission_per_share", 0.0), "risk.commission_per_share", minimum=0.0
        ),
        min_reward_risk=_as_float(risk_raw.get("min_reward_risk", 1.8), "risk.min_reward_risk", minimum=0.1),
        breakeven_hint_r=_as_float(
            risk_raw.get("breakeven_hint_r", 1.2), "risk.breakeven_hint_r", minimum=0.1
        ),
        take_profit_hint_r=tp_values,
    )

    orb = OrbParams(
        enabled_5m=_as_bool(orb_raw.get("enabled_5m", True), "strategies.orb.enabled_5m"),
        enabled_15m=_as_bool(orb_raw.get("enabled_15m", True), "strategies.orb.enabled_15m"),
        min_body_ratio=_as_float(
            orb_raw.get("min_body_ratio", 0.55), "strategies.orb.min_body_ratio", minimum=0.0, maximum=1.0
        ),
        min_volume_multiple=_as_float(
            orb_raw.get("min_volume_multiple", 1.5), "strategies.orb.min_volume_multiple", minimum=0.0
        ),
        trend_ema_period=_as_int(orb_raw.get("trend_ema_period", 21), "strategies.orb.trend_ema_period", minimum=1),
        atr_period=_as_int(orb_raw.get("atr_period", 14), "strategies.orb.atr_period", minimum=1),
        stop_atr_multiple=_as_float(
            orb_raw.get("stop_atr_multiple", 1.0), "strategies.orb.stop_atr_multiple", minimum=0.0
        ),
        min_stop_pct=_as_float(orb_raw.get("min_stop_pct", 0.0025), "strategies.orb.min_stop_pct", minimum=0.0),
        max_stop_pct=_as_float(orb_raw.get("max_stop_pct", 0.02), "strategies.orb.max_stop_pct", minimum=0.0),
        target_r=_as_float(orb_raw.get("target_r", 2.0), "strategies.orb.target_r", minimum=0.1),
        min_breakout_buffer_pct=_as_float(
            orb_raw.get("min_breakout_buffer_pct", 0.0005),
            "strategies.orb.min_breakout_buffer_pct",
            minimum=0.0,
        ),
    )
    vwap = VwapReclaimParams(
        enabled=_as_bool(vwap_raw.get("enabled", True), "strategies.vwap_reclaim.enabled"),
        min_volume_multiple=_as_float(
            vwap_raw.get("min_volume_multiple", 1.2), "strategies.vwap_reclaim.min_volume_multiple", minimum=0.0
        ),
        ema_fast=_as_int(vwap_raw.get("ema_fast", 9), "strategies.vwap_reclaim.ema_fast", minimum=1),
        ema_slow=_as_int(vwap_raw.get("ema_slow", 21), "strategies.vwap_reclaim.ema_slow", minimum=1),
        atr_period=_as_int(vwap_raw.get("atr_period", 14), "strategies.vwap_reclaim.atr_period", minimum=1),
        stop_atr_multiple=_as_float(
            vwap_raw.get("stop_atr_multiple", 1.2), "strategies.vwap_reclaim.stop_atr_multiple", minimum=0.0
        ),
        min_stop_pct=_as_float(vwap_raw.get("min_stop_pct", 0.003), "strategies.vwap_reclaim.min_stop_pct", minimum=0.0),
        max_stop_pct=_as_float(vwap_raw.get("max_stop_pct", 0.02), "strategies.vwap_reclaim.max_stop_pct", minimum=0.0),
        target_r=_as_float(vwap_raw.get("target_r", 2.0), "strategies.vwap_reclaim.target_r", minimum=0.1),
        reclaim_buffer_pct=_as_float(
            vwap_raw.get("reclaim_buffer_pct", 0.0005), "strategies.vwap_reclaim.reclaim_buffer_pct", minimum=0.0
        ),
    )
    pull = EmaPullbackParams(
        enabled=_as_bool(pull_raw.get("enabled", True), "strategies.ema_pullback.enabled"),
        ema_fast=_as_int(pull_raw.get("ema_fast", 9), "strategies.ema_pullback.ema_fast", minimum=1),
        ema_slow=_as_int(pull_raw.get("ema_slow", 21), "strategies.ema_pullback.ema_slow", minimum=1),
        vwap_trend_bars=_as_int(pull_raw.get("vwap_trend_bars", 3), "strategies.ema_pullback.vwap_trend_bars", minimum=1),
        atr_period=_as_int(pull_raw.get("atr_period", 14), "strategies.ema_pullback.atr_period", minimum=1),
        stop_atr_multiple=_as_float(
            pull_raw.get("stop_atr_multiple", 1.0), "strategies.ema_pullback.stop_atr_multiple", minimum=0.0
        ),
        min_stop_pct=_as_float(pull_raw.get("min_stop_pct", 0.0025), "strategies.ema_pullback.min_stop_pct", minimum=0.0),
        max_stop_pct=_as_float(pull_raw.get("max_stop_pct", 0.02), "strategies.ema_pullback.max_stop_pct", minimum=0.0),
        target_r=_as_float(pull_raw.get("target_r", 2.0), "strategies.ema_pullback.target_r", minimum=0.1),
        confirmation_body_ratio=_as_float(
            pull_raw.get("confirmation_body_ratio", 0.35),
            "strategies.ema_pullback.confirmation_body_ratio",
            minimum=0.0,
            maximum=1.0,
        ),
    )
    strategies = StrategyConfig(
        ema_fast=_as_int(str_raw.get("ema_fast", 9), "strategies.ema_fast", minimum=1),
        ema_slow=_as_int(str_raw.get("ema_slow", 21), "strategies.ema_slow", minimum=1),
        rsi_period=_as_int(str_raw.get("rsi_period", 14), "strategies.rsi_period", minimum=2),
        atr_period=_as_int(str_raw.get("atr_period", 14), "strategies.atr_period", minimum=1),
        orb=orb,
        vwap_reclaim=vwap,
        ema_pullback=pull,
    )

    symbols_raw = data_raw.get("symbols", ("SOXL", "SOXS"))
    symbols = tuple(str(s).strip().upper() for s in symbols_raw)
    if len(set(symbols)) != len(symbols):
        raise ValueError("data.symbols contains duplicates")
    data = DataConfig(
        symbols=symbols,
        ws_url=str(data_raw.get("ws_url", "wss://ws.finnhub.io")).strip(),
        staleness_seconds=_as_float(
            data_raw.get("staleness_seconds", 90.0), "data.staleness_seconds", minimum=0.1
        ),
        max_reconnect_attempts=_as_int(
            data_raw.get("max_reconnect_attempts", 50), "data.max_reconnect_attempts", minimum=0
        ),
        reconnect_backoff_initial_seconds=_as_float(
            data_raw.get("reconnect_backoff_initial_seconds", 1.0),
            "data.reconnect_backoff_initial_seconds",
            minimum=0.1,
        ),
        reconnect_backoff_max_seconds=_as_float(
            data_raw.get("reconnect_backoff_max_seconds", 60.0),
            "data.reconnect_backoff_max_seconds",
            minimum=0.1,
        ),
        reconnect_backoff_factor=_as_float(
            data_raw.get("reconnect_backoff_factor", 2.0), "data.reconnect_backoff_factor", minimum=1.0
        ),
        ws_open_timeout_seconds=_as_float(
            data_raw.get("ws_open_timeout_seconds", 15.0), "data.ws_open_timeout_seconds", minimum=0.1
        ),
        ws_ping_interval_seconds=_as_float(
            data_raw.get("ws_ping_interval_seconds", 20.0), "data.ws_ping_interval_seconds", minimum=0.0
        ),
        startup_quote_checks_max=_as_int(
            data_raw.get("startup_quote_checks_max", 2),
            "data.startup_quote_checks_max",
            minimum=0,
            maximum=2,
        ),
        rest_timeout_seconds=_as_float(
            data_raw.get("rest_timeout_seconds", 10.0), "data.rest_timeout_seconds", minimum=0.1
        ),
        rest_min_interval_seconds=_as_float(
            data_raw.get("rest_min_interval_seconds", 1.1), "data.rest_min_interval_seconds", minimum=0.0
        ),
        rest_max_retries_429=_as_int(
            data_raw.get("rest_max_retries_429", 3), "data.rest_max_retries_429", minimum=0
        ),
        max_tick_age_seconds=_as_float(
            data_raw.get("max_tick_age_seconds", 300.0), "data.max_tick_age_seconds", minimum=0.1
        ),
        require_true_vwap=_as_bool(data_raw.get("require_true_vwap", True), "data.require_true_vwap"),
    )

    session = SessionConfig(
        no_new_entries_et=_as_time(sess_raw.get("no_new_entries_et", "15:30"), "session.no_new_entries_et"),
        eod_check_et=_as_time(sess_raw.get("eod_check_et", "15:50"), "session.eod_check_et"),
        eod_urgent_et=_as_time(sess_raw.get("eod_urgent_et", "15:55"), "session.eod_urgent_et"),
        market_close_et=_as_time(sess_raw.get("market_close_et", "16:00"), "session.market_close_et"),
        opening_range_5m_et=_as_time(
            sess_raw.get("opening_range_5m_et", "09:35"), "session.opening_range_5m_et"
        ),
        opening_range_15m_et=_as_time(
            sess_raw.get("opening_range_15m_et", "09:45"), "session.opening_range_15m_et"
        ),
    )

    paths = PathsConfig(
        package_root=PACKAGE_ROOT,
        var_dir=PACKAGE_ROOT / "var",
        raw_tick_dir=PACKAGE_ROOT / "var" / "raw_ticks",
        bar_dir=PACKAGE_ROOT / "var" / "bars",
        log_dir=PACKAGE_ROOT / "var" / "logs",
        journal_db=PACKAGE_ROOT / "var" / "journal.sqlite3",
        backtest_dir=PACKAGE_ROOT / "var" / "backtest",
    )

    config = AppConfig(
        mode=_mode(mapping.get("mode", "dual_watch")),
        risk=risk,
        strategies=strategies,
        data=data,
        session=session,
        paths=paths,
    )
    return config.validate()


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load AppConfig from YAML (defaults to ``config/default.yaml``)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config_from_mapping(raw)
