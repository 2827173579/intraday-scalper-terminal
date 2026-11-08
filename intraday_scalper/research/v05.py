"""V0.5 R-Ladder runner exit modes + OR30 dead-fish regime filter.

**Entries are frozen** to V0.4 Step 3c-A NoCap (10:00-14:00 ET window, fractional
shares, VWAP_RECLAIM_5M trend filter, EMA_PULLBACK_5M, SMC_SWEEP_5M with strict
AND confirmation, no 8% exhaustion cap, no Step-4 scale-out).  V0.5 only changes
**exits** and adds an optional **OR30 dead-fish day filter**; nothing else.

B1 risk base (task-15): equity $10,000, risk $60/trade (sensitivity $80),
notional $10,000, fractional 0.01, 3 consecutive stops -> 15-minute cooldown,
daily -$320 circuit breaker, max 15 entries/day.

Exit modes
==========
``baseline``  B1: +1.2R breakeven shield moves the stop to +0.05R, then a fixed
              +2.0R target (this is exactly the V0.4 Step3c-NoCap exit logic, so
              the 2026-08-01..11-06 anchor 68 trades / +$385.89 / PF 1.2138 is
              reproduced bit-for-bit).
``runner_a``  delayed breakeven + R-ladder, fixed +4.5R target:
              MFE >= 1.5R -> stop +0.20R; MFE >= 2.2R -> stop +1.20R;
              MFE >= 3.0R -> stop +2.00R; target +4.5R; else EOD.
``runner_b``  lock and let it run: MFE >= 1.5R -> stop +0.20R; MFE >= 2.0R ->
              stop max(1.0R, MFE-1.0R) (no fixed R cap); extra exits: any
              completed 5m close < 5m EMA21 (next 1m open), or the trailing lock
              stop; otherwise hold to 15:59 ET EOD.

Execution conventions (no look-ahead)
-------------------------------------
* MFE and the ladder stop are updated on a **completed 5m bar** (its high /
  close) and only become effective from the **next 1m bar**.  Inside a bucket the
  previous stop/target apply; a 1m bar that touches both stop and target fills the
  **stop** (conservative).
* Targets are checked against each 1m bar's high.  Stop fills gap through at the
  bar open when the open is already past the stop.
* The EMA21 exit is evaluated on completed 5m closes only and executes at the
  next 1m bar's open.
* All stops are monotonic (only move up, never down).
* Slippage = ``config.risk.est_slippage_bps`` on both sides; commission from
  ``config.risk.commission_per_share`` (B1: 0).

OR30 dead-fish filter
---------------------
For each (session, symbol): OR30 = first six completed 5m bars (09:30-10:00 ET).
``Range_pct = (OR30_High - OR30_Low) / OR30_Low``; ``ATR14_5m(10:00)`` is the
5m ATR14 value as of the last OR30 bar.  Two sub-definitions, reported
separately:

* ``pct_only``   : skip new entries for the day when ``Range_pct < 1.8%``.
* ``pct_or_atr`` : the literal OR condition - skip when ``Range_pct < 1.8%``
  **or** ``OR30_Range < 1.2 * ATR14_5m(10:00)``.

The filter only blocks *new* entries after 10:00 for that symbol/session; open
positions are managed unchanged.  Thresholds are fixed a-priori, not tuned.

Data caveat: Alpaca IEX-only single venue (small share of US consolidated
volume); research output, not proof of profitability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from statistics import mean
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
    SignalCandidate,
    TradeRecord,
    et_date,
    load_config,
    to_et,
)
from intraday_scalper.research import v02, v03, v04
from intraday_scalper.research.v02 import (
    DATA_CAVEAT,
    build_strategy_view,
    prepare_v02_data,
)
from intraday_scalper.research.v04 import build_eval_context
from intraday_scalper.risk.manager import RiskManager
from intraday_scalper.risk.sizing import size_position

__all__ = [
    "B1_COOLDOWN_MINUTES",
    "B1_CONSECUTIVE_STOPS",
    "B1_DAILY_LOSS",
    "B1_EQUITY",
    "B1_FRACTIONAL_PRECISION",
    "B1_MAX_ENTRIES",
    "B1_NOTIONAL",
    "B1_RISK",
    "EXIT_BASELINE",
    "EXIT_MODES",
    "EXIT_RUNNER_A",
    "EXIT_RUNNER_B",
    "A_F1_F2_VARIANT",
    "A_F1_VARIANT",
    "A_F2_VARIANT",
    "A_PLUS_F_VARIANT",
    "A_VARIANT",
    "F2_BOX_FRACTION",
    "F2_EMA21_SLOPE_MIN",
    "OR30_ATR_MULTIPLE",
    "OR30_ATR_MULTIPLE_F1",
    "OR30_FILTER_F1",
    "OR30_FILTER_NONE",
    "OR30_PCT_MIN_F1",
    "OR30_FILTER_PCT_ONLY",
    "OR30_FILTER_PCT_OR_ATR",
    "OR30_PCT_MIN",
    "RUNNER_A_LADDER",
    "RUNNER_A_TARGET_R",
    "RUNNER_B_LADDER",
    "RUNNER_B_TRAIL_GAP_R",
    "V05B_VARIANTS",
    "V05_PRODUCTION_VARIANT",
    "V05Result",
    "V05Variant",
    "V05_VARIANTS",
    "V05WalkForwardResult",
    "build_b1_config",
    "compute_or30_blocks",
    "evaluate_v05_session",
    "ladder_stop",
    "or30_filter_blocks",
    "or30_session_metrics",
    "result_to_trade_dicts_v05",
    "run_v05_backtest",
    "run_v05_walkforward",
    "target_price_for_mode",
    "trend_box_allows",
]

# --------------------------------------------------------------------------- #
# Constants — a-priori, not tuned
# --------------------------------------------------------------------------- #
EXIT_BASELINE = "baseline"
EXIT_RUNNER_A = "runner_a"
EXIT_RUNNER_B = "runner_b"
EXIT_MODES: tuple[str, ...] = (EXIT_BASELINE, EXIT_RUNNER_A, EXIT_RUNNER_B)

#: Runner-A: (MFE threshold in R, stop level in R) — ascending thresholds.
RUNNER_A_LADDER: tuple[tuple[float, float], ...] = (
    (1.5, 0.20),
    (2.2, 1.20),
    (3.0, 2.00),
)
RUNNER_A_TARGET_R = 4.5

#: Runner-B: (MFE threshold in R, stop level in R) then lock / trailing.
RUNNER_B_LADDER: tuple[tuple[float, float], ...] = (
    (1.5, 0.20),
    (2.0, 1.00),
)
RUNNER_B_TRAIL_GAP_R = 1.0

#: OR30 dead-fish filter (a-priori, not tuned).
OR30_PCT_MIN = 0.018
OR30_ATR_MULTIPLE = 1.2
OR30_FILTER_NONE = "none"
OR30_FILTER_PCT_ONLY = "pct_only"
OR30_FILTER_PCT_OR_ATR = "pct_or_atr"
#: F1 (V0.5b pre-registered): 30m-scale thresholds (2.8% / 2.2 x ATR14_5m).
OR30_FILTER_F1 = "f1"
OR30_PCT_MIN_F1 = 0.028
OR30_ATR_MULTIPLE_F1 = 2.2
_OR30_FILTERS = (
    OR30_FILTER_NONE,
    OR30_FILTER_PCT_ONLY,
    OR30_FILTER_PCT_OR_ATR,
    OR30_FILTER_F1,
)

#: F2 (V0.5b pre-registered): trend strategies may only trade outside / in the
#: upper half of the OR30 box with a rising 3-bar EMA21 slope.
F2_BOX_FRACTION = 0.5
F2_EMA21_SLOPE_MIN = 0.0015
_TREND_LABELS: tuple[str, ...] = ("VWAP_RECLAIM_5M", "EMA_PULLBACK_5M")

#: B1 risk base (task-15).
B1_EQUITY = 10_000.0
B1_RISK = 60.0
B1_NOTIONAL = 10_000.0
B1_DAILY_LOSS = 320.0
B1_MAX_ENTRIES = 15
B1_CONSECUTIVE_STOPS = 3
B1_COOLDOWN_MINUTES = 15
B1_FRACTIONAL_PRECISION = 2

_BUCKET_MINUTES = 5
_PRICE_EPS = 1e-9
_V05_SIGNAL_PREFIX = "V05-"  # pragma: no cover - parent engine prefixes signals


# --------------------------------------------------------------------------- #
# Variant
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class V05Variant:
    """One V0.5 exit/regime combination (entries are fixed to Step3c-A NoCap)."""

    name: str
    exit_mode: str = EXIT_BASELINE
    or30_filter: str | None = None
    risk_dollars: float = B1_RISK
    trend_box_filter: bool = False

    def __post_init__(self) -> None:
        if self.exit_mode not in EXIT_MODES:
            raise ValueError(
                f"exit_mode must be one of {EXIT_MODES}, got {self.exit_mode!r}"
            )
        if self.or30_filter is None:
            object.__setattr__(self, "or30_filter", OR30_FILTER_NONE)
        elif self.or30_filter not in _OR30_FILTERS:
            raise ValueError(
                f"or30_filter must be one of {_OR30_FILTERS}, got {self.or30_filter!r}"
            )
        if not math.isfinite(float(self.risk_dollars)) or self.risk_dollars <= 0.0:
            raise ValueError(f"risk_dollars must be > 0, got {self.risk_dollars!r}")

    def flags(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "exit_mode": self.exit_mode,
            "or30_filter": self.or30_filter,
            "risk_dollars": float(self.risk_dollars),
            "trend_box_filter": bool(self.trend_box_filter),
        }


#: The six matrix rows (3 exits x with/without the literal OR30 filter).
BASELINE_VARIANT = V05Variant("B1", EXIT_BASELINE)
RUNNER_A_VARIANT = V05Variant("A", EXIT_RUNNER_A)
RUNNER_B_VARIANT = V05Variant("B", EXIT_RUNNER_B)
BASELINE_FILTER_VARIANT = V05Variant("B1+F", EXIT_BASELINE, OR30_FILTER_PCT_OR_ATR)
RUNNER_A_FILTER_VARIANT = V05Variant("A+F", EXIT_RUNNER_A, OR30_FILTER_PCT_OR_ATR)
RUNNER_B_FILTER_VARIANT = V05Variant("B+F", EXIT_RUNNER_B, OR30_FILTER_PCT_OR_ATR)
MATRIX_VARIANTS: tuple[V05Variant, ...] = (
    BASELINE_VARIANT,
    RUNNER_A_VARIANT,
    RUNNER_B_VARIANT,
    BASELINE_FILTER_VARIANT,
    RUNNER_A_FILTER_VARIANT,
    RUNNER_B_FILTER_VARIANT,
)

#: V0.5b pre-registered comparison set (all use the Runner-A 4.5R R-ladder).
A_VARIANT = V05Variant("A", EXIT_RUNNER_A)
A_PLUS_F_VARIANT = V05Variant(
    "A+F", EXIT_RUNNER_A, OR30_FILTER_PCT_OR_ATR
)
A_F1_VARIANT = V05Variant("A+F1", EXIT_RUNNER_A, OR30_FILTER_F1)
A_F2_VARIANT = V05Variant(
    "A+F2", EXIT_RUNNER_A, OR30_FILTER_NONE, trend_box_filter=True
)
A_F1_F2_VARIANT = V05Variant(
    "A+F1+F2", EXIT_RUNNER_A, OR30_FILTER_F1, trend_box_filter=True
)
V05B_VARIANTS: tuple[V05Variant, ...] = (
    A_PLUS_F_VARIANT,
    A_F1_VARIANT,
    A_F2_VARIANT,
    A_F1_F2_VARIANT,
)
V05_VARIANTS: dict[str, V05Variant] = {
    "A": A_VARIANT,
    "A+F": A_PLUS_F_VARIANT,
    "A+F1": A_F1_VARIANT,
    "A+F2": A_F2_VARIANT,
    "A+F1+F2": A_F1_F2_VARIANT,
}

#: Web production variant placeholder: A+F (Lead switches after this A/B report).
V05_PRODUCTION_VARIANT = A_PLUS_F_VARIANT


def build_b1_config(config: AppConfig, risk_dollars: float = B1_RISK) -> AppConfig:
    """Return the in-memory B1 risk override (task-15 spec; never persisted)."""

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not math.isfinite(float(risk_dollars)) or risk_dollars <= 0.0:
        raise ValueError(f"risk_dollars must be > 0, got {risk_dollars!r}")
    return replace(
        config,
        risk=replace(
            config.risk,
            account_equity=B1_EQUITY,
            max_risk_per_trade=float(risk_dollars),
            max_notional_per_trade=B1_NOTIONAL,
            max_daily_net_loss=B1_DAILY_LOSS,
            max_entries_per_day=B1_MAX_ENTRIES,
            consecutive_stop_threshold=B1_CONSECUTIVE_STOPS,
            consecutive_stop_cooldown_minutes=B1_COOLDOWN_MINUTES,
            allow_fractional_shares=True,
            fractional_precision=B1_FRACTIONAL_PRECISION,
            est_slippage_bps=1.5,
            commission_per_share=0.0,
        ),
    )


# --------------------------------------------------------------------------- #
# R-ladder helpers (pure)
# --------------------------------------------------------------------------- #
def _is_runner(mode: str) -> bool:
    return mode in (EXIT_RUNNER_A, EXIT_RUNNER_B)


def ladder_stop(mode: str, max_mfe_r: float) -> tuple[float, str]:
    """Return ``(stop_level_in_R, stop_kind)`` for the current MFE (monotonic).

    ``baseline`` keeps the initial stop here (the V0.3 shield is handled by the
    parent engine).  All levels are fixed a-priori, never tuned.
    """

    if mode == EXIT_RUNNER_A:
        level = -1.0
        kind = "INITIAL"
        for threshold, stop_level in RUNNER_A_LADDER:
            if max_mfe_r >= threshold - 1e-12:
                level = max(level, stop_level)
                kind = f"LADDER_{threshold:g}R"
        return level, kind
    if mode == EXIT_RUNNER_B:
        level = -1.0
        kind = "INITIAL"
        for threshold, stop_level in RUNNER_B_LADDER:
            if max_mfe_r >= threshold - 1e-12:
                level = max(level, stop_level)
                kind = f"LADDER_{threshold:g}R"
        if max_mfe_r >= RUNNER_B_LADDER[-1][0] - 1e-12:
            trailing_level = max(RUNNER_B_LADDER[-1][1], max_mfe_r - RUNNER_B_TRAIL_GAP_R)
            if trailing_level > level:
                level = trailing_level
                kind = "TRAIL_LOCK"
        return level, kind
    return -1.0, "INITIAL"


def target_price_for_mode(
    mode: str,
    entry_fill: float,
    risk_per_share: float,
) -> float | None:
    """Fixed-R target price for the mode (None = uncapped / no fixed target)."""

    if mode == EXIT_RUNNER_A:
        return float(entry_fill) + RUNNER_A_TARGET_R * float(risk_per_share)
    return None


# --------------------------------------------------------------------------- #
# OR30 range filter
# --------------------------------------------------------------------------- #
def or30_session_metrics(
    session: Any,
    symbol_session: Any,
    features: Any,
) -> dict[str, Any]:
    """Compute OR_30M high/low/range/pct and the 10:00 ATR14(5m) value."""

    expected = [
        session.open_utc + timedelta(minutes=_BUCKET_MINUTES * index)
        for index in range(6)
    ]
    if not all(start in symbol_session.bucket_by_start for start in expected):
        return {
            "or30_high": None,
            "or30_low": None,
            "or30_range": None,
            "range_pct": None,
            "atr14_5m_1000": None,
        }
    members = [symbol_session.bucket_by_start[start] for start in expected]
    high = max(item.bar.high for item in members)
    low = min(item.bar.low for item in members)
    or30_range = high - low
    range_pct = (or30_range / low) if low > 0.0 else None
    # ATR14(5m) as of 10:00 = value on the last OR30 bucket (09:55-10:00).
    last_bucket = members[-1]
    atr = features.atr[last_bucket.global_index]
    return {
        "or30_high": high,
        "or30_low": low,
        "or30_range": or30_range,
        "range_pct": range_pct,
        "atr14_5m_1000": atr,
    }


def or30_filter_blocks(or30_metrics: Mapping[str, Any], mode: str | None) -> bool:
    """True when the OR30 dead-fish filter blocks new entries for the session.

    ``pct_only``  : ``Range_pct < 1.8%``.
    ``pct_or_atr``: literal task OR - ``Range_pct < 1.8%`` **or**
                    ``OR30_Range < 1.2 * ATR14_5m(10:00)``.
    ``none`` / ``None``: never blocks.  Unavailable OR30 data never blocks.
    """

    if mode in (None, OR30_FILTER_NONE):
        return False
    range_pct = or30_metrics.get("range_pct")
    or30_range = or30_metrics.get("or30_range")
    atr = or30_metrics.get("atr14_5m_1000")
    if range_pct is None or or30_range is None:
        return False
    if mode == OR30_FILTER_F1:
        pct_min = OR30_PCT_MIN_F1
        atr_multiple = OR30_ATR_MULTIPLE_F1
    else:
        pct_min = OR30_PCT_MIN
        atr_multiple = OR30_ATR_MULTIPLE
    if range_pct < pct_min:
        return True
    if (
        mode in (OR30_FILTER_PCT_OR_ATR, OR30_FILTER_F1)
        and atr is not None
        and float(or30_range) < atr_multiple * float(atr)
    ):
        return True
    return False


def compute_or30_blocks(
    data: v02.V02Data,
    mode: str | None,
) -> tuple[frozenset[tuple[date, str]], list[dict[str, Any]]]:
    """Return the blocked ``(session_date, symbol)`` set plus per-session stats."""

    blocked: set[tuple[date, str]] = set()
    rows: list[dict[str, Any]] = []
    for session in data.sessions:
        for symbol, symbol_session in session.symbols.items():
            metrics = or30_session_metrics(
                session, symbol_session, data.features[symbol]
            )
            is_blocked = or30_filter_blocks(metrics, mode)
            if is_blocked:
                blocked.add((session.et_date, symbol))
            rows.append(
                {
                    "et_date": session.et_date.isoformat(),
                    "symbol": symbol,
                    **{
                        key: (round(value, 6) if isinstance(value, float) else value)
                        for key, value in metrics.items()
                    },
                    "blocked": bool(is_blocked),
                }
            )
    return frozenset(blocked), rows


def trend_box_allows(
    close: float,
    or30_high: float | None,
    or30_low: float | None,
    ema21_now: float | None,
    ema21_3bars_ago: float | None,
) -> bool:
    """F2: trend strategies may only trade outside / upper-half of the OR30 box.

    Pre-registered rule (fixed a-priori, causal):
    1) ``close >= OR30_High`` (box breakout), OR
    2) ``close >= OR30_Low + 0.5 * (OR30_High - OR30_Low)`` **and**
       ``(EMA21 - EMA21[3]) / EMA21[3] >= 0.15%`` (3 completed 5m bars slope up).

    Missing OR30/EMA references cannot be confirmed -> reject (conservative).
    SMC_SWEEP_5M is never passed through this helper.
    """

    if or30_high is None or or30_low is None:
        return False
    box_high = float(or30_high)
    box_low = float(or30_low)
    if float(close) >= box_high - _PRICE_EPS:
        return True
    box_mid = box_low + F2_BOX_FRACTION * (box_high - box_low)
    if float(close) < box_mid - _PRICE_EPS:
        return False
    if ema21_now is None or ema21_3bars_ago is None:
        return False
    reference = float(ema21_3bars_ago)
    if reference <= 0.0:
        return False
    slope = (float(ema21_now) - reference) / reference
    return slope >= F2_EMA21_SLOPE_MIN - 1e-12


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class V05Result:
    """One V0.5 run (full period, 3-month window, a segment or a WF fold)."""

    variant: V05Variant
    trades: list[TradeRecord]
    equity_curve: list[tuple[datetime, float]]
    metrics: BacktestMetrics
    trade_details: dict[str, dict[str, Any]]
    markers: list[dict[str, Any]]
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
    or30_skipped_evaluations: int = 0
    or30_blocked_sessions: int = 0
    trend_box_rejected: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    def extra_metrics(self) -> dict[str, Any]:
        exit_reasons: dict[str, int] = {}
        for trade in self.trades:
            reason = trade.exit_reason or "UNKNOWN"
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        target_r_values = [
            float(trade.r_multiple)
            for trade in self.trades
            if trade.exit_reason == "TARGET"
        ]
        return {
            "variant": self.variant.flags(),
            "ambiguous_bars": int(self.ambiguous_bars),
            "breaker_days": int(self.breaker_days),
            "pending_cancelled": int(self.pending_cancelled),
            "signals": int(self.signals),
            "buckets_evaluated": int(self.buckets_evaluated),
            "skipped_buckets": int(self.skipped_buckets),
            "sessions_run": int(self.sessions_run),
            "or30_skipped_evaluations": int(self.or30_skipped_evaluations),
            "or30_blocked_sessions": int(self.or30_blocked_sessions),
            "trend_box_rejected": int(self.trend_box_rejected),
            "exit_reasons": dict(sorted(exit_reasons.items())),
            "target_exits": len(target_r_values),
            "target_avg_r": round(mean(target_r_values), 4) if target_r_values else 0.0,
            "risk_block_counts": dict(self.risk_block_counts),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class V05FoldResult:
    fold: Fold
    train_result: V05Result
    test_result: V05Result
    leakage_checked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": {
                "index": self.fold.index,
                "train_start": self.fold.train_start.isoformat()
                if self.fold.train_start
                else None,
                "test_start": self.fold.test_start.isoformat()
                if self.fold.test_start
                else None,
                "test_end": self.fold.test_end.isoformat()
                if self.fold.test_end
                else None,
            },
            "train_trades": self.train_result.total_trades,
            "test_trades": self.test_result.total_trades,
            "test_metrics": self.test_result.metrics.to_dict(),
            "test_extra": self.test_result.extra_metrics(),
            "leakage_checked": self.leakage_checked,
        }


@dataclass(frozen=True, slots=True)
class V05WalkForwardResult:
    variant: V05Variant
    folds: tuple[V05FoldResult, ...]
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
# Replay engine (subclasses the frozen V0.4 replay; entries untouched)
# --------------------------------------------------------------------------- #
class _V05Replay(v04._V04Replay):
    """V0.4 Step3c-A NoCap entries + V0.5 R-ladder exits + optional OR30 filter."""

    def __init__(
        self,
        data: v02.V02Data,
        config: AppConfig,
        variant: V05Variant,
        *,
        risk: Any | None = None,
        initial_equity: float | None = None,
        or30_blocks: frozenset[tuple[date, str]] | None = None,
    ) -> None:
        if not isinstance(variant, V05Variant):
            raise TypeError("variant must be a V05Variant")
        # Entries are frozen to Step3c-A NoCap (AND SMC, no exhaustion cap).
        super().__init__(
            data,
            config,
            v04.STEP3C_NOCAP_VARIANT,
            risk=risk,
            initial_equity=initial_equity,
        )
        self.v05_variant = variant
        self._exit_mode = variant.exit_mode
        self._or30_blocks = frozenset(or30_blocks or ())
        self._trend_box_filter = bool(variant.trend_box_filter)
        self._f2_rejected = 0
        self._or30_skipped = 0
        self._or30_blocked_sessions = len(
            {session for session, _ in self._or30_blocks}
        )

    # -- strategy evaluation: identical to the frozen V0.4 engine, plus the
    #    per-(session, symbol) OR30 block for NEW entries only --------------
    def _evaluate_bucket(self, session: Any, bucket_start: datetime) -> None:
        if not self._evaluation_allowed(bucket_start):
            self._cutoff_skipped += 1
            return
        views: list[tuple[Any, Any]] = []
        for symbol in self.symbols:
            if (session.et_date, symbol) in self._or30_blocks:
                self._or30_skipped += 1
                continue
            symbol_session = session.symbols.get(symbol)
            if symbol_session is None:
                continue
            if bucket_start not in symbol_session.bucket_by_start:
                continue
            view = build_strategy_view(self.data, session, symbol, bucket_start)
            if view is None:
                continue
            context = build_eval_context(
                self.data, session, symbol, bucket_start, self._flat_buckets
            )
            views.append((view, context))
        if not views:
            return
        candidate: SignalCandidate | None = None
        candidate_label = ""
        candidate_context: Any = None
        for label, rule in self._rule_items:
            for view, context in views:
                candidate = rule(view, context)
                if candidate is not None:
                    candidate_label = label
                    candidate_context = context
                    break
            if candidate is not None:
                break
        if candidate is None:
            return
        if self._max_stop_pct is not None:
            entry_ref = float(candidate.entry_ref)
            stop_loss = float(candidate.stop_loss)
            if entry_ref > 0:
                stop_pct = (entry_ref - stop_loss) / entry_ref
                if stop_pct > float(self._max_stop_pct):
                    return
        if self._trend_box_filter and candidate_label in _TREND_LABELS:
            # F2 uses only completed data: current confirmed bar close, OR30 of
            # the same session and EMA21 vs the EMA21 value 3 completed bars back.
            symbol_session = session.symbols.get(candidate.symbol)
            bucket = (
                symbol_session.bucket_by_start.get(bucket_start)
                if symbol_session is not None
                else None
            )
            ema_now = None
            ema_prev3 = None
            if bucket is not None:
                features = self.data.features[candidate.symbol]
                global_index = bucket.global_index
                ema_now = features.ema_slow[global_index]
                if global_index >= 3:
                    ema_prev3 = features.ema_slow[global_index - 3]
            if not trend_box_allows(
                float(candidate.entry_ref),
                candidate_context.or30_high if candidate_context is not None else None,
                candidate_context.or30_low if candidate_context is not None else None,
                ema_now,
                ema_prev3,
            ):
                self._f2_rejected += 1
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
        signal_id = f"V05-{self._signal_seq:06d}"
        self._signal_labels[signal_id] = candidate_label
        self._pending = v03._PendingEntry(
            candidate=candidate,
            plan=allowed_plan,
            signal_id=signal_id,
            expected_bucket_start_utc=bucket_start + timedelta(minutes=_BUCKET_MINUTES),
        )

    # -- exits ---------------------------------------------------------- #
    def _manage_open_trade(self, session: Any, bucket_start: datetime) -> None:
        if not _is_runner(self._exit_mode):
            # Baseline (B1) = the frozen V0.4 Step3c-NoCap exit: 1.2R shield ->
            # +0.05R stop and fixed 2.0R target.
            super()._manage_open_trade(session, bucket_start)
            return
        self._manage_runner_trade(session, bucket_start)

    def _state(self, trade: Any) -> dict[str, Any]:
        state = getattr(trade, "v05_state", None)
        if state is None:
            risk_per_share = float(trade.entry_fill) - float(trade.stop)
            if risk_per_share <= 0.0:  # pragma: no cover - contract guard
                risk_per_share = 1e-12
            state = {
                "risk_per_share": risk_per_share,
                "initial_stop": float(trade.stop),
                "stop": float(trade.stop),
                "stop_kind": "INITIAL",
                "max_mfe_r": 0.0,
                "target": target_price_for_mode(
                    self._exit_mode, float(trade.entry_fill), risk_per_share
                ),
                "ema_exit_pending": False,
            }
            trade.v05_state = state
        return state

    def _manage_runner_trade(self, session: Any, bucket_start: datetime) -> None:
        trade = self._open
        if trade is None:  # pragma: no cover - caller guarantees
            return
        symbol = trade.candidate.symbol
        symbol_session = session.symbols.get(symbol)
        if symbol_session is None:  # pragma: no cover
            return
        bucket = symbol_session.bucket_by_start.get(bucket_start)
        if bucket is None:
            return
        state = self._state(trade)

        for bar in bucket.members:
            if bar.start_utc < trade.entry_time_utc:
                continue
            # 0) EMA21 exit (Runner-B only), effective from a completed 5m close.
            if self._exit_mode == EXIT_RUNNER_B and state["ema_exit_pending"]:
                reference = float(bar.open)
                fill = reference * (1.0 - self._slippage_rate)
                self._close_v05(
                    trade,
                    reference=reference,
                    fill=fill,
                    ts=bar.start_utc,
                    reason="TRAIL_EMA21",
                    stop_kind="EMA21_CLOSE",
                    state=state,
                )
                return
            # 1) Excursions (dollar terms for the aggregate trade record).
            trade.mae_dollars = max(
                trade.mae_dollars,
                max(0.0, (trade.entry_fill - bar.low) * trade.shares),
            )
            trade.mfe_dollars = max(
                trade.mfe_dollars,
                max(0.0, (bar.high - trade.entry_fill) * trade.shares),
            )
            # 2) Stop / target on this 1m bar. Collision -> conservative stop.
            stop_price = float(state["stop"])
            target = state["target"]
            stop_hit = bar.low <= stop_price + _PRICE_EPS
            target_hit = target is not None and bar.high >= float(target) - _PRICE_EPS
            if stop_hit or target_hit:
                if stop_hit:
                    reference = bar.open if bar.open <= stop_price else stop_price
                    fill = reference * (1.0 - self._slippage_rate)
                    self._close_v05(
                        trade,
                        reference=reference,
                        fill=fill,
                        ts=bar.end_utc,
                        reason="STOP",
                        stop_kind=state["stop_kind"],
                        state=state,
                    )
                else:
                    reference = float(target)
                    fill = reference * (1.0 - self._slippage_rate)
                    self._close_v05(
                        trade,
                        reference=reference,
                        fill=fill,
                        ts=bar.end_utc,
                        reason="TARGET",
                        stop_kind="TARGET",
                        state=state,
                    )
                return
            # 3) 15:59 ET EOD flatten.
            if bar.start_utc == symbol_session.last_bar_start_utc:
                reference = float(bar.close)
                fill = reference * (1.0 - self._slippage_rate)
                self._close_v05(
                    trade,
                    reference=reference,
                    fill=fill,
                    ts=bar.end_utc,
                    reason="EOD_EXIT",
                    stop_kind="EOD_EXIT",
                    state=state,
                )
                return

        # 4) Completed 5m bar: update MFE and ladder stop (effective next bar).
        self._update_ladder(trade, bucket, state)

    def _update_ladder(self, trade: Any, bucket: Any, state: dict[str, Any]) -> None:
        entry_fill = float(trade.entry_fill)
        risk_per_share = state["risk_per_share"]
        bucket_high = max(
            bar.high for bar in bucket.members if bar.start_utc >= trade.entry_time_utc
        )
        mfe_r = (bucket_high - entry_fill) / risk_per_share
        state["max_mfe_r"] = max(float(state["max_mfe_r"]), float(mfe_r))
        level_r, kind = ladder_stop(self._exit_mode, state["max_mfe_r"])
        new_stop = entry_fill + level_r * risk_per_share
        if new_stop > float(state["stop"]) + _PRICE_EPS:
            state["stop"] = float(new_stop)
            state["stop_kind"] = kind
        if self._exit_mode == EXIT_RUNNER_B:
            features = self.data.features[trade.candidate.symbol]
            ema21 = features.ema_slow[bucket.global_index]
            if ema21 is not None and float(bucket.bar.close) < float(ema21):
                state["ema_exit_pending"] = True

    def _close_v05(
        self,
        trade: Any,
        *,
        reference: float,
        fill: float,
        ts: datetime,
        reason: str,
        stop_kind: str | None,
        state: Mapping[str, Any],
    ) -> None:
        """Close the whole position and attach V0.5 metadata to the trade detail."""

        risk_per_share = float(state.get("risk_per_share", 0.0) or 0.0)
        exit_stop_r = (
            (float(reference) - float(trade.entry_fill)) / risk_per_share
            if risk_per_share > 0.0
            else 0.0
        )
        signal_id = trade.signal_id
        self._close_trade(
            reference_price=reference,
            fill_price=fill,
            ts=ts,
            reason=reason,
            stop_kind=stop_kind,
        )
        detail = self._trade_details.get(signal_id)
        if detail is not None:
            detail["exit_reason_detail"] = (
                f"{reason} ({stop_kind})" if stop_kind else reason
            )
            detail["max_mfe_r"] = round(float(state.get("max_mfe_r", 0.0)), 4)
            detail["exit_reference_r"] = round(float(exit_stop_r), 4)
            detail["ladder_stop_r"] = round(
                (float(state.get("stop", trade.stop)) - float(trade.entry_fill))
                / risk_per_share,
                4,
            ) if risk_per_share > 0.0 else 0.0
            detail["exit_stop_kind"] = stop_kind
        # Surface explicit Web marker types for trail/EOD exits (the frozen
        # parent maps those to the generic EXIT marker; type is presentation only).
        if reason in ("TRAIL_EMA21", "EOD_EXIT") and self.markers:
            marker = self.markers[-1]
            if marker.get("trade_id") == signal_id:
                marker["type"] = reason


# --------------------------------------------------------------------------- #
# Run wrappers
# --------------------------------------------------------------------------- #
def _build_v05_result(
    base: v03.V03Result,
    variant: V05Variant,
    replay: _V05Replay,
) -> V05Result:
    return V05Result(
        variant=variant,
        trades=list(base.trades),
        equity_curve=list(base.equity_curve),
        metrics=base.metrics,
        trade_details=dict(base.trade_details),
        markers=list(replay.markers),
        initial_equity=base.initial_equity,
        final_equity=base.final_equity,
        sessions_run=base.sessions_run,
        session_dates=list(base.session_dates),
        buckets_evaluated=base.buckets_evaluated,
        skipped_buckets=base.skipped_buckets,
        signals=base.signals,
        pending_cancelled=base.pending_cancelled,
        ambiguous_bars=base.ambiguous_bars,
        breaker_days=base.breaker_days,
        risk_block_counts=dict(base.risk_block_counts),
        notes=list(base.notes),
        or30_skipped_evaluations=int(replay._or30_skipped),
        or30_blocked_sessions=int(replay._or30_blocked_sessions),
        trend_box_rejected=int(replay._f2_rejected),
    )


def run_v05_backtest(
    data: v02.V02Data | Sequence[Bar],
    config: AppConfig | None = None,
    *,
    variant: V05Variant = BASELINE_VARIANT,
    risk_dollars: float | None = None,
    risk: Any | None = None,
    initial_equity: float | None = B1_EQUITY,
    session_dates: Sequence[date] | None = None,
) -> V05Result:
    """Run one V0.5 variant (B1 risk base, entries frozen to Step3c-A NoCap)."""

    base_config = load_config() if config is None else config
    effective_config = build_b1_config(
        base_config, variant.risk_dollars if risk_dollars is None else risk_dollars
    )
    prepared = data if isinstance(data, v02.V02Data) else prepare_v02_data(data)
    or30_blocks: frozenset[tuple[date, str]] = frozenset()
    if variant.or30_filter != OR30_FILTER_NONE:
        or30_blocks, _ = compute_or30_blocks(prepared, variant.or30_filter)
    replay = _V05Replay(
        prepared,
        effective_config,
        variant,
        risk=risk if risk is not None else RiskManager(effective_config),
        initial_equity=B1_EQUITY if initial_equity is None else initial_equity,
        or30_blocks=or30_blocks,
    )
    base = replay.run(session_dates)
    return _build_v05_result(base, variant, replay)


def _collect_5m_bars(data: v02.V02Data, dates: Sequence[date]) -> list[Bar]:
    wanted = set(dates)
    collected: list[Bar] = []
    for session in data.sessions:
        if session.et_date not in wanted:
            continue
        for symbol_session in session.symbols.values():
            collected.extend(bucket.bar for bucket in symbol_session.buckets)
    collected.sort(key=lambda bar: bar.start_utc)
    return collected


def run_v05_walkforward(
    data: v02.V02Data | Sequence[Bar],
    config: AppConfig | None = None,
    *,
    variant: V05Variant = BASELINE_VARIANT,
    risk_dollars: float | None = None,
    dates: Sequence[date] | None = None,
    train_days: int = 120,
    test_days: int = 20,
    step_days: int = 20,
    initial_equity: float | None = B1_EQUITY,
    progress: Callable[[int, int], None] | None = None,
) -> V05WalkForwardResult:
    """Strict chronological walk-forward (same 120/20/20 grid as V0.2-V0.4)."""

    base_config = load_config() if config is None else config
    effective_config = build_b1_config(
        base_config, variant.risk_dollars if risk_dollars is None else risk_dollars
    )
    prepared = data if isinstance(data, v02.V02Data) else prepare_v02_data(data)
    equity = float(B1_EQUITY if initial_equity is None else initial_equity)
    all_dates = list(prepared.dates if dates is None else dates)
    splits = walk_forward_splits(all_dates, train_days, test_days, step_days)
    notes: list[str] = []
    if not splits:
        empty = compute_metrics([], [], initial_equity=equity)
        notes.append(
            "WALK-FORWARD INCOMPLETE: not enough session dates for one full fold "
            f"(have {len(all_dates)}, need {train_days}+{test_days})"
        )
        return V05WalkForwardResult(
            variant=variant,
            folds=(),
            summary={"folds": 0, "total_test_trades": 0},
            aggregate_test_metrics=empty,
            notes=tuple(notes),
        )
    or30_blocks: frozenset[tuple[date, str]] = frozenset()
    if variant.or30_filter != OR30_FILTER_NONE:
        or30_blocks, _ = compute_or30_blocks(prepared, variant.or30_filter)

    fold_results: list[V05FoldResult] = []
    all_test_trades: list[TradeRecord] = []
    breakers = 0
    for position, fold in enumerate(splits, start=1):
        train_bars = _collect_5m_bars(prepared, fold.train_dates)
        test_bars = _collect_5m_bars(prepared, fold.test_dates)
        assert_no_leakage(train_bars, test_bars)
        train_result = run_v05_backtest(
            prepared,
            base_config,
            variant=variant,
            risk_dollars=risk_dollars,
            risk=RiskManager(effective_config),
            initial_equity=equity,
            session_dates=list(fold.train_dates),
        )
        test_result = run_v05_backtest(
            prepared,
            base_config,
            variant=variant,
            risk_dollars=risk_dollars,
            risk=RiskManager(effective_config),
            initial_equity=equity,
            session_dates=list(fold.test_dates),
        )
        all_test_trades.extend(test_result.trades)
        breakers += test_result.breaker_days
        fold_results.append(
            V05FoldResult(fold=fold, train_result=train_result, test_result=test_result)
        )
        if progress is not None:
            progress(position, len(splits))
    aggregate = compute_metrics(all_test_trades, [], initial_equity=equity)
    fold_pnls = [fold.test_result.metrics.net_pnl_dollars for fold in fold_results]
    fold_drawdowns = [
        fold.test_result.metrics.max_drawdown_dollars for fold in fold_results
    ]
    summary = {
        "folds": len(fold_results),
        "train_days": int(train_days),
        "test_days": int(test_days),
        "step_days": int(step_days),
        "session_dates": len(all_dates),
        "total_test_trades": aggregate.total_trades,
        "test_wins": aggregate.wins,
        "test_win_rate": aggregate.win_rate,
        "test_profit_factor": aggregate.profit_factor,
        "test_net_pnl_dollars": sum(trade.pnl for trade in all_test_trades),
        "test_avg_r": aggregate.avg_r,
        "test_max_drawdown_dollars_max_fold": max(fold_drawdowns)
        if fold_drawdowns
        else 0.0,
        "positive_folds": sum(1 for pnl in fold_pnls if pnl > 0.0),
        "best_fold_pnl": max(fold_pnls) if fold_pnls else 0.0,
        "worst_fold_pnl": min(fold_pnls) if fold_pnls else 0.0,
        "test_breaker_days_total": breakers,
    }
    notes.append("fresh RiskManager and replay state per fold; no cross-fold compounding")
    notes.append(DATA_CAVEAT)
    return V05WalkForwardResult(
        variant=variant,
        folds=tuple(fold_results),
        summary=summary,
        aggregate_test_metrics=aggregate,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Web-facing helpers (same style as v04.evaluate_v04_session)
# --------------------------------------------------------------------------- #
def _markers_by_trade(result: V05Result) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for marker in result.markers:
        grouped.setdefault(str(marker.get("trade_id")), []).append(dict(marker))
    for markers in grouped.values():
        markers.sort(key=lambda item: (item.get("t") or 0, str(item.get("type"))))
    return grouped


def result_to_trade_dicts_v05(
    result: V05Result,
    *,
    include_markers: bool = True,
) -> list[dict[str, Any]]:
    """Serialise a V0.5 result into Web-ready closed-trade dicts (with markers)."""

    grouped = _markers_by_trade(result)
    trade_dicts: list[dict[str, Any]] = []
    for trade in sorted(result.trades, key=lambda item: item.entry_time_utc):
        detail = result.trade_details.get(trade.signal_id, {}) or {}
        entry_et = to_et(trade.entry_time_utc)
        exit_et = to_et(trade.exit_time_utc) if trade.exit_time_utc else None
        trade_dicts.append(
            {
                "kind": "trade",
                "trade_id": trade.signal_id,
                "symbol": trade.symbol,
                "strategy": trade.strategy.value,
                "strategy_label": detail.get(
                    "strategy_label", trade.strategy.value
                ),
                "direction": "LONG",
                "shares": trade.shares,
                "entry_time_utc": trade.entry_time_utc.isoformat(),
                "entry_time_et": entry_et.strftime("%H:%M") if entry_et else None,
                "entry_price_signal": trade.entry_price_signal,
                "entry_price_fill": trade.entry_price_fill,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "exit_time_utc": trade.exit_time_utc.isoformat()
                if trade.exit_time_utc
                else None,
                "exit_time_et": exit_et.strftime("%H:%M") if exit_et else None,
                "exit_price_fill": trade.exit_price_fill,
                "exit_reason": trade.exit_reason or "",
                "exit_reason_detail": detail.get(
                    "exit_reason_detail", trade.exit_reason or ""
                ),
                "exit_stop_kind": detail.get("exit_stop_kind"),
                "pnl": round(trade.pnl, 4),
                "r_multiple": round(trade.r_multiple, 4),
                "holding_minutes": round(trade.holding_minutes, 2),
                "entry_slippage_bps": round(trade.entry_slippage_bps, 3),
                "max_mfe_r": detail.get("max_mfe_r"),
                "ladder_stop_r": detail.get("ladder_stop_r"),
                "markers": list(grouped.get(trade.signal_id, []))
                if include_markers
                else [],
            }
        )
    return trade_dicts


def evaluate_v05_session(
    bars_1m: Sequence[Bar] | v02.V02Data,
    session_date: date | str,
    variant: V05Variant = V05_PRODUCTION_VARIANT,
    config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Replay one session and return Web-ready closed trades with markers.

    Mirrors :func:`intraday_scalper.research.v04.evaluate_v04_session`: pass raw
    1m bars (history + the target session, needed for indicator warm-up) or an
    already prepared ``V02Data``.  The B1 risk base is applied and entries are the
    frozen Step3c-A NoCap set; exits/OR30 are taken from ``variant``.
    """

    prepared = (
        bars_1m if isinstance(bars_1m, v02.V02Data) else prepare_v02_data(bars_1m)
    )
    target = (
        session_date
        if isinstance(session_date, date)
        else date.fromisoformat(str(session_date))
    )
    if target not in prepared.dates:
        raise ValueError(
            f"session date {target.isoformat()} not present in the supplied bars "
            f"({prepared.dates[0].isoformat()}..{prepared.dates[-1].isoformat()})"
        )
    result = run_v05_backtest(
        prepared,
        config,
        variant=variant,
        session_dates=[target],
    )
    return result_to_trade_dicts_v05(result)
