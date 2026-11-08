"""V0.4 structural research: SOXL/SOXS 5m intraday upgrade on **raw** IEX bars.

V0.4 builds on the frozen V0.2/V0.3 research modules (imported read-only; never
modified) and implements four a-priori structural changes plus a Friday
single-session recap hook for the Web replay UI:

===========  ==================================================================
Step 0       V0.3 baseline (v0.3 STEP4 semantics) reproduced on raw bars
Step 1       + 10:00-14:00 ET entry window (09:30-10:00 only builds OR_30M/VWAP)
Step 2       + VWAP_RECLAIM_5M trend filter (EMA9/21 + session-range vs ATR14)
Step 3       + SMC_SWEEP_5M (liquidity sweep + reclaim, mirror-confirmed)
Step 4       + 1.2R scale-out (DEPRECATED: negative contribution in task-12)
Step 3b      Step 3 + EMA_PULLBACK daily-exhaustion guard (>= +8.0% above
             session low) + SMC right-side confirmation (close >= EMA21 OR
             close >= session VWAP)
Step 3c      SMC confirmation upgraded to strict AND (close >= EMA21 AND
             close >= session VWAP); A/B on keeping the 8% exhaustion cap
===========  ==================================================================

Design contract
---------------
* **Frozen dependencies**: ``research/v02.py`` and ``research/v03.py`` are only
  imported.  The replay engine here subclasses ``v03._V03Replay`` so Step 0 is
  bit-for-bit the V0.3 engine path (anchor: raw_full Step4 = 970 trades /
  -$214.76 / WF -$234.92 / 16 positive folds).
* **Raw (unadjusted) data only**: SOXS had multiple reverse splits; Alpaca's
  adjusted feed inflates 2024 prices (~$10.9k vs raw ~$6) and divides volume by
  the split factor (dropping low-volume minutes).  V0.4 therefore runs on
  ``adjustment=raw`` bars, exactly like the V0.3 raw cross-check.
* **No tuning**: every threshold below is a one-shot a-priori prior
  (``# a-priori, not tuned``); the walk-forward grid is the same V0.2/V0.3 grid
  (120/20/20), every fold gets a fresh ``RiskManager``, no compounding.
* **Data caveat**: Alpaca **IEX-only** single venue (small share of US
  consolidated volume).  This is research, not proof of profitability.

V0.4 rule details
-----------------
**Entry window (Step 1+).**  A completed 5m bar whose ``end_utc`` (the next 5m
bar's open = fill instant) is before 10:00 ET or at/after 14:00 ET creates no
signal.  Pending orders whose fill instant would violate the window are
cancelled.  Open positions are managed normally to stop/target/EOD.

**VWAP_RECLAIM_5M trend filter (Step 2+).**  In addition to the V0.2 reclaim
rule: the traded symbol's session range so far must be >= 1.2 x its ATR14(5m)
series value, and: long SOXL requires SOXL EMA9 > EMA21; long SOXS uses the
*mirror* of the semis trend, i.e. SOXL EMA9 < EMA21 (SOXS is the inverse leg).
The SOXS-side parenthetical in the task text ("EMA9 < EMA21") is interpreted as
the mirror-symbol (SOXL) trend, otherwise it would contradict the base reclaim
rule that already requires the traded symbol's EMA9 > EMA21; documented here as
the deterministic interpretation.

**SMC_SWEEP_5M (Step 3+).**  For the traded symbol (both legs are long ETFs):
a completed 5m bar whose lower wick pierces a support (trailing 20-bar 5m low or
the OR_30M low, i.e. min low of the first six 5m bars) and closes back above it;
bar range >= 1.2 x the mean range of the previous 20 completed 5m bars; mirror
regime requires the *other* ETF to close below its own session VWAP.
*Deterministic stop choice* (the task left this open): stop anchor is the nearer
support below the entry, ``max(sweep_bar.low, OR30_low)``; the stop is placed
``0.25 * ATR14`` below that anchor and then clamped to the V0.2 [-0.25%,
-2.0%] entry band (the most conservative of the two constraints).  Target 2.0R.
For long SOXS the symmetric "mirror" setup (lower-wick sweep + reclaim on SOXS
itself, other ETF below VWAP) is used: sweeping SOXS lows corresponds exactly to
sweeping SOXL highs, so this is the mirror of the SOXL rule.  Using an
upper-wick rejection on SOXS itself would be short-side price action and would
contradict a long SOXS entry, so it is not used.

**+1.2R scale-out (Step 4).**  Per 1m bar: check stop/target on the remaining
position first (same-bar TP/SL collision -> conservative stop), then (if not yet
scaled) take 50% at the 1.2R limit, move the remaining stop to
``entry_fill + 0.10R`` (effective next bar, no same-bar look-ahead), and keep
the 2.0R target on the remainder.  If the position has already scaled and the
previous completed 5m bar closed below EMA9 (long), the remainder is closed at
the next 1m bar open (TRAIL_EMA9).  The trade is recorded as **one aggregated
entry**: one :class:`~intraday_scalper.core.models.TradeRecord` whose exit fill
is the quantity-weighted average of the partial legs; per-leg detail is stored in
``trade_details['partial_legs']`` and ``exit_reason`` is the final leg reason
(STOP / TARGET / TRAIL_EMA9 / EOD_EXIT) with ``exit_reason_detail`` carrying the
full leg path (e.g. ``SCALE_TARGET+STOP``).  This keeps the aggregate P&L exactly
equal to the sum of legs while preserving a single-risk-unit record.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
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
    StrategyName,
    TradeRecord,
    et_date,
    load_config,
    to_et,
)
from intraday_scalper.risk.manager import RiskManager
from intraday_scalper.risk.sizing import size_position
from intraday_scalper.research import v02, v03
from intraday_scalper.research.v02 import (
    DATA_CAVEAT,
    EMA_PULLBACK_5M,
    ENUM_TO_LABEL,
    MAX_STOP_PCT,
    MIN_STOP_PCT,
    ORB_15M_5M,
    TARGET_R,
    V02Data,
    VWAP_RECLAIM_5M,
    build_strategy_view,
    prepare_v02_data,
)

__all__ = [
    "ABLATION_STEPS",
    "AB_COMPARE_VARIANTS",
    "BREAKEVEN_OFFSET_R",
    "COMPARE_VARIANTS",
    "DATA_CAVEAT",
    "EMA_EXHAUSTION_GAIN_LIMIT",
    "EMA_PULLBACK_5M",
    "ENTRY_WINDOW_END_ET",
    "ENTRY_WINDOW_START_ET",
    "FRACTIONAL_PRECISION",
    "OR30_BARS",
    "PRODUCTION_VARIANT",
    "SMC_LOOKBACK_BARS",
    "SMC_RANGE_MULTIPLE",
    "SMC_STOP_ATR_BUFFER",
    "SMC_SWEEP_5M",
    "STEP0_VARIANT",
    "STEP1_VARIANT",
    "STEP2_VARIANT",
    "STEP3_VARIANT",
    "STEP3B_VARIANT",
    "STEP3C_NOCAP_VARIANT",
    "STEP3C_WITHCAP_VARIANT",
    "STEP4_VARIANT",
    "SCALE_PORTION",
    "SCALE_TRIGGER_R",
    "V04_FINAL_VARIANT",
    "V04EvalContext",
    "V04FoldResult",
    "V04Result",
    "V04Variant",
    "V04WalkForwardResult",
    "VWAP_TREND_RANGE_ATR_MULTIPLE",
    "VWAP_RECLAIM_5M",
    "build_eval_context",
    "ema_pullback_exhaustion_blocked",
    "evaluate_v04_session",
    "result_to_trade_dicts",
    "run_v04_backtest",
    "run_v04_walkforward",
    "smc_sweep_5m_candidate",
    "smc_trend_confirm_ok",
    "smc_trend_confirm_require_both",
    "strategy_label_for_candidate",
    "vwap_reclaim_trend_ok",
]

# --------------------------------------------------------------------------- #
# V0.4 constants — a-priori, not tuned
# --------------------------------------------------------------------------- #
#: Entry window: first 30 minutes only build OR_30M/VWAP/indicators.
ENTRY_WINDOW_START_ET = time(10, 0)
ENTRY_WINDOW_END_ET = time(14, 0)

#: OR_30M = first six completed 5m bars of the session.
OR30_BARS = 6

#: SMC sweep lookback and range filter (a-priori, not tuned).
SMC_LOOKBACK_BARS = 20
SMC_RANGE_MULTIPLE = 1.2
SMC_STOP_ATR_BUFFER = 0.25

#: VWAP_RECLAIM trend filter: session range so far >= 1.2 x ATR14(5m).
VWAP_TREND_RANGE_ATR_MULTIPLE = 1.2

#: EMA_PULLBACK daily-exhaustion guard (a-priori, not tuned): block new
#: EMA_PULLBACK long signals once the signal 5m close is >= +8.0% above the
#: session low so far (09:30 ET onward, completed 5m bars only -> causal).
EMA_EXHAUSTION_GAIN_LIMIT = 0.08

#: 1.2R scale-out geometry (a-priori, not tuned).
SCALE_TRIGGER_R = 1.2
SCALE_PORTION = 0.5
BREAKEVEN_OFFSET_R = 0.10

#: Firstrade fractional-share granularity, kept consistent with V0.3.
FRACTIONAL_PRECISION = 2

#: V0.4-only strategy label (not a core.StrategyName member).
SMC_SWEEP_5M = "SMC_SWEEP_5M"

_BUCKET_MINUTES = 5
_PRICE_EPS = 1e-9
_V04_SIGNAL_PREFIX = "V04-"

_TWO_STRATEGIES: tuple[str, ...] = (VWAP_RECLAIM_5M, EMA_PULLBACK_5M)
_THREE_STRATEGIES: tuple[str, ...] = (VWAP_RECLAIM_5M, EMA_PULLBACK_5M, SMC_SWEEP_5M)

# SMC has no core.StrategyName member (core is frozen); map it to the
# breakout/reversal family member and keep the true label in the report/CSV.
_ENUM_FOR_SMC = StrategyName.ORB_15M


# --------------------------------------------------------------------------- #
# Variant description
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class V04Variant:
    """One cumulative V0.4 ablation step."""

    step: int
    name: str
    enabled_strategies: tuple[str, ...] = _THREE_STRATEGIES
    allow_fractional_shares: bool = True
    entry_window_start_et: time | None = None
    entry_window_end_et: time | None = None
    vwap_trend_filter: bool = False
    breakeven_shield: bool = True
    scale_out: bool = False
    exhaustion_filter: bool = False
    smc_trend_confirm: bool = False
    smc_require_both: bool = False
    max_stop_pct: float | None = None

    def __post_init__(self) -> None:
        allowed = {VWAP_RECLAIM_5M, EMA_PULLBACK_5M, SMC_SWEEP_5M}
        unknown = [label for label in self.enabled_strategies if label not in allowed]
        if unknown:
            raise ValueError(f"unknown V0.4 strategy label(s): {unknown}")
        if not self.enabled_strategies:
            raise ValueError("a V0.4 variant must enable at least one strategy")
        if self.scale_out and self.breakeven_shield:
            raise ValueError(
                "scale_out and the V0.3 breakeven shield are mutually exclusive "
                "(scale-out supersedes the single-stop shield)"
            )

    def flags(self) -> dict[str, Any]:
        """JSON-safe switch summary for the report."""
        return {
            "step": int(self.step),
            "name": self.name,
            "strategies": list(self.enabled_strategies),
            "allow_fractional_shares": bool(self.allow_fractional_shares),
            "fractional_precision": (
                FRACTIONAL_PRECISION if self.allow_fractional_shares else None
            ),
            "entry_window_start_et": (
                self.entry_window_start_et.strftime("%H:%M")
                if self.entry_window_start_et is not None
                else None
            ),
            "entry_window_end_et": (
                self.entry_window_end_et.strftime("%H:%M")
                if self.entry_window_end_et is not None
                else None
            ),
            "vwap_trend_filter": bool(self.vwap_trend_filter),
            "smc_sweep": SMC_SWEEP_5M in self.enabled_strategies,
            "breakeven_shield": bool(self.breakeven_shield),
            "scale_out_1_2r": bool(self.scale_out),
            "exhaustion_filter": bool(self.exhaustion_filter),
            "smc_trend_confirm": bool(self.smc_trend_confirm),
            "smc_require_both": bool(self.smc_require_both),
            "max_stop_pct": self.max_stop_pct,
        }


#: Step 0 is the V0.3 STEP4 baseline reproduced through this engine on raw bars:
#: fractional shares + VWAP/EMA strategies + 14:00 ET cutoff + the V0.3 1.2R
#: single-stop breakeven shield (no 10:00 lower bound, no trend filter, no SMC).
STEP0_VARIANT = V04Variant(
    step=0,
    name="Step 0 · V0.3 baseline (raw)",
    enabled_strategies=_TWO_STRATEGIES,
    allow_fractional_shares=True,
    entry_window_start_et=None,
    entry_window_end_et=v03.ENTRY_CUTOFF_ET,
    vwap_trend_filter=False,
    breakeven_shield=True,
    scale_out=False,
)
STEP1_VARIANT = V04Variant(
    step=1,
    name="Step 1 · + 10:00-14:00 入场窗",
    enabled_strategies=_TWO_STRATEGIES,
    allow_fractional_shares=True,
    entry_window_start_et=ENTRY_WINDOW_START_ET,
    entry_window_end_et=ENTRY_WINDOW_END_ET,
    vwap_trend_filter=False,
    breakeven_shield=True,
    scale_out=False,
)
STEP2_VARIANT = V04Variant(
    step=2,
    name="Step 2 · + VWAP_RECLAIM 趋势过滤",
    enabled_strategies=_TWO_STRATEGIES,
    allow_fractional_shares=True,
    entry_window_start_et=ENTRY_WINDOW_START_ET,
    entry_window_end_et=ENTRY_WINDOW_END_ET,
    vwap_trend_filter=True,
    breakeven_shield=True,
    scale_out=False,
)
STEP3_VARIANT = V04Variant(
    step=3,
    name="Step 3 · + SMC_SWEEP_5M",
    enabled_strategies=_THREE_STRATEGIES,
    allow_fractional_shares=True,
    entry_window_start_et=ENTRY_WINDOW_START_ET,
    entry_window_end_et=ENTRY_WINDOW_END_ET,
    vwap_trend_filter=True,
    breakeven_shield=True,
    scale_out=False,
)
STEP4_VARIANT = V04Variant(
    step=4,
    name="Step 4 · + 1.2R 分批止盈 = V0.4 最终",
    enabled_strategies=_THREE_STRATEGIES,
    allow_fractional_shares=True,
    entry_window_start_et=ENTRY_WINDOW_START_ET,
    entry_window_end_et=ENTRY_WINDOW_END_ET,
    vwap_trend_filter=True,
    breakeven_shield=False,
    scale_out=True,
)

#: Step 3b (a-priori filters on top of Step 3): EMA_PULLBACK daily-exhaustion
#: guard + SMC right-side trend confirmation.  Step 4 (scale-out) is kept for
#: archive/reproduction only and is deprecated (negative contribution task-12).
STEP3B_VARIANT = replace(
    STEP3_VARIANT,
    step=5,
    name="Step 3b · + 防追高/防接刀",
    exhaustion_filter=True,
    smc_trend_confirm=True,
)

#: Production/default variant placeholder; the Lead makes the final call based
#: on the Step1/Step3/Step3b comparison (one-line switch here).
PRODUCTION_VARIANT = STEP3B_VARIANT

#: The three variants compared by ``tools/run_v04b_compare.py``.
COMPARE_VARIANTS: tuple[V04Variant, ...] = (
    STEP1_VARIANT,
    STEP3_VARIANT,
    STEP3B_VARIANT,
)

#: Step 3c A/B: SMC trend confirmation upgraded from OR to strict AND
#: (close >= EMA21 AND close >= session VWAP); both variants keep the Step3
#: strategy set and the V0.3 shield.
STEP3C_NOCAP_VARIANT = replace(
    STEP3_VARIANT,
    step=6,
    name="Step 3c-NoCap · SMC AND 双确认（无防追高）",
    smc_trend_confirm=True,
    smc_require_both=True,
    exhaustion_filter=False,
)
STEP3C_WITHCAP_VARIANT = replace(
    STEP3_VARIANT,
    step=7,
    name="Step 3c-WithCap · SMC AND 双确认 + 8% 防追高",
    smc_trend_confirm=True,
    smc_require_both=True,
    exhaustion_filter=True,
)

#: A/B comparison ladder for ``tools/run_v04c_ab.py``.
AB_COMPARE_VARIANTS: tuple[V04Variant, ...] = (
    STEP3C_NOCAP_VARIANT,
    STEP3C_WITHCAP_VARIANT,
)

#: DEPRECATED archive variant (1.2R scale-out); kept for reproduction only.
V04_FINAL_VARIANT = STEP4_VARIANT

#: Cumulative ablation ladder (each step keeps every earlier change).
ABLATION_STEPS: tuple[V04Variant, ...] = (
    STEP0_VARIANT,
    STEP1_VARIANT,
    STEP2_VARIANT,
    STEP3_VARIANT,
    STEP4_VARIANT,
)


# --------------------------------------------------------------------------- #
# Labels / small helpers
# --------------------------------------------------------------------------- #
def strategy_label_for_candidate(candidate: SignalCandidate) -> str:
    """Return the true V0.4 strategy label for a candidate."""

    if candidate.rationale.startswith(SMC_SWEEP_5M):
        return SMC_SWEEP_5M
    return ENUM_TO_LABEL.get(candidate.strategy, candidate.strategy.value)


def _truncate_quantity(value: float, allow_fractional: bool, precision: int) -> float:
    """Floor a share quantity; never round up."""
    if value <= 0.0:
        return 0.0
    if not allow_fractional:
        return float(math.floor(value))
    scale = 10 ** int(precision)
    return math.floor(value * scale) / scale


# --------------------------------------------------------------------------- #
# Evaluation context
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class V04EvalContext:
    """Extra context needed by the V0.4-only filters (all backward-looking)."""

    session_range_so_far: float | None
    atr: float | None
    or30_high: float | None
    or30_low: float | None
    prev20_high: float | None
    prev20_low: float | None
    avg_range20: float | None
    mirror_ema_fast: float | None
    mirror_ema_slow: float | None
    mirror_ok: bool
    session_low_so_far: float | None = None
    session_high_so_far: float | None = None


def _flat_buckets_by_symbol(data: V02Data) -> dict[str, list[Any]]:
    """Per-symbol list of 5m buckets ordered by their global index."""

    flat: dict[str, list[Any]] = {}
    for session in data.sessions:
        for symbol, symbol_session in session.symbols.items():
            flat.setdefault(symbol, []).extend(symbol_session.buckets)
    return flat


def build_eval_context(
    data: V02Data,
    session: Any,
    symbol: str,
    bucket_start_utc: datetime,
    flat_buckets: Mapping[str, Sequence[Any]],
) -> V04EvalContext:
    """Assemble the V0.4 filter context for one completed 5m bucket."""

    symbol_session = session.symbols.get(symbol)
    if symbol_session is None:
        return V04EvalContext(None, None, None, None, None, None, None, None, None, False)
    bucket = symbol_session.bucket_by_start.get(bucket_start_utc)
    if bucket is None:
        return V04EvalContext(None, None, None, None, None, None, None, None, None, False)

    index = bucket.global_index
    session_index = bucket.index_in_session
    session_buckets = symbol_session.buckets[: session_index + 1]
    highs = [item.bar.high for item in session_buckets]
    lows = [item.bar.low for item in session_buckets]
    session_range = (max(highs) - min(lows)) if highs else None
    session_low = min(lows) if lows else None
    session_high = max(highs) if highs else None

    features = data.features[symbol]
    atr = features.atr[index] if 0 <= index < len(features.atr) else None

    # OR_30M: first six completed 5m bars (09:30..10:00 ET).
    or30_high: float | None = None
    or30_low: float | None = None
    expected = [
        session.open_utc + timedelta(minutes=_BUCKET_MINUTES * offset)
        for offset in range(OR30_BARS)
    ]
    if all(start in symbol_session.bucket_by_start for start in expected):
        or_buckets = [symbol_session.bucket_by_start[start] for start in expected]
        or30_high = max(item.bar.high for item in or_buckets)
        or30_low = min(item.bar.low for item in or_buckets)

    # Trailing 20 completed 5m bars on the continuous (cross-session) series.
    flat = flat_buckets.get(symbol, [])
    prior = flat[max(0, index - SMC_LOOKBACK_BARS) : index]
    prev20_high: float | None = None
    prev20_low: float | None = None
    avg_range20: float | None = None
    if len(prior) >= SMC_LOOKBACK_BARS:
        prev20_high = max(item.bar.high for item in prior)
        prev20_low = min(item.bar.low for item in prior)
        avg_range20 = sum(item.bar.range for item in prior) / float(len(prior))

    # Mirror leg (other ETF): latest completed 5m bar at/before this bucket start.
    mirror_ema_fast: float | None = None
    mirror_ema_slow: float | None = None
    other = "SOXS" if symbol == "SOXL" else "SOXL"
    other_session = session.symbols.get(other)
    if other_session is not None and other_session.buckets:
        other_index = bisect_right(other_session.bucket_starts, bucket_start_utc) - 1
        if other_index >= 0:
            other_bucket = other_session.buckets[other_index]
            mirror_features = data.features[other]
            mirror_global = other_bucket.global_index
            mirror_ema_fast = mirror_features.ema_fast[mirror_global]
            mirror_ema_slow = mirror_features.ema_slow[mirror_global]

    return V04EvalContext(
        session_range_so_far=session_range,
        atr=atr,
        or30_high=or30_high,
        or30_low=or30_low,
        prev20_high=prev20_high,
        prev20_low=prev20_low,
        avg_range20=avg_range20,
        mirror_ema_fast=mirror_ema_fast,
        mirror_ema_slow=mirror_ema_slow,
        mirror_ok=v02.mirror_relation_ok(data, session, symbol, bucket_start_utc),
        session_low_so_far=session_low,
        session_high_so_far=session_high,
    )


# --------------------------------------------------------------------------- #
# V0.4-only filters / SMC strategy
# --------------------------------------------------------------------------- #
def vwap_reclaim_trend_ok(symbol: str, view: Any, ctx: V04EvalContext) -> bool:
    """VWAP_RECLAIM trend filter (a-priori, not tuned).

    * session range so far must be >= 1.2 x ATR14(5m) for the traded symbol;
    * long SOXL: SOXL EMA9 > EMA21;
    * long SOXS: mirror-symbol (SOXL) EMA9 < EMA21.
    """

    if ctx.session_range_so_far is None or ctx.atr is None or ctx.atr <= 0.0:
        return False
    if ctx.session_range_so_far < VWAP_TREND_RANGE_ATR_MULTIPLE * ctx.atr:
        return False
    if symbol == "SOXL":
        return (
            view.ema_fast is not None
            and view.ema_slow is not None
            and view.ema_fast > view.ema_slow
        )
    return (
        ctx.mirror_ema_fast is not None
        and ctx.mirror_ema_slow is not None
        and ctx.mirror_ema_fast < ctx.mirror_ema_slow
    )


def ema_pullback_exhaustion_blocked(
    bar_close: float,
    session_low_so_far: float | None,
    limit: float = EMA_EXHAUSTION_GAIN_LIMIT,
) -> bool:
    """Step-3b anti-chase guard for EMA_PULLBACK_5M (a-priori, not tuned).

    Blocks a **new** EMA_PULLBACK long signal when the signal 5m close has moved
    at least ``limit`` (8.0%) above the session low so far (09:30 ET onward,
    completed 5m bars only -> causal).  A missing session low cannot be
    evaluated and therefore does not block.
    """

    if session_low_so_far is None or session_low_so_far <= 0.0:
        return False
    gain = (float(bar_close) - float(session_low_so_far)) / float(session_low_so_far)
    return gain >= limit - 1e-12


def smc_trend_confirm_ok(
    bar_close: float,
    ema21: float | None,
    session_vwap: float | None,
) -> bool:
    """Step-3b right-side confirmation for SMC_SWEEP_5M (a-priori, not tuned).

    The sweep may only be bought when at least one reference is reclaimed:
    ``close >= EMA21(5m)`` or ``close >= session VWAP``.  This rejects
    underwater knife-catching (close below both).  When both references are
    unavailable the sweep cannot be confirmed and is rejected (conservative).
    """

    close = float(bar_close)
    if ema21 is not None and close >= float(ema21) - 1e-12:
        return True
    if session_vwap is not None and close >= float(session_vwap) - 1e-12:
        return True
    return False


def smc_trend_confirm_require_both(
    bar_close: float,
    ema21: float | None,
    session_vwap: float | None,
) -> bool:
    """Step-3c strict AND confirmation for SMC_SWEEP_5M (a-priori, not tuned).

    The sweep may only be bought when **both** references are reclaimed:
    ``close >= EMA21(5m)`` **and** ``close >= session VWAP``.  This is stricter
    than :func:`smc_trend_confirm_ok` (OR).  Missing references make the
    confirmation fail (conservative).
    """

    if ema21 is None or session_vwap is None:
        return False
    close = float(bar_close)
    return (
        close >= float(ema21) - 1e-12
        and close >= float(session_vwap) - 1e-12
    )


def smc_sweep_5m_candidate(
    view: Any,
    ctx: V04EvalContext,
) -> SignalCandidate | None:
    """SMC_SWEEP_5M: liquidity sweep below support + close back above it.

    Deterministic implementation (the task left the stop definition open; the
    most conservative of the listed options is used):

    * supports: trailing 20-bar 5m low and the OR_30M low; the bar must pierce
      at least one of them (``low < level``) and close back above it;
    * range filter: ``bar.range >= 1.2 * mean(previous 20 5m ranges)``;
    * mirror regime: the other ETF closes below its own session VWAP;
    * stop: ``max(sweep_bar.low, OR30_low) - 0.25 * ATR14`` (the nearer support,
      with a volatility buffer), then clamped into the V0.2 [0.25%, 2.0%] band;
    * target ``2.0R``; entry at the next 5m bar open (engine-level).
    """

    data_bar = view.bar
    if not ctx.mirror_ok:
        return None
    if ctx.avg_range20 is None or ctx.avg_range20 <= 0.0:
        return None
    if data_bar.range < SMC_RANGE_MULTIPLE * ctx.avg_range20:
        return None
    if ctx.atr is None or not math.isfinite(ctx.atr) or ctx.atr <= 0.0:
        return None

    supports: list[float] = []
    if ctx.prev20_low is not None:
        supports.append(ctx.prev20_low)
    if ctx.or30_low is not None:
        supports.append(ctx.or30_low)
    pierced = [
        level
        for level in supports
        if data_bar.low < level - _PRICE_EPS and data_bar.close > level + _PRICE_EPS
    ]
    if not pierced:
        return None
    nearest_support = max(pierced)

    entry = data_bar.close
    anchor = (
        max(data_bar.low, ctx.or30_low)
        if ctx.or30_low is not None
        else data_bar.low
    )
    raw_stop = anchor - SMC_STOP_ATR_BUFFER * ctx.atr
    widest_allowed = entry * (1.0 - MAX_STOP_PCT)
    tightest_allowed = entry * (1.0 - MIN_STOP_PCT)
    stop = min(max(raw_stop, widest_allowed), tightest_allowed)
    if not stop < entry:
        return None
    stop_pct = (entry - stop) / entry
    rationale = (
        f"{SMC_SWEEP_5M}: 下影线 {data_bar.low:.4f} 刺穿支撑 {nearest_support:.4f}，"
        f"收回 {entry:.4f}；range={data_bar.range:.4f} >= "
        f"{SMC_RANGE_MULTIPLE:.1f}×avg20={ctx.avg_range20:.4f}；mirror=OK"
    )
    note = (
        f"action=SMC_SWEEP; anchor=max(sweep_low={data_bar.low:.4f}, "
        f"or30_low={ctx.or30_low if ctx.or30_low is not None else float('nan'):.4f}); "
        f"atr5m={ctx.atr:.4f}; buffer={SMC_STOP_ATR_BUFFER:.2f}*ATR; "
        f"stop_pct={stop_pct:.5%} (clamp {MIN_STOP_PCT:.2%}-{MAX_STOP_PCT:.2%})"
    )
    return SignalCandidate(
        symbol=view.symbol,
        strategy=_ENUM_FOR_SMC,
        entry_ref=entry,
        stop_loss=stop,
        target_r=TARGET_R,
        rationale=rationale,
        created_utc=view.now_utc,
        confirmation_bar_start_utc=data_bar.start_utc,
        structural_note=note,
    )


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class V04Result:
    """One V0.4 run (full period, a segment, or a single walk-forward fold)."""

    variant: V04Variant
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
    window_skipped_evaluations: int = 0
    scale_target_trades: int = 0
    trail_exit_trades: int = 0
    breakeven_armed_trades: int = 0
    elapsed_seconds: float = 0.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def trades_per_session(self) -> float:
        return (len(self.trades) / self.sessions_run) if self.sessions_run else 0.0

    def extra_metrics(self) -> dict[str, Any]:
        return {
            "step": int(self.variant.step),
            "variant": self.variant.name,
            "ambiguous_bars": int(self.ambiguous_bars),
            "breaker_days": int(self.breaker_days),
            "pending_cancelled": int(self.pending_cancelled),
            "window_skipped_evaluations": int(self.window_skipped_evaluations),
            "signals": int(self.signals),
            "buckets_evaluated": int(self.buckets_evaluated),
            "skipped_buckets": int(self.skipped_buckets),
            "sessions_run": int(self.sessions_run),
            "trades_per_session": round(self.trades_per_session, 4),
            "scale_target_trades": int(self.scale_target_trades),
            "trail_exit_trades": int(self.trail_exit_trades),
            "breakeven_armed_trades": int(self.breakeven_armed_trades),
            "risk_block_counts": dict(self.risk_block_counts),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class V04FoldResult:
    """One walk-forward fold: fresh engine/risk state for train *and* test."""

    fold: Fold
    train_result: V04Result
    test_result: V04Result
    leakage_checked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": {
                "index": self.fold.index,
                "train_start": self.fold.train_start.isoformat()
                if self.fold.train_start
                else None,
                "train_end": self.fold.train_end.isoformat()
                if self.fold.train_end
                else None,
                "test_start": self.fold.test_start.isoformat()
                if self.fold.test_start
                else None,
                "test_end": self.fold.test_end.isoformat()
                if self.fold.test_end
                else None,
                "train_days": len(self.fold.train_dates),
                "test_days": len(self.fold.test_dates),
            },
            "train_trades": self.train_result.total_trades,
            "test_trades": self.test_result.total_trades,
            "train_metrics": self.train_result.metrics.to_dict(),
            "test_metrics": self.test_result.metrics.to_dict(),
            "test_extra": self.test_result.extra_metrics(),
            "leakage_checked": self.leakage_checked,
        }


@dataclass(frozen=True, slots=True)
class V04WalkForwardResult:
    """All walk-forward folds plus the aggregated out-of-sample summary."""

    variant: V04Variant
    folds: tuple[V04FoldResult, ...]
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
# Replay engine (subclasses v03._V03Replay; Step 0 is the V0.3 path)
# --------------------------------------------------------------------------- #
@dataclass
class _V04OpenTrade(v03._OpenTrade):
    """V0.3 open-trade record extended with scale-out state."""

    remaining_shares: float = 0.0
    partial_done: bool = False
    partial_trigger: float = 0.0
    current_stop: float = 0.0
    legs: list[dict[str, Any]] | None = None


def _parent_variant(variant: V04Variant) -> v03.V03Variant:
    """Map a V0.4 variant onto the v03 variant the parent engine expects."""

    parent_strategies = tuple(
        label for label in variant.enabled_strategies if label in (VWAP_RECLAIM_5M, EMA_PULLBACK_5M)
    ) or (VWAP_RECLAIM_5M,)
    return v03.V03Variant(
        step=variant.step,
        name=variant.name,
        enabled_strategies=parent_strategies,
        allow_fractional_shares=variant.allow_fractional_shares,
        entry_cutoff_et=variant.entry_window_end_et,
        breakeven_shield=bool(variant.breakeven_shield),
    )


class _V04Replay(v03._V03Replay):
    """V0.3 replay engine + V0.4 entry window / trend filter / SMC / scale-out."""

    def __init__(
        self,
        data: V02Data,
        config: AppConfig,
        variant: V04Variant,
        *,
        risk: Any | None = None,
        initial_equity: float | None = None,
    ) -> None:
        if not isinstance(variant, V04Variant):
            raise TypeError("variant must be a V04Variant")
        super().__init__(
            data,
            config,
            _parent_variant(variant),
            risk=risk,
            initial_equity=initial_equity,
        )
        self.v04_variant = variant
        self._window_start = variant.entry_window_start_et
        self._window_end = variant.entry_window_end_et
        self._vwap_trend_filter = bool(variant.vwap_trend_filter)
        self._scale_out = bool(variant.scale_out)
        self._exhaustion_filter = bool(variant.exhaustion_filter)
        self._smc_trend_confirm = bool(variant.smc_trend_confirm)
        self._smc_require_both = bool(variant.smc_require_both)
        self._max_stop_pct = variant.max_stop_pct
        self._flat_buckets = _flat_buckets_by_symbol(data)
        # Strategy labels in the fixed ablation order (strategy-major evaluation).
        self._rule_items: tuple[
            tuple[str, Callable[[Any, V04EvalContext], SignalCandidate | None]], ...
        ] = tuple((label, self._make_rule(label)) for label in variant.enabled_strategies)
        # v04-only state
        self._signal_labels: dict[str, str] = {}
        self.markers: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Rule plumbing
    # ------------------------------------------------------------------ #
    def _make_rule(
        self, label: str
    ) -> Callable[[Any, V04EvalContext], SignalCandidate | None]:
        if label == VWAP_RECLAIM_5M:
            return self._vwap_reclaim_rule
        if label == EMA_PULLBACK_5M:
            return self._ema_pullback_rule
        if label == SMC_SWEEP_5M:
            return self._smc_rule
        raise ValueError(f"unsupported V0.4 strategy label: {label}")  # pragma: no cover

    def _ema_pullback_rule(
        self, view: Any, ctx: V04EvalContext
    ) -> SignalCandidate | None:
        candidate = v02.ema_pullback_5m_candidate(view)
        if candidate is None:
            return None
        if self._exhaustion_filter and ema_pullback_exhaustion_blocked(
            view.bar.close, ctx.session_low_so_far
        ):
            return None
        return candidate

    def _vwap_reclaim_rule(
        self, view: Any, ctx: V04EvalContext
    ) -> SignalCandidate | None:
        candidate = v02.vwap_reclaim_5m_candidate(view)
        if candidate is None:
            return None
        if self._vwap_trend_filter and not vwap_reclaim_trend_ok(
            view.symbol, view, ctx
        ):
            return None
        return candidate

    def _smc_rule(self, view: Any, ctx: V04EvalContext) -> SignalCandidate | None:
        candidate = smc_sweep_5m_candidate(view, ctx)
        if candidate is None:
            return None
        if self._smc_trend_confirm:
            if self._smc_require_both:
                confirmed = smc_trend_confirm_require_both(
                    view.bar.close, view.ema_slow, view.session_vwap
                )
            else:
                confirmed = smc_trend_confirm_ok(
                    view.bar.close, view.ema_slow, view.session_vwap
                )
            if not confirmed:
                return None
        return candidate

    # ------------------------------------------------------------------ #
    # Entry window
    # ------------------------------------------------------------------ #
    def _evaluation_allowed(self, bucket_start_utc: datetime) -> bool:
        # Parent enforces the upper cutoff (= window end for every V0.4 step).
        if not super()._evaluation_allowed(bucket_start_utc):
            return False
        if self._window_start is not None:
            bucket_end = bucket_start_utc + timedelta(minutes=_BUCKET_MINUTES)
            if to_et(bucket_end).time() < self._window_start:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Strategy evaluation (V0.4 context + SMC)
    # ------------------------------------------------------------------ #
    def _evaluate_bucket(self, session: Any, bucket_start: datetime) -> None:
        if not self._evaluation_allowed(bucket_start):
            self._cutoff_skipped += 1
            return
        views: list[tuple[Any, V04EvalContext]] = []
        for symbol in self.symbols:
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
        for label, rule in self._rule_items:
            for view, context in views:
                candidate = rule(view, context)
                if candidate is not None:
                    candidate_label = label
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
        signal_id = f"{_V04_SIGNAL_PREFIX}{self._signal_seq:06d}"
        self._signal_labels[signal_id] = candidate_label
        self._pending = v03._PendingEntry(
            candidate=candidate,
            plan=allowed_plan,
            signal_id=signal_id,
            expected_bucket_start_utc=bucket_start + timedelta(minutes=_BUCKET_MINUTES),
        )

    # ------------------------------------------------------------------ #
    # Fills / markers
    # ------------------------------------------------------------------ #
    def _strategy_label(self, signal_id: str, candidate: SignalCandidate) -> str:
        return self._signal_labels.get(signal_id) or strategy_label_for_candidate(
            candidate
        )

    def _record_marker(
        self,
        *,
        marker_type: str,
        ts: datetime,
        price: float,
        trade: Any,
        detail: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        marker: dict[str, Any] = {
            "type": marker_type,
            "time_utc": ts.isoformat(),
            "price": round(float(price), 6),
            "symbol": trade.candidate.symbol,
            "strategy": self._strategy_label(trade.signal_id, trade.candidate),
            "trade_id": trade.signal_id,
            "t": int(ts.timestamp() * 1000),
            "detail": detail,
        }
        if extra:
            marker.update(dict(extra))
        self.markers.append(marker)

    def _try_fill_pending(self, session: Any, bucket_start: datetime) -> None:
        had_open = self._open is not None
        super()._try_fill_pending(session, bucket_start)
        if self._open is None or had_open:
            return
        parent_trade = self._open
        if self._scale_out:
            risk_per_share = parent_trade.entry_fill - parent_trade.stop
            self._open = _V04OpenTrade(
                signal_id=parent_trade.signal_id,
                candidate=parent_trade.candidate,
                plan=parent_trade.plan,
                shares=parent_trade.shares,
                entry_fill=parent_trade.entry_fill,
                entry_time_utc=parent_trade.entry_time_utc,
                stop=parent_trade.stop,
                target=parent_trade.target,
                remaining_shares=parent_trade.shares,
                partial_done=False,
                partial_trigger=parent_trade.entry_fill
                + SCALE_TRIGGER_R * risk_per_share,
                current_stop=parent_trade.stop,
                legs=[],
            )
        trade = self._open
        self._record_marker(
            marker_type="ENTRY",
            ts=trade.entry_time_utc,
            price=trade.entry_fill,
            trade=trade,
            extra={"stop": round(trade.stop, 6), "target": round(trade.target, 6)},
        )

    # ------------------------------------------------------------------ #
    # Exits: non-scale path delegates to V0.3 (shield), scale path is V0.4
    # ------------------------------------------------------------------ #
    def _close_trade(
        self,
        *,
        reference_price: float,
        fill_price: float,
        ts: datetime,
        reason: str,
        stop_kind: str | None,
    ) -> None:
        trade = self._open
        super()._close_trade(
            reference_price=reference_price,
            fill_price=fill_price,
            ts=ts,
            reason=reason,
            stop_kind=stop_kind,
        )
        if trade is None:  # pragma: no cover - parent guarantees an open trade
            return
        label = self._strategy_label(trade.signal_id, trade.candidate)
        detail = self._trade_details.get(trade.signal_id)
        if detail is not None:
            detail["strategy_label"] = label
            detail.setdefault("partial_legs", [])
            detail.setdefault("scale_target_hit", False)
            detail.setdefault("exit_reason_detail", reason)
        marker_type = "STOP" if reason == "STOP" else ("TARGET" if reason == "TARGET" else "EXIT")
        self._record_marker(
            marker_type=marker_type,
            ts=ts,
            price=fill_price,
            trade=trade,
            detail=reason,
        )

    def _manage_open_trade(self, session: Any, bucket_start: datetime) -> None:
        if not self._scale_out:
            super()._manage_open_trade(session, bucket_start)
            return
        self._manage_open_trade_scale(session, bucket_start)

    def _manage_open_trade_scale(self, session: Any, bucket_start: datetime) -> None:
        trade = self._open
        if trade is None:  # pragma: no cover - caller guarantees
            return
        if not isinstance(trade, _V04OpenTrade):  # pragma: no cover - defensive
            return
        symbol = trade.candidate.symbol
        symbol_session = session.symbols.get(symbol)
        if symbol_session is None:  # pragma: no cover - position implies bars
            return
        bucket = symbol_session.bucket_by_start.get(bucket_start)
        if bucket is None:
            return

        # EMA9 trail: only for the remaining leg, and only after the 1.2R scale.
        trail_due = False
        if trade.partial_done and trade.remaining_shares > 0:
            previous_bucket = symbol_session.bucket_by_start.get(
                bucket_start - timedelta(minutes=_BUCKET_MINUTES)
            )
            if previous_bucket is not None:
                features = self.data.features[symbol]
                ema9 = features.ema_fast[previous_bucket.global_index]
                if ema9 is not None and previous_bucket.bar.close < ema9:
                    trail_due = True

        for bar in bucket.members:
            if bar.start_utc < trade.entry_time_utc:
                continue
            if trade.remaining_shares <= 0.0:  # pragma: no cover
                return
            if trail_due:
                reference = bar.open
                fill = reference * (1.0 - self._slippage_rate)
                self._close_scale_trade(
                    remaining_qty=trade.remaining_shares,
                    remaining_fill=fill,
                    remaining_reference=reference,
                    ts=bar.start_utc,
                    final_reason="TRAIL_EMA9",
                    stop_kind="TRAIL_EMA9",
                )
                return

            # 1) Stop / target on the remaining position (conservative stop on collision).
            current_stop = trade.current_stop
            stop_hit = bar.low <= current_stop + _PRICE_EPS
            target_hit = bar.high >= trade.target - _PRICE_EPS
            if stop_hit or target_hit:
                if stop_hit and target_hit:
                    # Same-bar TP/SL collision -> conservative stop fill.
                    self._ambiguous_bars += 1
                    trade.ambiguous = True
                prefer_stop = stop_hit
                if prefer_stop:
                    reference = bar.open if bar.open <= current_stop else current_stop
                    fill = reference * (1.0 - self._slippage_rate)
                    self._close_scale_trade(
                        remaining_qty=trade.remaining_shares,
                        remaining_fill=fill,
                        remaining_reference=reference,
                        ts=bar.end_utc,
                        final_reason="STOP",
                        stop_kind="SCALE_STOP" if trade.partial_done else "INITIAL",
                    )
                else:
                    reference = trade.target
                    fill = reference * (1.0 - self._slippage_rate)
                    self._close_scale_trade(
                        remaining_qty=trade.remaining_shares,
                        remaining_fill=fill,
                        remaining_reference=reference,
                        ts=bar.end_utc,
                        final_reason="TARGET",
                        stop_kind="TARGET",
                    )
                return

            # 2) Excursions (dollar terms, aggregate position).
            trade.mae_dollars = max(
                trade.mae_dollars, max(0.0, (trade.entry_fill - bar.low) * trade.shares)
            )
            trade.mfe_dollars = max(
                trade.mfe_dollars, max(0.0, (bar.high - trade.entry_fill) * trade.shares)
            )

            # 3) First 1.2R scale-out (effective from the next bar for the moved stop).
            risk_per_share_positive = (trade.entry_fill - trade.stop) > 0.0
            if (
                not trade.partial_done
                and risk_per_share_positive
                and bar.high >= trade.partial_trigger - _PRICE_EPS
            ):
                partial_qty = _truncate_quantity(
                    trade.shares * SCALE_PORTION,
                    bool(self.config.risk.allow_fractional_shares),
                    int(self.config.risk.fractional_precision),
                )
                if partial_qty > 0.0 and partial_qty < trade.shares:
                    partial_fill = trade.partial_trigger * (1.0 - self._slippage_rate)
                    trade.legs = trade.legs or []
                    trade.legs.append(
                        {
                            "qty": float(partial_qty),
                            "fill": float(partial_fill),
                            "reference": float(trade.partial_trigger),
                            "time_utc": bar.end_utc.isoformat(),
                            "reason": "SCALE_TARGET",
                        }
                    )
                    trade.remaining_shares = round(
                        trade.shares - float(partial_qty), 10
                    )
                    trade.partial_done = True
                    trade.breakeven_armed = True
                    risk_per_share = trade.entry_fill - trade.stop
                    trade.current_stop = max(
                        trade.stop,
                        trade.entry_fill + BREAKEVEN_OFFSET_R * risk_per_share,
                    )
                    self._record_marker(
                        marker_type="TARGET",
                        ts=bar.end_utc,
                        price=partial_fill,
                        trade=trade,
                        detail="SCALE_1.2R",
                        extra={
                            "qty": float(partial_qty),
                            "limit_price": round(float(trade.partial_trigger), 6),
                        },
                    )
                    self._record_marker(
                        marker_type="BREAKEVEN",
                        ts=bar.end_utc,
                        price=trade.current_stop,
                        trade=trade,
                        detail="STOP_TO_ENTRY_PLUS_0.10R",
                    )

            # 4) Intraday system: flatten the remainder on the session's last 1m bar.
            if bar.start_utc == symbol_session.last_bar_start_utc:
                reference = bar.close
                fill = reference * (1.0 - self._slippage_rate)
                self._close_scale_trade(
                    remaining_qty=trade.remaining_shares,
                    remaining_fill=fill,
                    remaining_reference=reference,
                    ts=bar.end_utc,
                    final_reason="EOD_EXIT",
                    stop_kind="EOD_EXIT",
                )
                return

    def _close_scale_trade(
        self,
        *,
        remaining_qty: float,
        remaining_fill: float,
        remaining_reference: float,
        ts: datetime,
        final_reason: str,
        stop_kind: str | None,
    ) -> None:
        """Aggregate all partial legs into one TradeRecord (quantity-weighted fill)."""

        trade = self._open
        if trade is None:  # pragma: no cover - caller guarantees
            return
        legs = list(trade.legs or [])
        if remaining_qty > 0.0:
            legs.append(
                {
                    "qty": float(remaining_qty),
                    "fill": float(remaining_fill),
                    "reference": float(remaining_reference),
                    "time_utc": ts.isoformat(),
                    "reason": final_reason,
                }
            )
        total_qty = sum(leg["qty"] for leg in legs)
        if total_qty <= 0.0:  # pragma: no cover - defensive
            return
        weighted_fill = sum(leg["fill"] * leg["qty"] for leg in legs) / total_qty
        entry_slippage = (trade.entry_fill - trade.candidate.entry_ref) * total_qty
        exit_slippage = sum(
            (leg["reference"] - leg["fill"]) * leg["qty"] for leg in legs
        )
        reason_detail = "+".join(leg["reason"] for leg in legs)
        self._scale_leg_slippage = getattr(self, "_scale_leg_slippage", {})
        self._scale_leg_slippage[trade.signal_id] = (entry_slippage, exit_slippage)
        # Parent close path builds the single aggregated TradeRecord through the
        # RiskManager (daily PnL / cooldown state updated exactly once).
        self._close_trade(
            reference_price=weighted_fill,
            fill_price=weighted_fill,
            ts=ts,
            reason=final_reason,
            stop_kind=stop_kind,
        )
        record = self._trades[-1]
        detail = self._trade_details.get(record.signal_id, {})
        detail.update(
            {
                "strategy_label": self._strategy_label(
                    record.signal_id, trade.candidate
                ),
                "partial_legs": legs,
                "scale_target_hit": any(
                    leg["reason"] == "SCALE_TARGET" for leg in legs
                ),
                "exit_reason_detail": reason_detail,
                "exit_stop_kind": stop_kind,
                "entry_slippage_dollars": entry_slippage,
                "exit_slippage_dollars": exit_slippage,
                "legs_pnl": round(
                    sum((leg["fill"] - trade.entry_fill) * leg["qty"] for leg in legs), 6
                ),
                "weighted_exit_fill": round(weighted_fill, 8),
            }
        )
        # Markers for the final leg are emitted by _close_trade above.

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def _count_breaker_days(self) -> int:
        return super()._count_breaker_days()


def _build_v04_result(
    base: v03.V03Result,
    variant: V04Variant,
    markers: list[dict[str, Any]],
) -> V04Result:
    details = dict(base.trade_details)
    scale_target_trades = sum(
        1 for detail in details.values() if detail.get("scale_target_hit")
    )
    trail_exit_trades = sum(
        1 for trade in base.trades if trade.exit_reason == "TRAIL_EMA9"
    )
    return V04Result(
        variant=variant,
        trades=list(base.trades),
        equity_curve=list(base.equity_curve),
        metrics=base.metrics,
        trade_details=details,
        markers=list(markers),
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
        window_skipped_evaluations=base.cutoff_skipped_evaluations,
        scale_target_trades=scale_target_trades,
        trail_exit_trades=trail_exit_trades,
        breakeven_armed_trades=base.breakeven_armed_trades,
    )


# --------------------------------------------------------------------------- #
# Public run / walk-forward entry points
# --------------------------------------------------------------------------- #
def run_v04_backtest(
    data: V02Data | Sequence[Bar],
    config: AppConfig | None = None,
    *,
    variant: V04Variant = V04_FINAL_VARIANT,
    risk: Any | None = None,
    initial_equity: float | None = None,
    session_dates: Sequence[date] | None = None,
) -> V04Result:
    """Run one V0.4 ablation variant over *data*.

    Parameters
    ----------
    data:
        Prepared :class:`~intraday_scalper.research.v02.V02Data` (preferred:
        built once and reused by all steps/folds) or a sorted sequence of
        completed regular-session raw 1m bars.
    config:
        Validated ``AppConfig``; default config when omitted.  The fractional
        share override (precision 2) is applied in memory for V0.4 variants that
        request it — ``config/default.yaml`` is never modified.
    variant:
        Ablation step (default = :data:`V04_FINAL_VARIANT`).
    risk:
        Optional ``RiskManager``-compatible instance; ``None`` creates a fresh
        production manager.
    session_dates:
        Optional ET-date filter (segments / walk-forward folds / Friday recap).
    """

    app_config = load_config() if config is None else config
    prepared = data if isinstance(data, V02Data) else prepare_v02_data(data)
    replay = _V04Replay(
        prepared,
        app_config,
        variant,
        risk=risk,
        initial_equity=initial_equity,
    )
    base = replay.run(session_dates)
    return _build_v04_result(base, variant, replay.markers)


def _collect_5m_bars(data: V02Data, dates: Sequence[date]) -> list[Bar]:
    wanted = set(dates)
    collected: list[Bar] = []
    for session in data.sessions:
        if session.et_date not in wanted:
            continue
        for symbol_session in session.symbols.values():
            collected.extend(bucket.bar for bucket in symbol_session.buckets)
    collected.sort(key=lambda bar: bar.start_utc)
    return collected


def run_v04_walkforward(
    data: V02Data | Sequence[Bar],
    config: AppConfig | None = None,
    *,
    variant: V04Variant = V04_FINAL_VARIANT,
    train_days: int = 120,
    test_days: int = 20,
    step_days: int = 20,
    initial_equity: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> V04WalkForwardResult:
    """Strict chronological walk-forward with the V0.2/V0.3 fold discipline."""

    app_config = load_config() if config is None else config
    prepared = data if isinstance(data, V02Data) else prepare_v02_data(data)
    equity = float(
        app_config.risk.account_equity if initial_equity is None else initial_equity
    )
    if not math.isfinite(equity) or equity <= 0.0:
        raise ValueError("initial_equity must be finite and > 0")

    dates = list(prepared.dates)
    splits = walk_forward_splits(dates, train_days, test_days, step_days)
    notes: list[str] = []
    if not splits:
        empty = compute_metrics([], [], initial_equity=equity)
        notes.append(
            "WALK-FORWARD INCOMPLETE: not enough session dates for one full fold "
            f"(have {len(dates)}, need {train_days}+{test_days})"
        )
        return V04WalkForwardResult(
            variant=variant,
            folds=(),
            summary={
                "folds": 0,
                "train_days": int(train_days),
                "test_days": int(test_days),
                "step_days": int(step_days),
                "session_dates": len(dates),
                "total_test_trades": 0,
            },
            aggregate_test_metrics=empty,
            notes=tuple(notes),
        )

    fold_results: list[V04FoldResult] = []
    all_test_trades: list[TradeRecord] = []
    all_train_trades: list[TradeRecord] = []
    breakers = 0
    ambiguous = 0
    for position, fold in enumerate(splits, start=1):
        train_bars = _collect_5m_bars(prepared, fold.train_dates)
        test_bars = _collect_5m_bars(prepared, fold.test_dates)
        assert_no_leakage(train_bars, test_bars)
        train_result = run_v04_backtest(
            prepared,
            app_config,
            variant=variant,
            risk=v03.RiskManager(app_config),
            initial_equity=equity,
            session_dates=list(fold.train_dates),
        )
        test_result = run_v04_backtest(
            prepared,
            app_config,
            variant=variant,
            risk=v03.RiskManager(app_config),
            initial_equity=equity,
            session_dates=list(fold.test_dates),
        )
        all_train_trades.extend(train_result.trades)
        all_test_trades.extend(test_result.trades)
        breakers += test_result.breaker_days
        ambiguous += test_result.ambiguous_bars
        fold_results.append(
            V04FoldResult(
                fold=fold,
                train_result=train_result,
                test_result=test_result,
                leakage_checked=True,
            )
        )
        if progress is not None:
            progress(position, len(splits))

    aggregate = compute_metrics(all_test_trades, [], initial_equity=equity)
    test_net = sum(trade.pnl for trade in all_test_trades)
    fold_pnls = [fold.test_result.metrics.net_pnl_dollars for fold in fold_results]
    fold_drawdowns = [
        fold.test_result.metrics.max_drawdown_dollars for fold in fold_results
    ]
    fold_wins = sum(1 for pnl in fold_pnls if pnl > 0.0)
    summary = {
        "folds": len(fold_results),
        "train_days": int(train_days),
        "test_days": int(test_days),
        "step_days": int(step_days),
        "session_dates": len(dates),
        "total_train_trades": len(all_train_trades),
        "total_test_trades": aggregate.total_trades,
        "test_wins": aggregate.wins,
        "test_win_rate": aggregate.win_rate,
        "test_profit_factor": aggregate.profit_factor,
        "test_net_pnl_dollars": test_net,
        "test_avg_r": aggregate.avg_r,
        "test_max_drawdown_dollars_max_fold": max(fold_drawdowns)
        if fold_drawdowns
        else 0.0,
        "test_max_drawdown_pct_max_fold": max(
            (fold.test_result.metrics.max_drawdown_pct for fold in fold_results),
            default=0.0,
        ),
        "positive_folds": fold_wins,
        "best_fold_pnl": max(fold_pnls) if fold_pnls else 0.0,
        "worst_fold_pnl": min(fold_pnls) if fold_pnls else 0.0,
        "test_breaker_days_total": breakers,
        "test_ambiguous_bars_total": ambiguous,
        "test_trades_per_session": (
            len(all_test_trades)
            / sum(len(fold.fold.test_dates) for fold in fold_results)
            if fold_results
            else 0.0
        ),
    }
    notes.append(
        "each fold starts from the same initial equity (no compounding across folds); "
        "train metrics are in-sample diagnostics, only test metrics are out-of-sample"
    )
    notes.append(DATA_CAVEAT)
    return V04WalkForwardResult(
        variant=variant,
        folds=tuple(fold_results),
        summary=summary,
        aggregate_test_metrics=aggregate,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Web / Friday recap API
# --------------------------------------------------------------------------- #

def _to_et_hm(ts: datetime | None) -> str | None:
    """Compact ET clock string for the Web recap tables."""

    return to_et(ts).strftime("%H:%M:%S") if ts is not None else None


def _legs_for_detail(detail: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not detail:
        return []
    legs = detail.get("partial_legs") or []
    return [dict(leg) for leg in legs]


def result_to_trade_dicts(result: V04Result) -> list[dict[str, Any]]:
    """Convert a V04Result into Web-friendly per-trade dicts with markers."""

    markers_by_trade: dict[str, list[dict[str, Any]]] = {}
    for marker in result.markers:
        markers_by_trade.setdefault(str(marker.get("trade_id")), []).append(dict(marker))

    trade_dicts: list[dict[str, Any]] = []
    for trade in sorted(result.trades, key=lambda item: item.entry_time_utc):
        detail = result.trade_details.get(trade.signal_id, {})
        risk_dollars = trade.risk_dollars
        trade_dicts.append(
            {
                "kind": "trade",
                "trade_id": trade.signal_id,
                "symbol": trade.symbol,
                "strategy": detail.get("strategy_label", trade.strategy.value),
                "direction": "LONG",
                "mode": trade.mode.value,
                "shares": trade.shares,
                "entry_time_utc": trade.entry_time_utc.isoformat(),
                "entry_time_et": _to_et_hm(trade.entry_time_utc),
                "entry_price_signal": trade.entry_price_signal,
                "entry_price_fill": trade.entry_price_fill,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "exit_time_utc": trade.exit_time_utc.isoformat()
                if trade.exit_time_utc
                else None,
                "exit_time_et": _to_et_hm(trade.exit_time_utc),
                "exit_price_fill": trade.exit_price_fill,
                "exit_reason": trade.exit_reason,
                "exit_reason_detail": detail.get("exit_reason_detail", trade.exit_reason),
                "exit_stop_kind": detail.get("exit_stop_kind"),
                "pnl": round(trade.pnl, 4),
                "r_multiple": round(trade.r_multiple, 4),
                "holding_minutes": round(trade.holding_minutes, 2),
                "entry_slippage_bps": round(trade.entry_slippage_bps, 3),
                "mae_r": round(float(detail.get("mae_r", 0.0) or 0.0), 4),
                "mfe_r": round(float(detail.get("mfe_r", 0.0) or 0.0), 4),
                "scale_target_hit": bool(detail.get("scale_target_hit", False)),
                "breakeven_armed": bool(detail.get("breakeven_armed", False)),
                "partial_legs": _legs_for_detail(detail),
                "risk_dollars": round(risk_dollars, 4),
                "markers": markers_by_trade.get(trade.signal_id, []),
            }
        )
    return trade_dicts


def evaluate_v04_session(
    bars_1m: V02Data | Sequence[Bar],
    session_date: date | str | datetime,
    variant: V04Variant = V04_FINAL_VARIANT,
    config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one session and return Web-friendly trade dicts with markers.

    Intended for the Friday replay UI and for live-session evaluation.  Pass a
    prepared :class:`V02Data` when available, or the raw 1m bars **including
    enough prior history** for the 5m EMA/ATR warm-up (the engine replays only
    ``session_date``, but indicators are continuous across sessions, exactly as
    in the backtest).

    Returns
    -------
    list[dict]
        One dict per trade (aggregated per entry, partial legs merged), in
        chronological entry order.  Each trade dict contains the aggregate trade
        fields plus ``markers``: a list of dicts with ``type``
        (ENTRY/STOP/BREAKEVEN/TARGET/EXIT), ``time_utc``, ``price``, ``symbol``,
        ``strategy`` and ``trade_id`` — directly consumable by the Web chart.
    """

    if isinstance(session_date, datetime):
        target_date = et_date(session_date)
    elif isinstance(session_date, date):
        target_date = session_date
    else:
        target_date = date.fromisoformat(str(session_date)[:10])

    prepared = bars_1m if isinstance(bars_1m, V02Data) else prepare_v02_data(bars_1m)
    if target_date not in set(prepared.dates):
        raise ValueError(
            f"session_date {target_date.isoformat()} not present in the supplied bars "
            f"(have {len(prepared.dates)} sessions, last {prepared.dates[-1]})"
        )
    result = run_v04_backtest(
        prepared,
        config,
        variant=variant,
        session_dates=[target_date],
    )
    return result_to_trade_dicts(result)
