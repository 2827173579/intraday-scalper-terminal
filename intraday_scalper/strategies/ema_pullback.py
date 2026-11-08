"""Strategy C: EMA trend pullback.

Setup: an established uptrend (the last few confirmed bars all closed above the
true session VWAP and the fast EMA is above the slow EMA), a shallow pullback
that wicks into the fast EMA, and a bullish confirmation bar that closes above
the pullback bar's high while a strong body shows real buying.

Everything is evaluated on already-closed bars: the pullback bar is the
previous confirmed 1m bar and the confirmation bar is the latest confirmed 1m
bar. No future bar is read, and the structural stop is anchored below the
pullback low / fast EMA.
"""

from __future__ import annotations

from typing import Optional

from intraday_scalper.core import EmaPullbackParams, SignalCandidate, StrategyName
from intraday_scalper.features.indicators import ema

from .base import Strategy, StrategyContext

__all__ = ["EmaTrendPullback", "PULLBACK_TOUCH_TOLERANCE_PCT", "PULLBACK_CLOSE_TOLERANCE_PCT"]

#: The pullback bar's low may be this far above the fast EMA and still count as
#: "touching" it (0.05%).
PULLBACK_TOUCH_TOLERANCE_PCT = 0.0005
#: The pullback bar's close may be this far below the fast EMA and still count
#: as "holding" it (0.05%).
PULLBACK_CLOSE_TOLERANCE_PCT = 0.0005


class EmaTrendPullback(Strategy):
    """Buy a shallow pullback to the fast EMA inside an established uptrend."""

    def __init__(
        self,
        params: EmaPullbackParams,
        *,
        touch_tolerance_pct: float = PULLBACK_TOUCH_TOLERANCE_PCT,
        close_tolerance_pct: float = PULLBACK_CLOSE_TOLERANCE_PCT,
    ) -> None:
        if not isinstance(params, EmaPullbackParams):
            raise TypeError("params must be an EmaPullbackParams instance")
        if touch_tolerance_pct < 0.0 or close_tolerance_pct < 0.0:
            raise ValueError("pullback tolerances must be >= 0")
        self.params = params
        self.touch_tolerance_pct = float(touch_tolerance_pct)
        self.close_tolerance_pct = float(close_tolerance_pct)
        self.name = StrategyName.EMA_PULLBACK

    # -- helpers --------------------------------------------------------- #
    def _ema_pair(self, ctx: StrategyContext) -> tuple[Optional[float], Optional[float]]:
        fast = None if ctx.ema_fast is None else float(ctx.ema_fast)
        slow = None if ctx.ema_slow is None else float(ctx.ema_slow)
        closes = ctx.closes()
        if fast is None and len(closes) >= self.params.ema_fast:
            fast = ema(closes, self.params.ema_fast)[-1]
        if slow is None and len(closes) >= self.params.ema_slow:
            slow = ema(closes, self.params.ema_slow)[-1]
        return fast, slow

    # -- evaluation ------------------------------------------------------ #
    def evaluate(self, ctx: StrategyContext) -> Optional[SignalCandidate]:
        bar = self._confirmed_bar(ctx)
        if bar is None or len(ctx.bars_1m) < 2:
            return None
        previous = ctx.bars_1m[-2]
        if not previous.is_complete:
            return None
        vwap = ctx.vwap
        if vwap is None:
            return None
        vwap = float(vwap)
        atr = ctx.atr
        if atr is None:
            return None

        ema_fast, ema_slow = self._ema_pair(ctx)
        if ema_fast is None or ema_slow is None:
            return None
        if not ema_fast > ema_slow:
            return None

        # Uptrend filter: the bars before the confirmation bar (the pullback bar
        # included) all closed above the true session VWAP.
        trend_window = self.params.vwap_trend_bars
        trend_bars = ctx.bars_1m[-(trend_window + 1) : -1]
        if len(trend_bars) < trend_window:
            return None
        for trend_bar in trend_bars:
            if not trend_bar.is_complete:
                return None
            if trend_bar.close <= vwap:
                return None

        # Pullback: previous bar wicked down into the fast EMA but held it.
        if previous.low > ema_fast * (1.0 + self.touch_tolerance_pct):
            return None
        if previous.close < ema_fast * (1.0 - self.close_tolerance_pct):
            return None

        # Confirmation: bullish, solid body, closes above the pullback high.
        if bar.close <= bar.open:
            return None
        if bar.body_ratio < self.params.confirmation_body_ratio:
            return None
        if bar.close <= previous.high:
            return None

        structural_stop = min(previous.low, ema_fast)
        rationale = (
            f"EMA 回踩：最近 {trend_window} 根收盘均 > VWAP {vwap:.3f}，"
            f"EMA{self.params.ema_fast}({ema_fast:.3f})>EMA{self.params.ema_slow}"
            f"({ema_slow:.3f})；前一根低点 {previous.low:.3f} 回踩快线，"
            f"确认收盘 {bar.close:.3f} > 前高 {previous.high:.3f}，"
            f"实体比 {bar.body_ratio:.2f}"
        )
        structural_note = (
            f"结构止损 = min(前一根.low {previous.low:.3f}, EMA{self.params.ema_fast} "
            f"{ema_fast:.3f}) - ATR({atr:.4f}) x {self.atr_stop_multiple:.2f}，"
            f"再夹紧到 {self.params.min_stop_pct:.2%}-{self.params.max_stop_pct:.2%}"
        )
        return self._build_candidate(
            symbol=ctx.symbol,
            entry_ref=bar.close,
            structural_stop=structural_stop,
            target_r=self.params.target_r,
            atr=atr,
            min_stop_pct=self.params.min_stop_pct,
            max_stop_pct=self.params.max_stop_pct,
            rationale=rationale,
            structural_note=structural_note,
            now_utc=ctx.now_utc,
            confirmation_bar_start_utc=bar.start_utc,
        )
