"""Strategy A: opening-range breakout (5-minute and 15-minute variants).

The opening range (OR) is the high/low of the first N regular-session minutes.
A long candidate is generated only when the latest **confirmed** 1m bar
finishes above the OR high plus a small buffer, with a strong body, above its
volume average and above the trend EMA. SOXL and SOXS are structurally
identical here: buying whichever ETF is breaking out expresses the view.

No look-ahead: the OR is built only from completed bars inside the window, the
signal bar must start at/after ``opening_range.end_utc`` (so a bar inside the
range can never trigger), and every indicator referenced is computed on bars
no newer than the signal bar.
"""

from __future__ import annotations

from typing import Optional

from intraday_scalper.core import OrbParams, SignalCandidate, StrategyName, to_utc
from intraday_scalper.features.indicators import OpeningRange, ema

from .base import Strategy, StrategyContext

__all__ = ["OpeningRangeBreakout"]


class OpeningRangeBreakout(Strategy):
    """Breakout of the 5-minute or 15-minute opening range."""

    def __init__(self, params: OrbParams, timeframe_minutes: int = 5) -> None:
        if timeframe_minutes not in (5, 15):
            raise ValueError("opening-range timeframe must be 5 or 15 minutes")
        if not isinstance(params, OrbParams):
            raise TypeError("params must be an OrbParams instance")
        self.params = params
        self.timeframe_minutes = int(timeframe_minutes)
        self.name = (
            StrategyName.ORB_5M if self.timeframe_minutes == 5 else StrategyName.ORB_15M
        )

    # -- helpers --------------------------------------------------------- #
    def _range(self, ctx: StrategyContext) -> Optional[OpeningRange]:
        return (
            ctx.opening_range_5m if self.timeframe_minutes == 5 else ctx.opening_range_15m
        )

    def _trend_ema(self, ctx: StrategyContext) -> Optional[float]:
        """EMA(trend_ema_period) on confirmed 1m closes (context fallback)."""
        period = self.params.trend_ema_period
        closes = ctx.closes()
        if len(closes) >= period:
            series = ema(closes, period)
            if series[-1] is not None:
                return series[-1]
        return None if ctx.ema_slow is None else float(ctx.ema_slow)

    # -- evaluation ------------------------------------------------------ #
    def evaluate(self, ctx: StrategyContext) -> Optional[SignalCandidate]:
        bar = self._confirmed_bar(ctx)
        if bar is None:
            return None
        opening_range = self._range(ctx)
        if opening_range is None or not opening_range.complete:
            return None
        # The signal bar must start at or after the OR end: never inside the OR.
        if to_utc(bar.start_utc) < to_utc(opening_range.end_utc):
            return None
        atr = ctx.atr
        if atr is None:
            return None

        buffer = self.params.min_breakout_buffer_pct
        breakout_level = opening_range.high * (1.0 + buffer)
        if bar.close <= breakout_level:
            return None
        if bar.body_ratio < self.params.min_body_ratio:
            return None

        average_volume = self._prior_average_volume(ctx)
        if average_volume is None:
            return None
        if bar.volume < self.params.min_volume_multiple * average_volume:
            return None

        trend_ema = self._trend_ema(ctx)
        if trend_ema is None or bar.close <= trend_ema:
            return None

        # Structural reference: the range high and the breakout bar low, whichever
        # is lower. The shared builder subtracts ATR and clamps to the configured
        # percentage band.
        structural_stop = min(opening_range.high, bar.low)
        rationale = (
            f"ORB {self.timeframe_minutes}m 突破：收盘 {bar.close:.3f} > 区间上沿 "
            f"{opening_range.high:.3f}(+{buffer:.2%})，实体比 {bar.body_ratio:.2f}，"
            f"量 {bar.volume:.0f} >= {self.params.min_volume_multiple:.1f}x 均量 "
            f"{average_volume:.0f}，收盘在 EMA{self.params.trend_ema_period} 之上"
        )
        structural_note = (
            f"结构止损 = min(OR.high {opening_range.high:.3f}, bar.low {bar.low:.3f}) "
            f"- ATR({atr:.4f}) x {self.atr_stop_multiple:.2f}，"
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
