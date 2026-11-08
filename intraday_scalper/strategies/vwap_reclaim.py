"""Strategy B: VWAP reclaim.

Setup: price was below the true session VWAP on the previous confirmed bar and
the latest confirmed bar reclaims it with a bullish, above-average-volume bar
while the fast EMA is above the slow EMA and price is above the fast EMA.

"VWAP as of the previous bar" is reconstructed from the confirmed bars before
the latest one (``sum(bar.vwap * bar.volume) / sum(bar.volume)``), so the check
uses only information that was available before the signal bar closed. The
current bar is compared against ``ctx.vwap`` (the session VWAP including the
already-closed latest bar), which is the standard reclaim definition.
"""

from __future__ import annotations

from typing import Optional

from intraday_scalper.core import SignalCandidate, StrategyName, VwapReclaimParams
from intraday_scalper.features.indicators import ema

from .base import Strategy, StrategyContext

__all__ = ["VwapReclaim"]


class VwapReclaim(Strategy):
    """Buy the first bullish reclaim of the session VWAP after a loss of it."""

    def __init__(self, params: VwapReclaimParams) -> None:
        if not isinstance(params, VwapReclaimParams):
            raise TypeError("params must be a VwapReclaimParams instance")
        self.params = params
        self.name = StrategyName.VWAP_RECLAIM

    # -- helpers --------------------------------------------------------- #
    def _ema_pair(self, ctx: StrategyContext) -> tuple[Optional[float], Optional[float]]:
        """Fast/slow EMA values, computed from confirmed bars when absent."""
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

        previous_vwap = self._cumulative_vwap(ctx.bars_1m[:-1])
        if previous_vwap is None:
            return None
        # Reclaim = price was below VWAP before and is above it now.
        if not previous.close < previous_vwap:
            return None
        if bar.close <= vwap * (1.0 + self.params.reclaim_buffer_pct):
            return None
        if bar.close <= bar.open:  # must close green
            return None

        average_volume = self._prior_average_volume(ctx)
        if average_volume is None:
            return None
        if bar.volume < self.params.min_volume_multiple * average_volume:
            return None

        ema_fast, ema_slow = self._ema_pair(ctx)
        if ema_fast is None or ema_slow is None:
            return None
        if not ema_fast > ema_slow:
            return None
        if bar.close <= ema_fast:
            return None

        structural_stop = min(bar.low, vwap)
        rationale = (
            f"VWAP 收复：前一根收盘 {previous.close:.3f} < 前值VWAP {previous_vwap:.3f}，"
            f"当前收盘 {bar.close:.3f} > VWAP {vwap:.3f}(+{self.params.reclaim_buffer_pct:.2%})，"
            f"阳线且量 {bar.volume:.0f} >= {self.params.min_volume_multiple:.1f}x 均量 "
            f"{average_volume:.0f}，EMA{self.params.ema_fast}>EMA{self.params.ema_slow}"
        )
        structural_note = (
            f"结构止损 = min(bar.low {bar.low:.3f}, VWAP {vwap:.3f}) "
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
