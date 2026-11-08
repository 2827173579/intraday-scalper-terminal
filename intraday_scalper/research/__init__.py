"""Structural research backtests (isolated from the production package).

This subpackage holds *research only* experiments.  Nothing here is imported by
the live signal engine, dashboard or journal; it exists so that structural
hypotheses can be measured on real IEX bars **without** touching the frozen
production modules.

* :mod:`intraday_scalper.research.v02` — V0.2 "5-minute driven + mirror-regime
  filter + IEX-activity adaptation" backtest.
* :mod:`intraday_scalper.research.v03` — V0.3 structural ablation on top of
  V0.2 (fractional shares / strategy subset / 14:00 ET cutoff / +1.2R breakeven
  shield), imported read-only from V0.2 and never modifying it.

Data caveat: every V0.2/V0.3 number is computed on Alpaca **IEX-only** 1-minute
bars (single venue, a small share of US consolidated volume).  Volume, VWAP and
"relative activity" are therefore IEX-scoped, not consolidated-tape values.
The output is *research*, not proof of profitability.
"""

from .v02 import (
    ATR_PERIOD,
    EMA_FAST,
    EMA_SLOW,
    EMA_PULLBACK_5M,
    ORB_15M_5M,
    STRATEGY_LABELS,
    TARGET_R,
    VWAP_RECLAIM_5M,
    SymbolFeatures,
    StrategyView,
    V02Data,
    V02FoldResult,
    V02Result,
    V02WalkForwardResult,
    build_strategy_view,
    ema_pullback_5m_candidate,
    evaluate_v02,
    mirror_relation_ok,
    orb_15m_5m_candidate,
    prepare_v02_data,
    resample_session_5m,
    run_v02_backtest,
    run_v02_walkforward,
    vwap_reclaim_5m_candidate,
)
from .v03 import (
    ABLATION_VARIANTS,
    BREAKEVEN_OFFSET_R,
    BREAKEVEN_TRIGGER_R,
    ENTRY_CUTOFF_ET,
    FRACTIONAL_PRECISION,
    STEP0_VARIANT,
    STEP1_VARIANT,
    STEP2_VARIANT,
    STEP3_VARIANT,
    STEP4_VARIANT,
    V03_SIGNAL_PREFIX,
    V03FoldResult,
    V03Result,
    V03Variant,
    V03WalkForwardResult,
    build_effective_config,
    run_v03_backtest,
    run_v03_walkforward,
)

from .v04 import (
    ABLATION_STEPS as V04_ABLATION_STEPS,
    BREAKEVEN_OFFSET_R as V04_BREAKEVEN_OFFSET_R,
    ENTRY_WINDOW_END_ET,
    ENTRY_WINDOW_START_ET,
    SMC_SWEEP_5M,
    V04_FINAL_VARIANT,
    V04EvalContext,
    V04FoldResult,
    V04Result,
    V04Variant,
    V04WalkForwardResult,
    build_eval_context,
    evaluate_v04_session,
    result_to_trade_dicts,
    run_v04_backtest,
    run_v04_walkforward,
    smc_sweep_5m_candidate,
    strategy_label_for_candidate,
    vwap_reclaim_trend_ok,
)

__all__ = [
    # V0.2
    "ATR_PERIOD",
    "EMA_FAST",
    "EMA_SLOW",
    "EMA_PULLBACK_5M",
    "ORB_15M_5M",
    "STRATEGY_LABELS",
    "TARGET_R",
    "VWAP_RECLAIM_5M",
    "StrategyView",
    "SymbolFeatures",
    "V02Data",
    "V02FoldResult",
    "V02Result",
    "V02WalkForwardResult",
    "build_strategy_view",
    "ema_pullback_5m_candidate",
    "evaluate_v02",
    "mirror_relation_ok",
    "orb_15m_5m_candidate",
    "prepare_v02_data",
    "resample_session_5m",
    "run_v02_backtest",
    "run_v02_walkforward",
    "vwap_reclaim_5m_candidate",
    # V0.3
    "ABLATION_VARIANTS",
    "BREAKEVEN_OFFSET_R",
    "BREAKEVEN_TRIGGER_R",
    "ENTRY_CUTOFF_ET",
    "FRACTIONAL_PRECISION",
    "STEP0_VARIANT",
    "STEP1_VARIANT",
    "STEP2_VARIANT",
    "STEP3_VARIANT",
    "STEP4_VARIANT",
    "V03_SIGNAL_PREFIX",
    "V03FoldResult",
    "V03Result",
    "V03Variant",
    "V03WalkForwardResult",
    "build_effective_config",
    "run_v03_backtest",
    "run_v03_walkforward",
    # V0.4
    "ENTRY_WINDOW_END_ET",
    "ENTRY_WINDOW_START_ET",
    "SMC_SWEEP_5M",
    "V04_ABLATION_STEPS",
    "V04_BREAKEVEN_OFFSET_R",
    "V04_FINAL_VARIANT",
    "V04EvalContext",
    "V04FoldResult",
    "V04Result",
    "V04Variant",
    "V04WalkForwardResult",
    "build_eval_context",
    "evaluate_v04_session",
    "result_to_trade_dicts",
    "run_v04_backtest",
    "run_v04_walkforward",
    "smc_sweep_5m_candidate",
    "strategy_label_for_candidate",
    "vwap_reclaim_trend_ok",
]
