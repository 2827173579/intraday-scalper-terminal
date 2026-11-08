"""SOXL / SOXS Intraday Scalper.

A standalone, minute-level intraday signal/research system for US-listed
3x leveraged semiconductor ETFs (SOXL / SOXS). It is intentionally isolated
from the legacy ``integrated_market_radar.py`` / ``ml_predictor_5y.py``
projects that live at the repository root and must not be modified.

The package never places orders. It only produces analysis, strategy
signals, risk calculations, audible alerts and a manual-trade journal.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
