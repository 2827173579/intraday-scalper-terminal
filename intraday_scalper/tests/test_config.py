"""Tests for the YAML configuration loader and validation rules."""

from __future__ import annotations

from dataclasses import replace
from datetime import time

import pytest

from intraday_scalper.core import Mode, load_config
from intraday_scalper.core.config import config_from_mapping


class TestDefaultConfig:
    def test_defaults_load(self):
        config = load_config()
        assert config.mode is Mode.DUAL_WATCH
        assert config.data.symbols == ("SOXL", "SOXS")
        assert config.risk.account_equity == 3000.0
        assert config.risk.max_risk_per_trade == 20.0
        assert config.risk.max_notional_per_trade == 1200.0
        assert config.risk.max_daily_net_loss == 60.0
        assert config.risk.max_entries_per_day == 15
        assert config.risk.consecutive_stop_cooldown_minutes == 20
        assert config.risk.take_profit_hint_r == (2.0, 2.5)
        assert config.session.no_new_entries_et == time(15, 30)
        assert config.session.eod_check_et == time(15, 50)
        assert config.session.eod_urgent_et == time(15, 55)
        assert config.data.startup_quote_checks_max == 2

    def test_default_config_passes_validation(self):
        load_config().validate()


class TestOverrides:
    def test_minimal_mapping_uses_defaults(self):
        config = config_from_mapping({"mode": "long_soxl"})
        assert config.mode is Mode.LONG_SOXL
        assert config.risk.max_risk_per_trade == 20.0

    def test_nested_overrides(self):
        config = config_from_mapping(
            {
                "mode": "long_soxs",
                "risk": {"max_risk_per_trade": 10.0, "max_entries_per_day": 5},
                "data": {"symbols": ["SOXS"], "staleness_seconds": 30},
            }
        )
        assert config.mode is Mode.LONG_SOXS
        assert config.risk.max_risk_per_trade == 10.0
        assert config.risk.max_entries_per_day == 5
        assert config.data.symbols == ("SOXS",)
        assert config.data.staleness_seconds == 30.0

    def test_take_profit_hint_scalar_is_normalised_to_tuple(self):
        config = config_from_mapping({"risk": {"take_profit_hint_r": 2.0}})
        assert config.risk.take_profit_hint_r == (2.0,)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            config_from_mapping({"mode": "moon_mission"})

    def test_duplicate_symbols_rejected(self):
        with pytest.raises(ValueError, match="duplicates"):
            config_from_mapping({"data": {"symbols": ["SOXL", "SOXL"]}})

    def test_negative_risk_rejected(self):
        with pytest.raises(ValueError):
            config_from_mapping({"risk": {"max_risk_per_trade": -1}})

    def test_entries_per_day_must_be_positive(self):
        with pytest.raises(ValueError):
            config_from_mapping({"risk": {"max_entries_per_day": 0}})

    def test_bad_time_format_rejected(self):
        with pytest.raises(ValueError):
            config_from_mapping({"session": {"no_new_entries_et": "3:30 PM"}})

    def test_risk_cannot_exceed_equity(self):
        with pytest.raises(ValueError, match="account_equity"):
            config_from_mapping({"risk": {"account_equity": 10.0, "max_risk_per_trade": 20.0}})


class TestFrozen:
    def test_config_is_immutable(self):
        config = load_config()
        with pytest.raises(Exception):
            config.mode = Mode.LONG_SOXL  # type: ignore[misc]

    def test_with_mode_returns_new_copy(self):
        config = load_config()
        other = config.with_mode("long_soxs")
        assert other is not config
        assert other.mode is Mode.LONG_SOXS
        assert config.mode is Mode.DUAL_WATCH

    def test_replace_nested_risk(self):
        config = load_config()
        risk = replace(config.risk, max_risk_per_trade=5.0)
        assert risk.max_risk_per_trade == 5.0
        assert config.risk.max_risk_per_trade == 20.0
