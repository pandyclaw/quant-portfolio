"""Tests for risk/ modules: stress testing and correlation monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk.correlation_monitor import correlation_break_alert, rolling_correlation
from risk.stress_test import SCENARIOS, apply_scenario, format_stress_report, stress_test_all


@pytest.fixture
def long_returns() -> pd.Series:
    """Returns spanning 2006-2024 for stress testing."""
    rng = np.random.RandomState(42)
    n = 4500
    dates = pd.bdate_range("2006-01-03", periods=n)
    return pd.Series(rng.normal(0.0003, 0.012, n), index=dates)


@pytest.fixture
def multi_strategy_returns() -> pd.DataFrame:
    """Multi-strategy return DataFrame for correlation tests."""
    rng = np.random.RandomState(42)
    n = 504
    dates = pd.bdate_range("2022-01-03", periods=n)
    common = rng.normal(0, 0.005, n)  # Common market factor
    return pd.DataFrame({
        "momentum": common + rng.normal(0.0003, 0.008, n),
        "mean_rev": -0.3 * common + rng.normal(0.0002, 0.006, n),
        "carry": 0.2 * common + rng.normal(0.0004, 0.007, n),
        "vol_sell": -0.5 * common + rng.normal(0.0001, 0.010, n),
    }, index=dates)


class TestStressTest:
    def test_scenarios_defined(self) -> None:
        assert len(SCENARIOS) >= 5
        assert "GFC_2008" in SCENARIOS
        assert "COVID_2020" in SCENARIOS

    def test_apply_scenario(self, long_returns: pd.Series) -> None:
        result = apply_scenario(long_returns, "GFC_2008")
        assert result is not None
        assert result.n_days > 0
        assert result.scenario_name == "GFC_2008"

    def test_stress_test_all(self, long_returns: pd.Series) -> None:
        results = stress_test_all(long_returns)
        assert len(results) >= 3  # At least GFC, COVID, Rate Hike should be covered

    def test_format_report(self, long_returns: pd.Series) -> None:
        results = stress_test_all(long_returns)
        report = format_stress_report(results)
        assert "STRESS TEST REPORT" in report
        assert "GFC_2008" in report

    def test_missing_period_returns_none(self) -> None:
        # Short series that doesn't cover any scenario
        dates = pd.bdate_range("2024-01-02", periods=50)
        short = pd.Series(np.random.normal(0, 0.01, 50), index=dates)
        result = apply_scenario(short, "GFC_2008")
        assert result is None


class TestCorrelationMonitor:
    def test_rolling_correlation(self, multi_strategy_returns: pd.DataFrame) -> None:
        corr = rolling_correlation(multi_strategy_returns, window=63)
        assert "avg_correlation" in corr.columns
        # Should have values after warm-up period
        valid = corr["avg_correlation"].dropna()
        assert len(valid) > 0

    def test_break_alert(self, multi_strategy_returns: pd.DataFrame) -> None:
        alerts = correlation_break_alert(multi_strategy_returns, threshold=0.5)
        assert "alert" in alerts.columns
        assert "regime" in alerts.columns

    def test_correlation_bounded(self, multi_strategy_returns: pd.DataFrame) -> None:
        corr = rolling_correlation(multi_strategy_returns)
        valid = corr["avg_correlation"].dropna()
        assert valid.min() >= -1.01
        assert valid.max() <= 1.01
