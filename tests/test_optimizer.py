"""Tests for portfolio/optimizer.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.optimizer import (
    compare_methods,
    hierarchical_risk_parity,
    mean_variance,
    min_variance,
    risk_parity,
)


@pytest.fixture
def strategy_returns() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 504
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "strat_a": rng.normal(0.0004, 0.010, n),
        "strat_b": rng.normal(0.0003, 0.008, n),
        "strat_c": rng.normal(0.0002, 0.012, n),
        "strat_d": rng.normal(0.0005, 0.015, n),
    }, index=dates)


class TestMeanVariance:
    def test_weights_sum_to_one(self, strategy_returns: pd.DataFrame) -> None:
        result = mean_variance(strategy_returns)
        assert abs(np.sum(result.weights) - 1.0) < 0.01

    def test_no_negative_weights(self, strategy_returns: pd.DataFrame) -> None:
        result = mean_variance(strategy_returns)
        assert np.all(result.weights >= -0.01)

    def test_max_weight_respected(self, strategy_returns: pd.DataFrame) -> None:
        result = mean_variance(strategy_returns, max_weight=0.30)
        assert np.all(result.weights <= 0.31)


class TestMinVariance:
    def test_lower_vol_than_equal(self, strategy_returns: pd.DataFrame) -> None:
        mv = min_variance(strategy_returns)
        n = len(strategy_returns.columns)
        eq_w = np.ones(n) / n
        from portfolio.optimizer import _shrunk_covariance
        cov = _shrunk_covariance(strategy_returns) * 252
        eq_vol = float(np.sqrt(eq_w @ cov @ eq_w))
        assert mv.expected_vol <= eq_vol + 0.001


class TestRiskParity:
    def test_weights_positive(self, strategy_returns: pd.DataFrame) -> None:
        result = risk_parity(strategy_returns)
        assert np.all(result.weights > 0)

    def test_weights_sum_to_one(self, strategy_returns: pd.DataFrame) -> None:
        result = risk_parity(strategy_returns)
        assert abs(np.sum(result.weights) - 1.0) < 0.01


class TestHRP:
    def test_weights_sum_to_one(self, strategy_returns: pd.DataFrame) -> None:
        result = hierarchical_risk_parity(strategy_returns)
        assert abs(np.sum(result.weights) - 1.0) < 0.01

    def test_no_negative_weights(self, strategy_returns: pd.DataFrame) -> None:
        result = hierarchical_risk_parity(strategy_returns)
        assert np.all(result.weights >= 0)

    def test_no_zero_weights(self, strategy_returns: pd.DataFrame) -> None:
        result = hierarchical_risk_parity(strategy_returns)
        assert np.all(result.weights > 0.01)


class TestCompare:
    def test_compare_returns_table(self, strategy_returns: pd.DataFrame) -> None:
        table = compare_methods(strategy_returns)
        assert len(table) == 5  # equal, mv, minvar, rp, hrp
        assert "method" in table.columns
        assert "sharpe" in table.columns
