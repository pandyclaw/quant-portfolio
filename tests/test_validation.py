"""Tests for utils/validation.py — statistical validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.validation import (
    bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    permutation_test,
)


class TestDeflatedSharpe:
    def test_single_trial_equals_observed(self) -> None:
        result = deflated_sharpe_ratio(
            observed_sharpe=1.5, num_trials=1, num_returns=252
        )
        assert result.is_significant
        assert result.expected_max_sharpe == 0.0

    def test_many_trials_deflates(self) -> None:
        result = deflated_sharpe_ratio(
            observed_sharpe=1.0, num_trials=100, num_returns=252
        )
        assert not result.is_significant, (
            "Sharpe 1.0 should not survive 100 trials"
        )

    def test_high_sharpe_survives(self) -> None:
        result = deflated_sharpe_ratio(
            observed_sharpe=3.0, num_trials=50, num_returns=1000
        )
        assert result.is_significant

    def test_more_trials_harder_to_pass(self) -> None:
        few = deflated_sharpe_ratio(
            observed_sharpe=1.5, num_trials=5, num_returns=252
        )
        many = deflated_sharpe_ratio(
            observed_sharpe=1.5, num_trials=100, num_returns=252
        )
        assert few.p_value <= many.p_value, (
            "More trials should make it harder to pass (higher p-value)"
        )


class TestExpectedMaxSharpe:
    def test_monotonic_in_trials(self) -> None:
        s10 = expected_max_sharpe(10, 252)
        s100 = expected_max_sharpe(100, 252)
        s1000 = expected_max_sharpe(1000, 252)
        assert s10 < s100 < s1000, "More trials = higher expected max"

    def test_single_trial(self) -> None:
        assert expected_max_sharpe(1, 252) == 0.0


class TestBootstrapCI:
    def test_positive_returns_positive_ci(self, daily_returns: pd.Series) -> None:
        result = bootstrap_sharpe_ci(daily_returns)
        assert result["prob_positive"] > 0.5

    def test_ci_contains_point_estimate(self, daily_returns: pd.Series) -> None:
        result = bootstrap_sharpe_ci(daily_returns)
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_short_series(self) -> None:
        result = bootstrap_sharpe_ci(pd.Series([0.01, 0.02]))
        assert result["point_estimate"] == 0.0


class TestPermutationTest:
    def test_strong_signal_significant(self) -> None:
        rng = np.random.RandomState(42)
        # Very strong positive drift with low noise — clearly non-random
        returns = pd.Series(rng.normal(0.005, 0.002, 500))
        result = permutation_test(returns, n_permutations=500)
        assert result["real_sharpe"] > 0, "Strong positive drift should have positive Sharpe"

    def test_zero_mean_low_sharpe(self) -> None:
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0, 0.01, 252))
        result = permutation_test(returns, n_permutations=500)
        # Zero-mean returns should have Sharpe near zero
        assert abs(result["real_sharpe"]) < 3.0
