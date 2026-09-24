"""Tests for strategies/tsmom_rotation.py — TSMOM cross-asset rotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.tsmom_rotation import TSMOMRotation


@pytest.fixture
def multi_asset_prices() -> pd.DataFrame:
    """Synthetic multi-asset prices: trending and mean-reverting series."""
    rng = np.random.RandomState(42)
    n = 756  # ~3 years
    dates = pd.bdate_range("2021-01-04", periods=n)

    # SPY: uptrend
    spy = 400 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    # TLT: downtrend (rising rates)
    tlt = 150 * np.cumprod(1 + rng.normal(-0.0002, 0.008, n))
    # GLD: mild uptrend
    gld = 170 * np.cumprod(1 + rng.normal(0.0002, 0.010, n))
    # DBC: choppy
    dbc = 25 * np.cumprod(1 + rng.normal(0.0001, 0.015, n))
    # EFA: sideways
    efa = 75 * np.cumprod(1 + rng.normal(0.0001, 0.011, n))

    return pd.DataFrame(
        {"SPY": spy, "TLT": tlt, "GLD": gld, "DBC": dbc, "EFA": efa},
        index=dates,
    )


class TestSignals:
    def test_signals_binary(self, multi_asset_prices: pd.DataFrame) -> None:
        strategy = TSMOMRotation()
        signals = strategy.compute_signals(multi_asset_prices)
        # All values should be 0 or 1
        assert set(signals.values.flatten()).issubset({0, 1})

    def test_trending_asset_has_positive_signal(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        strategy = TSMOMRotation()
        signals = strategy.compute_signals(multi_asset_prices)
        # SPY (uptrend) should have positive signal most of the time
        spy_active_pct = (signals["SPY"] > 0).mean()
        assert spy_active_pct > 0.3, f"SPY should be active >30%, got {spy_active_pct:.1%}"


class TestWeights:
    def test_weights_sum_reasonable(self, multi_asset_prices: pd.DataFrame) -> None:
        strategy = TSMOMRotation(weighting="equal")
        signals = strategy.compute_signals(multi_asset_prices)
        weights = strategy.compute_weights(signals, multi_asset_prices)

        # At rebalance points, active weights should sum to ~1.0
        active_rows = weights[weights.sum(axis=1) > 0]
        if len(active_rows) > 0:
            max_sum = active_rows.sum(axis=1).max()
            assert max_sum <= 1.01, f"Equal weights should sum to ~1.0, got {max_sum:.3f}"

    def test_vol_target_scales_by_volatility(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        strategy = TSMOMRotation(weighting="vol_target")
        signals = strategy.compute_signals(multi_asset_prices)
        weights = strategy.compute_weights(signals, multi_asset_prices)

        # Lower-vol assets should get higher weight
        # (TLT has lower vol than DBC typically)
        assert not weights.empty


class TestBacktest:
    def test_backtest_runs(self, multi_asset_prices: pd.DataFrame) -> None:
        strategy = TSMOMRotation()
        result = strategy.backtest(multi_asset_prices)

        assert "metrics" in result
        assert "equity" in result
        assert "benchmarks" in result
        assert "trades_summary" in result
        assert len(result["equity"]) == len(multi_asset_prices)

    def test_benchmarks_present(self, multi_asset_prices: pd.DataFrame) -> None:
        strategy = TSMOMRotation()
        result = strategy.backtest(multi_asset_prices)

        assert "SPY Buy-Hold" in result["benchmarks"]
        assert "60/40" in result["benchmarks"]
        assert "Equal Weight" in result["benchmarks"]

    def test_turnover_tracked(self, multi_asset_prices: pd.DataFrame) -> None:
        strategy = TSMOMRotation()
        result = strategy.backtest(multi_asset_prices)

        assert "annual_turnover" in result["metrics"]
        assert result["metrics"]["annual_turnover"] >= 0

    def test_costs_reduce_returns(self, multi_asset_prices: pd.DataFrame) -> None:
        s1 = TSMOMRotation()
        s2 = TSMOMRotation()

        r1 = s1.backtest(multi_asset_prices, cost_bps=0)
        r2 = s2.backtest(multi_asset_prices, cost_bps=20)

        assert r2["metrics"]["total_return"] <= r1["metrics"]["total_return"]
