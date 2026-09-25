"""Tests for all 5 new strategies (D through H)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sector_prices() -> pd.DataFrame:
    """Synthetic sector ETF prices."""
    rng = np.random.RandomState(42)
    n = 756
    dates = pd.bdate_range("2021-01-04", periods=n)
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLU", "XLP"]
    data = {}
    for i, s in enumerate(sectors):
        drift = 0.0003 + (i - 3) * 0.0001
        data[s] = 100 * np.cumprod(1 + rng.normal(drift, 0.012, n))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def carry_prices() -> pd.DataFrame:
    """Synthetic carry trade universe."""
    rng = np.random.RandomState(42)
    n = 756
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({
        "USO": 60 * np.cumprod(1 + rng.normal(0.0002, 0.018, n)),
        "GLD": 170 * np.cumprod(1 + rng.normal(0.0002, 0.010, n)),
        "TLT": 140 * np.cumprod(1 + rng.normal(-0.0001, 0.009, n)),
        "UUP": 25 * np.cumprod(1 + rng.normal(0.0001, 0.005, n)),
    }, index=dates)


@pytest.fixture
def treasury_prices() -> pd.DataFrame:
    """Synthetic Treasury ETF prices."""
    rng = np.random.RandomState(42)
    n = 756
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({
        "SHY": 85 * np.cumprod(1 + rng.normal(0.00005, 0.002, n)),
        "IEI": 110 * np.cumprod(1 + rng.normal(0.0001, 0.004, n)),
        "IEF": 105 * np.cumprod(1 + rng.normal(-0.0001, 0.006, n)),
        "TLT": 140 * np.cumprod(1 + rng.normal(-0.0002, 0.010, n)),
    }, index=dates)


@pytest.fixture
def spy_vix_data() -> tuple[pd.Series, pd.Series]:
    """Synthetic SPY + VIX data."""
    rng = np.random.RandomState(42)
    n = 756
    dates = pd.bdate_range("2021-01-04", periods=n)
    spy = pd.Series(450 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)), index=dates, name="SPY")
    vix = pd.Series(np.clip(20 + np.cumsum(rng.normal(0, 0.5, n)), 10, 60), index=dates, name="VIX")
    return spy, vix


@pytest.fixture
def multi_asset_prices() -> pd.DataFrame:
    """Synthetic multi-asset universe."""
    rng = np.random.RandomState(42)
    n = 756
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({
        "SPY": 400 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)),
        "TLT": 150 * np.cumprod(1 + rng.normal(-0.0002, 0.008, n)),
        "GLD": 170 * np.cumprod(1 + rng.normal(0.0002, 0.010, n)),
        "DBC": 25 * np.cumprod(1 + rng.normal(0.0001, 0.015, n)),
        "EFA": 75 * np.cumprod(1 + rng.normal(0.0001, 0.011, n)),
        "IEF": 105 * np.cumprod(1 + rng.normal(-0.0001, 0.006, n)),
    }, index=dates)


class TestCrossSectionalMomentum:
    def test_backtest_runs(self, sector_prices: pd.DataFrame) -> None:
        from strategies.equities.cross_sectional_momentum import CrossSectionalMomentum
        s = CrossSectionalMomentum(n_long=2)
        result = s.backtest(sector_prices)
        assert "metrics" in result
        assert "equity" in result
        assert len(result["equity"]) == len(sector_prices)

    def test_rankings_computed(self, sector_prices: pd.DataFrame) -> None:
        from strategies.equities.cross_sectional_momentum import CrossSectionalMomentum
        s = CrossSectionalMomentum()
        rankings = s.compute_rankings(sector_prices)
        assert rankings.shape == sector_prices.shape

    def test_costs_reduce_returns(self, sector_prices: pd.DataFrame) -> None:
        from strategies.equities.cross_sectional_momentum import CrossSectionalMomentum
        r0 = CrossSectionalMomentum(n_long=2).backtest(sector_prices, cost_bps=0)
        r20 = CrossSectionalMomentum(n_long=2).backtest(sector_prices, cost_bps=20)
        assert r20["metrics"]["total_return"] <= r0["metrics"]["total_return"]


class TestFuturesCarry:
    def test_backtest_runs(self, carry_prices: pd.DataFrame) -> None:
        from strategies.futures.carry_trade import FuturesCarry
        s = FuturesCarry()
        result = s.backtest(carry_prices)
        assert "metrics" in result
        assert "carry_signals" in result

    def test_carry_signals_computed(self, carry_prices: pd.DataFrame) -> None:
        from strategies.futures.carry_trade import FuturesCarry
        s = FuturesCarry()
        carry = s.estimate_carry(carry_prices)
        assert carry.shape == carry_prices.shape


class TestYieldCurveMomentum:
    def test_backtest_runs(self, treasury_prices: pd.DataFrame) -> None:
        from strategies.fixed_income.yield_curve_momentum import YieldCurveMomentum
        s = YieldCurveMomentum()
        result = s.backtest(treasury_prices)
        assert "metrics" in result
        assert "signals_df" in result

    def test_slope_estimated(self, treasury_prices: pd.DataFrame) -> None:
        from strategies.fixed_income.yield_curve_momentum import YieldCurveMomentum
        s = YieldCurveMomentum()
        slope = s.estimate_slope(treasury_prices)
        assert len(slope) == len(treasury_prices)


class TestVarianceRiskPremium:
    def test_backtest_runs(self, spy_vix_data: tuple[pd.Series, pd.Series]) -> None:
        from strategies.volatility.variance_risk_premium import VarianceRiskPremium
        spy, vix = spy_vix_data
        s = VarianceRiskPremium()
        result = s.backtest(spy, vix)
        assert "metrics" in result
        assert result["metrics"]["time_in_market"] >= 0

    def test_vrp_computed(self, spy_vix_data: tuple[pd.Series, pd.Series]) -> None:
        from strategies.volatility.variance_risk_premium import VarianceRiskPremium
        spy, vix = spy_vix_data
        s = VarianceRiskPremium()
        vrp_df = s.compute_vrp(spy, vix)
        assert "vrp" in vrp_df.columns
        assert "implied_vol" in vrp_df.columns


class TestRiskParityMomentum:
    def test_backtest_runs(self, multi_asset_prices: pd.DataFrame) -> None:
        from strategies.multi_asset.risk_parity_momentum import RiskParityMomentum
        s = RiskParityMomentum()
        result = s.backtest(multi_asset_prices)
        assert "metrics" in result
        assert "benchmarks" in result
        assert "Pure Risk Parity" in result["benchmarks"]

    def test_weights_sum_reasonable(self, multi_asset_prices: pd.DataFrame) -> None:
        from strategies.multi_asset.risk_parity_momentum import RiskParityMomentum
        s = RiskParityMomentum()
        returns = multi_asset_prices.pct_change().fillna(0)
        w = s.compute_risk_parity_weights(returns)
        assert abs(w.sum() - 1.0) < 0.01, f"Risk parity weights should sum to 1, got {w.sum():.3f}"

    def test_momentum_filter_reduces_assets(self, multi_asset_prices: pd.DataFrame) -> None:
        from strategies.multi_asset.risk_parity_momentum import RiskParityMomentum
        s = RiskParityMomentum()
        result = s.backtest(multi_asset_prices)
        # Some assets should be filtered out by momentum
        assert result["metrics"]["avg_active_assets"] < len(multi_asset_prices.columns)

    def test_costs_reduce_returns(self, multi_asset_prices: pd.DataFrame) -> None:
        from strategies.multi_asset.risk_parity_momentum import RiskParityMomentum
        r0 = RiskParityMomentum().backtest(multi_asset_prices, cost_bps=0)
        r20 = RiskParityMomentum().backtest(multi_asset_prices, cost_bps=20)
        assert r20["metrics"]["total_return"] <= r0["metrics"]["total_return"]
