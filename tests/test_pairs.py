"""Tests for strategies/pairs_gold_silver.py — Gold/Silver stat arb."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.pairs_gold_silver import GoldSilverPairs


class TestFit:
    def test_cointegrated_pair_detected(
        self, cointegrated_pair: tuple[pd.Series, pd.Series]
    ) -> None:
        y, x = cointegrated_pair
        strategy = GoldSilverPairs()
        result = strategy.fit(y, x)
        assert result.is_cointegrated, (
            f"Should detect cointegration (p={result.adf_pvalue:.4f})"
        )
        assert result.adf_pvalue < 0.05

    def test_half_life_positive_for_cointegrated(
        self, cointegrated_pair: tuple[pd.Series, pd.Series]
    ) -> None:
        y, x = cointegrated_pair
        strategy = GoldSilverPairs()
        result = strategy.fit(y, x)
        assert result.half_life > 0, "Cointegrated pair should have finite half-life"
        assert result.half_life < 200, "Half-life should be reasonable (< 200 bars)"

    def test_random_walks_not_cointegrated(self) -> None:
        rng = np.random.RandomState(123)
        n = 500
        dates = pd.bdate_range("2022-01-03", periods=n)
        # Two independent random walks — should NOT be cointegrated
        y = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100, index=dates)
        x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50, index=dates)
        strategy = GoldSilverPairs()
        result = strategy.fit(y, x)
        # With high probability, random walks are not cointegrated
        # (not guaranteed, so we use a softer check)
        assert result.adf_pvalue > 0.01 or not result.is_cointegrated

    def test_hurst_below_half_for_mean_reverting(
        self, cointegrated_pair: tuple[pd.Series, pd.Series]
    ) -> None:
        y, x = cointegrated_pair
        strategy = GoldSilverPairs()
        result = strategy.fit(y, x)
        assert result.hurst_exponent < 0.6, (
            f"Mean-reverting spread should have Hurst < 0.5-0.6, got {result.hurst_exponent:.3f}"
        )


class TestSignals:
    def test_signals_generated(
        self, cointegrated_pair: tuple[pd.Series, pd.Series]
    ) -> None:
        y, x = cointegrated_pair
        strategy = GoldSilverPairs(entry_z=1.5)
        strategy.fit(y, x)
        signals_df = strategy.generate_signals(y, x)

        assert "signal" in signals_df.columns
        assert "z_score" in signals_df.columns
        assert "position" in signals_df.columns
        assert len(signals_df) == len(y)

        # Should have some non-zero signals
        non_zero = (signals_df["signal"] != 0).sum()
        assert non_zero > 0, "Should generate at least some signals"

    def test_no_consecutive_entries(
        self, cointegrated_pair: tuple[pd.Series, pd.Series]
    ) -> None:
        y, x = cointegrated_pair
        strategy = GoldSilverPairs()
        strategy.fit(y, x)
        signals_df = strategy.generate_signals(y, x)

        # Position should never jump from +1 to -1 without going through 0
        positions = signals_df["position"].values
        for i in range(1, len(positions)):
            if positions[i - 1] == 1:
                assert positions[i] in (0, 1), "Cannot go from long to short without flat"
            if positions[i - 1] == -1:
                assert positions[i] in (0, -1), "Cannot go from short to long without flat"

    def test_max_holding_enforced(
        self, cointegrated_pair: tuple[pd.Series, pd.Series]
    ) -> None:
        y, x = cointegrated_pair
        max_hold = 20
        strategy = GoldSilverPairs(
            entry_z=1.0, exit_z=0.1, stop_z=10.0, max_holding_days=max_hold
        )
        result = strategy.backtest(y, x)
        # Check actual trade holding days from extracted trades
        for t in result["trades"]:
            assert t["holding_days"] <= max_hold + 2, (
                f"Trade held {t['holding_days']} days, max should be ~{max_hold}"
            )


class TestBacktest:
    def test_backtest_runs(
        self, cointegrated_pair: tuple[pd.Series, pd.Series]
    ) -> None:
        y, x = cointegrated_pair
        strategy = GoldSilverPairs(entry_z=1.5)
        result = strategy.backtest(y, x, initial_capital=100_000)

        assert "metrics" in result
        assert "equity" in result
        assert "trades" in result
        assert "fit_result" in result
        assert isinstance(result["equity"], pd.Series)
        assert len(result["equity"]) == len(y)

    def test_costs_reduce_returns(
        self, cointegrated_pair: tuple[pd.Series, pd.Series]
    ) -> None:
        y, x = cointegrated_pair
        strategy_no_cost = GoldSilverPairs(entry_z=1.5)
        strategy_cost = GoldSilverPairs(entry_z=1.5)

        result_no_cost = strategy_no_cost.backtest(y, x, cost_bps=0)
        result_cost = strategy_cost.backtest(y, x, cost_bps=20)

        # With costs, total return should be lower
        ret_no_cost = result_no_cost["metrics"]["total_return"]
        ret_cost = result_cost["metrics"]["total_return"]
        assert ret_cost <= ret_no_cost, "Transaction costs should reduce returns"

    def test_equity_starts_at_capital(
        self, cointegrated_pair: tuple[pd.Series, pd.Series]
    ) -> None:
        y, x = cointegrated_pair
        capital = 50_000
        strategy = GoldSilverPairs()
        result = strategy.backtest(y, x, initial_capital=capital)
        assert result["equity"].iloc[0] == capital
