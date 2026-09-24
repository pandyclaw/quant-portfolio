"""Tests for utils/metrics.py — performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.metrics import (
    compute_all_metrics,
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


class TestSharpeRatio:
    def test_positive_returns(self, daily_returns: pd.Series) -> None:
        sr = sharpe_ratio(daily_returns)
        assert sr > 0, "Positive drift should produce positive Sharpe"

    def test_zero_volatility(self) -> None:
        flat = pd.Series([0.001] * 100)
        sr = sharpe_ratio(flat)
        assert sr > 0, "Constant positive returns should give positive Sharpe"

    def test_empty_series(self) -> None:
        assert sharpe_ratio(pd.Series([], dtype=float)) == 0.0

    def test_single_return(self) -> None:
        assert sharpe_ratio(pd.Series([0.01])) == 0.0

    def test_annualization(self, daily_returns: pd.Series) -> None:
        daily = sharpe_ratio(daily_returns, annualization=252)
        monthly = sharpe_ratio(daily_returns, annualization=12)
        assert daily != monthly, "Different annualization should give different values"


class TestSortinoRatio:
    def test_higher_than_sharpe_for_positive_skew(self) -> None:
        rng = np.random.RandomState(42)
        # Positive skew: small losses, large wins
        returns = pd.Series(np.concatenate([rng.normal(0.002, 0.005, 200), [0.05, 0.04, 0.03]]))
        sr = sharpe_ratio(returns)
        so = sortino_ratio(returns)
        assert so >= sr, "Sortino should be >= Sharpe for positive-skew returns"

    def test_all_positive_returns(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.005, 0.015])
        sr = sortino_ratio(returns)
        assert sr == 0.0 or sr > 0, "All-positive returns: Sortino should be 0 or positive"


class TestMaxDrawdown:
    def test_no_drawdown(self) -> None:
        equity = pd.Series([100, 101, 102, 103, 104])
        assert max_drawdown(equity) == 0.0

    def test_known_drawdown(self) -> None:
        equity = pd.Series([100, 110, 90, 95, 105])
        mdd = max_drawdown(equity)
        expected = (110 - 90) / 110  # 18.18%
        assert abs(mdd - expected) < 0.001

    def test_single_point(self) -> None:
        assert max_drawdown(pd.Series([100])) == 0.0


class TestMaxDrawdownDuration:
    def test_quick_recovery(self) -> None:
        equity = pd.Series([100, 90, 100, 110])
        dur = max_drawdown_duration(equity)
        assert dur == 1  # Only 1 bar in drawdown before recovery

    def test_extended_drawdown(self) -> None:
        equity = pd.Series([100, 95, 90, 85, 88, 92, 100])
        dur = max_drawdown_duration(equity)
        assert dur == 5  # Bars 1-5 are below peak of 100


class TestProfitFactor:
    def test_profitable(self, trade_returns: pd.Series) -> None:
        pf = profit_factor(trade_returns)
        assert pf > 1.0, "Net profitable trades should have PF > 1"

    def test_all_winners(self) -> None:
        assert profit_factor(pd.Series([10, 20, 30])) == float("inf")

    def test_all_losers(self) -> None:
        assert profit_factor(pd.Series([-10, -20, -30])) == 0.0


class TestWinRate:
    def test_mixed_trades(self, trade_returns: pd.Series) -> None:
        wr = win_rate(trade_returns)
        assert 0 < wr < 1

    def test_empty(self) -> None:
        assert win_rate(pd.Series([], dtype=float)) == 0.0


class TestComputeAllMetrics:
    def test_returns_dict(self, daily_returns: pd.Series) -> None:
        result = compute_all_metrics(daily_returns)
        assert isinstance(result, dict)
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "sortino_ratio" in result
        assert "var_95" in result

    def test_with_benchmark(self, daily_returns: pd.Series) -> None:
        benchmark = daily_returns * 0.8  # Weaker benchmark
        result = compute_all_metrics(daily_returns, benchmark=benchmark)
        assert "information_ratio" in result

    def test_with_trades(self, daily_returns: pd.Series, trade_returns: pd.Series) -> None:
        result = compute_all_metrics(daily_returns, trade_returns=trade_returns)
        assert "profit_factor" in result
        assert "win_rate" in result
