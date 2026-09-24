"""Tests for strategies/intraday_mean_reversion.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.intraday_mean_reversion import IntradayMeanReversion


class TestFit:
    def test_ou_params_estimated(self, spy_prices: pd.DataFrame) -> None:
        strategy = IntradayMeanReversion()
        params = strategy.fit(spy_prices)
        assert params.theta > 0, "Mean reversion speed should be positive"
        assert params.half_life > 0, "Half-life should be positive"

    def test_short_data(self) -> None:
        short = pd.DataFrame({"close": [100, 101, 99, 100, 102]})
        strategy = IntradayMeanReversion()
        params = strategy.fit(short)
        assert params.half_life == float("inf"), "Too short for meaningful OU estimate"


class TestSignals:
    def test_signals_generated(self, spy_prices: pd.DataFrame) -> None:
        strategy = IntradayMeanReversion(entry_z=1.5)
        strategy.fit(spy_prices)
        signals_df = strategy.generate_signals(spy_prices)

        assert "z_score" in signals_df.columns
        assert "signal" in signals_df.columns
        assert "position" in signals_df.columns
        assert "kill_switch" in signals_df.columns
        assert len(signals_df) == len(spy_prices)

    def test_signals_only_at_extremes(self, spy_prices: pd.DataFrame) -> None:
        strategy = IntradayMeanReversion(entry_z=3.0)  # High threshold = rare signals
        signals_df = strategy.generate_signals(spy_prices)
        entry_signals = (signals_df["signal"] != 0).sum()
        # With z=3.0, signals should be rare
        assert entry_signals < len(spy_prices) * 0.2, "3-sigma entry should be infrequent"

    def test_kill_switch_halts_trading(self) -> None:
        n = 200
        dates = pd.bdate_range("2023-01-02", periods=n)
        # Create a crash scenario: sharp drop then recovery
        close = np.ones(n) * 100
        close[50:60] = np.linspace(100, 85, 10)  # 15% crash
        close[60:70] = np.linspace(85, 95, 10)
        prices = pd.DataFrame({"close": close}, index=dates)

        strategy = IntradayMeanReversion(
            entry_z=1.0, daily_dd_limit=0.01
        )
        signals_df = strategy.generate_signals(prices)
        # Kill switch should activate during the crash
        assert signals_df["kill_switch"].any(), "Kill switch should fire during crash"


class TestBacktest:
    def test_backtest_runs(self, spy_prices: pd.DataFrame) -> None:
        strategy = IntradayMeanReversion(entry_z=1.5)
        result = strategy.backtest(spy_prices)

        assert "metrics" in result
        assert "equity" in result
        assert "ou_params" in result
        assert len(result["equity"]) == len(spy_prices)

    def test_costs_reduce_returns(self, spy_prices: pd.DataFrame) -> None:
        s1 = IntradayMeanReversion(entry_z=1.5)
        s2 = IntradayMeanReversion(entry_z=1.5)

        r1 = s1.backtest(spy_prices, cost_bps=0)
        r2 = s2.backtest(spy_prices, cost_bps=20)

        assert r2["metrics"]["total_return"] <= r1["metrics"]["total_return"]

    def test_equity_starts_at_capital(self, spy_prices: pd.DataFrame) -> None:
        strategy = IntradayMeanReversion()
        result = strategy.backtest(spy_prices, initial_capital=75_000)
        assert result["equity"].iloc[0] == 75_000
