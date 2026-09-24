"""Shared test fixtures for all strategy and utility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_returns() -> pd.Series:
    """Synthetic daily returns: slight positive drift with noise."""
    rng = np.random.RandomState(42)
    n = 504  # ~2 years
    returns = rng.normal(0.0004, 0.01, n)  # ~10% annual, 16% vol
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def equity_curve(daily_returns: pd.Series) -> pd.Series:
    """Equity curve from daily returns starting at $100,000."""
    return (1 + daily_returns).cumprod() * 100_000


@pytest.fixture
def losing_returns() -> pd.Series:
    """Synthetic losing strategy — negative drift."""
    rng = np.random.RandomState(99)
    n = 252
    returns = rng.normal(-0.0005, 0.015, n)
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def trade_returns() -> pd.Series:
    """Synthetic per-trade P&L series."""
    return pd.Series(
        [120, -50, 80, -30, 200, -100, 45, -20, 150, -60, 90, -40],
        name="trade_pnl",
    )


@pytest.fixture
def spy_prices() -> pd.DataFrame:
    """Synthetic SPY-like OHLCV data."""
    rng = np.random.RandomState(42)
    n = 504
    dates = pd.bdate_range("2022-01-03", periods=n)

    close = 450.0
    closes = []
    for _ in range(n):
        close *= 1 + rng.normal(0.0004, 0.01)
        closes.append(close)

    closes_arr = np.array(closes)
    return pd.DataFrame(
        {
            "open": closes_arr * (1 + rng.normal(0, 0.002, n)),
            "high": closes_arr * (1 + abs(rng.normal(0, 0.005, n))),
            "low": closes_arr * (1 - abs(rng.normal(0, 0.005, n))),
            "close": closes_arr,
            "volume": rng.randint(50_000_000, 120_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def cointegrated_pair() -> tuple[pd.Series, pd.Series]:
    """Synthetic cointegrated price pair for pairs trading tests."""
    rng = np.random.RandomState(42)
    n = 504
    dates = pd.bdate_range("2022-01-03", periods=n)

    # Random walk (common factor)
    common = np.cumsum(rng.normal(0, 1, n)) + 100

    # Y = 1.5 * X + stationary noise
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.5)  # AR(1) mean-reverting

    x = pd.Series(common + rng.normal(0, 0.3, n), index=dates, name="X")
    y = pd.Series(1.5 * common + noise + rng.normal(0, 0.3, n), index=dates, name="Y")

    return y, x
