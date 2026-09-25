"""Abstract base class for all trading strategies.

Every strategy in this repository follows this protocol:
    fit()              — calibrate on training data
    generate_signals() — produce trading signals
    backtest()         — run full backtest with costs and validation

This ensures consistent interfaces across asset classes and
enables the portfolio construction module to combine strategies
programmatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class StrategyBase(ABC):
    """Abstract base for systematic trading strategies.

    Subclasses must define metadata properties (name, asset_class, style)
    and implement the three core methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'pairs_gold_silver')."""

    @property
    @abstractmethod
    def asset_class(self) -> str:
        """Asset class: 'equities', 'futures', 'fixed_income', 'volatility', 'multi_asset'."""

    @property
    @abstractmethod
    def style(self) -> str:
        """Style premium: 'momentum', 'mean_reversion', 'carry', 'volatility', 'relative_value'."""

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Calibrate the strategy on training data.

        Returns diagnostics (e.g., cointegration test result, OU parameters).
        """

    @abstractmethod
    def generate_signals(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Generate trading signals from price data.

        Returns a DataFrame with at minimum 'signal' and 'position' columns.
        Signals are generated using only information available at time t
        (no look-ahead).
        """

    @abstractmethod
    def backtest(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run a full backtest with transaction costs.

        Returns a dict with at minimum:
            'metrics': dict of performance metrics
            'equity': pd.Series equity curve
            'returns': pd.Series return series
        """
