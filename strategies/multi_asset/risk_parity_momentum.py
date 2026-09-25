"""Risk Parity with Momentum Overlay.

Combines two well-documented premia in a single portfolio:
1. Risk parity: inverse-vol weighting to equalize risk contribution
   across asset classes (diversification benefit)
2. Momentum overlay: reduce weight to zero for assets with negative
   6-month momentum (avoid trend losers)

This creates a defensive, regime-aware multi-asset portfolio that
avoids the worst drawdowns of pure risk parity (which suffers in
rising rate environments like 2022) while maintaining diversification.

Academic basis:
    Asness, C., Frazzini, A. & Pedersen, L. (2012). "Leverage
    Aversion and Risk Parity." Financial Analysts Journal.

    Roncalli, T. (2013). "Introduction to Risk Parity and Budgeting."
    Chapman & Hall.

    Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012).
    "Time Series Momentum." Journal of Financial Economics.

Universe: SPY, TLT, GLD, DBC, EFA, IEF.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class RiskParityMomentum:
    """Risk parity allocation with TSMOM filter.

    Parameters
    ----------
    vol_lookback
        Volatility estimation window.
    mom_lookback
        Momentum evaluation window.
    rebalance_freq
        Rebalance frequency in trading days.
    vol_target
        Portfolio-level annualized vol target.
    """

    def __init__(
        self,
        vol_lookback: int = 63,
        mom_lookback: int = 126,
        rebalance_freq: int = 21,
        vol_target: float = 0.10,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.mom_lookback = mom_lookback
        self.rebalance_freq = rebalance_freq
        self.vol_target = vol_target

    def compute_risk_parity_weights(
        self, returns: pd.DataFrame, lookback: int = 63
    ) -> pd.Series:
        """Compute inverse-vol weights (risk parity).

        Each asset gets weight proportional to 1/vol so all assets
        contribute equal risk to the portfolio.
        """
        vols = returns.iloc[-lookback:].std() * np.sqrt(252)
        inv_vol = 1.0 / vols.replace(0, np.nan)
        inv_vol = inv_vol.fillna(0)
        total = inv_vol.sum()
        if total > 0:
            return inv_vol / total
        return pd.Series(1.0 / len(returns.columns), index=returns.columns)

    def apply_momentum_filter(
        self, weights: pd.Series, prices: pd.DataFrame, idx: int
    ) -> pd.Series:
        """Zero out assets with negative momentum.

        If 6-month return is negative, set weight to zero and
        redistribute to remaining assets.
        """
        filtered = weights.copy()
        for col in prices.columns:
            if idx >= self.mom_lookback:
                mom = (prices[col].iloc[idx] - prices[col].iloc[idx - self.mom_lookback]) / prices[
                    col
                ].iloc[idx - self.mom_lookback]
                if mom < 0:
                    filtered[col] = 0.0

        # Renormalize
        total = filtered.sum()
        if total > 0:
            filtered = filtered / total
        return filtered

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        cost_bps: float = 5.0,
    ) -> dict[str, Any]:
        """Run risk parity + momentum backtest with benchmarks."""
        from utils.metrics import compute_all_metrics

        n = len(prices)
        split = n // 2
        returns = prices.pct_change().fillna(0)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights_pure_rp = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        start = max(self.vol_lookback, self.mom_lookback)

        for i in range(start, n):
            if (i - start) % self.rebalance_freq != 0:
                if i > start:
                    weights.iloc[i] = weights.iloc[i - 1]
                    weights_pure_rp.iloc[i] = weights_pure_rp.iloc[i - 1]
                continue

            # Base: risk parity weights
            rp = self.compute_risk_parity_weights(returns.iloc[:i], self.vol_lookback)
            weights_pure_rp.iloc[i] = rp

            # Overlay: momentum filter
            filtered = self.apply_momentum_filter(rp, prices, i)

            # Scale to vol target
            port_vol = float((filtered * returns.iloc[i - 63 : i].std() * np.sqrt(252)).sum())
            if port_vol > 0:
                scale = min(2.0, self.vol_target / port_vol)
                filtered = filtered * scale

            weights.iloc[i] = filtered

        # Forward-fill
        is_rebal = weights.abs().sum(axis=1) > 0
        for i in range(1, len(weights)):
            if not is_rebal.iloc[i]:
                weights.iloc[i] = weights.iloc[i - 1]
                weights_pure_rp.iloc[i] = weights_pure_rp.iloc[i - 1]

        # Portfolio returns
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
        pure_rp_returns = (weights_pure_rp.shift(1) * returns).sum(axis=1)

        # Costs
        weight_changes = weights.diff().abs().sum(axis=1).fillna(0)
        costs = weight_changes * cost_bps / 10_000
        portfolio_returns = portfolio_returns - costs

        equity = initial_capital * (1 + portfolio_returns).cumprod()
        equity_series = pd.Series(equity.values, index=prices.index)

        rp_equity = initial_capital * (1 + pure_rp_returns).cumprod()

        # Benchmarks
        benchmarks = {
            "Pure Risk Parity": pd.Series(rp_equity.values, index=prices.index),
        }
        ew_ret = returns.mean(axis=1)
        benchmarks["Equal Weight"] = pd.Series(
            (initial_capital * (1 + ew_ret).cumprod()).values, index=prices.index
        )

        # OOS metrics
        oos_returns = portfolio_returns.iloc[split:]
        oos_equity = equity_series.iloc[split:]
        metrics = compute_all_metrics(oos_returns, equity_curve=oos_equity)
        metrics["annual_turnover"] = float(weight_changes.mean() * 252)
        metrics["avg_active_assets"] = float((weights > 0.01).sum(axis=1).mean())
        metrics["momentum_filter_rate"] = float(
            1 - (weights > 0.01).sum(axis=1).mean() / len(prices.columns)
        )

        return {
            "metrics": metrics,
            "equity": equity_series,
            "returns": portfolio_returns,
            "weights": weights,
            "benchmarks": benchmarks,
        }
