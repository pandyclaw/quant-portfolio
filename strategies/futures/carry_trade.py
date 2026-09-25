"""Futures carry trade across asset classes.

Carry is the expected return from holding an asset assuming prices
don't change. In futures, carry = (front_price - back_price) / front,
which captures the term structure slope (contango vs backwardation).

Implemented using ETF proxies since actual futures data requires
paid subscriptions. The carry signal is estimated from rolling
return differentials between short and long-duration ETFs.

Academic basis:
    Koijen, R., Moskowitz, T., Pedersen, L. & Vrugt, E. (2018).
    "Carry." Journal of Financial Economics.

    Erb, C. & Harvey, C. (2006). "The Strategic and Tactical Value
    of Commodity Futures." Financial Analysts Journal.

Universe: USO (crude), GLD (gold), TLT (bonds), UUP (dollar).
Carry proxy: rolling 1-month return as term structure estimate.

Limitation: ETF carry is a proxy — real carry uses futures roll yield.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FuturesCarry:
    """Carry trade across asset classes using ETF proxies.

    Parameters
    ----------
    carry_lookback
        Window for estimating carry (trading days).
    rebalance_freq
        Rebalance frequency in trading days.
    vol_target
        Annualized vol target per position.
    """

    def __init__(
        self,
        carry_lookback: int = 63,
        rebalance_freq: int = 21,
        vol_target: float = 0.10,
    ) -> None:
        self.carry_lookback = carry_lookback
        self.rebalance_freq = rebalance_freq
        self.vol_target = vol_target

    def estimate_carry(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Estimate carry signal from rolling returns.

        Carry proxy = 1-month rolling return (captures term structure).
        Positive carry → backwardation (favorable for longs).
        Negative carry → contango (unfavorable).
        """
        carry = prices.pct_change(21).fillna(0)  # 1-month rolling return
        # Normalize by rolling volatility to get carry/vol ratio
        vol = prices.pct_change().rolling(self.carry_lookback).std() * np.sqrt(252)
        carry_vol = carry / vol.replace(0, np.nan)
        return carry_vol.fillna(0)

    def compute_weights(
        self, carry_signals: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute weights: long positive carry, short negative carry."""
        returns = prices.pct_change().fillna(0)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n = len(prices)

        for i in range(self.carry_lookback, n):
            if (i - self.carry_lookback) % self.rebalance_freq != 0:
                if i > self.carry_lookback:
                    weights.iloc[i] = weights.iloc[i - 1]
                continue

            row = carry_signals.iloc[i]
            active = [c for c in prices.columns if np.isfinite(row[c]) and abs(row[c]) > 0.1]

            if not active:
                continue

            for c in active:
                direction = 1.0 if row[c] > 0 else -1.0
                start = max(0, i - 63)
                vol = returns[c].iloc[start:i].std() * np.sqrt(252)
                if vol > 0:
                    w = (self.vol_target / max(len(active), 1)) / vol
                    weights.loc[weights.index[i], c] = direction * min(w, 0.40)

        # Forward-fill between rebalances
        is_rebal = weights.abs().sum(axis=1) > 0
        for i in range(1, len(weights)):
            if not is_rebal.iloc[i]:
                weights.iloc[i] = weights.iloc[i - 1]

        return weights

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        cost_bps: float = 5.0,
    ) -> dict[str, Any]:
        """Run full carry trade backtest."""
        from utils.metrics import compute_all_metrics

        n = len(prices)
        split = n // 2

        carry_signals = self.estimate_carry(prices)
        weights = self.compute_weights(carry_signals, prices)

        asset_returns = prices.pct_change().fillna(0)
        portfolio_returns = (weights.shift(1) * asset_returns).sum(axis=1)

        weight_changes = weights.diff().abs().sum(axis=1).fillna(0)
        costs = weight_changes * cost_bps / 10_000
        portfolio_returns = portfolio_returns - costs

        equity = initial_capital * (1 + portfolio_returns).cumprod()
        equity_series = pd.Series(equity.values, index=prices.index)

        oos_returns = portfolio_returns.iloc[split:]
        oos_equity = equity_series.iloc[split:]

        metrics = compute_all_metrics(oos_returns, equity_curve=oos_equity)
        metrics["annual_turnover"] = float(weight_changes.mean() * 252)

        return {
            "metrics": metrics,
            "equity": equity_series,
            "returns": portfolio_returns,
            "weights": weights,
            "carry_signals": carry_signals,
        }
