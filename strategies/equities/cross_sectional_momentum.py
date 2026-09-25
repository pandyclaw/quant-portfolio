"""Cross-sectional equity momentum across US sector ETFs.

Ranks sectors by 12-1 month momentum (same lookback as TSMOM but
applied cross-sectionally — assets are ranked against each other,
not evaluated independently). Goes long the top-N sectors and
optionally short the bottom-N.

This is the most replicated anomaly in asset pricing: winners
continue winning and losers continue losing over 3-12 month
horizons. The 1-month skip avoids short-term reversal effects.

Academic basis:
    Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners
    and Selling Losers." Journal of Finance.

    Asness, C., Moskowitz, T. & Pedersen, L. (2013). "Value and
    Momentum Everywhere." Journal of Finance.

Universe: 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLU, XLP,
XLY, XLI, XLB, XLRE, XLC).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class CrossSectionalMomentum:
    """Cross-sectional momentum on US sector ETFs.

    Parameters
    ----------
    lookback_long
        Long-term momentum window (trading days, default 252 = 12M).
    lookback_short
        Short-term skip window (default 21 = 1M).
    n_long
        Number of top sectors to go long.
    n_short
        Number of bottom sectors to short (0 = long-only).
    rebalance_freq
        Rebalance frequency in trading days.
    vol_target
        Annualized vol target per position.
    """

    def __init__(
        self,
        lookback_long: int = 252,
        lookback_short: int = 21,
        n_long: int = 3,
        n_short: int = 0,
        rebalance_freq: int = 21,
        vol_target: float = 0.10,
    ) -> None:
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_freq = rebalance_freq
        self.vol_target = vol_target

    def compute_rankings(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum score and rank for each sector at each date.

        Momentum = return from t-252 to t-21 (12M minus 1M skip).
        Rank 1 = strongest momentum, Rank N = weakest.
        """
        n = len(prices)
        momentum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for i in range(self.lookback_long, n):
            for col in prices.columns:
                p = prices[col].values
                # 12M return minus 1M (skip recent month)
                ret = (p[i - self.lookback_short] - p[i - self.lookback_long]) / p[
                    i - self.lookback_long
                ]
                momentum.loc[momentum.index[i], col] = ret

        return momentum

    def compute_weights(
        self, rankings: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute portfolio weights from momentum rankings."""
        returns = prices.pct_change().fillna(0)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n_assets = len(prices.columns)

        for i in range(self.lookback_long, len(prices)):
            if (i - self.lookback_long) % self.rebalance_freq != 0:
                if i > self.lookback_long:
                    weights.iloc[i] = weights.iloc[i - 1]
                continue

            row = rankings.iloc[i]
            ranked = row.sort_values(ascending=False)

            # Long top-N
            long_assets = ranked.index[: self.n_long].tolist()
            # Short bottom-N
            short_assets = ranked.index[-self.n_short :].tolist() if self.n_short > 0 else []

            # Vol-targeted weights
            for c in long_assets:
                start = max(0, i - 63)
                vol = returns[c].iloc[start:i].std() * np.sqrt(252)
                if vol > 0:
                    w = (self.vol_target / len(long_assets)) / vol
                    weights.loc[weights.index[i], c] = min(w, 0.40)

            for c in short_assets:
                start = max(0, i - 63)
                vol = returns[c].iloc[start:i].std() * np.sqrt(252)
                if vol > 0:
                    w = (self.vol_target / len(short_assets)) / vol
                    weights.loc[weights.index[i], c] = -min(w, 0.40)

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
        """Run full backtest with benchmarks."""
        from utils.metrics import compute_all_metrics

        n = len(prices)
        split = n // 2

        rankings = self.compute_rankings(prices)
        weights = self.compute_weights(rankings, prices)

        asset_returns = prices.pct_change().fillna(0)
        portfolio_returns = (weights.shift(1) * asset_returns).sum(axis=1)

        # Rebalance costs
        weight_changes = weights.diff().abs().sum(axis=1).fillna(0)
        costs = weight_changes * cost_bps / 10_000
        portfolio_returns = portfolio_returns - costs

        equity = initial_capital * (1 + portfolio_returns).cumprod()
        equity_series = pd.Series(equity.values, index=prices.index)

        # OOS metrics
        oos_returns = portfolio_returns.iloc[split:]
        oos_equity = equity_series.iloc[split:]

        metrics = compute_all_metrics(oos_returns, equity_curve=oos_equity)
        metrics["annual_turnover"] = float(weight_changes.mean() * 252)
        metrics["avg_long_count"] = float((weights > 0.01).sum(axis=1).mean())

        # Top/bottom sector frequency
        sector_longs = {}
        for col in prices.columns:
            sector_longs[col] = int((weights[col] > 0.01).sum())

        return {
            "metrics": metrics,
            "equity": equity_series,
            "returns": portfolio_returns,
            "weights": weights,
            "rankings": rankings,
            "sector_activity": sector_longs,
        }
