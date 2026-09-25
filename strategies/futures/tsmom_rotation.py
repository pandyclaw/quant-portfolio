"""Cross-asset time-series momentum (TSMOM) rotation.

Applies the Moskowitz, Ooi & Pedersen (2012) TSMOM framework across
a diversified asset class universe: equities (SPY), bonds (TLT),
gold (GLD), commodities (DBC), and international equities (EFA).

Each asset is evaluated independently: go long if 12-month return
minus 1-month return is positive, flat otherwise. Positions are
vol-targeted so each asset contributes equal risk to the portfolio.

This is a medium-frequency strategy (monthly rebalance) with zero
security selection — pure systematic momentum exposure across
asset classes.

Academic basis:
    Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012).
    "Time Series Momentum." Journal of Financial Economics.

    Baltas, A. N. & Kosowski, R. (2013). "Momentum Strategies in
    Futures Markets and Trend-Following Funds." SSRN.

    Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2017).
    "A Century of Evidence on Trend-Following Investing." AQR.

Signal construction:
    momentum_signal = sign(return_12m - return_1m)
    This "12-1" lookback skips the most recent month to avoid
    short-term reversal effects (Jegadeesh & Titman, 1993).

Portfolio construction:
    - Vol-targeted: each asset sized to contribute target_vol/N
    - Risk parity: inverse-vol weighting as comparison
    - Equal weight: 1/N allocation as baseline
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class TSMOMRotation:
    """Time-series momentum rotation across asset classes.

    Parameters
    ----------
    lookback_long
        Long-term momentum window (trading days, default 252 = 12 months).
    lookback_short
        Short-term skip window (default 21 = 1 month).
    rebalance_freq
        Rebalance frequency in trading days (default 21 = monthly).
    vol_target
        Target annualized volatility per asset (default 0.10 = 10%).
    vol_lookback
        Volatility estimation window (default 63 = 3 months).
    weighting
        Position sizing method: 'vol_target', 'equal', or 'risk_parity'.
    """

    def __init__(
        self,
        lookback_long: int = 252,
        lookback_short: int = 21,
        rebalance_freq: int = 21,
        vol_target: float = 0.10,
        vol_lookback: int = 63,
        weighting: str = "vol_target",
    ) -> None:
        self.lookback_long = lookback_long
        self.lookback_short = lookback_short
        self.rebalance_freq = rebalance_freq
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.weighting = weighting

    def compute_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute TSMOM signal for each asset at each rebalance date.

        signal = sign(return_12m - return_1m)
        +1 = positive momentum (go long)
         0 = negative momentum (stay flat)

        Long-only: we do not short ETFs (borrow costs, uptick rules).

        Parameters
        ----------
        prices
            DataFrame of adjusted close prices, one column per asset.

        Returns
        -------
        pd.DataFrame
            Binary signals (1 or 0) for each asset at each date.
        """
        n = len(prices)
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        for col in prices.columns:
            p = prices[col].values.astype(float)
            for i in range(self.lookback_long, n):
                # 12-month return minus 1-month return (skip recent month)
                ret_long = (p[i - self.lookback_short] - p[i - self.lookback_long]) / p[
                    i - self.lookback_long
                ]
                signals.loc[signals.index[i], col] = 1 if ret_long > 0 else 0

        return signals

    def compute_weights(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute portfolio weights from signals and volatility.

        Parameters
        ----------
        signals
            Binary TSMOM signals (1/0 per asset).
        prices
            Price DataFrame for volatility estimation.

        Returns
        -------
        pd.DataFrame
            Portfolio weights for each asset at each date.
        """
        returns = prices.pct_change().fillna(0)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for i in range(self.lookback_long, len(prices)):
            # Only rebalance at intervals
            if (i - self.lookback_long) % self.rebalance_freq != 0:
                if i > self.lookback_long:
                    weights.iloc[i] = weights.iloc[i - 1]
                continue

            active_assets = [c for c in prices.columns if signals.loc[signals.index[i], c] > 0]

            if not active_assets:
                continue

            if self.weighting == "equal":
                w = 1.0 / len(active_assets)
                for c in active_assets:
                    weights.loc[weights.index[i], c] = w

            elif self.weighting == "risk_parity":
                vols = {}
                for c in active_assets:
                    start = max(0, i - self.vol_lookback)
                    vol = returns[c].iloc[start:i].std() * np.sqrt(252)
                    vols[c] = max(vol, 0.01)
                inv_vol = {c: 1.0 / v for c, v in vols.items()}
                total = sum(inv_vol.values())
                for c in active_assets:
                    weights.loc[weights.index[i], c] = inv_vol[c] / total

            else:  # vol_target
                per_asset_target = self.vol_target / max(len(active_assets), 1)
                for c in active_assets:
                    start = max(0, i - self.vol_lookback)
                    vol = returns[c].iloc[start:i].std() * np.sqrt(252)
                    if vol > 0:
                        w = per_asset_target / vol
                        weights.loc[weights.index[i], c] = min(w, 0.5)  # Cap at 50% per asset

        # Forward-fill weights between rebalances
        # Only forward-fill rows that are all-zero (between rebalances)
        is_rebalance = weights.sum(axis=1) > 0
        for i in range(1, len(weights)):
            if not is_rebalance.iloc[i]:
                weights.iloc[i] = weights.iloc[i - 1]

        return weights

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        cost_bps: float = 5.0,
    ) -> dict[str, Any]:
        """Run a full backtest with comparison benchmarks.

        Benchmarks:
            - Buy-and-hold SPY (if SPY is in universe)
            - 60/40 SPY/TLT (if both in universe)
            - Equal-weight buy-and-hold

        Parameters
        ----------
        prices
            Adjusted close prices for all assets.
        initial_capital
            Starting capital.
        cost_bps
            Round-trip transaction cost per rebalance.

        Returns
        -------
        dict
            Keys: metrics, equity, returns, weights, signals,
            benchmarks, trades_summary
        """
        from utils.metrics import compute_all_metrics

        n = len(prices)
        split = n // 2

        # Compute signals and weights
        signals = self.compute_signals(prices)
        weights = self.compute_weights(signals, prices)

        # Portfolio returns
        asset_returns = prices.pct_change().fillna(0)
        portfolio_returns = (weights.shift(1) * asset_returns).sum(axis=1)  # T+1 execution

        # Apply rebalance costs
        weight_changes = weights.diff().abs().sum(axis=1).fillna(0)
        costs = weight_changes * cost_bps / 10_000
        portfolio_returns = portfolio_returns - costs

        returns_series = portfolio_returns
        equity = initial_capital * (1 + returns_series).cumprod()
        equity_series = pd.Series(equity.values, index=prices.index)

        # Benchmarks
        benchmarks: dict[str, pd.Series] = {}

        if "SPY" in prices.columns:
            spy_ret = asset_returns["SPY"]
            spy_equity = initial_capital * (1 + spy_ret).cumprod()
            benchmarks["SPY Buy-Hold"] = spy_equity

        if "SPY" in prices.columns and "TLT" in prices.columns:
            sixty_forty = 0.6 * asset_returns["SPY"] + 0.4 * asset_returns["TLT"]
            sf_equity = initial_capital * (1 + sixty_forty).cumprod()
            benchmarks["60/40"] = sf_equity

        ew_ret = asset_returns.mean(axis=1)
        ew_equity = initial_capital * (1 + ew_ret).cumprod()
        benchmarks["Equal Weight"] = ew_equity

        # OOS metrics
        oos_returns = returns_series.iloc[split:]
        oos_equity = equity_series.iloc[split:]

        spy_benchmark = asset_returns.get("SPY")
        oos_spy = spy_benchmark.iloc[split:] if spy_benchmark is not None else None

        metrics = compute_all_metrics(
            oos_returns,
            equity_curve=oos_equity,
            benchmark=oos_spy,
        )

        # Turnover
        metrics["annual_turnover"] = float(weight_changes.mean() * 252)
        metrics["rebalance_count"] = int((weight_changes > 0.01).sum())
        metrics["avg_active_assets"] = float((weights > 0.01).sum(axis=1).mean())

        # Trades summary
        trades_summary: list[dict[str, Any]] = []
        for col in prices.columns:
            col_signals = signals[col]
            entries = ((col_signals == 1) & (col_signals.shift(1) == 0)).sum()
            active_pct = (col_signals > 0).mean() * 100
            trades_summary.append({
                "asset": col,
                "entries": int(entries),
                "active_pct": round(float(active_pct), 1),
            })

        return {
            "metrics": metrics,
            "equity": equity_series,
            "returns": returns_series,
            "weights": weights,
            "signals": signals,
            "benchmarks": benchmarks,
            "trades_summary": trades_summary,
        }
