"""Variance Risk Premium (VRP) strategy.

The variance risk premium is the persistent spread between implied
volatility (what the market expects) and realized volatility (what
actually happens). IV consistently overprices RV by 2-4% annualized,
creating a structural risk premium that can be harvested.

Signal: VRP = IV (VIX) - RV (realized vol on SPY).
When VRP is elevated → sell premium (long SPY as vol-short proxy).
When VRP compresses → reduce exposure (vol regime shift).

Academic basis:
    Carr, P. & Wu, L. (2009). "Variance Risk Premiums."
    Review of Financial Studies.

    Bollerslev, T., Tauchen, G. & Zhou, H. (2009). "Expected Stock
    Returns and Variance Risk Premia." Review of Financial Studies.

Data: VIX index (^VIX) as IV proxy, SPY for RV and execution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class VarianceRiskPremium:
    """Harvest the variance risk premium via VIX/RV spread.

    Parameters
    ----------
    rv_window
        Realized volatility estimation window (trading days).
    vrp_lookback
        Rolling window for VRP z-score normalization.
    entry_z
        Z-score threshold for entry (VRP is elevated).
    exit_z
        Z-score threshold for exit (VRP compressed).
    vol_target
        Annualized vol target for position sizing.
    """

    def __init__(
        self,
        rv_window: int = 21,
        vrp_lookback: int = 252,
        entry_z: float = 0.5,
        exit_z: float = -0.5,
        vol_target: float = 0.10,
    ) -> None:
        self.rv_window = rv_window
        self.vrp_lookback = vrp_lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.vol_target = vol_target

    def compute_vrp(
        self, spy_prices: pd.Series, vix_prices: pd.Series
    ) -> pd.DataFrame:
        """Compute the variance risk premium time series.

        VRP = VIX/100 - realized_vol
        Positive VRP = IV > RV = premium exists = favorable for selling vol.
        """
        # Realized volatility: annualized std of daily log returns
        log_returns = np.log(spy_prices / spy_prices.shift(1))
        realized_vol = log_returns.rolling(self.rv_window).std() * np.sqrt(252)

        # Implied vol: VIX is quoted in annualized % terms
        implied_vol = vix_prices / 100.0

        vrp = implied_vol - realized_vol

        # Z-score of VRP
        vrp_mean = vrp.rolling(self.vrp_lookback).mean()
        vrp_std = vrp.rolling(self.vrp_lookback).std()
        vrp_z = (vrp - vrp_mean) / vrp_std.replace(0, np.nan)

        return pd.DataFrame({
            "implied_vol": implied_vol,
            "realized_vol": realized_vol,
            "vrp": vrp,
            "vrp_z": vrp_z.fillna(0),
        }, index=spy_prices.index)

    def generate_signals(
        self, spy_prices: pd.Series, vix_prices: pd.Series
    ) -> pd.DataFrame:
        """Generate VRP signals.

        Long SPY when VRP is elevated (z > entry_z) — selling vol.
        Flat when VRP compresses (z < exit_z) — vol regime uncertain.
        """
        vrp_df = self.compute_vrp(spy_prices, vix_prices)
        n = len(vrp_df)

        signals = np.zeros(n)
        positions = np.zeros(n)
        position = 0

        for i in range(self.vrp_lookback, n):
            z = vrp_df["vrp_z"].iloc[i]

            exited = False
            if position != 0 and z < self.exit_z:
                position = 0
                exited = True

            if position == 0 and not exited:
                if z > self.entry_z:
                    position = 1  # Long SPY (sell vol proxy)
                    signals[i] = 1

            positions[i] = position

        vrp_df["signal"] = signals
        vrp_df["position"] = positions
        return vrp_df

    def backtest(
        self,
        spy_prices: pd.Series,
        vix_prices: pd.Series,
        initial_capital: float = 100_000.0,
        cost_bps: float = 5.0,
    ) -> dict[str, Any]:
        """Run VRP strategy backtest."""
        from utils.metrics import compute_all_metrics

        signals_df = self.generate_signals(spy_prices, vix_prices)
        n = len(signals_df)
        split = n // 2

        spy_ret = spy_prices.pct_change().fillna(0).values
        positions = signals_df["position"].values
        sigs = signals_df["signal"].values

        returns_arr = np.zeros(n)
        for i in range(1, n):
            if positions[i - 1] != 0:
                returns_arr[i] = positions[i - 1] * spy_ret[i]
            if sigs[i] != 0:
                returns_arr[i] -= cost_bps / 10_000

        # Vol-target
        for i in range(max(self.vrp_lookback + 1, 63), n):
            recent_vol = np.std(returns_arr[i - 63 : i]) * np.sqrt(252)
            if recent_vol > 0:
                scale = min(3.0, self.vol_target / recent_vol)
                returns_arr[i] *= scale

        equity = initial_capital * np.cumprod(1 + returns_arr)
        equity_series = pd.Series(equity, index=spy_prices.index)
        returns_series = pd.Series(returns_arr, index=spy_prices.index)

        oos_returns = returns_series.iloc[split:]
        oos_equity = equity_series.iloc[split:]
        metrics = compute_all_metrics(oos_returns, equity_curve=oos_equity)

        # VRP diagnostics
        vrp = signals_df["vrp"].dropna()
        metrics["avg_vrp"] = float(vrp.mean())
        metrics["vrp_positive_pct"] = float((vrp > 0).mean())
        metrics["time_in_market"] = float((signals_df["position"] != 0).mean())

        return {
            "metrics": metrics,
            "equity": equity_series,
            "returns": returns_series,
            "signals_df": signals_df,
        }
