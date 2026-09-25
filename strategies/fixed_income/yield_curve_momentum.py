"""Yield curve momentum strategy on Treasury ETFs.

Trades the slope of the yield curve using Treasury ETFs across
maturities. When the slope steepens beyond historical norm (z > entry),
goes long duration (TLT). When it flattens (z < -entry), goes short
duration (overweight SHY).

The slope is measured as the ratio TLT/SHY, which captures the
yield curve steepness via price differentials across maturities.

Academic basis:
    Diebold, F. X. & Li, C. (2006). "Forecasting the Term Structure
    of Government Bond Yields." Journal of Econometrics.

    Phoa, W. (2001). "The Predictive Power of the Yield Curve."
    Journal of Fixed Income.

Universe: SHY (1-3Y), IEI (3-7Y), IEF (7-10Y), TLT (20+Y).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class YieldCurveMomentum:
    """Yield curve slope momentum on Treasury ETFs.

    Parameters
    ----------
    slope_lookback
        Rolling window for slope z-score (trading days).
    entry_z
        Z-score threshold for entry.
    exit_z
        Z-score threshold for exit.
    vol_target
        Annualized vol target.
    """

    def __init__(
        self,
        slope_lookback: int = 126,
        entry_z: float = 1.0,
        exit_z: float = 0.3,
        vol_target: float = 0.10,
    ) -> None:
        self.slope_lookback = slope_lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.vol_target = vol_target

    def estimate_slope(self, prices: pd.DataFrame) -> pd.Series:
        """Estimate yield curve slope from Treasury ETF ratio.

        Uses TLT/SHY ratio as slope proxy. Rising ratio = steepening.
        """
        if "TLT" not in prices.columns or "SHY" not in prices.columns:
            # Fallback: use the longest and shortest maturity available
            cols = sorted(prices.columns)
            ratio = prices[cols[-1]] / prices[cols[0]]
        else:
            ratio = prices["TLT"] / prices["SHY"]

        return ratio

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate slope momentum signals.

        Long duration when slope is steepening (momentum up).
        Short duration when slope is flattening (momentum down).
        """
        slope = self.estimate_slope(prices)
        n = len(slope)

        # Slope momentum: rate of change of slope
        slope_mom = slope.pct_change(21).fillna(0)  # 1-month change

        # Z-score of slope momentum
        z_scores = np.zeros(n)
        for i in range(self.slope_lookback, n):
            window = slope_mom.iloc[i - self.slope_lookback : i]
            mu = window.mean()
            sigma = window.std()
            if sigma > 0:
                z_scores[i] = (slope_mom.iloc[i] - mu) / sigma

        # Signal generation
        signals = np.zeros(n)
        positions = np.zeros(n)
        position = 0

        for i in range(self.slope_lookback, n):
            z = z_scores[i]

            # Exit
            exited = False
            if position != 0 and abs(z) < self.exit_z:
                position = 0
                exited = True

            # Entry
            if position == 0 and not exited:
                if z > self.entry_z:
                    position = 1  # Long duration (steepening)
                    signals[i] = 1
                elif z < -self.entry_z:
                    position = -1  # Short duration (flattening)
                    signals[i] = -1

            positions[i] = position

        return pd.DataFrame(
            {"slope": slope, "slope_momentum": slope_mom, "z_score": z_scores,
             "signal": signals, "position": positions},
            index=prices.index,
        )

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        cost_bps: float = 3.0,
    ) -> dict[str, Any]:
        """Run yield curve momentum backtest.

        When long duration: overweight TLT, underweight SHY.
        When short duration: overweight SHY, underweight TLT.
        """
        from utils.metrics import compute_all_metrics

        n = len(prices)
        split = n // 2
        signals_df = self.generate_signals(prices)
        positions = signals_df["position"].values

        # Use TLT returns for duration trade, SHY for cash equivalent
        tlt_col = "TLT" if "TLT" in prices.columns else prices.columns[-1]
        shy_col = "SHY" if "SHY" in prices.columns else prices.columns[0]

        tlt_ret = prices[tlt_col].pct_change().fillna(0).values
        shy_ret = prices[shy_col].pct_change().fillna(0).values

        returns_arr = np.zeros(n)
        for i in range(1, n):
            if positions[i - 1] == 1:
                returns_arr[i] = tlt_ret[i]  # Long duration
            elif positions[i - 1] == -1:
                returns_arr[i] = shy_ret[i]  # Short duration (cash proxy)
            else:
                returns_arr[i] = 0.5 * tlt_ret[i] + 0.5 * shy_ret[i]  # Neutral

            if signals_df["signal"].values[i] != 0:
                returns_arr[i] -= cost_bps / 10_000

        # Vol-target
        for i in range(max(self.slope_lookback + 1, 63), n):
            recent_vol = np.std(returns_arr[i - 63 : i]) * np.sqrt(252) if i > 63 else 0.10
            if recent_vol > 0:
                scale = min(3.0, self.vol_target / recent_vol)
                returns_arr[i] *= scale

        equity = initial_capital * np.cumprod(1 + returns_arr)
        equity_series = pd.Series(equity, index=prices.index)
        returns_series = pd.Series(returns_arr, index=prices.index)

        oos_returns = returns_series.iloc[split:]
        oos_equity = equity_series.iloc[split:]
        metrics = compute_all_metrics(oos_returns, equity_curve=oos_equity)

        return {
            "metrics": metrics,
            "equity": equity_series,
            "returns": returns_series,
            "signals_df": signals_df,
        }
