"""Intraday mean reversion on equity index futures (ES/NQ proxy).

Backtested on daily SPY/QQQ data as a proxy for ES/NQ futures.
A production implementation would use 5-minute bars with RTH session
filters and tick-level execution. This is documented as a limitation.

The strategy exploits short-term price deviations from fair value
(measured by Bollinger Bands and z-score). When price stretches beyond
2 standard deviations, it tends to revert — particularly during
low-volatility, mean-reverting regimes.

Academic basis:
    Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security
    Returns." Journal of Finance.

    Poterba, J. M. & Summers, L. H. (1988). "Mean Reversion in Stock
    Prices: Evidence and Implications." Journal of Financial Economics.

    Ornstein, L. S. & Uhlenbeck, G. E. (1930). Theory of Brownian
    motion — OU process for mean-reversion parameter estimation.

Limitations:
    - Backtested on daily SPY/QQQ as proxy for ES/NQ futures
    - Actual intraday slippage is 1 tick ($12.50/ES, $5.00/NQ)
    - RTH session filter not applied (daily data has no session info)
    - A production system requires 5-min bars from Databento/Polygon
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class OUParams:
    """Ornstein-Uhlenbeck process parameters for the price deviation."""

    theta: float  # Mean reversion speed
    mu: float  # Long-term mean
    sigma: float  # Volatility
    half_life: float  # Half-life in bars


class IntradayMeanReversion:
    """Intraday mean reversion using Bollinger Bands and OU process.

    Parameters
    ----------
    bb_period
        Bollinger Band lookback period.
    bb_std
        Number of standard deviations for bands.
    entry_z
        Z-score threshold for entry (e.g., 2.0 = 2 sigma deviation).
    exit_z
        Z-score threshold for exit (mean reversion target).
    max_holding_bars
        Maximum bars to hold before forced exit.
        78 bars = one RTH session at 5-min frequency.
    daily_dd_limit
        Maximum daily drawdown before kill-switch (fraction, e.g., 0.02 = 2%).
    vol_target
        Annualized volatility target for position sizing.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_holding_bars: int = 78,
        daily_dd_limit: float = 0.02,
        vol_target: float = 0.10,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_holding_bars = max_holding_bars
        self.daily_dd_limit = daily_dd_limit
        self.vol_target = vol_target

        self._ou_params: OUParams | None = None

    def fit(self, prices: pd.DataFrame) -> OUParams:
        """Fit OU process parameters to price deviations from SMA.

        Estimates mean-reversion speed (theta), which determines
        the expected half-life of price dislocations. A half-life
        of 5-20 bars is ideal for intraday mean reversion.

        Parameters
        ----------
        prices
            DataFrame with 'close' column.

        Returns
        -------
        OUParams
            Fitted Ornstein-Uhlenbeck process parameters.
        """
        close = prices["close"].values.astype(float)

        # Deviation from moving average
        sma = pd.Series(close).rolling(self.bb_period).mean().values
        deviation = close - sma

        # Remove NaN from rolling window warm-up
        valid = ~np.isnan(deviation)
        dev = deviation[valid]

        if len(dev) < 20:
            self._ou_params = OUParams(theta=0.0, mu=0.0, sigma=1.0, half_life=float("inf"))
            return self._ou_params

        # AR(1) regression: dev_t = a + b * dev_{t-1} + eps
        dev_lag = dev[:-1]
        dev_delta = np.diff(dev)
        X = sm.add_constant(dev_lag)
        ols = sm.OLS(dev_delta, X).fit()

        b = ols.params[1]
        a = ols.params[0]

        # OU parameters: theta = -b, mu = -a/b, sigma = std(residuals)
        theta = float(-b) if b < 0 else 0.001
        mu = float(-a / b) if abs(b) > 1e-8 else 0.0
        sigma = float(np.std(ols.resid, ddof=1))
        half_life = float(np.log(2) / theta) if theta > 0 else float("inf")

        self._ou_params = OUParams(theta=theta, mu=mu, sigma=sigma, half_life=half_life)
        return self._ou_params

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals from Bollinger Band z-scores.

        Signal logic:
            - LONG when z < -entry_z (oversold, expect reversion up)
            - SHORT when z > +entry_z (overbought, expect reversion down)
            - EXIT when |z| < exit_z (reverted to mean)
            - EXIT when holding > max_holding_bars (time stop)
            - HALT when daily drawdown exceeds limit (kill-switch)

        Parameters
        ----------
        prices
            DataFrame with 'close' column.

        Returns
        -------
        pd.DataFrame
            Columns: sma, upper_band, lower_band, z_score, signal,
            position, kill_switch
        """
        close = prices["close"].astype(float)
        n = len(close)

        # Bollinger Bands
        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std(ddof=1)
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std

        # Z-score: normalized distance from SMA
        z_scores = np.zeros(n)
        for i in range(self.bb_period, n):
            s = float(std.iloc[i])
            if s > 0:
                z_scores[i] = (float(close.iloc[i]) - float(sma.iloc[i])) / s

        # Signal generation with state tracking
        signals = np.zeros(n)
        positions = np.zeros(n)
        kill_switch = np.zeros(n, dtype=bool)
        position = 0
        holding_bars = 0

        # Daily drawdown tracking
        daily_peak = 0.0
        daily_pnl = 0.0
        halted = False

        for i in range(self.bb_period, n):
            z = z_scores[i]
            abs_z = abs(z)

            # Daily drawdown kill-switch
            if i > 0:
                ret = (float(close.iloc[i]) - float(close.iloc[i - 1])) / float(close.iloc[i - 1])
                if position != 0:
                    daily_pnl += position * ret
                daily_peak = max(daily_peak, daily_pnl)
                dd = daily_peak - daily_pnl
                if dd > self.daily_dd_limit:
                    halted = True

            # Reset daily tracking (simplified: reset every 20 bars as proxy for session)
            if i % 20 == 0 and i > 0:
                daily_pnl = 0.0
                daily_peak = 0.0
                halted = False

            kill_switch[i] = halted

            # If halted, force exit and skip entries
            if halted and position != 0:
                signals[i] = -position
                position = 0
                holding_bars = 0

            if halted:
                positions[i] = position
                continue

            # Exit logic
            exited = False
            if position != 0:
                should_exit = (
                    abs_z < self.exit_z
                    or holding_bars >= self.max_holding_bars
                    # Profit target: take profit at 1 sigma reversion
                    or (position == 1 and z > 0)  # Long reverted past mean
                    or (position == -1 and z < 0)  # Short reverted past mean
                )
                if should_exit:
                    signals[i] = -position
                    position = 0
                    holding_bars = 0
                    exited = True

            # Entry logic
            if position == 0 and not exited:
                if z < -self.entry_z:
                    signals[i] = 1  # Long (oversold)
                    position = 1
                    holding_bars = 0
                elif z > self.entry_z:
                    signals[i] = -1  # Short (overbought)
                    position = -1
                    holding_bars = 0

            if position != 0:
                holding_bars += 1
            positions[i] = position

        return pd.DataFrame(
            {
                "sma": sma,
                "upper_band": upper,
                "lower_band": lower,
                "z_score": z_scores,
                "signal": signals,
                "position": positions,
                "kill_switch": kill_switch,
            },
            index=prices.index,
        )

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        cost_bps: float = 5.0,
    ) -> dict[str, Any]:
        """Run a full backtest with vol-targeted sizing and costs.

        Parameters
        ----------
        prices
            DataFrame with 'close' (and optionally 'open') columns.
        initial_capital
            Starting capital.
        cost_bps
            Round-trip cost in basis points.

        Returns
        -------
        dict
            Keys: metrics, equity, returns, signals_df, trades, ou_params
        """
        from utils.metrics import compute_all_metrics

        # Fit on first half
        n = len(prices)
        split = n // 2
        ou_params = self.fit(prices.iloc[:split])

        # Generate signals on full series
        signals_df = self.generate_signals(prices)

        close = prices["close"].values.astype(float)
        positions = signals_df["position"].values
        sigs = signals_df["signal"].values

        # Simulate returns
        returns_arr = np.zeros(n)
        cost_per_trade = cost_bps / 10_000

        for i in range(1, n):
            price_ret = (close[i] - close[i - 1]) / close[i - 1] if close[i - 1] > 0 else 0

            if positions[i - 1] != 0:
                returns_arr[i] = positions[i - 1] * price_ret

            # Apply costs on trade events
            if sigs[i] != 0:
                returns_arr[i] -= cost_per_trade

        # Vol-target scaling
        if self.vol_target > 0:
            for i in range(max(self.bb_period + 1, 63), n):
                recent_vol = np.std(returns_arr[i - 63 : i]) * np.sqrt(252) if i > 63 else 0.16
                if recent_vol > 0:
                    scale = min(3.0, self.vol_target / recent_vol)
                    returns_arr[i] *= scale

        # Build equity curve
        equity = initial_capital * np.cumprod(1 + returns_arr)
        equity_series = pd.Series(equity, index=prices.index)
        returns_series = pd.Series(returns_arr, index=prices.index)

        # Extract trades
        trades = self._extract_trades(signals_df, returns_arr, prices.index)
        trade_pnl = pd.Series([t["pnl_pct"] for t in trades]) if trades else pd.Series(dtype=float)

        # OOS metrics
        oos_returns = returns_series.iloc[split:]
        oos_equity = equity_series.iloc[split:]
        metrics = compute_all_metrics(oos_returns, equity_curve=oos_equity, trade_returns=trade_pnl)
        metrics["ou_half_life"] = ou_params.half_life
        metrics["ou_theta"] = ou_params.theta

        return {
            "metrics": metrics,
            "equity": equity_series,
            "returns": returns_series,
            "signals_df": signals_df,
            "trades": trades,
            "ou_params": ou_params,
        }

    @staticmethod
    def _extract_trades(
        signals_df: pd.DataFrame,
        returns: np.ndarray,
        index: pd.DatetimeIndex,
    ) -> list[dict[str, Any]]:
        """Extract individual trades from signal DataFrame."""
        trades: list[dict[str, Any]] = []
        sigs = signals_df["signal"].values
        positions = signals_df["position"].values

        in_trade = False
        entry_idx = 0
        entry_dir = 0
        cum_ret = 0.0

        for i in range(len(positions)):
            if not in_trade and sigs[i] != 0 and positions[i] != 0:
                in_trade = True
                entry_idx = i
                entry_dir = int(sigs[i])
                cum_ret = 0.0
            elif in_trade:
                cum_ret += returns[i]
                if positions[i] == 0:
                    trades.append({
                        "entry_date": str(index[entry_idx].date()),
                        "exit_date": str(index[i].date()),
                        "direction": "long" if entry_dir > 0 else "short",
                        "holding_bars": i - entry_idx,
                        "pnl_pct": round(cum_ret * 100, 4),
                        "z_entry": round(signals_df["z_score"].values[entry_idx], 2),
                        "z_exit": round(signals_df["z_score"].values[i], 2),
                        "kill_switch_exit": bool(signals_df["kill_switch"].values[i]),
                    })
                    in_trade = False

        return trades
