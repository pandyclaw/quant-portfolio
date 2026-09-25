"""Statistical arbitrage: Gold/Silver cointegration pairs trading.

Trades the mean-reverting spread between GLD and SLV using the
Engle-Granger cointegration framework with a Kalman filter for
dynamic hedge ratio estimation.

The gold/silver ratio has exhibited mean-reverting behavior historically
due to shared monetary metal fundamentals. When the spread deviates
significantly from equilibrium (measured by z-score), it tends to revert.

Academic basis:
    Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006).
    "Pairs Trading: Performance of a Relative-Value Arbitrage Rule."
    Review of Financial Studies.

    Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and
    Analysis." Wiley.

Signal construction:
    1. Test for cointegration (Engle-Granger ADF on OLS residuals)
    2. Estimate dynamic hedge ratio via Kalman filter
    3. Compute rolling z-score of the spread
    4. Entry: |z| > entry_z (default 2.0)
    5. Exit: |z| < exit_z (default 0.5) or stop at |z| > stop_z

Position sizing:
    Volatility-targeted: scale notional so the spread's contribution
    to portfolio vol matches the target (default 10% annualized).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass(frozen=True)
class FitResult:
    """Diagnostics from fitting the cointegration model."""

    is_cointegrated: bool
    adf_statistic: float
    adf_pvalue: float
    hedge_ratio_ols: float
    half_life: float
    spread_mean: float
    spread_std: float
    hurst_exponent: float
    num_observations: int


@dataclass
class KalmanState:
    """Internal state of the Kalman filter for hedge ratio estimation."""

    beta: np.ndarray  # [hedge_ratio, intercept]
    P: np.ndarray  # State covariance
    R: float  # Observation noise
    Q: np.ndarray  # State transition noise


class GoldSilverPairs:
    """Cointegration-based pairs trading on GLD/SLV.

    Parameters
    ----------
    entry_z
        Z-score threshold for entry (default 2.0 = 2 sigma).
    exit_z
        Z-score threshold for exit (default 0.5 = mean reversion).
    stop_z
        Z-score threshold for stop-loss (default 4.0 = divergence).
    max_holding_days
        Maximum position duration before forced exit.
    spread_lookback
        Rolling window for z-score computation.
    vol_target
        Target annualized volatility for position sizing.
    kalman_delta
        Kalman filter transition covariance. Smaller = smoother hedge ratio.
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        max_holding_days: int = 63,
        spread_lookback: int = 63,
        vol_target: float = 0.10,
        kalman_delta: float = 1e-5,
    ) -> None:
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_holding_days = max_holding_days
        self.spread_lookback = spread_lookback
        self.vol_target = vol_target

        self._kalman_delta = kalman_delta
        self._kalman_state: KalmanState | None = None
        self._fit_result: FitResult | None = None

    def fit(self, prices_y: pd.Series, prices_x: pd.Series) -> FitResult:
        """Fit the cointegration model on training data.

        Steps:
            1. OLS regression: y = beta * x + alpha + epsilon
            2. ADF test on residuals (Engle-Granger cointegration test)
            3. OU process half-life estimation
            4. Hurst exponent for mean-reversion confirmation

        Parameters
        ----------
        prices_y
            Price series for the dependent asset (e.g., GLD).
        prices_x
            Price series for the independent asset (e.g., SLV).

        Returns
        -------
        FitResult
            Diagnostics including cointegration status and parameters.
        """
        # Drop NaN/inf values (yfinance data can have gaps)
        mask = np.isfinite(prices_y.values) & np.isfinite(prices_x.values)
        y = prices_y.values[mask].astype(float)
        x = prices_x.values[mask].astype(float)
        n = len(y)

        # Step 1: OLS for static hedge ratio
        X = sm.add_constant(x)
        ols_result = sm.OLS(y, X).fit()
        hedge_ratio_ols = float(ols_result.params[1])
        residuals = ols_result.resid

        # Step 2: ADF test on residuals (Engle-Granger)
        from statsmodels.tsa.stattools import adfuller

        adf = adfuller(residuals, maxlag=int(np.sqrt(n)), autolag="AIC")
        adf_stat = float(adf[0])
        adf_pvalue = float(adf[1])
        is_cointegrated = adf_pvalue < 0.05

        # Step 3: Half-life via AR(1) on spread
        spread_lag = residuals[:-1]
        spread_delta = np.diff(residuals)
        X_hl = sm.add_constant(spread_lag)
        ols_hl = sm.OLS(spread_delta, X_hl).fit()
        b = ols_hl.params[1]
        half_life = float(-np.log(2) / b) if b < 0 else float("inf")

        # Step 4: Hurst exponent (simplified R/S method)
        hurst = self._hurst_exponent(residuals)

        spread_mean = float(np.mean(residuals))
        spread_std = float(np.std(residuals, ddof=1))

        self._fit_result = FitResult(
            is_cointegrated=is_cointegrated,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pvalue,
            hedge_ratio_ols=hedge_ratio_ols,
            half_life=half_life,
            spread_mean=spread_mean,
            spread_std=max(spread_std, 1e-8),
            hurst_exponent=hurst,
            num_observations=n,
        )

        return self._fit_result

    def generate_signals(
        self,
        prices_y: pd.Series,
        prices_x: pd.Series,
    ) -> pd.DataFrame:
        """Generate trading signals from the spread z-score.

        Runs the Kalman filter over the full series, computes rolling
        z-scores, and generates +1 (long spread) / -1 (short spread) / 0
        (flat) signals.

        Parameters
        ----------
        prices_y
            Full price series for Y (dependent asset).
        prices_x
            Full price series for X (independent asset).

        Returns
        -------
        pd.DataFrame
            Columns: hedge_ratio, spread, z_score, signal, position
        """
        y = prices_y.ffill().bfill().values.astype(float)
        x = prices_x.ffill().bfill().values.astype(float)
        n = len(y)

        # Run Kalman filter (skip NaN/inf values)
        self._kalman_state = None
        hedge_ratios = np.zeros(n)
        spreads = np.zeros(n)

        for i in range(n):
            if np.isfinite(y[i]) and np.isfinite(x[i]):
                hr, sp = self._kalman_update(y[i], x[i])
                hedge_ratios[i] = hr
                spreads[i] = sp
            elif i > 0:
                hedge_ratios[i] = hedge_ratios[i - 1]
                spreads[i] = spreads[i - 1]

        # Rolling z-score (using only past data — no look-ahead)
        z_scores = np.zeros(n)
        for i in range(self.spread_lookback, n):
            window = spreads[i - self.spread_lookback : i]
            mu = np.mean(window)
            sigma = np.std(window, ddof=1)
            if sigma > 0:
                z_scores[i] = (spreads[i] - mu) / sigma

        # Generate signals with state tracking
        signals = np.zeros(n)
        positions = np.zeros(n)
        position = 0  # 0=flat, 1=long spread, -1=short spread
        holding_days = 0

        for i in range(self.spread_lookback, n):
            z = z_scores[i]
            abs_z = abs(z)

            # Exit logic (check BEFORE incrementing holding_days)
            exited_this_bar = False
            if position != 0:
                should_exit = (
                    abs_z < self.exit_z  # Mean reversion
                    or abs_z > self.stop_z  # Stop-loss
                    or holding_days >= self.max_holding_days  # Time stop
                )
                if should_exit:
                    signals[i] = -position  # Close signal
                    position = 0
                    holding_days = 0
                    exited_this_bar = True

            # Entry logic (only when flat AND did not just exit — no same-bar re-entry)
            if position == 0 and not exited_this_bar:
                if z < -self.entry_z:
                    signals[i] = 1  # Long spread
                    position = 1
                    holding_days = 0
                elif z > self.entry_z:
                    signals[i] = -1  # Short spread
                    position = -1
                    holding_days = 0

            if position != 0:
                holding_days += 1
            positions[i] = position

        idx = prices_y.index if hasattr(prices_y, "index") else range(n)
        return pd.DataFrame(
            {
                "hedge_ratio": hedge_ratios,
                "spread": spreads,
                "z_score": z_scores,
                "signal": signals,
                "position": positions,
            },
            index=idx,
        )

    def backtest(
        self,
        prices_y: pd.Series,
        prices_x: pd.Series,
        initial_capital: float = 100_000.0,
        cost_bps: float = 10.0,
    ) -> dict[str, Any]:
        """Run a full backtest of the pairs strategy.

        Computes equity curve, returns, and performance metrics.
        Transaction costs are applied on every entry and exit.

        Parameters
        ----------
        prices_y
            Full price series for Y.
        prices_x
            Full price series for X.
        initial_capital
            Starting capital.
        cost_bps
            Round-trip transaction cost in basis points.

        Returns
        -------
        dict
            Keys: metrics (dict), equity (pd.Series), signals_df (pd.DataFrame),
            trades (list[dict]), fit_result (FitResult)
        """
        from utils.metrics import compute_all_metrics

        # Fit on first half, generate signals on full series
        n = len(prices_y)
        split = n // 2

        fit_result = self.fit(prices_y.iloc[:split], prices_x.iloc[:split])
        signals_df = self.generate_signals(prices_y, prices_x)

        # Simulate P&L from spread positions
        positions = signals_df["position"].values
        y_returns = prices_y.pct_change().fillna(0).values
        x_returns = prices_x.pct_change().fillna(0).values
        hedge_ratios = signals_df["hedge_ratio"].values

        # Spread return: long Y, short hedge_ratio * X
        spread_returns = np.zeros(n)
        for i in range(1, n):
            if positions[i - 1] != 0:
                spread_ret = y_returns[i] - hedge_ratios[i - 1] * x_returns[i]
                spread_returns[i] = positions[i - 1] * spread_ret

        # Vol-targeted position sizing
        if self.vol_target > 0 and n > self.spread_lookback:
            for i in range(self.spread_lookback + 1, n):
                recent_vol = np.std(spread_returns[i - 63 : i]) * np.sqrt(252) if i > 63 else 0.16
                if recent_vol > 0:
                    vol_scale = self.vol_target / recent_vol
                    vol_scale = min(vol_scale, 3.0)  # Cap leverage at 3x
                    spread_returns[i] *= vol_scale

        # Apply transaction costs
        cost_per_trade = cost_bps / 10_000
        for i in range(1, n):
            if signals_df["signal"].values[i] != 0:
                spread_returns[i] -= cost_per_trade

        # Build equity curve
        equity = initial_capital * np.cumprod(1 + spread_returns)
        equity_series = pd.Series(equity, index=prices_y.index)
        returns_series = pd.Series(spread_returns, index=prices_y.index)

        # Extract trades
        trades = self._extract_trades(signals_df, spread_returns, prices_y.index)
        trade_pnl = pd.Series([t["pnl_pct"] for t in trades]) if trades else pd.Series(dtype=float)

        # Compute metrics (OOS only = second half)
        oos_returns = returns_series.iloc[split:]
        oos_equity = equity_series.iloc[split:]
        metrics = compute_all_metrics(
            oos_returns, equity_curve=oos_equity, trade_returns=trade_pnl
        )
        metrics["is_cointegrated"] = fit_result.is_cointegrated
        metrics["half_life"] = fit_result.half_life
        metrics["hurst_exponent"] = fit_result.hurst_exponent
        metrics["adf_pvalue"] = fit_result.adf_pvalue
        metrics["is_period"] = f"{prices_y.index[0].date()} to {prices_y.index[split].date()}"
        metrics["oos_period"] = f"{prices_y.index[split].date()} to {prices_y.index[-1].date()}"

        return {
            "metrics": metrics,
            "equity": equity_series,
            "returns": returns_series,
            "signals_df": signals_df,
            "trades": trades,
            "fit_result": fit_result,
        }

    def _kalman_update(self, y: float, x: float) -> tuple[float, float]:
        """Run one Kalman filter step. Returns (hedge_ratio, spread)."""
        if self._kalman_state is None:
            self._kalman_state = KalmanState(
                beta=np.zeros(2),
                P=np.eye(2) / self._kalman_delta,
                R=1.0,
                Q=np.eye(2) * self._kalman_delta,
            )

        st = self._kalman_state
        F = np.array([x, 1.0])

        # Predict
        P_prior = st.P + st.Q

        # Update
        y_hat = F @ st.beta
        e = y - y_hat
        S = float(F @ P_prior @ F) + st.R
        K = (P_prior @ F) / S

        st.beta = st.beta + K * e
        st.P = P_prior - np.outer(K, K) * S

        hedge_ratio = float(st.beta[0])
        intercept = float(st.beta[1])
        spread = y - hedge_ratio * x - intercept

        return hedge_ratio, spread

    @staticmethod
    def _hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
        """Simplified Hurst exponent via variance ratio.

        H < 0.5: mean-reverting (good for pairs trading)
        H = 0.5: random walk (no edge)
        H > 0.5: trending
        """
        n = len(series)
        if n < max_lag * 2:
            return 0.5

        lags = range(2, min(max_lag, n // 4))
        tau = []
        var_tau = []

        for lag in lags:
            diff = series[lag:] - series[:-lag]
            tau.append(lag)
            var_tau.append(np.std(diff))

        if len(tau) < 2 or any(v <= 0 for v in var_tau):
            return 0.5

        log_tau = np.log(tau)
        log_var = np.log(var_tau)

        slope, _, _, _, _ = stats.linregress(log_tau, log_var)
        return float(slope)

    @staticmethod
    def _extract_trades(
        signals_df: pd.DataFrame,
        spread_returns: np.ndarray,
        index: pd.DatetimeIndex,
    ) -> list[dict[str, Any]]:
        """Extract individual trades from the signal DataFrame."""
        trades: list[dict[str, Any]] = []
        positions = signals_df["position"].values
        sigs = signals_df["signal"].values

        in_trade = False
        entry_idx = 0
        entry_dir = 0
        cumulative_ret = 0.0

        for i in range(len(positions)):
            if not in_trade and sigs[i] != 0:
                in_trade = True
                entry_idx = i
                entry_dir = int(sigs[i])
                cumulative_ret = 0.0
            elif in_trade:
                cumulative_ret += spread_returns[i]
                if positions[i] == 0:
                    trades.append({
                        "entry_date": str(index[entry_idx].date()),
                        "exit_date": str(index[i].date()),
                        "direction": "long" if entry_dir > 0 else "short",
                        "holding_days": i - entry_idx,
                        "pnl_pct": round(cumulative_ret * 100, 4),
                        "z_entry": round(signals_df["z_score"].values[entry_idx], 2),
                        "z_exit": round(signals_df["z_score"].values[i], 2),
                    })
                    in_trade = False

        return trades
