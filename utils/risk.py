"""Position sizing and risk management utilities.

Implements Kelly criterion, volatility-targeted sizing, and drawdown
circuit breakers — the core components of institutional risk management.

Kelly criterion sizes positions to maximize geometric growth rate.
Full Kelly is too aggressive for real portfolios (assumes perfect
parameter estimates). Quarter-Kelly (fraction=0.25) is standard
practice — it captures ~75% of the growth with ~50% of the volatility.

References:
    Kelly, J. L. (1956). "A New Interpretation of Information Rate."
        Bell System Technical Journal.
    Thorp, E. O. (2006). "The Kelly Criterion in Blackjack, Sports
        Betting, and the Stock Market."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_mult: float = 0.25,
) -> float:
    """Kelly criterion for position sizing.

    f* = (p * b - q) / b

    where:
        p = probability of winning
        q = 1 - p
        b = avg_win / avg_loss (odds ratio)

    Parameters
    ----------
    win_rate
        Probability of a winning trade (0 to 1).
    avg_win
        Average winning trade return (positive).
    avg_loss
        Average losing trade return (positive, magnitude only).
    kelly_mult
        Fraction of full Kelly to use (0.25 = quarter-Kelly).
        Full Kelly is theoretically optimal but assumes perfect
        parameter estimates. Quarter-Kelly is standard practice.

    Returns
    -------
    float
        Recommended position size as fraction of capital (0 to 1).
        Returns 0.0 if edge is negative.
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    b = avg_win / avg_loss
    q = 1 - win_rate
    f_star = (win_rate * b - q) / b

    if f_star <= 0:
        return 0.0

    return float(min(1.0, f_star * kelly_mult))


def continuous_kelly(
    expected_return: float,
    variance: float,
    kelly_mult: float = 0.25,
) -> float:
    """Continuous Kelly for normally-distributed returns.

    f* = mu / sigma^2

    Simpler formulation when you have return distribution parameters
    rather than discrete win/loss statistics.

    Parameters
    ----------
    expected_return
        Expected periodic return (e.g., daily mean return).
    variance
        Variance of periodic returns.
    kelly_mult
        Fraction of full Kelly to use.
    """
    if variance <= 0:
        return 0.0
    f_star = expected_return / variance
    if f_star <= 0:
        return 0.0
    return float(min(1.0, f_star * kelly_mult))


def vol_target_shares(
    target_vol: float,
    realized_vol: float,
    capital: float,
    price: float,
) -> int:
    """Volatility-targeted position sizing.

    Sizes the position so that the strategy's contribution to portfolio
    volatility matches the target. This is the standard approach for
    institutional multi-strategy portfolios.

    shares = (capital * target_vol) / (price * realized_vol * sqrt(252))

    Parameters
    ----------
    target_vol
        Target annualized volatility for this position (e.g., 0.10 = 10%).
    realized_vol
        Realized daily volatility of the instrument.
    capital
        Capital allocated to this strategy.
    price
        Current price per share/contract.

    Returns
    -------
    int
        Number of shares/contracts to trade.
    """
    if realized_vol <= 0 or price <= 0:
        return 0
    daily_vol = realized_vol  # already daily if using daily returns
    annualized_vol = daily_vol * np.sqrt(252)
    if annualized_vol <= 0:
        return 0
    notional = capital * target_vol / annualized_vol
    return max(0, int(notional / price))


def drawdown_kill_switch(
    equity_curve: pd.Series,
    max_dd_threshold: float = 0.10,
) -> pd.Series:
    """Generate a boolean mask for drawdown-based trading halt.

    When drawdown exceeds the threshold, the kill switch activates.
    It remains active until equity recovers to 95% of the previous peak
    (allowing re-entry before full recovery to avoid missing the bounce).

    This implements behavioral risk management: during deep drawdowns,
    reducing exposure prevents emotional decision-making and preserves
    capital for recovery. Regulatory precedent exists (CME circuit breakers).

    Parameters
    ----------
    equity_curve
        Equity values over time.
    max_dd_threshold
        Maximum allowed drawdown before halting (default 10%).

    Returns
    -------
    pd.Series
        Boolean series: True = trading allowed, False = halted.
    """
    values = equity_curve.values
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / np.where(peak == 0, 1.0, peak)

    trading_allowed = np.ones(len(values), dtype=bool)
    halted = False

    for i in range(len(values)):
        if drawdown[i] >= max_dd_threshold:
            halted = True
        elif halted and values[i] >= peak[i] * 0.95:
            halted = False
            peak[i] = values[i]  # Reset peak on recovery

        trading_allowed[i] = not halted

    return pd.Series(trading_allowed, index=equity_curve.index)


def circuit_breaker_equity(
    equity_curve: pd.Series,
    max_dd_pct: float = 0.05,
    scale_factor: float = 0.5,
) -> pd.Series:
    """Circuit breaker that halves position size during drawdowns.

    Instead of full halt, reduces exposure by scale_factor when
    drawdown exceeds threshold. Restores full sizing when equity
    recovers to previous peak.

    Parameters
    ----------
    equity_curve
        Original equity curve.
    max_dd_pct
        Drawdown threshold to trigger scaling (default 5%).
    scale_factor
        Position size multiplier during drawdown (default 0.5 = half).

    Returns
    -------
    pd.Series
        Adjusted equity curve with circuit breaker applied.
    """
    values = equity_curve.values.copy()
    adjusted = np.zeros_like(values)
    adjusted[0] = values[0]

    peak = values[0]
    in_breaker = False

    for i in range(1, len(values)):
        ret = (values[i] - values[i - 1]) / values[i - 1] if values[i - 1] != 0 else 0.0

        dd = (peak - adjusted[i - 1]) / peak if peak > 0 else 0.0

        if dd >= max_dd_pct:
            in_breaker = True
        elif adjusted[i - 1] >= peak:
            in_breaker = False
            peak = adjusted[i - 1]

        effective_ret = ret * scale_factor if in_breaker else ret
        adjusted[i] = adjusted[i - 1] * (1 + effective_ret)
        peak = max(peak, adjusted[i])

    return pd.Series(adjusted, index=equity_curve.index)
