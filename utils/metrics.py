"""Institutional-grade performance metrics for strategy evaluation.

All metrics operate on return series and handle edge cases gracefully.
Annualization uses sqrt(252) for daily returns — the standard convention
because volatility scales with the square root of time under i.i.d.
assumptions. Sharpe uses total standard deviation (symmetric risk);
Sortino uses downside deviation only (penalizes losses, not gains).

References:
    Sharpe, W. F. (1994). "The Sharpe Ratio." Journal of Portfolio Management.
    Sortino, F. A. & Price, L. N. (1994). "Performance Measurement in a
        Downside Risk Framework." Journal of Investing.
    Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio."
        Journal of Portfolio Management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> float:
    """Annualized Sharpe ratio.

    Sharpe = (mean excess return / std of excess returns) * sqrt(annualization)

    Uses standard deviation of ALL returns (both up and down moves).
    This penalizes upside volatility equally with downside — a known
    limitation. See sortino_ratio() for a downside-only alternative.

    Parameters
    ----------
    returns
        Periodic (e.g., daily) return series.
    risk_free_rate
        Periodic risk-free rate matching return frequency.
        Default 0.0 for simplicity; production systems use
        the 3-month Treasury rate / 252.
    annualization
        252 for daily, 12 for monthly, 52 for weekly.

    Returns
    -------
    float
        Annualized Sharpe ratio. Returns 0.0 if insufficient data
        or zero volatility.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate
    std = excess.std(ddof=1)
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(annualization))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> float:
    """Annualized Sortino ratio (downside deviation only).

    Sortino = (mean excess return / downside deviation) * sqrt(annualization)

    Unlike Sharpe, Sortino only penalizes negative returns. A strategy
    with high upside volatility (large winning trades) is not penalized.
    Preferred for asymmetric return distributions (e.g., trend-following).

    Parameters
    ----------
    returns
        Periodic return series.
    risk_free_rate
        Minimum acceptable return (MAR).
    annualization
        Annualization factor.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate
    downside = np.minimum(excess.values, 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2)))
    if downside_std == 0.0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(annualization))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline as a positive fraction.

    Drawdown at time t = (peak_t - equity_t) / peak_t
    where peak_t = max(equity_0, equity_1, ..., equity_t).

    A drawdown of 0.10 means 10% decline from the running peak.
    This is the standard institutional risk metric — it captures
    the worst-case loss an investor would have experienced.

    Parameters
    ----------
    equity_curve
        Equity values over time (not returns).

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (0.0 to 1.0).
    """
    if len(equity_curve) < 2:
        return 0.0
    values = equity_curve.values
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / np.where(peak == 0, 1.0, peak)
    return float(np.max(drawdown))


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Maximum drawdown duration in trading days.

    The longest period the strategy spent below its previous peak.
    Complements max_drawdown magnitude — a 10% drawdown lasting
    3 days is very different from one lasting 200 days.

    Parameters
    ----------
    equity_curve
        Equity values over time.

    Returns
    -------
    int
        Duration of longest drawdown in number of periods.
    """
    if len(equity_curve) < 2:
        return 0
    values = equity_curve.values
    peak = np.maximum.accumulate(values)
    in_drawdown = values < peak

    max_duration = 0
    current_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_duration


def calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    annualization: float = 252.0,
) -> float:
    """Calmar ratio: annualized return divided by max drawdown.

    A Calmar of 2.0 means annualized return is 2x the worst drawdown.
    Institutional minimum is typically > 1.0.

    Parameters
    ----------
    returns
        Periodic returns.
    equity_curve
        Equity curve values.
    annualization
        Annualization factor.
    """
    mdd = max_drawdown(equity_curve)
    if mdd == 0.0:
        return 0.0
    ann_return = float(returns.mean() * annualization)
    return ann_return / mdd


def profit_factor(trade_returns: pd.Series) -> float:
    """Profit factor: gross profits / gross losses.

    PF > 1.0 means profitable. PF > 2.0 is strong.
    Calculated on per-trade returns, not daily returns.

    Parameters
    ----------
    trade_returns
        Series of individual trade P&L values.

    Returns
    -------
    float
        Profit factor. Returns inf if no losses, 0.0 if no gains.
    """
    gains = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of profitable trades.

    Parameters
    ----------
    trade_returns
        Series of individual trade P&L values.
    """
    if len(trade_returns) == 0:
        return 0.0
    return float((trade_returns > 0).sum() / len(trade_returns))


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    annualization: float = 252.0,
) -> float:
    """Information ratio: annualized active return / tracking error.

    Measures risk-adjusted alpha relative to a benchmark.
    IR > 0.5 is generally considered skilled management.

    Parameters
    ----------
    returns
        Strategy periodic returns.
    benchmark
        Benchmark periodic returns (same frequency).
    annualization
        Annualization factor.
    """
    if len(returns) < 2 or len(benchmark) < 2:
        return 0.0
    active = returns - benchmark
    te = active.std(ddof=1)
    if te == 0.0 or np.isnan(te):
        return 0.0
    return float(active.mean() / te * np.sqrt(annualization))


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Value at Risk — the loss threshold at a given confidence level.

    VaR(95%) = 2.0% means there is a 5% chance of losing more than 2%
    in a single period.

    Parameters
    ----------
    returns
        Periodic return series.
    confidence
        Confidence level (0.95 = 95th percentile).
    method
        'historical' (empirical quantile) or 'parametric' (Gaussian assumption).

    Returns
    -------
    float
        VaR as a positive number (loss magnitude).
    """
    if len(returns) == 0:
        return 0.0
    if method == "parametric":
        from scipy.stats import norm

        z = norm.ppf(1 - confidence)
        return float(-(returns.mean() + z * returns.std(ddof=1)))
    else:
        return float(-np.percentile(returns.values, (1 - confidence) * 100))


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Conditional VaR (Expected Shortfall) — average loss beyond VaR.

    CVaR is always >= VaR. It captures tail risk better than VaR
    because it considers the magnitude of extreme losses, not just
    their probability.

    Parameters
    ----------
    returns
        Periodic return series.
    confidence
        Confidence level.
    """
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


def tail_ratio(returns: pd.Series, percentile: float = 5.0) -> float:
    """Ratio of right tail to left tail magnitude.

    tail_ratio > 1.0 means winners are larger than losers on average.
    Calculated as |95th percentile| / |5th percentile|.

    Parameters
    ----------
    returns
        Periodic return series.
    percentile
        Tail percentile (default 5th and 95th).
    """
    if len(returns) < 10:
        return 0.0
    right = np.percentile(returns.values, 100 - percentile)
    left = np.percentile(returns.values, percentile)
    if left == 0.0:
        return 0.0
    return float(abs(right / left))


def compute_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    trade_returns: pd.Series | None = None,
    annualization: float = 252.0,
) -> dict[str, float]:
    """Compute a comprehensive metrics dictionary.

    Parameters
    ----------
    returns
        Daily return series.
    equity_curve
        Equity values. If None, computed from returns assuming $100k start.
    benchmark
        Benchmark return series (for Information Ratio).
    trade_returns
        Per-trade P&L series (for profit factor, win rate).
    annualization
        Annualization factor.

    Returns
    -------
    dict[str, float]
        Dictionary of all computed metrics.
    """
    if equity_curve is None:
        equity_curve = (1 + returns).cumprod() * 100_000

    result: dict[str, float] = {
        "total_return": float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)
        if len(equity_curve) > 0
        else 0.0,
        "annualized_return": float(returns.mean() * annualization),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(annualization)),
        "sharpe_ratio": sharpe_ratio(returns, annualization=annualization),
        "sortino_ratio": sortino_ratio(returns, annualization=annualization),
        "max_drawdown": max_drawdown(equity_curve),
        "max_drawdown_duration": max_drawdown_duration(equity_curve),
        "calmar_ratio": calmar_ratio(returns, equity_curve, annualization),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "tail_ratio": tail_ratio(returns),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "num_observations": len(returns),
    }

    if benchmark is not None:
        result["information_ratio"] = information_ratio(
            returns, benchmark, annualization
        )

    if trade_returns is not None and len(trade_returns) > 0:
        result["profit_factor"] = profit_factor(trade_returns)
        result["win_rate"] = win_rate(trade_returns)
        result["num_trades"] = len(trade_returns)
        result["avg_trade_return"] = float(trade_returns.mean())

    return result
