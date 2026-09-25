"""Strategy correlation monitoring and regime break detection.

In a multi-strategy portfolio, correlation between strategies determines
diversification benefit. When correlations spike (crisis = "all
correlations go to 1"), the portfolio loses its diversification edge
precisely when it's most needed.

This module tracks rolling correlations and flags regime breaks when
correlation exceeds a threshold.

References:
    Kritzman, M. et al. (2011). "Principal Components as a Measure
    of Systemic Risk." Journal of Portfolio Management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_correlation(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Compute rolling pairwise correlation matrix.

    Returns a DataFrame of average pairwise correlation at each date.
    """
    n = len(returns)
    avg_corr = pd.Series(np.nan, index=returns.index)

    for i in range(window, n):
        corr_matrix = returns.iloc[i - window : i].corr()
        # Average off-diagonal correlation
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr.iloc[i] = float(corr_matrix.values[mask].mean())

    return avg_corr.to_frame("avg_correlation")


def detect_correlation_regime(
    avg_corr: pd.Series,
    high_threshold: float = 0.60,
    low_threshold: float = 0.20,
) -> pd.Series:
    """Classify correlation regime: normal, elevated, crisis.

    Returns a categorical Series: 'low', 'normal', 'elevated', 'crisis'.
    """
    regime = pd.Series("normal", index=avg_corr.index)
    regime[avg_corr < low_threshold] = "low"
    regime[avg_corr > high_threshold] = "elevated"
    regime[avg_corr > 0.80] = "crisis"
    return regime


def correlation_break_alert(
    returns: pd.DataFrame,
    window: int = 63,
    threshold: float = 0.60,
) -> pd.DataFrame:
    """Monitor for correlation breakouts.

    Flags dates where average pairwise correlation exceeds threshold.
    These are early warning signals for regime shifts.
    """
    corr_df = rolling_correlation(returns, window)
    corr_df["alert"] = corr_df["avg_correlation"] > threshold
    corr_df["regime"] = detect_correlation_regime(
        corr_df["avg_correlation"], high_threshold=threshold
    )
    return corr_df
