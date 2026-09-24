"""Statistical validation for strategy backtests.

The single most important module in this repository. Without these
tests, a backtest Sharpe ratio is meaningless — it could be the best
of 100 failed experiments (selection bias), or a lucky random walk.

Implements:
    1. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
       — corrects Sharpe for multiple testing
    2. Bootstrap confidence intervals
       — non-parametric uncertainty quantification
    3. Permutation test
       — null hypothesis: "this strategy has no edge"

A strategy that survives all three tests has a defensible claim
to genuine alpha. One that fails any test requires honest disclosure.

References:
    Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe
        Ratio: Correcting for Selection Bias, Backtest Overfitting,
        and Non-Normality." Journal of Portfolio Management.
    White, H. (2000). "A Reality Check for Data Snooping."
        Econometrica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Result of the Deflated Sharpe Ratio test.

    Attributes
    ----------
    observed_sharpe
        The raw (unadjusted) Sharpe ratio from the backtest.
    deflated_sharpe
        Sharpe adjusted for number of trials (always <= observed).
    p_value
        Probability of observing this Sharpe by chance given
        the number of trials. < 0.05 = statistically significant.
    num_trials
        Number of strategy variants tested.
    expected_max_sharpe
        Expected maximum Sharpe from num_trials random strategies.
        If observed < expected_max, the strategy is likely noise.
    is_significant
        True if p_value < 0.05.
    """

    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    num_trials: int
    expected_max_sharpe: float
    is_significant: bool


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int,
    num_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std: float | None = None,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Corrects the observed Sharpe ratio for the number of strategy
    variants tested. If you test 100 strategies and pick the best,
    the expected maximum Sharpe of random noise is ~2.3 — meaning
    any strategy with Sharpe < 2.3 could be pure luck.

    The deflated Sharpe subtracts this expected-maximum-under-null
    and normalizes by the standard error of the Sharpe estimate.

    Parameters
    ----------
    observed_sharpe
        The backtest Sharpe ratio.
    num_trials
        Total number of strategy variants tested (including failures).
        This is the key input — underreporting num_trials inflates
        the deflated Sharpe, which is intellectual dishonesty.
    num_returns
        Number of return observations (e.g., 252 * num_years).
    skewness
        Skewness of the return distribution (0 = symmetric).
    kurtosis
        Kurtosis (3.0 = normal). Excess kurtosis = kurtosis - 3.
    sharpe_std
        Standard error of the Sharpe estimate. If None, computed
        from num_returns, skewness, and kurtosis.

    Returns
    -------
    DeflatedSharpeResult
        Contains deflated Sharpe, p-value, and significance flag.
    """
    if num_trials < 1:
        num_trials = 1
    if num_returns < 2:
        return DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=0.0,
            p_value=1.0,
            num_trials=num_trials,
            expected_max_sharpe=0.0,
            is_significant=False,
        )

    # Standard error of the Sharpe ratio (Lo, 2002)
    if sharpe_std is None:
        excess_kurt = kurtosis - 3.0
        se_sr = np.sqrt(
            (1 - skewness * observed_sharpe + (excess_kurt / 4) * observed_sharpe**2)
            / num_returns
        )
    else:
        se_sr = sharpe_std

    if se_sr <= 0:
        se_sr = 1.0 / np.sqrt(num_returns)

    # Expected maximum Sharpe under null
    e_max_sr = expected_max_sharpe(num_trials, num_returns, skewness, kurtosis)

    # Deflated Sharpe = (observed - E[max]) / SE(SR)
    test_stat = (observed_sharpe - e_max_sr) / se_sr
    p_value = float(1.0 - stats.norm.cdf(test_stat))

    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe=float(test_stat),
        p_value=p_value,
        num_trials=num_trials,
        expected_max_sharpe=e_max_sr,
        is_significant=p_value < 0.05,
    )


def expected_max_sharpe(
    num_trials: int,
    num_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Expected maximum Sharpe ratio from N random strategies.

    Under the null hypothesis (no edge), the best of N independent
    strategies has an expected Sharpe that grows with sqrt(2 * ln(N)).
    This is the benchmark — your strategy must beat this to be real.

    Parameters
    ----------
    num_trials
        Number of independent strategies tested.
    num_returns
        Number of return observations per strategy.
    skewness
        Return distribution skewness.
    kurtosis
        Return distribution kurtosis.
    """
    if num_trials <= 1:
        return 0.0

    euler_mascheroni = 0.5772156649
    z = stats.norm.ppf(1 - 1 / num_trials)

    e_max = z * (1 - euler_mascheroni) + euler_mascheroni * stats.norm.ppf(
        1 - 1 / (num_trials * np.e)
    )

    # Adjust for non-normality
    excess_kurt = kurtosis - 3.0
    if abs(skewness) > 0 or abs(excess_kurt) > 0:
        se = np.sqrt(
            (1 - skewness * e_max + (excess_kurt / 4) * e_max**2)
            / max(num_returns, 1)
        )
        e_max = e_max * (1 + se * 0.1)

    return float(e_max)


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for the Sharpe ratio.

    Non-parametric method: resample returns with replacement,
    compute Sharpe on each sample, report the distribution.

    A wide CI (e.g., Sharpe 0.5 to 2.5) means the estimate is
    unreliable — you need more data. A narrow CI that excludes
    zero (e.g., 0.8 to 1.5) is a strong signal.

    Parameters
    ----------
    returns
        Periodic return series.
    n_bootstrap
        Number of bootstrap samples.
    confidence
        Confidence level (default 95%).
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict
        point_estimate, ci_lower, ci_upper, prob_positive
    """
    rng = np.random.RandomState(seed)
    arr = returns.values
    n = len(arr)

    if n < 10:
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "prob_positive": 0.0,
        }

    sharpes = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        std = np.std(sample, ddof=1)
        sharpes[i] = (np.mean(sample) / std * np.sqrt(252)) if std > 0 else 0.0

    alpha = (1 - confidence) / 2
    return {
        "point_estimate": float(np.median(sharpes)),
        "ci_lower": float(np.percentile(sharpes, alpha * 100)),
        "ci_upper": float(np.percentile(sharpes, (1 - alpha) * 100)),
        "prob_positive": float(np.mean(sharpes > 0)),
    }


def permutation_test(
    returns: pd.Series,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation test for strategy significance.

    Null hypothesis: the ordering of returns doesn't matter
    (i.e., the strategy has no predictive power).

    Shuffles the return series and computes Sharpe on each
    permutation. The p-value is the fraction of permuted Sharpes
    that exceed the real Sharpe.

    This is a powerful non-parametric test because it makes no
    distributional assumptions. If p < 0.05, the strategy's
    time-series structure (which captures the edge) is unlikely
    to be random.

    Parameters
    ----------
    returns
        Periodic return series.
    n_permutations
        Number of random shuffles.
    seed
        Random seed.

    Returns
    -------
    dict
        real_sharpe, p_value, is_significant
    """
    rng = np.random.RandomState(seed)
    arr = returns.values
    n = len(arr)

    if n < 10:
        return {"real_sharpe": 0.0, "p_value": 1.0, "is_significant": False}

    std = np.std(arr, ddof=1)
    real_sharpe = (np.mean(arr) / std * np.sqrt(252)) if std > 0 else 0.0

    count_exceeding = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(arr)
        s_std = np.std(shuffled, ddof=1)
        s_sharpe = (np.mean(shuffled) / s_std * np.sqrt(252)) if s_std > 0 else 0.0
        if s_sharpe >= real_sharpe:
            count_exceeding += 1

    p_value = (count_exceeding + 1) / (n_permutations + 1)

    return {
        "real_sharpe": float(real_sharpe),
        "p_value": float(p_value),
        "is_significant": p_value < 0.05,
    }
