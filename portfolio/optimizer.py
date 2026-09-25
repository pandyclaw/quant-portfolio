"""Portfolio optimization: MVO, HRP, Risk Parity, Min Variance.

Implements four portfolio construction methods with realistic
constraints. Each method offers a different approach to the
return-estimation problem:

    MVO: requires expected returns AND covariance (most fragile)
    Min Variance: requires only covariance (robust)
    Risk Parity: requires only covariance, equalizes risk (robust)
    HRP: uses hierarchical clustering on covariance (most robust)

All methods use Ledoit-Wolf shrinkage on the covariance matrix
to improve estimation stability with limited sample sizes.

References:
    Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.
    Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample
        Covariance Matrix." Journal of Portfolio Management.
    Lopez de Prado, M. (2016). "Building Diversified Portfolios that
        Outperform Out of Sample." Journal of Portfolio Management.
    Maillard, S., Roncalli, T. & Teiletche, J. (2010). "The Properties
        of Equally Weighted Risk Contribution Portfolios."
        Journal of Portfolio Management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class OptimizationResult:
    """Portfolio optimization output."""

    weights: np.ndarray
    method: str
    asset_names: list[str]
    expected_return: float
    expected_vol: float
    sharpe_ratio: float


def _shrunk_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Estimate covariance matrix with Ledoit-Wolf shrinkage.

    Shrinkage toward a structured target (constant correlation)
    dramatically improves out-of-sample performance vs sample covariance,
    especially when T/N < 5 (few observations relative to assets).
    """
    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_


def mean_variance(
    returns: pd.DataFrame,
    target_return: float | None = None,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.40,
) -> OptimizationResult:
    """Mean-variance optimization with Ledoit-Wolf shrinkage.

    Maximizes Sharpe ratio (or targets a specific return) subject
    to weight constraints.

    Parameters
    ----------
    returns
        Historical return DataFrame (assets as columns).
    target_return
        If specified, minimize variance for this target return.
        If None, maximize Sharpe ratio.
    risk_free_rate
        Risk-free rate for Sharpe computation.
    max_weight
        Maximum weight per asset (prevents concentration).
    """
    n_assets = len(returns.columns)
    mu = returns.mean().values * 252  # Annualized
    cov = _shrunk_covariance(returns) * 252

    if target_return is not None:
        # Minimize variance subject to target return
        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: float(w @ mu) - target_return},
        ]
    else:
        # Maximize Sharpe ratio
        def objective(w: np.ndarray) -> float:
            port_ret = float(w @ mu) - risk_free_rate
            port_vol = float(np.sqrt(w @ cov @ w))
            if port_vol < 1e-8:
                return 1e10
            return -port_ret / port_vol  # Negative for minimization

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    bounds = [(0, max_weight)] * n_assets
    x0 = np.ones(n_assets) / n_assets

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = result.x

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    sr = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

    return OptimizationResult(
        weights=w,
        method="mean_variance",
        asset_names=list(returns.columns),
        expected_return=port_ret,
        expected_vol=port_vol,
        sharpe_ratio=sr,
    )


def min_variance(
    returns: pd.DataFrame,
    max_weight: float = 0.40,
) -> OptimizationResult:
    """Global minimum variance portfolio.

    Requires only a covariance estimate — no expected return inputs.
    This avoids the biggest weakness of MVO (garbage-in returns).
    """
    n_assets = len(returns.columns)
    cov = _shrunk_covariance(returns) * 252

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, max_weight)] * n_assets
    x0 = np.ones(n_assets) / n_assets

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = result.x

    mu = returns.mean().values * 252
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    sr = port_ret / port_vol if port_vol > 0 else 0

    return OptimizationResult(
        weights=w,
        method="min_variance",
        asset_names=list(returns.columns),
        expected_return=port_ret,
        expected_vol=port_vol,
        sharpe_ratio=sr,
    )


def risk_parity(
    returns: pd.DataFrame,
) -> OptimizationResult:
    """Equal risk contribution (risk parity) portfolio.

    Each asset contributes equally to portfolio risk:
        w_i * (Σw)_i / (w'Σw) = 1/N for all i

    This produces the most diversified portfolio by risk, without
    requiring any return estimates.
    """
    n_assets = len(returns.columns)
    cov = _shrunk_covariance(returns) * 252
    target_risk = 1.0 / n_assets

    def objective(w: np.ndarray) -> float:
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 1e10
        marginal_risk = cov @ w
        risk_contrib = w * marginal_risk / port_vol
        # Minimize deviation from equal risk contribution
        return float(np.sum((risk_contrib / port_vol - target_risk) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.01, 0.50)] * n_assets
    x0 = np.ones(n_assets) / n_assets

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = result.x

    mu = returns.mean().values * 252
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    sr = port_ret / port_vol if port_vol > 0 else 0

    return OptimizationResult(
        weights=w,
        method="risk_parity",
        asset_names=list(returns.columns),
        expected_return=port_ret,
        expected_vol=port_vol,
        sharpe_ratio=sr,
    )


def hierarchical_risk_parity(
    returns: pd.DataFrame,
) -> OptimizationResult:
    """Hierarchical Risk Parity (HRP) — Lopez de Prado (2016).

    Uses hierarchical clustering to group correlated assets, then
    allocates by inverse-variance within each cluster. This avoids
    covariance matrix inversion (unlike MVO) and produces more
    stable out-of-sample results.

    Steps:
    1. Compute distance matrix from correlation
    2. Hierarchical clustering (single linkage)
    3. Quasi-diagonalization (reorder by cluster)
    4. Recursive bisection with inverse-variance allocation
    """
    cov = _shrunk_covariance(returns)
    corr = returns.corr().values
    n_assets = len(returns.columns)

    # Step 1: Distance matrix from correlation
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)

    # Step 2: Hierarchical clustering
    link = linkage(condensed, method="single")

    # Step 3: Quasi-diagonalization (reorder assets by cluster)
    sort_idx = list(leaves_list(link).astype(int))

    # Step 4: Recursive bisection
    weights = np.ones(n_assets)
    cluster_items = [sort_idx]

    while cluster_items:
        new_clusters = []
        for items in cluster_items:
            if len(items) <= 1:
                continue
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]

            # Variance of each sub-cluster
            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)
            total_var = left_var + right_var

            if total_var > 0:
                alpha = 1 - left_var / total_var  # Weight to left
            else:
                alpha = 0.5

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1 - alpha)

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        cluster_items = new_clusters

    # Normalize
    weights = weights / weights.sum()

    mu = returns.mean().values * 252
    ann_cov = cov * 252
    port_ret = float(weights @ mu)
    port_vol = float(np.sqrt(weights @ ann_cov @ weights))
    sr = port_ret / port_vol if port_vol > 0 else 0

    return OptimizationResult(
        weights=weights,
        method="hrp",
        asset_names=list(returns.columns),
        expected_return=port_ret,
        expected_vol=port_vol,
        sharpe_ratio=sr,
    )


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    """Compute variance of an equally-weighted sub-portfolio."""
    sub_cov = cov[np.ix_(indices, indices)]
    n = len(indices)
    w = np.ones(n) / n
    return float(w @ sub_cov @ w)


def compare_methods(
    returns: pd.DataFrame,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Compare multiple portfolio construction methods.

    Parameters
    ----------
    returns
        Historical returns DataFrame.
    methods
        List of methods to compare. Default: all four.

    Returns
    -------
    pd.DataFrame
        Comparison table with weights, expected return, vol, Sharpe.
    """
    if methods is None:
        methods = ["equal", "mean_variance", "min_variance", "risk_parity", "hrp"]

    results: list[dict[str, Any]] = []

    for method in methods:
        if method == "equal":
            n = len(returns.columns)
            w = np.ones(n) / n
            mu = returns.mean().values * 252
            cov = _shrunk_covariance(returns) * 252
            ret = float(w @ mu)
            vol = float(np.sqrt(w @ cov @ w))
            results.append({
                "method": "equal_weight",
                "expected_return": ret,
                "expected_vol": vol,
                "sharpe": ret / vol if vol > 0 else 0,
                **{returns.columns[i]: round(w[i], 4) for i in range(n)},
            })
        elif method == "mean_variance":
            opt = mean_variance(returns)
            results.append({
                "method": opt.method,
                "expected_return": opt.expected_return,
                "expected_vol": opt.expected_vol,
                "sharpe": opt.sharpe_ratio,
                **{opt.asset_names[i]: round(opt.weights[i], 4) for i in range(len(opt.weights))},
            })
        elif method == "min_variance":
            opt = min_variance(returns)
            results.append({
                "method": opt.method,
                "expected_return": opt.expected_return,
                "expected_vol": opt.expected_vol,
                "sharpe": opt.sharpe_ratio,
                **{opt.asset_names[i]: round(opt.weights[i], 4) for i in range(len(opt.weights))},
            })
        elif method == "risk_parity":
            opt = risk_parity(returns)
            results.append({
                "method": opt.method,
                "expected_return": opt.expected_return,
                "expected_vol": opt.expected_vol,
                "sharpe": opt.sharpe_ratio,
                **{opt.asset_names[i]: round(opt.weights[i], 4) for i in range(len(opt.weights))},
            })
        elif method == "hrp":
            opt = hierarchical_risk_parity(returns)
            results.append({
                "method": opt.method,
                "expected_return": opt.expected_return,
                "expected_vol": opt.expected_vol,
                "sharpe": opt.sharpe_ratio,
                **{opt.asset_names[i]: round(opt.weights[i], 4) for i in range(len(opt.weights))},
            })

    return pd.DataFrame(results)
