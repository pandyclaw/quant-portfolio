"""Yield curve modeling: Nelson-Siegel and PCA decomposition.

The Nelson-Siegel model parameterizes the yield curve with three
factors — level (β₀), slope (β₁), and curvature (β₂) — plus a
decay parameter (τ). This parsimonious model captures >95% of
yield curve variation and is used by central banks worldwide.

Nelson-Siegel yield at maturity m:
    y(m) = β₀ + β₁ × [(1-e^(-m/τ)) / (m/τ)]
         + β₂ × [(1-e^(-m/τ)) / (m/τ) - e^(-m/τ)]

Factor interpretation:
    β₀ = long-term rate (level)
    β₁ = short-long spread (slope, negative = normal curve)
    β₂ = mid-term hump (curvature)
    τ = decay speed (controls where curvature peaks)

References:
    Nelson, C. R. & Siegel, A. F. (1987). "Parsimonious Modeling of
        Yield Curves." Journal of Business.
    Diebold, F. X. & Li, C. (2006). "Forecasting the Term Structure
        of Government Bond Yields." Journal of Econometrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class NelsonSiegelParams:
    """Fitted Nelson-Siegel parameters."""

    beta0: float  # Level
    beta1: float  # Slope
    beta2: float  # Curvature
    tau: float  # Decay parameter
    rmse: float  # Fit quality


def nelson_siegel_yield(
    maturities: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float,
) -> np.ndarray:
    """Compute Nelson-Siegel yield curve at given maturities."""
    m_tau = maturities / tau
    # Avoid division by zero for very short maturities
    m_tau = np.maximum(m_tau, 1e-6)

    factor1 = (1 - np.exp(-m_tau)) / m_tau
    factor2 = factor1 - np.exp(-m_tau)

    return beta0 + beta1 * factor1 + beta2 * factor2


def nelson_siegel_fit(
    maturities: np.ndarray,
    yields: np.ndarray,
    tau_init: float = 1.5,
) -> NelsonSiegelParams:
    """Fit Nelson-Siegel model to observed yields.

    Uses scipy optimization to minimize sum of squared errors
    between model and observed yields.

    Parameters
    ----------
    maturities
        Array of maturities in years (e.g., [0.25, 0.5, 1, 2, 5, 10, 30]).
    yields
        Array of observed yields (as decimals, e.g., 0.045 = 4.5%).
    tau_init
        Initial guess for decay parameter.

    Returns
    -------
    NelsonSiegelParams
        Fitted parameters with RMSE.
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)

    def objective(params: np.ndarray) -> float:
        b0, b1, b2, tau = params
        if tau <= 0.01:
            return 1e10
        fitted = nelson_siegel_yield(maturities, b0, b1, b2, tau)
        return float(np.sum((yields - fitted) ** 2))

    # Initial guess: level ≈ long rate, slope ≈ short-long spread
    b0_init = float(yields[-1]) if len(yields) > 0 else 0.04
    b1_init = float(yields[0] - yields[-1]) if len(yields) > 1 else -0.01
    b2_init = 0.0

    result = minimize(
        objective,
        x0=[b0_init, b1_init, b2_init, tau_init],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8},
    )

    b0, b1, b2, tau = result.x
    fitted = nelson_siegel_yield(maturities, b0, b1, b2, max(tau, 0.01))
    rmse = float(np.sqrt(np.mean((yields - fitted) ** 2)))

    return NelsonSiegelParams(
        beta0=float(b0),
        beta1=float(b1),
        beta2=float(b2),
        tau=float(max(tau, 0.01)),
        rmse=rmse,
    )


@dataclass(frozen=True)
class PCAResult:
    """PCA decomposition of yield curve history."""

    loadings: np.ndarray  # (n_components, n_maturities)
    explained_variance: np.ndarray  # Fraction explained per component
    scores: np.ndarray  # (n_dates, n_components)
    maturities: np.ndarray
    component_names: list[str]


def yield_curve_pca(
    yield_history: np.ndarray,
    maturities: np.ndarray | None = None,
    n_components: int = 3,
) -> PCAResult:
    """PCA decomposition of yield curve movements.

    The first three principal components of yield curve changes
    correspond to well-known factors:
        PC1: level (parallel shift) — explains ~85-90%
        PC2: slope (steepening/flattening) — explains ~8-12%
        PC3: curvature (butterfly) — explains ~2-4%

    This decomposition is used for:
        - Hedging: immunize a portfolio against level/slope/curvature moves
        - Trading: express views on specific curve dynamics
        - Risk: measure exposure to each factor

    Parameters
    ----------
    yield_history
        Matrix of yields: (n_dates, n_maturities).
    maturities
        Array of maturity labels.
    n_components
        Number of principal components (default 3).
    """
    from sklearn.decomposition import PCA

    # Work with yield changes (differences), not levels
    changes = np.diff(yield_history, axis=0)

    # Remove any rows with NaN
    valid = ~np.any(np.isnan(changes), axis=1)
    changes = changes[valid]

    pca = PCA(n_components=min(n_components, changes.shape[1]))
    scores = pca.fit_transform(changes)

    names = ["Level", "Slope", "Curvature"][:n_components]
    if n_components > 3:
        names += [f"PC{i+1}" for i in range(3, n_components)]

    if maturities is None:
        maturities = np.arange(yield_history.shape[1])

    return PCAResult(
        loadings=pca.components_,
        explained_variance=pca.explained_variance_ratio_,
        scores=scores,
        maturities=np.asarray(maturities),
        component_names=names,
    )
