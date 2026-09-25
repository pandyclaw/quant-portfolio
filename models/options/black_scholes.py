"""Black-Scholes-Merton option pricing, Greeks, and implied volatility.

The BSM model assumes log-normal price dynamics under risk-neutral measure:
    dS = (r - q) S dt + sigma S dW

where S is the underlying price, r is the risk-free rate, q is the
dividend yield, sigma is volatility, and W is a Wiener process.

The model is wrong (volatility is not constant, returns are not normal),
but it remains the lingua franca of derivatives markets. Every trader
knows BSM — they use it not because it's accurate, but because it provides
a common pricing framework and a set of risk sensitivities (Greeks) that
are additive across a portfolio.

References:
    Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate
        Liabilities." Journal of Political Economy.
    Merton, R. C. (1973). "Theory of Rational Option Pricing."
        Bell Journal of Economics and Management Science.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class GreeksResult:
    """Option Greeks (risk sensitivities).

    delta: dV/dS — sensitivity to underlying price
    gamma: d²V/dS² — sensitivity of delta to underlying price
    theta: dV/dt — time decay (per calendar day)
    vega: dV/dsigma — sensitivity to volatility (per 1% vol change)
    rho: dV/dr — sensitivity to interest rate (per 1% rate change)
    """

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BSM d1 parameter."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BSM d2 parameter."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Black-Scholes call option price.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to expiry in years
    r : risk-free rate (annualized, continuous)
    sigma : volatility (annualized)
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def put_price(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Black-Scholes put option price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> GreeksResult:
    """Compute all BSM Greeks for an option.

    Parameters
    ----------
    S, K, T, r, sigma : standard BSM parameters
    option_type : 'call' or 'put'
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        d = 1.0 if (option_type == "call" and S > K) else (-1.0 if option_type == "put" and K > S else 0.0)
        return GreeksResult(delta=d, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-r * T)

    # Gamma is the same for calls and puts
    gamma = float(pdf_d1 / (S * sigma * sqrt_T))

    # Vega is the same for calls and puts (per 1% vol move)
    vega = float(S * pdf_d1 * sqrt_T / 100)

    if option_type == "call":
        delta = float(norm.cdf(d1))
        theta = float(
            (-S * pdf_d1 * sigma / (2 * sqrt_T)) - r * K * discount * norm.cdf(d2)
        ) / 365  # Per calendar day
        rho = float(K * T * discount * norm.cdf(d2) / 100)
    else:
        delta = float(norm.cdf(d1) - 1)
        theta = float(
            (-S * pdf_d1 * sigma / (2 * sqrt_T)) + r * K * discount * norm.cdf(-d2)
        ) / 365
        rho = float(-K * T * discount * norm.cdf(-d2) / 100)

    return GreeksResult(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Solve for implied volatility using Newton-Raphson.

    The Newton-Raphson method converges quickly for IV because
    vega (the derivative of price w.r.t. sigma) is always positive
    for options with time remaining, making the function monotonic.

    Parameters
    ----------
    market_price : observed option price
    S, K, T, r : standard BSM parameters
    option_type : 'call' or 'put'
    max_iter : maximum Newton-Raphson iterations
    tol : convergence tolerance

    Returns
    -------
    float
        Implied volatility. Returns NaN if no convergence.
    """
    if T <= 0:
        return float("nan")

    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * market_price / S

    # Clamp initial guess
    sigma = max(0.01, min(sigma, 5.0))

    price_fn = call_price if option_type == "call" else put_price

    for _ in range(max_iter):
        price = price_fn(S, K, T, r, sigma)
        diff = price - market_price

        if abs(diff) < tol:
            return float(sigma)

        # Vega (not scaled — raw dPrice/dSigma)
        d1 = _d1(S, K, T, r, sigma)
        vega_raw = S * norm.pdf(d1) * np.sqrt(T)

        if vega_raw < 1e-12:
            break

        sigma -= diff / vega_raw
        sigma = max(0.001, min(sigma, 10.0))

    return float(sigma)


def put_call_parity_check(
    call_px: float, put_px: float, S: float, K: float, T: float, r: float
) -> float:
    """Check put-call parity: C - P = S - K*exp(-rT).

    Returns the deviation from parity. Large deviations indicate
    either an arbitrage opportunity or stale quotes.
    """
    theoretical_diff = S - K * np.exp(-r * T)
    actual_diff = call_px - put_px
    return float(actual_diff - theoretical_diff)
