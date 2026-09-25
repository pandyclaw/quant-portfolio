"""Delta hedging simulation with P&L decomposition.

Simulates the process of dynamically hedging a short option position
by trading the underlying. Decomposes the hedging P&L into its
component parts: delta, gamma, theta, and vega.

This decomposition is fundamental to options market making:
    - Theta P&L: premium collected from time decay (positive for sellers)
    - Gamma P&L: cost of discrete hedging (negative — gamma scalping)
    - Delta P&L: residual directional exposure from hedge lag
    - Vega P&L: impact of vol changes on option value

In a perfect BSM world with continuous hedging, theta and gamma
exactly offset. In practice, discrete hedging creates gamma drag,
and realized vol differing from implied vol creates the core P&L.

References:
    Taleb, N. N. (1997). "Dynamic Hedging." Wiley.
    Hull, J. C. (2021). "Options, Futures, and Other Derivatives." Pearson.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.options.black_scholes import call_price, greeks


@dataclass(frozen=True)
class DeltaHedgeResult:
    """Result of a delta hedging simulation."""

    pnl_total: float
    pnl_delta: float
    pnl_gamma: float
    pnl_theta: float
    pnl_components: pd.DataFrame  # Per-step breakdown
    hedge_trades: int
    avg_gamma_cost: float


def simulate_delta_hedge(
    price_path: np.ndarray,
    K: float,
    T: float,
    r: float = 0.05,
    sigma: float = 0.20,
    hedge_freq: int = 1,
    option_type: str = "call",
) -> DeltaHedgeResult:
    """Simulate delta hedging a short option over a price path.

    Parameters
    ----------
    price_path
        Array of underlying prices (one per time step).
    K
        Strike price.
    T
        Time to expiry in years at the start.
    r
        Risk-free rate.
    sigma
        Implied volatility used for hedging (may differ from realized).
    hedge_freq
        Rebalance every N steps (1 = every step).
    option_type
        'call' or 'put'.

    Returns
    -------
    DeltaHedgeResult
        Total P&L and per-component decomposition.
    """
    n = len(price_path)
    dt = T / n

    # Initialize: sell option, receive premium
    initial_premium = call_price(price_path[0], K, T, r, sigma)
    if option_type == "put":
        from models.options.black_scholes import put_price
        initial_premium = put_price(price_path[0], K, T, r, sigma)

    # Track P&L components
    delta_pnl = np.zeros(n)
    gamma_pnl = np.zeros(n)
    theta_pnl = np.zeros(n)
    shares_held = 0.0
    hedge_trades = 0

    for i in range(n - 1):
        S = price_path[i]
        tau = T - i * dt  # Time remaining

        if tau <= 0:
            break

        g = greeks(S, K, tau, r, sigma, option_type)

        # Rebalance hedge at specified frequency
        if i % hedge_freq == 0:
            target_shares = -g.delta  # Short option → hedge with +delta shares
            trade = target_shares - shares_held
            if abs(trade) > 1e-6:
                hedge_trades += 1
            shares_held = target_shares

        # Price change
        dS = price_path[i + 1] - price_path[i]

        # P&L decomposition for this step
        delta_pnl[i] = shares_held * dS  # Hedge P&L
        gamma_pnl[i] = 0.5 * g.gamma * dS**2  # Gamma P&L (from option, not hedge)
        theta_pnl[i] = -g.theta * 365 * dt  # Theta decay (option seller earns)

    # Final settlement: option payoff
    S_final = price_path[-1]
    if option_type == "call":
        payoff = max(S_final - K, 0)
    else:
        payoff = max(K - S_final, 0)

    # Total P&L = premium received - payoff + hedge P&L
    total_hedge_pnl = float(np.sum(delta_pnl))
    total_gamma = float(np.sum(gamma_pnl))
    total_theta = float(np.sum(theta_pnl))
    total_pnl = initial_premium - payoff + total_hedge_pnl

    components = pd.DataFrame({
        "delta_pnl": delta_pnl,
        "gamma_pnl": gamma_pnl,
        "theta_pnl": theta_pnl,
        "price": price_path,
    })

    return DeltaHedgeResult(
        pnl_total=total_pnl,
        pnl_delta=total_hedge_pnl,
        pnl_gamma=total_gamma,
        pnl_theta=total_theta,
        pnl_components=components,
        hedge_trades=hedge_trades,
        avg_gamma_cost=total_gamma / max(hedge_trades, 1),
    )
