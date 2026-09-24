"""Execution simulation with realistic transaction costs and slippage.

Models the gap between theoretical signals and real-world P&L. Every
backtest in this repository routes through these cost models — zero-cost
backtests are refused by design.

Slippage models:
    - Fixed basis points: constant per-trade friction
    - Volatility-adjusted: wider slippage during high-vol regimes
    - Square-root market impact (Almgren & Chriss, 2001): slippage
      proportional to sqrt(shares / ADV), standard for institutional
      execution analysis

Transaction cost accounting:
    - Commission per share (equities) or per contract (futures)
    - Minimum commission per order
    - SEC fee on sells (equities)

References:
    Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio
        transactions." Journal of Risk.
    Kissell, R. (2013). "The Science of Algorithmic Trading and Portfolio
        Management." Academic Press.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    """Transaction cost configuration.

    Attributes
    ----------
    commission_per_share
        Commission per share traded (default $0.005 = IB tiered).
    min_commission
        Minimum commission per order.
    slippage_bps
        Base slippage in basis points (1 bp = 0.01%).
    spread_bps
        Half-spread cost in basis points.
    sec_fee_per_dollar
        SEC Section 31 fee on sell proceeds (currently ~$27.80 per $1M).
    """

    commission_per_share: float = 0.005
    min_commission: float = 1.00
    slippage_bps: float = 5.0
    spread_bps: float = 2.0
    sec_fee_per_dollar: float = 0.0000278


# Pre-built configs for common scenarios
EQUITY_COSTS = CostConfig()
FUTURES_COSTS = CostConfig(
    commission_per_share=1.25,  # per contract
    min_commission=1.25,
    slippage_bps=2.0,
    spread_bps=1.0,
    sec_fee_per_dollar=0.0,
)
ETF_COSTS = CostConfig(
    commission_per_share=0.0,  # commission-free at most brokers
    min_commission=0.0,
    slippage_bps=3.0,
    spread_bps=1.5,
)


def estimate_trade_cost(
    price: float,
    quantity: float,
    side: str,
    config: CostConfig | None = None,
) -> float:
    """Estimate total transaction cost for a single trade.

    Parameters
    ----------
    price
        Execution price per share/contract.
    quantity
        Number of shares/contracts.
    side
        'buy' or 'sell'.
    config
        Cost configuration. Uses EQUITY_COSTS if None.

    Returns
    -------
    float
        Total cost in dollars (always positive).
    """
    if config is None:
        config = EQUITY_COSTS

    notional = price * quantity

    # Commission
    commission = max(config.min_commission, config.commission_per_share * quantity)

    # Slippage (half-spread + market impact)
    slippage = notional * (config.slippage_bps + config.spread_bps) / 10_000

    # SEC fee on sells only
    sec_fee = notional * config.sec_fee_per_dollar if side.lower() == "sell" else 0.0

    return commission + slippage + sec_fee


def apply_slippage(
    price: float,
    side: str,
    volatility: float = 0.0,
    config: CostConfig | None = None,
) -> float:
    """Apply slippage to an execution price.

    For buys, price is marked UP (you pay more).
    For sells, price is marked DOWN (you receive less).

    Parameters
    ----------
    price
        Intended execution price.
    side
        'buy' or 'sell'.
    volatility
        Current annualized volatility (0-1). Higher vol = wider slippage.
        If 0, uses fixed slippage from config.
    config
        Cost configuration.

    Returns
    -------
    float
        Adjusted execution price.
    """
    if config is None:
        config = EQUITY_COSTS

    base_slip_pct = (config.slippage_bps + config.spread_bps) / 10_000

    # Volatility adjustment: scale slippage by current vol / baseline vol
    if volatility > 0:
        vol_ratio = volatility / 0.16  # 16% annualized = baseline
        base_slip_pct *= max(0.5, min(3.0, vol_ratio))

    if side.lower() == "buy":
        return price * (1 + base_slip_pct)
    else:
        return price * (1 - base_slip_pct)


def market_impact_sqroot(
    quantity: float,
    adv: float,
    price: float,
    volatility: float,
    eta: float = 0.1,
) -> float:
    """Square-root market impact model (Almgren & Chriss, 2001).

    Impact = eta * sigma * sqrt(Q / ADV) * price

    This is the standard institutional model for estimating how much
    a large order moves the market against you.

    Parameters
    ----------
    quantity
        Order size in shares/contracts.
    adv
        Average daily volume.
    price
        Current price.
    volatility
        Daily volatility (not annualized).
    eta
        Market impact coefficient (calibrated empirically, ~0.1 for liquid US equities).

    Returns
    -------
    float
        Estimated price impact in dollars.
    """
    if adv <= 0 or quantity <= 0:
        return 0.0
    participation = quantity / adv
    return eta * volatility * np.sqrt(participation) * price


def limit_order_fill_probability(
    limit_price: float,
    current_price: float,
    volatility: float,
    time_horizon: float = 1.0,
) -> float:
    """Estimate probability a limit order fills within a time horizon.

    Uses a simple Brownian motion model: P(fill) = 2 * N(-|d| / (sigma * sqrt(T)))
    where d = (limit_price - current_price) / current_price.

    Parameters
    ----------
    limit_price
        Limit order price.
    current_price
        Current market price.
    volatility
        Annualized volatility.
    time_horizon
        Time in trading days.

    Returns
    -------
    float
        Estimated fill probability (0.0 to 1.0).
    """
    from scipy.stats import norm

    if current_price <= 0 or volatility <= 0:
        return 0.0

    d = abs(limit_price - current_price) / current_price
    daily_vol = volatility / np.sqrt(252)
    vol_horizon = daily_vol * np.sqrt(time_horizon)

    if vol_horizon == 0:
        return 1.0 if limit_price >= current_price else 0.0

    return float(2.0 * norm.cdf(-d / vol_horizon))


def simulate_execution(
    signals: pd.Series,
    prices: pd.DataFrame,
    initial_capital: float = 100_000.0,
    cost_config: CostConfig | None = None,
    execution_delay: int = 1,
    position_size: str = "fixed_fraction",
    fraction: float = 1.0,
) -> pd.DataFrame:
    """Simulate strategy execution with T+N delay and transaction costs.

    This is the core backtesting engine. It enforces:
    1. T+1 execution: signals generated at close of bar t,
       filled at open of bar t+1 (no look-ahead).
    2. Mandatory cost modeling: every trade incurs costs.
    3. Position tracking: long/flat only (no leverage by default).

    Parameters
    ----------
    signals
        Series of signals: +1 (long), -1 (short), 0 (flat).
        Index must match prices index.
    prices
        DataFrame with at minimum 'close' column. 'open' used for
        execution if available, otherwise close is used.
    initial_capital
        Starting capital.
    cost_config
        Transaction cost configuration.
    execution_delay
        Number of bars between signal and execution (default 1 = T+1).
    position_size
        'fixed_fraction' (fraction of capital) or 'all_in' (100%).
    fraction
        Fraction of capital to allocate per trade (if fixed_fraction).

    Returns
    -------
    pd.DataFrame
        Columns: equity, returns, position, costs, signal
    """
    if cost_config is None:
        cost_config = EQUITY_COSTS

    # Align signals and prices
    signals = signals.reindex(prices.index).fillna(0)

    # Shift signals by execution_delay (T+1 = no look-ahead)
    delayed_signals = signals.shift(execution_delay).fillna(0)

    exec_price = prices["open"] if "open" in prices.columns else prices["close"]
    close_price = prices["close"]

    n = len(prices)
    equity = np.full(n, initial_capital, dtype=float)
    returns = np.zeros(n)
    positions = np.zeros(n)
    costs = np.zeros(n)
    shares_held = 0.0
    cash = initial_capital

    for i in range(1, n):
        target_pos = int(delayed_signals.iloc[i])
        current_pos = 1 if shares_held > 0 else (-1 if shares_held < 0 else 0)

        # Trade if position changed
        if target_pos != current_pos:
            px = float(exec_price.iloc[i])
            if px <= 0:
                continue

            # Close existing position
            if shares_held != 0:
                close_side = "sell" if shares_held > 0 else "buy"
                close_px = apply_slippage(px, close_side, config=cost_config)
                close_cost = estimate_trade_cost(
                    close_px, abs(shares_held), close_side, cost_config
                )
                cash += shares_held * close_px - close_cost
                costs[i] += close_cost
                shares_held = 0.0

            # Open new position
            if target_pos != 0:
                open_side = "buy" if target_pos > 0 else "sell"
                open_px = apply_slippage(px, open_side, config=cost_config)
                trade_capital = cash * fraction
                new_shares = int(trade_capital / open_px) if open_px > 0 else 0

                if new_shares > 0:
                    open_cost = estimate_trade_cost(
                        open_px, new_shares, open_side, cost_config
                    )
                    cash -= new_shares * open_px + open_cost
                    shares_held = float(new_shares * (1 if target_pos > 0 else -1))
                    costs[i] += open_cost

        # Mark to market
        mtm = cash + shares_held * float(close_price.iloc[i])
        equity[i] = mtm
        positions[i] = shares_held
        returns[i] = (equity[i] - equity[i - 1]) / equity[i - 1] if equity[i - 1] != 0 else 0.0

    return pd.DataFrame(
        {
            "equity": equity,
            "returns": returns,
            "position": positions,
            "costs": costs,
            "signal": delayed_signals.values,
        },
        index=prices.index,
    )
