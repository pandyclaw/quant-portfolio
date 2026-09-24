#!/usr/bin/env python3
"""Trade Analyzer — institutional risk management tool.

Reads a CSV of historical trades and produces:
    1. Total return and cumulative equity curve
    2. Sharpe ratio and Sortino ratio (annualized)
    3. Profit factor (gross profits / gross losses)
    4. Maximum drawdown percentage and duration (trading days)
    5. Win rate and average risk-to-reward ratio

Risk Circuit Breaker:
    Monitors per-trade equity against running peak. When drawdown
    exceeds 5% from peak, halves position size until equity recovers
    to 95% of peak. Outputs side-by-side comparison of original vs
    circuit-breaker-adjusted equity curves with metric deltas.

    Why circuit breakers exist:
    - Behavioral: prevents emotional revenge trading during drawdowns
    - Capital preservation: smaller losses compound less against you
    - Regulatory precedent: CME Level 1/2/3 circuit breakers halt
      trading during extreme moves for the same structural reason

CSV format:
    Date,Ticker,Action,Quantity,Entry_Price,Exit_Price,PnL

Usage:
    python scripts/trade_analyzer.py                    # Use sample data
    python scripts/trade_analyzer.py --file trades.csv  # Your data
    python scripts/trade_analyzer.py --generate         # Create sample CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import (
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
)


def generate_sample_trades(output_path: str = "sample_trades.csv") -> pd.DataFrame:
    """Generate a synthetic trade CSV for testing.

    Creates 50 futures trades with realistic P&L distribution:
    ~55% win rate, mix of small wins and occasional large losses.
    """
    rng = np.random.RandomState(42)
    n = 50
    dates = pd.bdate_range("2024-01-02", periods=n)
    tickers = rng.choice(["ES", "NQ", "GC", "CL"], n)
    actions = ["BUY"] * n

    entry_prices = []
    exit_prices = []
    quantities = []
    pnls = []

    for i in range(n):
        ticker = tickers[i]
        base = {"ES": 4800, "NQ": 16500, "GC": 2050, "CL": 75}[ticker]
        qty = rng.choice([1, 2, 3])
        entry = base + rng.normal(0, base * 0.02)
        # Skewed returns: wins are smaller than losses on average
        if rng.random() < 0.55:  # 55% win rate
            pnl = abs(rng.normal(200, 100)) * qty  # Avg $200 win per contract
        else:
            pnl = -abs(rng.normal(350, 150)) * qty  # Avg $350 loss per contract
        tick_value = {"ES": 12.50, "NQ": 5.00, "GC": 10.00, "CL": 10.00}[ticker]
        price_change = pnl / (qty * tick_value)
        exit_price = entry + price_change

        entry_prices.append(round(entry, 2))
        exit_prices.append(round(exit_price, 2))
        quantities.append(qty)
        pnls.append(round(pnl, 2))

    df = pd.DataFrame({
        "Date": dates[:n].strftime("%Y-%m-%d"),
        "Ticker": tickers,
        "Action": actions,
        "Quantity": quantities,
        "Entry_Price": entry_prices,
        "Exit_Price": exit_prices,
        "PnL": pnls,
    })

    df.to_csv(output_path, index=False)
    print(f"Generated {n} sample trades → {output_path}")
    return df


def load_trades(file_path: str) -> pd.DataFrame:
    """Load and validate trade CSV."""
    df = pd.read_csv(file_path)
    required = ["Date", "Ticker", "Action", "Quantity", "Entry_Price", "Exit_Price", "PnL"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_equity_curve(trades: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.Series:
    """Build cumulative equity curve from trade P&L.

    Each trade's P&L is applied sequentially to the running equity.
    """
    equity = [initial_capital]
    for pnl in trades["PnL"]:
        equity.append(equity[-1] + pnl)
    return pd.Series(equity[1:], index=trades["Date"].values)


def apply_circuit_breaker(
    trades: pd.DataFrame,
    initial_capital: float = 100_000.0,
    dd_threshold: float = 0.05,
    scale_factor: float = 0.5,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply circuit breaker to trade sequence.

    When drawdown from peak exceeds dd_threshold, subsequent trades
    have their quantity (and PnL) scaled by scale_factor until equity
    recovers to 95% of peak.

    Parameters
    ----------
    trades
        Trade DataFrame with PnL column.
    initial_capital
        Starting capital.
    dd_threshold
        Drawdown threshold to trigger breaker (default 5%).
    scale_factor
        Position size multiplier during breaker (default 0.5 = half).

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        (adjusted_equity, adjusted_trades with 'Scale' and 'Adjusted_PnL' columns)
    """
    equity = initial_capital
    peak = equity
    in_breaker = False

    adjusted = trades.copy()
    adjusted["Scale"] = 1.0
    adjusted["Adjusted_PnL"] = 0.0
    equities = []

    for i in range(len(trades)):
        dd = (peak - equity) / peak if peak > 0 else 0.0

        if dd >= dd_threshold:
            in_breaker = True
        elif equity >= peak * 0.95:
            in_breaker = False

        scale = scale_factor if in_breaker else 1.0
        adj_pnl = trades["PnL"].iloc[i] * scale

        adjusted.iloc[i, adjusted.columns.get_loc("Scale")] = scale
        adjusted.iloc[i, adjusted.columns.get_loc("Adjusted_PnL")] = adj_pnl

        equity += adj_pnl
        peak = max(peak, equity)
        equities.append(equity)

    return pd.Series(equities, index=trades["Date"].values), adjusted


def analyze(trades: pd.DataFrame, initial_capital: float = 100_000.0) -> None:
    """Run full analysis and print results."""
    print("=" * 70)
    print("TRADE ANALYSIS REPORT")
    print("=" * 70)

    # Basic stats
    n_trades = len(trades)
    n_winners = (trades["PnL"] > 0).sum()
    n_losers = (trades["PnL"] <= 0).sum()
    total_pnl = trades["PnL"].sum()
    wr = n_winners / n_trades if n_trades > 0 else 0
    avg_win = trades[trades["PnL"] > 0]["PnL"].mean() if n_winners > 0 else 0
    avg_loss = abs(trades[trades["PnL"] <= 0]["PnL"].mean()) if n_losers > 0 else 0
    rr = avg_win / avg_loss if avg_loss > 0 else float("inf")

    print(f"\n{'Trades':.<40} {n_trades}")
    print(f"{'Winners':.<40} {n_winners} ({wr:.1%})")
    print(f"{'Losers':.<40} {n_losers} ({1-wr:.1%})")
    print(f"{'Total P&L':.<40} ${total_pnl:,.2f}")
    print(f"{'Total Return':.<40} {total_pnl/initial_capital:.2%}")
    print(f"{'Avg Win':.<40} ${avg_win:,.2f}")
    print(f"{'Avg Loss':.<40} ${avg_loss:,.2f}")
    print(f"{'Avg Risk:Reward':.<40} {rr:.2f}")

    # Equity curve
    equity = build_equity_curve(trades, initial_capital)
    returns = equity.pct_change().fillna(0)
    trade_rets = pd.Series(trades["PnL"].values)

    # Risk metrics
    sr = sharpe_ratio(returns)
    so = sortino_ratio(returns)
    mdd = max_drawdown(equity)
    mdd_dur = max_drawdown_duration(equity)
    pf = profit_factor(trade_rets)

    print(f"\n{'─ Risk Metrics ─':─^70}")
    # Sharpe uses total standard deviation (up + down moves) — this penalizes
    # large winning trades equally with large losing trades.
    print(f"{'Sharpe Ratio (annualized)':.<40} {sr:.3f}")
    # Sortino uses only downside deviation — only penalizes negative returns.
    # Better for asymmetric strategies (trend-following, momentum).
    print(f"{'Sortino Ratio (annualized)':.<40} {so:.3f}")
    # Annualization: daily metrics × sqrt(252) because volatility scales
    # with the square root of time under i.i.d. assumptions.
    print(f"{'Profit Factor':.<40} {pf:.3f}")
    # Drawdown = (peak - current) / peak, normalized by running maximum.
    print(f"{'Max Drawdown':.<40} {mdd:.2%}")
    print(f"{'Max DD Duration (days)':.<40} {mdd_dur}")

    # Circuit breaker analysis
    print(f"\n{'─ Circuit Breaker Analysis ─':─^70}")
    print("Threshold: 5% drawdown → halve position size until 95% recovery")
    print()

    cb_equity, cb_trades = apply_circuit_breaker(trades, initial_capital)
    cb_returns = cb_equity.pct_change().fillna(0)

    cb_sr = sharpe_ratio(cb_returns)
    cb_mdd = max_drawdown(cb_equity)
    cb_total = cb_equity.iloc[-1] - initial_capital

    breaker_active = (cb_trades["Scale"] < 1.0).sum()

    orig_ret = total_pnl / initial_capital
    cb_ret = cb_total / initial_capital
    delta_ret = (cb_total - total_pnl) / initial_capital
    print(f"{'Metric':<30} {'Original':>15} {'With Breaker':>15} {'Delta':>10}")
    print("-" * 70)
    print(f"{'Total Return':<30} {orig_ret:>14.2%} {cb_ret:>14.2%} {delta_ret:>+9.2%}")
    print(f"{'Sharpe Ratio':<30} {sr:>15.3f} {cb_sr:>15.3f} {cb_sr-sr:>+10.3f}")
    print(f"{'Max Drawdown':<30} {mdd:>14.2%} {cb_mdd:>14.2%} {cb_mdd-mdd:>+9.2%}")
    print(f"{'Breaker Active':<30} {'':>15} {breaker_active:>12} trades")

    # Equity curve comparison
    print(f"\n{'─ Equity Curve (last 20 trades) ─':─^70}")
    comparison = pd.DataFrame({
        "Original": equity.tail(20).values,
        "With_Breaker": cb_equity.tail(20).values,
    }, index=range(max(0, len(equity) - 20), len(equity)))
    print(comparison.to_string())

    print(f"\n{'=' * 70}")
    print("Analysis complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trade Analyzer")
    parser.add_argument("--file", type=str, help="Path to trades CSV")
    parser.add_argument("--generate", action="store_true", help="Generate sample CSV")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    args = parser.parse_args()

    if args.generate:
        generate_sample_trades()
        return

    if args.file:
        trades = load_trades(args.file)
    else:
        # Use inline sample data
        print("No file specified — using generated sample data.\n")
        generate_sample_trades("/tmp/sample_trades.csv")
        trades = load_trades("/tmp/sample_trades.csv")

    analyze(trades, args.capital)


if __name__ == "__main__":
    main()
