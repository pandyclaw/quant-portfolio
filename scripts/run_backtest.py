#!/usr/bin/env python3
"""CLI entrypoint for running strategy backtests.

Usage:
    python scripts/run_backtest.py --strategy pairs
    python scripts/run_backtest.py --strategy mean_reversion
    python scripts/run_backtest.py --strategy tsmom
    python scripts/run_backtest.py --strategy all
    python scripts/run_backtest.py --strategy pairs --start 2015-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data import load_ohlcv, load_prices


def run_pairs(start: str, end: str, capital: float) -> None:
    """Run Gold/Silver pairs strategy backtest."""
    from strategies.pairs_gold_silver import GoldSilverPairs

    print("=" * 60)
    print("Strategy A: Gold/Silver Statistical Arbitrage")
    print("=" * 60)

    prices = load_prices(["GLD", "SLV"], start=start, end=end)
    if prices.empty or "GLD" not in prices.columns or "SLV" not in prices.columns:
        print("ERROR: Could not load GLD/SLV data. Run with --source csv for offline mode.")
        return

    strategy = GoldSilverPairs()
    result = strategy.backtest(prices["GLD"], prices["SLV"], initial_capital=capital)

    fr = result["fit_result"]
    m = result["metrics"]
    print(f"\n{'─ Cointegration Diagnostics ─':─^60}")
    print(f"  Cointegrated: {fr.is_cointegrated} (ADF p={fr.adf_pvalue:.4f})")
    print(f"  Half-life: {fr.half_life:.1f} bars")
    print(f"  Hurst exponent: {fr.hurst_exponent:.3f}")
    print(f"  Hedge ratio (OLS): {fr.hedge_ratio_ols:.4f}")

    _print_metrics(m)
    print(f"  Trades: {len(result['trades'])}")


def run_mean_reversion(start: str, end: str, capital: float) -> None:
    """Run intraday mean reversion backtest."""
    from strategies.intraday_mean_reversion import IntradayMeanReversion

    print("=" * 60)
    print("Strategy B: Intraday Mean Reversion (SPY proxy)")
    print("=" * 60)

    prices = load_ohlcv("SPY", start=start, end=end)
    if prices.empty:
        print("ERROR: Could not load SPY data.")
        return

    strategy = IntradayMeanReversion()
    result = strategy.backtest(prices, initial_capital=capital)

    m = result["metrics"]
    ou = result["ou_params"]
    print(f"\n{'─ OU Process ─':─^60}")
    print(f"  Theta: {ou.theta:.4f}")
    print(f"  Half-life: {ou.half_life:.1f} bars")

    _print_metrics(m)
    print(f"  Trades: {len(result['trades'])}")


def run_tsmom(start: str, end: str, capital: float) -> None:
    """Run TSMOM cross-asset rotation backtest."""
    from strategies.tsmom_rotation import TSMOMRotation

    print("=" * 60)
    print("Strategy C: TSMOM Cross-Asset Rotation")
    print("=" * 60)

    symbols = ["SPY", "TLT", "GLD", "DBC", "EFA"]
    prices = load_prices(symbols, start=start, end=end)
    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        print(f"WARNING: Missing data for {missing}")

    strategy = TSMOMRotation()
    result = strategy.backtest(prices, initial_capital=capital)

    _print_metrics(result["metrics"])
    print(f"\n{'─ Asset Activity ─':─^60}")
    for t in result["trades_summary"]:
        print(f"  {t['asset']:6s}  entries={t['entries']:3d}  active={t['active_pct']:.0f}%")


def _print_metrics(m: dict) -> None:
    """Print formatted metrics table."""
    print(f"\n{'─ Performance (OOS) ─':─^60}")
    print(f"  {'Total Return':.<35} {m.get('total_return', 0):.2%}")
    print(f"  {'Annualized Return':.<35} {m.get('annualized_return', 0):.2%}")
    print(f"  {'Annualized Volatility':.<35} {m.get('annualized_volatility', 0):.2%}")
    print(f"  {'Sharpe Ratio':.<35} {m.get('sharpe_ratio', 0):.3f}")
    print(f"  {'Sortino Ratio':.<35} {m.get('sortino_ratio', 0):.3f}")
    print(f"  {'Calmar Ratio':.<35} {m.get('calmar_ratio', 0):.3f}")
    print(f"  {'Max Drawdown':.<35} {m.get('max_drawdown', 0):.2%}")
    print(f"  {'Max DD Duration':.<35} {m.get('max_drawdown_duration', 0):.0f} days")
    print(f"  {'VaR (95%)':.<35} {m.get('var_95', 0):.3%}")
    print(f"  {'CVaR (95%)':.<35} {m.get('cvar_95', 0):.3%}")
    if "profit_factor" in m:
        print(f"  {'Profit Factor':.<35} {m['profit_factor']:.3f}")
    if "win_rate" in m:
        print(f"  {'Win Rate':.<35} {m['win_rate']:.1%}")
    if "information_ratio" in m:
        print(f"  {'Information Ratio':.<35} {m['information_ratio']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy backtests")
    parser.add_argument(
        "--strategy",
        choices=["pairs", "mean_reversion", "tsmom", "all"],
        default="all",
        help="Which strategy to backtest",
    )
    parser.add_argument("--start", default="2012-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    args = parser.parse_args()

    runners = {
        "pairs": run_pairs,
        "mean_reversion": run_mean_reversion,
        "tsmom": run_tsmom,
    }

    if args.strategy == "all":
        for name, runner in runners.items():
            try:
                runner(args.start, args.end, args.capital)
            except Exception as e:
                print(f"\n[{name}] FAILED: {e}")
            print()
    else:
        runners[args.strategy](args.start, args.end, args.capital)


if __name__ == "__main__":
    main()
