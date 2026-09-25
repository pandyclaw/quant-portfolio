"""Systematic trading strategy implementations.

Each strategy implements fit(), generate_signals(), and backtest() methods.
All strategies enforce T+1 execution and mandatory transaction cost modeling.

Strategies are organized by asset class:
    equities/    — equity and ETF strategies
    futures/     — futures and commodity strategies
    fixed_income/ — bond and yield curve strategies
    volatility/  — vol surface and VRP strategies
    multi_asset/ — cross-asset allocation strategies
"""

# Backward-compatible imports (existing strategies moved to subdirectories)
from strategies.equities.intraday_mean_reversion import IntradayMeanReversion
from strategies.equities.pairs_gold_silver import GoldSilverPairs
from strategies.futures.tsmom_rotation import TSMOMRotation

__all__ = ["GoldSilverPairs", "IntradayMeanReversion", "TSMOMRotation"]

# Compatibility shims: allow `from strategies.pairs_gold_silver import ...`
# These exist because tests import from the old flat paths.
import sys as _sys

_sys.modules["strategies.pairs_gold_silver"] = _sys.modules.get(
    "strategies.equities.pairs_gold_silver"
) or __import__("strategies.equities.pairs_gold_silver", fromlist=["GoldSilverPairs"])
_sys.modules["strategies.intraday_mean_reversion"] = _sys.modules.get(
    "strategies.equities.intraday_mean_reversion"
) or __import__("strategies.equities.intraday_mean_reversion", fromlist=["IntradayMeanReversion"])
_sys.modules["strategies.tsmom_rotation"] = _sys.modules.get(
    "strategies.futures.tsmom_rotation"
) or __import__("strategies.futures.tsmom_rotation", fromlist=["TSMOMRotation"])
