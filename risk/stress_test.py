"""Historical stress testing framework.

Replays strategy returns through historical market crises to assess
tail risk behavior. Every strategy should be evaluated against at
least GFC 2008, COVID 2020, and the 2022 rate hike cycle.

The stress test answers: "If this crisis happened again, how much
would the strategy lose, and how long would recovery take?"

Defined scenarios:
    GFC_2008:         Sep 2008 - Mar 2009 (credit crisis, -50% equity)
    VOLMAGEDDON_2018: Jan-Feb 2018 (XIV blowup, VIX spike)
    COVID_2020:       Feb-Mar 2020 (pandemic crash, -34% in 23 days)
    RATE_HIKE_2022:   Jan-Oct 2022 (aggressive Fed tightening)
    TAPER_TANTRUM_2013: May-Sep 2013 (bond market sell-off)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from utils.metrics import max_drawdown, sharpe_ratio

# Scenario definitions: date ranges for historical crises
SCENARIOS: dict[str, dict[str, str]] = {
    "GFC_2008": {"start": "2008-09-01", "end": "2009-03-31", "description": "Global Financial Crisis — credit collapse, equity -50%"},
    "VOLMAGEDDON_2018": {"start": "2018-01-26", "end": "2018-03-31", "description": "XIV implosion, VIX spike from 11 to 50"},
    "COVID_2020": {"start": "2020-02-19", "end": "2020-04-30", "description": "Pandemic crash — fastest bear market in history"},
    "RATE_HIKE_2022": {"start": "2022-01-03", "end": "2022-10-31", "description": "Aggressive Fed tightening, bonds and equities both down"},
    "TAPER_TANTRUM_2013": {"start": "2013-05-01", "end": "2013-09-30", "description": "Bond market sell-off on Fed tapering signal"},
}


@dataclass(frozen=True)
class ScenarioResult:
    """Result of a single stress scenario."""

    scenario_name: str
    description: str
    period: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    worst_day: float
    recovery_days: int  # Days to recover from max drawdown
    n_days: int


def apply_scenario(
    returns: pd.Series,
    scenario_name: str,
) -> ScenarioResult | None:
    """Extract strategy performance during a historical scenario.

    Parameters
    ----------
    returns
        Full strategy return series (must span the scenario period).
    scenario_name
        Key from SCENARIOS dict.

    Returns
    -------
    ScenarioResult or None if the return series doesn't cover the period.
    """
    if scenario_name not in SCENARIOS:
        return None

    scenario = SCENARIOS[scenario_name]
    start, end = scenario["start"], scenario["end"]

    # Filter returns to scenario period
    mask = (returns.index >= start) & (returns.index <= end)
    scenario_returns = returns[mask]

    if len(scenario_returns) < 5:
        return None

    equity = (1 + scenario_returns).cumprod()
    total_ret = float(equity.iloc[-1] - 1)
    mdd = max_drawdown(equity)
    sr = sharpe_ratio(scenario_returns)
    worst = float(scenario_returns.min())

    # Recovery: days from max drawdown trough to previous peak
    peak = equity.cummax()
    in_dd = equity < peak
    recovery = 0
    if in_dd.any():
        last_dd_idx = in_dd[::-1].idxmax()
        recovery = int((returns.index[-1] - last_dd_idx).days) if last_dd_idx < returns.index[-1] else 0

    return ScenarioResult(
        scenario_name=scenario_name,
        description=scenario["description"],
        period=f"{start} to {end}",
        total_return=total_ret,
        max_drawdown=mdd,
        sharpe_ratio=sr,
        worst_day=worst,
        recovery_days=recovery,
        n_days=len(scenario_returns),
    )


def stress_test_all(
    returns: pd.Series,
    scenarios: list[str] | None = None,
) -> list[ScenarioResult]:
    """Run all defined stress scenarios on a return series.

    Parameters
    ----------
    returns
        Strategy return series.
    scenarios
        List of scenario names. Default: all defined scenarios.

    Returns
    -------
    list[ScenarioResult]
        Results for each scenario the return series covers.
    """
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    results = []
    for name in scenarios:
        result = apply_scenario(returns, name)
        if result is not None:
            results.append(result)

    return results


def stress_test_portfolio(
    strategy_returns: dict[str, pd.Series],
    weights: dict[str, float],
    scenario_name: str,
) -> ScenarioResult | None:
    """Stress test a portfolio of strategies.

    Parameters
    ----------
    strategy_returns
        Dict mapping strategy name to return series.
    weights
        Portfolio weights per strategy.
    scenario_name
        Scenario to apply.
    """
    # Combine strategy returns using weights
    combined = pd.Series(0.0, dtype=float)
    for name, returns in strategy_returns.items():
        w = weights.get(name, 0.0)
        if w > 0:
            combined = combined.add(returns * w, fill_value=0.0)

    if combined.empty:
        return None

    return apply_scenario(combined, scenario_name)


def format_stress_report(results: list[ScenarioResult]) -> str:
    """Format stress test results as a readable report."""
    lines = ["=" * 70, "STRESS TEST REPORT", "=" * 70, ""]

    for r in results:
        lines.append(f"Scenario: {r.scenario_name}")
        lines.append(f"  Period:     {r.period} ({r.n_days} days)")
        lines.append(f"  {r.description}")
        lines.append(f"  Return:     {r.total_return:+.2%}")
        lines.append(f"  Max DD:     {r.max_drawdown:.2%}")
        lines.append(f"  Sharpe:     {r.sharpe_ratio:.3f}")
        lines.append(f"  Worst Day:  {r.worst_day:+.2%}")
        lines.append("")

    return "\n".join(lines)
