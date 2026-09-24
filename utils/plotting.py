"""Standardized backtest visualization.

Consistent chart styling across all strategy research notebooks.
Dark theme, institutional color palette, minimal chart junk.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Institutional color palette
COLORS = {
    "strategy": "#2196f3",
    "benchmark": "#888888",
    "positive": "#00e676",
    "negative": "#ff1744",
    "neutral": "#9e9e9e",
    "accent": "#ff9800",
}


def _style_axis(ax: plt.Axes) -> None:
    """Apply institutional chart styling."""
    ax.set_facecolor("#0a0a1a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")
    ax.tick_params(colors="#888")
    ax.xaxis.label.set_color("#888")
    ax.yaxis.label.set_color("#888")
    ax.title.set_color("#fff")


def plot_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    labels: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot equity curve with optional benchmark overlay.

    Parameters
    ----------
    equity
        Strategy equity curve.
    benchmark
        Benchmark equity curve (same index).
    title
        Chart title.
    labels
        Custom legend labels: {'strategy': '...', 'benchmark': '...'}.
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0a0a1a")
    _style_axis(ax)

    strat_label = (labels or {}).get("strategy", "Strategy")
    bench_label = (labels or {}).get("benchmark", "Benchmark")

    ax.plot(equity.index, equity.values, color=COLORS["strategy"], linewidth=1.5, label=strat_label)
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, color=COLORS["benchmark"],
                linewidth=1, linestyle="--", alpha=0.7, label=bench_label)

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel("Equity ($)", color="#888")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> plt.Figure:
    """Plot drawdown from running peak."""
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0a0a1a")
    _style_axis(ax)

    peak = equity.cummax()
    dd = (equity - peak) / peak

    ax.fill_between(dd.index, dd.values, 0, color=COLORS["negative"], alpha=0.4)
    ax.plot(dd.index, dd.values, color=COLORS["negative"], linewidth=0.8)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel("Drawdown", color="#888")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    title: str = "Rolling 1-Year Sharpe Ratio",
) -> plt.Figure:
    """Plot rolling Sharpe ratio over time."""
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0a0a1a")
    _style_axis(ax)

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std(ddof=1)
    rolling_sr = (rolling_mean / rolling_std * np.sqrt(252)).dropna()

    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in rolling_sr.values]
    ax.bar(rolling_sr.index, rolling_sr.values, color=colors, alpha=0.6, width=1)
    ax.axhline(y=0, color="#444", linewidth=0.5)
    ax.axhline(y=1.0, color=COLORS["accent"], linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel("Sharpe Ratio", color="#888")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns (%)",
) -> plt.Figure:
    """Plot monthly returns as a heatmap (year × month)."""
    import seaborn as sns

    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, max(3, len(table) * 0.5)), facecolor="#0a0a1a")
    _style_axis(ax)

    sns.heatmap(
        table, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
        linewidths=0.5, linecolor="#333", ax=ax,
        cbar_kws={"label": "Return (%)"},
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_trade_distribution(
    trade_returns: pd.Series,
    title: str = "Trade Return Distribution",
) -> plt.Figure:
    """Plot histogram of per-trade returns."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a1a")
    _style_axis(ax)

    colors = [COLORS["positive"] if r > 0 else COLORS["negative"] for r in trade_returns]
    ax.bar(range(len(trade_returns)), trade_returns.values, color=colors, alpha=0.7)
    ax.axhline(y=0, color="#444", linewidth=0.5)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Trade #", color="#888")
    ax.set_ylabel("P&L (%)", color="#888")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig
