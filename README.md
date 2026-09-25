# quant-portfolio

Multi-asset systematic trading framework with institutional-grade risk metrics, portfolio construction, and overfitting controls.

This repository implements eight systematic strategies across five asset classes (equities, futures, fixed income, volatility, multi-asset), supported by options pricing models, yield curve analytics, portfolio optimization (MVO, HRP, Risk Parity), and a historical stress testing framework.

## Strategies

| # | Strategy | Asset Class | Style | Academic Source | Key Metric |
|---|----------|-------------|-------|----------------|------------|
| A | Gold/Silver Pairs | Precious Metals | Relative Value | Gatev et al. 2006 | Sharpe, Half-Life |
| B | Intraday Mean Reversion | Equity Index | Mean Reversion | Jegadeesh 1990 | Sharpe, OU Half-Life |
| C | TSMOM Rotation | Cross-Asset | Momentum | Moskowitz et al. 2012 | Sharpe, Turnover |
| D | Cross-Sectional Momentum | Equity Sectors | Momentum | Jegadeesh & Titman 1993 | Sharpe, IC |
| E | Futures Carry | Commodities | Carry | Koijen et al. 2018 | Sharpe, Carry/Vol |
| F | Yield Curve Momentum | Fixed Income | Momentum | Diebold & Li 2006 | Sharpe, Duration |
| G | Variance Risk Premium | Volatility | Vol Premium | Carr & Wu 2009 | Sharpe, VRP |
| H | Risk Parity + Momentum | Multi-Asset | Diversification | Asness et al. 2012 | Sharpe, Risk Contribution |

## Philosophy

- **No zero-cost backtests.** Every simulation includes transaction costs, slippage, and spread modeling. T+1 execution enforced (no look-ahead on fill prices).
- **No uncorrected Sharpe.** Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) corrects for multiple testing. Bootstrap confidence intervals quantify uncertainty.
- **Honest limitations.** Strategies that underperform out-of-sample are reported as-is. Regime dependency is documented, not hidden. GLD/SLV cointegration breakdown post-2020 is shown honestly.
- **Multi-asset depth.** Not just equities — options pricing (Black-Scholes, Greeks, IV solver), fixed income analytics (Nelson-Siegel, duration, convexity, DV01), volatility surface construction, and yield curve PCA.
- **Portfolio construction.** Four optimization methods compared: Mean-Variance (Ledoit-Wolf shrinkage), Hierarchical Risk Parity, Risk Parity, and Minimum Variance.
- **Stress tested.** Every strategy evaluated against GFC 2008, COVID 2020, 2022 Rate Hikes, Volmageddon 2018, and Taper Tantrum 2013.

## Quick Start

```bash
git clone https://github.com/pandyclaw/quant-portfolio.git
cd quant-portfolio
pip install -e ".[dev]"

# Run all backtests
python scripts/run_backtest.py --strategy all

# Compare portfolio construction methods
python -c "from portfolio.optimizer import compare_methods; import pandas as pd; r = pd.DataFrame({'a': [0.01]*100, 'b': [0.005]*100}); print(compare_methods(r))"

# Analyze trades with circuit breaker
python scripts/trade_analyzer.py --generate
python scripts/trade_analyzer.py --file sample_trades.csv

# Run tests (118 tests)
pytest tests/ -v
```

## Architecture

```
quant-portfolio/
├── strategies/                    # 8 strategies across 5 asset classes
│   ├── base.py                    # Strategy protocol (ABC)
│   ├── equities/                  # Pairs, mean reversion, cross-sectional momentum
│   ├── futures/                   # TSMOM, carry
│   ├── fixed_income/              # Yield curve momentum
│   ├── volatility/                # Variance risk premium
│   └── multi_asset/               # Risk parity + momentum
├── models/                        # Quantitative models
│   ├── options/                   # Black-Scholes, vol surface, delta hedging
│   └── fixed_income/              # Nelson-Siegel, bond math (duration/convexity/DV01)
├── portfolio/                     # Portfolio construction
│   └── optimizer.py               # MVO, HRP, Risk Parity, Min Variance
├── risk/                          # Risk framework
│   ├── stress_test.py             # 5 historical crisis scenarios
│   └── correlation_monitor.py     # Rolling correlation + regime detection
├── utils/                         # Reusable infrastructure
│   ├── metrics.py                 # Sharpe, Sortino, Calmar, VaR/CVaR, 14 metrics
│   ├── execution_sim.py           # Slippage, market impact (Almgren-Chriss)
│   ├── risk.py                    # Kelly, vol-targeting, circuit breakers
│   ├── validation.py              # Deflated Sharpe, bootstrap CI, permutation test
│   ├── data.py                    # yfinance + CSV fallback
│   └── plotting.py                # Institutional dark-theme charts
├── tests/                         # 118 tests
├── research/                      # Jupyter notebooks (hedge fund memo format)
├── scripts/                       # CLI tools
└── docs/                          # Methodology documentation
```

## Portfolio Construction

Four methods implemented and compared on strategy return streams:

| Method | Requires Returns? | Requires Covariance? | Robustness | Reference |
|--------|:-:|:-:|:-:|-----------|
| Mean-Variance (Ledoit-Wolf) | Yes | Yes (shrunk) | Low | Markowitz 1952 |
| Minimum Variance | No | Yes (shrunk) | Medium | - |
| Risk Parity | No | Yes | High | Maillard et al. 2010 |
| HRP | No | Yes | Highest | Lopez de Prado 2016 |

## Risk Framework

### Stress Testing
Every strategy is evaluated against five historical crises:
- **GFC 2008** (Sep 08 - Mar 09): Credit collapse, -50% equities
- **Volmageddon 2018** (Jan-Mar 18): VIX spike, XIV implosion
- **COVID 2020** (Feb-Apr 20): Fastest bear market in history
- **Rate Hike 2022** (Jan-Oct 22): Bonds and equities both down
- **Taper Tantrum 2013** (May-Sep 13): Bond sell-off on Fed signal

### Validation Framework
1. **Deflated Sharpe** — corrects for multiple testing
2. **Bootstrap CI** — non-parametric Sharpe uncertainty
3. **Permutation Test** — tests time-series structure significance

## Quantitative Models

### Options
- Black-Scholes pricing (calls, puts, analytical Greeks)
- Newton-Raphson implied volatility solver
- Put-call parity verification
- Delta hedging simulation with P&L decomposition (delta/gamma/theta/vega)

### Fixed Income
- Nelson-Siegel yield curve fitting (level, slope, curvature)
- Bond pricing, Macaulay/modified duration, convexity, DV01
- Yield curve PCA decomposition (level/slope/curvature factors)

## References

- Asness, C., Frazzini, A. & Pedersen, L. (2012). "Leverage Aversion and Risk Parity." *Financial Analysts Journal*.
- Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.
- Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*.
- Carr, P. & Wu, L. (2009). "Variance Risk Premiums." *Review of Financial Studies*.
- Diebold, F. X. & Li, C. (2006). "Forecasting the Term Structure of Government Bond Yields." *Journal of Econometrics*.
- Gatev, E. et al. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies*.
- Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*.
- Koijen, R. et al. (2018). "Carry." *Journal of Financial Economics*.
- Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample." *Journal of Portfolio Management*.
- Moskowitz, T. J. et al. (2012). "Time Series Momentum." *Journal of Financial Economics*.
- Nelson, C. R. & Siegel, A. F. (1987). "Parsimonious Modeling of Yield Curves." *Journal of Business*.

## Author

**Anthony Shi** — Quantitative researcher building systematic trading infrastructure across equity indices, precious metals, fixed income, and cross-asset momentum. Background in venture capital analysis and software engineering. Paper trading systematic strategies with institutional-grade risk controls.

## Disclaimers

This repository is for educational and research purposes only. It does not constitute investment advice. Past performance does not guarantee future results. All backtest results include transaction cost and slippage estimates, but actual execution may differ materially.

## License

MIT
