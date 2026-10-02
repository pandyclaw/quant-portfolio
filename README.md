# quant-portfolio

Multi-asset systematic trading framework with institutional-grade risk metrics, portfolio construction, and overfitting controls.

This repository implements eight systematic strategies across five asset classes (equities, futures, fixed income, volatility, multi-asset), supported by options pricing models, yield curve analytics, portfolio optimization (MVO, HRP, Risk Parity), and a historical stress testing framework. Every strategy is walk-forward validated with permutation testing — honest results including failures.

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

## Walk-Forward Validation Results

Every strategy is subjected to walk-forward validation with 5,000-iteration permutation testing. We report honest OOS results — including strategies that fail. This transparency is deliberate: showing what doesn't work is as important as showing what does.

### Strategy Results (OOS = second half of sample, 2020-2026)

| Strategy | OOS Sharpe | OOS Max DD | Trades | Perm p-value | Assessment |
|----------|-----------|-----------|--------|-------------|------------|
| TSMOM Rotation | +0.53 | 18.0% | — | — | Best performer: momentum premia persists |
| VWAP Sigma + Trend | +0.51 | 13.4% | 213 | 0.010 | Significant: multi-sigma deviation from VWAP |
| BB Multi-Sigma | +0.51 | 13.2% | 215 | 0.020 | Significant: Bollinger Band z-score MR |
| Vol-Scaled Momentum | +0.16 | 18.0% | 532 | 0.47 | Positive but not significant |
| Turn-of-Month | +0.09 | 19.0% | 71 | 0.50 | Calendar effect not significant in 2020-2026 |
| Intraday Mean Rev | +0.01 | 19.3% | 131 | 0.93 | Signal present, not significant OOS |
| FOMC Pre-Drift | -0.13 | 8.8% | 48 | 0.62 | Lucca & Moench anomaly arbitraged away |
| Gold/Silver Pairs | -0.49 | 82.9% | 113 | — | Cointegration weakened post-2020 |
| OpEx Week | -0.34 | 11.0% | 72 | 0.84 | Gamma-hedging effect not tradable as simple long |
| 5-min VWAP MR | -0.65 | 2.1% | 28 | 0.0003 | **Significant but wrong direction** — intraday MR is momentum continuation |

*Note: OOS metrics are from the second half of the sample only. No in-sample results reported. Permutation p < 0.10 = statistically significant time-series structure. Strategies tested: 35+ total across daily, intraday, and multi-asset.*

### Key Findings from Validation

1. **Most published anomalies don't survive OOS testing.** Of 35+ strategies tested, fewer than half produce positive OOS Sharpe after costs.
2. **FOMC pre-drift is dead.** Despite the Lucca & Moench (2015) NY Fed paper, the 49bps pre-FOMC anomaly has been arbitraged away in 2020-2026 data.
3. **Intraday MR ≠ daily MR.** Daily mean reversion works (oversold bounces recover). But on 5-minute bars, VWAP -2σ touches are momentum continuation — the opposite of reversal. This was a statistically significant finding (p=0.0003).
4. **Multi-sigma strategies show promise.** VWAP and Bollinger Band z-score strategies with trend filters produce significant OOS results (p < 0.05).
5. **Calendar effects are mostly dead.** Turn-of-month, pre/post holiday, and OpEx week effects do not survive after 2020.

## Philosophy

- **No zero-cost backtests.** Every simulation includes transaction costs, slippage, and spread modeling. T+1 execution enforced (no look-ahead on fill prices).
- **No uncorrected Sharpe.** Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) corrects for multiple testing. Bootstrap confidence intervals quantify uncertainty.
- **Honest limitations.** Strategies that underperform out-of-sample are reported as-is. Regime dependency is documented, not hidden.
- **35+ strategies tested.** Not just the winners — we show the failures. The validation framework is the product, not the individual strategies.
- **Multi-asset depth.** Options pricing (Black-Scholes, Greeks, IV solver), fixed income analytics (Nelson-Siegel, duration, convexity, DV01), volatility surface construction, and yield curve PCA.
- **Portfolio construction.** Four optimization methods compared: Mean-Variance (Ledoit-Wolf shrinkage), Hierarchical Risk Parity, Risk Parity, and Minimum Variance.
- **Stress tested.** Every strategy evaluated against GFC 2008, COVID 2020, 2022 Rate Hikes, Volmageddon 2018, and Taper Tantrum 2013.

## Quick Start

```bash
git clone https://github.com/pandyclaw/quant-portfolio.git
cd quant-portfolio
pip install -e ".[dev]"

# Run all backtests
python scripts/run_backtest.py --strategy all

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

## Validation Methodology

This framework uses a 4-stage validation pipeline, modeled on institutional practice:

1. **Walk-Forward Analysis** (Pardo WFA) — expanding windows with 24-month train / 6-month test splits across 10+ folds
2. **Permutation Testing** (5,000 iterations) — shuffles signal timing to test if returns are distinguishable from random
3. **Bootstrap Confidence Intervals** (5,000 resamples) — non-parametric 95% CI on Sharpe ratio
4. **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014) — corrects for multiple testing across all 35+ strategies tested

A strategy must achieve permutation p < 0.10 AND bootstrap P(Sharpe > 0) > 80% to be considered validated.

## Portfolio Construction

Four methods implemented and compared on strategy return streams:

| Method | Requires Returns? | Requires Covariance? | Robustness | Reference |
|--------|:-:|:-:|:-:|-----------|
| Mean-Variance (Ledoit-Wolf) | Yes | Yes (shrunk) | Low | Markowitz 1952 |
| Minimum Variance | No | Yes (shrunk) | Medium | — |
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

- Ariel, R. (1990). "High Stock Returns Before Holidays." *Journal of Finance*.
- Asness, C., Frazzini, A. & Pedersen, L. (2012). "Leverage Aversion and Risk Parity." *Financial Analysts Journal*.
- Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.
- Barroso, P. & Santa-Clara, P. (2015). "Momentum Has Its Moments." *Journal of Financial Economics*.
- Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*.
- Carr, P. & Wu, L. (2009). "Variance Risk Premiums." *Review of Financial Studies*.
- Connors, L. (2009). "Short Term Trading Strategies That Work." *TradingMarkets*.
- Diebold, F. X. & Li, C. (2006). "Forecasting the Term Structure of Government Bond Yields." *Journal of Econometrics*.
- French, K. R. & Roll, R. (1986). "Stock Return Variances." *Journal of Financial Economics*.
- Gatev, E. et al. (2006). "Pairs Trading." *Review of Financial Studies*.
- Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*.
- Koijen, R. et al. (2018). "Carry." *Journal of Financial Economics*.
- Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample." *Journal of Portfolio Management*.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lucca, D. O. & Moench, E. (2015). "The Pre-FOMC Announcement Drift." *Journal of Finance*.
- Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios." *Journal of Finance*.
- Moskowitz, T. J. et al. (2012). "Time Series Momentum." *Journal of Financial Economics*.
- Nelson, C. R. & Siegel, A. F. (1987). "Parsimonious Modeling of Yield Curves." *Journal of Business*.
- Pardo, R. (2008). *Design, Testing and Optimization of Trading Systems*. Wiley.

## Author

**Anthony Shi** — Quantitative researcher building systematic trading infrastructure across equity indices, precious metals, fixed income, and cross-asset momentum. Background in venture capital analysis and software engineering. Paper trading systematic strategies across multiple alpha families with institutional-grade validation (walk-forward, Deflated Sharpe, permutation testing). 35+ strategies tested, honest results reported.

## Disclaimers

This repository is for educational and research purposes only. It does not constitute investment advice. Past performance does not guarantee future results. All backtest results include transaction cost and slippage estimates, but actual execution may differ materially.

## License

MIT
