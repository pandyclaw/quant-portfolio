# quant-portfolio

Systematic quantitative trading strategies with institutional-grade risk metrics, execution simulation, and overfitting controls.

This repository contains research and implementation of three systematic strategies across precious metals pairs (statistical arbitrage), equity index futures (intraday mean reversion), and cross-asset momentum (TSMOM rotation). All strategies are evaluated using risk-adjusted performance metrics with realistic execution assumptions including slippage, transaction costs, and market impact modeling.

## Strategies

| Strategy | Asset Class | Method | Key Metric | Sample Period |
|----------|-------------|--------|------------|---------------|
| Gold/Silver Pairs | Precious Metals | Cointegration + Kalman | Sharpe, Half-Life | 2006-2024 |
| Intraday Mean Reversion | Equity Index (SPY/QQQ proxy) | Bollinger + OU Process | Sharpe, Max DD | 2012-2024 |
| TSMOM Rotation | Cross-Asset (SPY/TLT/GLD/DBC/EFA) | Time-Series Momentum | Sharpe, Turnover | 2012-2024 |

## Philosophy

- **No zero-cost backtests.** Every simulation includes transaction costs, slippage, and spread modeling. The execution simulator enforces T+1 execution (no look-ahead on fill prices).
- **No uncorrected Sharpe.** The Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) corrects for multiple testing. A Sharpe of 1.5 from 100 trials is meaningless — this repo quantifies that.
- **Honest limitations.** Strategies that underperform out-of-sample are reported as-is. Regime dependency (when a strategy works and when it doesn't) is documented, not hidden.
- **Reproducible.** Clone, install, run. All data from public sources (yfinance). Sample CSVs included for offline testing.

## Quick Start

```bash
git clone https://github.com/pandyclaw/quant-portfolio.git
cd quant-portfolio
pip install -e ".[dev]"

# Run all backtests
python scripts/run_backtest.py --strategy all

# Run a specific strategy
python scripts/run_backtest.py --strategy pairs --start 2015-01-01

# Analyze a CSV of trades
python scripts/trade_analyzer.py --generate  # Create sample data
python scripts/trade_analyzer.py --file sample_trades.csv

# Run tests
pytest tests/ -v
```

## Repository Structure

```
quant-portfolio/
├── strategies/             # Strategy implementations (fit, signal, backtest)
│   ├── pairs_gold_silver.py        # A: Cointegration pairs (Engle-Granger + Kalman)
│   ├── intraday_mean_reversion.py  # B: Bollinger/z-score mean reversion
│   └── tsmom_rotation.py           # C: Cross-asset time-series momentum
├── utils/                  # Reusable quantitative finance infrastructure
│   ├── metrics.py          # Sharpe, Sortino, Calmar, VaR/CVaR, profit factor
│   ├── execution_sim.py    # Slippage models, VWAP sim, market impact
│   ├── risk.py             # Kelly criterion, vol-targeting, circuit breakers
│   ├── validation.py       # Deflated Sharpe, bootstrap CI, permutation tests
│   ├── data.py             # Multi-source data loading (yfinance + CSV fallback)
│   └── plotting.py         # Equity curves, drawdown, monthly heatmaps
├── scripts/                # CLI tools
│   ├── run_backtest.py     # Run any strategy from command line
│   └── trade_analyzer.py   # Analyze trade CSV with circuit breaker comparison
├── tests/                  # pytest suite (57 tests)
├── research/               # Jupyter notebooks (hedge fund memo format)
├── docs/                   # Methodology documentation
└── data/sample/            # Small CSVs for offline CI testing
```

## Validation Framework

Every strategy undergoes three statistical tests before results are reported:

1. **Deflated Sharpe Ratio** — Corrects the observed Sharpe for the number of strategy variants tested. If you test 100 strategies and report the best, the expected maximum Sharpe of pure noise is ~2.3. The deflated Sharpe subtracts this bias.

2. **Bootstrap Confidence Interval** — Resamples the return series 5,000 times to produce a non-parametric 95% CI on the Sharpe ratio. A wide CI (e.g., 0.2 to 2.8) signals insufficient data.

3. **Permutation Test** — Shuffles the return time series and computes Sharpe on each permutation. If the real Sharpe exceeds 95% of permuted Sharpes (p < 0.05), the time-series structure contains signal.

## Methodology Notes

- **Annualization**: Daily metrics are annualized using sqrt(252) for volatility-based measures. This assumes i.i.d. returns, which is approximately true for daily data but breaks down for intraday.
- **Transaction costs**: Default 5 bps slippage + 2 bps half-spread for equities. Futures use contract-specific costs (see `execution_sim.py`).
- **Look-ahead bias prevention**: Signals are generated at bar close, execution at next bar open (T+1). Rolling statistics use only past data. IS/OOS split at the midpoint of the sample.
- **Position sizing**: Volatility-targeted by default — each position is sized so its contribution to portfolio vol matches a target (10% annualized). This normalizes risk across asset classes.

## References

- Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies*.
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). "Time Series Momentum." *Journal of Financial Economics*.
- Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security Returns." *Journal of Finance*.
- Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk*.
- Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal*.
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

## Author

**Anthony Shi** — Quantitative researcher focused on systematic trading across equity indices, precious metals, and cross-asset momentum. Background in venture capital analysis and software engineering. Currently building automated trading infrastructure with institutional-grade risk controls, paper trading systematic strategies, and developing custom TradingView indicators for real-time market analysis.

## Disclaimers

This repository is for educational and research purposes only. It does not constitute investment advice. Past performance, whether simulated or real, does not guarantee future results. All backtest results include transaction cost and slippage estimates, but actual execution may differ materially. The strategies presented here have not been validated on live capital.

## License

MIT
