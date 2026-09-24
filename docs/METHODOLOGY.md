# Methodology

## Backtest Framework

### Execution Model

All backtests use T+1 execution: signals generated at bar t close are filled at bar t+1 open. This prevents look-ahead bias in fill prices — the most common source of unrealistic backtest results.

The execution simulator (`utils/execution_sim.py`) models three cost components:

1. **Commission**: Per-share or per-contract, with minimum per order
2. **Slippage**: Base rate in basis points, optionally scaled by realized volatility
3. **Spread**: Half-spread cost applied on entry and exit

For large orders, the square-root market impact model (Almgren & Chriss, 2001) estimates price impact as:

```
Impact = η × σ × √(Q/ADV) × P
```

where η is the market impact coefficient (~0.1 for liquid US equities), σ is daily volatility, Q is order size, and ADV is average daily volume.

### Position Sizing

Three methods are implemented in `utils/risk.py`:

1. **Kelly Criterion**: Optimal growth-rate sizing. Quarter-Kelly (f* × 0.25) is used in practice — it captures ~75% of the geometric growth with ~50% of the variance.

2. **Volatility Targeting**: Each position is sized so its contribution to portfolio volatility matches a target (default 10% annualized). This normalizes risk across asset classes with different volatilities.

3. **Risk Parity**: Inverse-volatility weighting — lower-vol assets get higher weight. Used as a comparison benchmark for the TSMOM strategy.

### Statistical Validation

#### Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)

The Deflated Sharpe Ratio corrects the observed Sharpe for multiple testing. If a researcher tests N strategy variants and reports only the best, the expected maximum Sharpe of random noise grows as √(2 ln N). For N=100 trials, this is approximately 2.3.

The test statistic:

```
DSR = (SR_observed - E[max(SR)]) / SE(SR)
```

where SE(SR) accounts for non-normality (skewness and kurtosis) of the return distribution.

A p-value < 0.05 indicates the observed Sharpe is unlikely to have arisen from multiple testing alone.

#### Bootstrap Confidence Interval

The return series is resampled with replacement 5,000 times. For each sample, the Sharpe ratio is computed. The 2.5th and 97.5th percentiles form the 95% CI.

This is preferable to the parametric CI (which assumes normality) because financial returns are typically fat-tailed and skewed.

#### Permutation Test

The return series is randomly shuffled 1,000 times. For each permutation, the Sharpe is computed. The p-value is the fraction of permuted Sharpes that exceed the real Sharpe.

This tests whether the temporal structure of returns (which captures the strategy's signal) is significant. Random shuffling destroys serial correlation, mean-reversion, and momentum — if the real Sharpe is still not significantly better, the strategy has no time-series edge.

## Strategy-Specific Methodology

### A: Gold/Silver Pairs

**Cointegration test**: Engle-Granger two-step method. OLS regression of GLD on SLV, then ADF test on residuals. p < 0.05 required for entry.

**Dynamic hedge ratio**: Kalman filter with random-walk state transition. The hedge ratio β evolves as β_t = β_{t-1} + η_t, estimated online.

**Entry/exit**: Rolling z-score of the Kalman spread. Entry at |z| > 2.0, exit at |z| < 0.5, stop at |z| > 4.0.

**Known limitation**: GLD/SLV cointegration has weakened post-2020. Rolling cointegration tests in the research notebook show this honestly.

### B: Intraday Mean Reversion

**OU process**: The price deviation from the 20-period SMA is modeled as an Ornstein-Uhlenbeck process. The mean-reversion speed θ and half-life are estimated via AR(1) regression on the deviation series.

**Daily proxy**: Backtested on daily SPY/QQQ data as a proxy for ES/NQ futures. A production implementation would use 5-minute bars with RTH session filters (09:30-16:00 ET) and contract-specific slippage ($12.50/tick for ES).

**Kill-switch**: Trading halts when daily drawdown exceeds 2%. Resumes when equity recovers to 95% of the session peak.

### C: TSMOM Rotation

**Signal**: 12-month return minus 1-month return (the "12-1" momentum signal from Moskowitz et al. 2012). The 1-month skip avoids short-term reversal effects.

**Universe**: SPY (US equities), TLT (US Treasuries), GLD (gold), DBC (commodities), EFA (international equities).

**Rebalance**: Monthly. Turnover is tracked and costs are applied proportional to weight changes.

**Benchmarks**: Compared against SPY buy-and-hold, 60/40 SPY/TLT, and equal-weight buy-and-hold.

## Assumptions and Limitations

1. **Data quality**: yfinance provides adjusted close prices with corporate action adjustments. No bid-ask data is available, so spread costs are estimated.

2. **Survivorship bias**: The universe (GLD, SLV, SPY, TLT, etc.) consists of highly liquid ETFs that have existed throughout the sample period. No survivorship bias concern for these instruments.

3. **Regime dependency**: All strategies exhibit regime-dependent performance. Mean reversion works in ranging markets but fails in trends. Momentum works in trends but fails in chop. The research notebooks decompose performance by volatility regime.

4. **Sample size**: Strategies with fewer than 30 trades should be treated as preliminary. The Deflated Sharpe p-value reflects sample size limitations.

5. **Slippage estimation**: Fixed basis point slippage underestimates costs during high-volatility events. The vol-adjusted slippage model partially addresses this but remains an estimate.
