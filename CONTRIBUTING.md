# Contributing

Thank you for your interest in this project. Contributions are welcome in the following areas:

## Strategy Improvements

- Additional robustness checks (regime-conditional analysis, stress testing)
- Alternative cost models or slippage assumptions
- Extended validation methods (CSCV, PBO)
- Bug fixes in signal generation logic

## Infrastructure

- Additional performance metrics
- Improved data providers
- Enhanced plotting/visualization
- Documentation improvements

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for any new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Ensure lint passes: `ruff check .`
6. Submit a pull request with a clear description

## Code Standards

- Python 3.11+, type hints, PEP 8
- All strategies must include transaction cost modeling (no zero-cost backtests)
- Signal generation must enforce T+1 execution (no look-ahead)
- Academic citations for any new methodology
- Edge cases handled gracefully (empty series, zero volatility, etc.)

## Reporting Issues

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS
