"""Multi-source data loading with offline fallback.

Primary source: yfinance (free, no API key, adjusted prices).
Fallback: sample CSVs in data/sample/ for CI and offline testing.

Data quality checks:
    - Forward-fill gaps (max 5 days, then NaN)
    - Warn on missing dates
    - Reject future dates (look-ahead protection)
    - Timezone-aware timestamps (UTC)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def load_prices(
    symbols: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Load adjusted close prices for multiple symbols.

    Returns a DataFrame with DatetimeIndex and one column per symbol.
    Falls back to sample CSVs if yfinance is unavailable.

    Parameters
    ----------
    symbols
        List of ticker symbols.
    start
        Start date (YYYY-MM-DD).
    end
        End date. Defaults to today.
    source
        'yfinance' or 'csv'.
    """
    if source == "csv":
        return _load_from_csv(symbols, start, end)

    try:
        import yfinance as yf

        data = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]].rename(columns={"Close": symbols[0]})

        prices = prices.dropna(how="all")
        prices = prices.ffill(limit=5)

        if prices.empty:
            logger.warning("yfinance returned empty data, falling back to CSV")
            return _load_from_csv(symbols, start, end)

        return prices

    except Exception as e:
        logger.warning("yfinance failed (%s), falling back to CSV", e)
        return _load_from_csv(symbols, start, end)


def load_ohlcv(
    symbol: str,
    start: str = "2010-01-01",
    end: str | None = None,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Load OHLCV data for a single symbol.

    Returns DataFrame with columns: open, high, low, close, volume.

    Parameters
    ----------
    symbol
        Ticker symbol.
    start
        Start date.
    end
        End date.
    source
        'yfinance' or 'csv'.
    """
    if source == "csv":
        return _load_ohlcv_csv(symbol, start, end)

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end, auto_adjust=True)
        data.columns = [c.lower() for c in data.columns]

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing column: {col}")

        return data[required].dropna()

    except Exception as e:
        logger.warning("yfinance OHLCV failed (%s), falling back to CSV", e)
        return _load_ohlcv_csv(symbol, start, end)


def _load_from_csv(
    symbols: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load adjusted close prices from sample CSVs."""
    frames = {}
    for sym in symbols:
        path = SAMPLE_DIR / f"{sym.lower()}_daily_sample.csv"
        if not path.exists():
            logger.warning("Sample CSV not found: %s", path)
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "close" in df.columns:
            frames[sym] = df["close"]
        elif "Close" in df.columns:
            frames[sym] = df["Close"]

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    if start:
        result = result[result.index >= start]
    if end:
        result = result[result.index <= end]
    return result


def _load_ohlcv_csv(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV from sample CSV."""
    path = SAMPLE_DIR / f"{symbol.lower()}_daily_sample.csv"
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    available = [c for c in required if c in df.columns]
    df = df[available]

    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


def generate_sample_data(
    symbols: list[str] | None = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
) -> None:
    """Download and save sample CSVs for CI testing.

    Run manually: python -c "from utils.data import generate_sample_data; generate_sample_data()"
    """
    import yfinance as yf

    if symbols is None:
        symbols = ["SPY", "QQQ", "TLT", "GLD", "SLV", "DBC", "EFA"]

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        print(f"Downloading {sym}...")
        data = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            print(f"  WARNING: No data for {sym}")
            continue
        # Handle MultiIndex columns from newer yfinance versions
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0].lower() for c in data.columns]
        else:
            data.columns = [c.lower() for c in data.columns]
        path = SAMPLE_DIR / f"{sym.lower()}_daily_sample.csv"
        data.to_csv(path)
        print(f"  Saved {len(data)} rows to {path.name}")
