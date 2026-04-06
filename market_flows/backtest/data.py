"""Data fetching and caching for backtesting.

Provides functions to fetch daily and intraday OHLCV data via yfinance,
with parquet-based disk caching to avoid redundant downloads.  Also
integrates with the orb-strategy project's intraday data when available.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

BACKTEST_DATA_DIR = DATA_DIR / "backtest"

# Path to the orb-strategy intraday cache (separate project)
ORB_INTRADAY_DIR = Path("C:/Users/vrike/orb-strategy/data/intraday")

# yfinance caps intraday downloads at ~60 calendar days per request
_YF_INTRADAY_MAX_DAYS = 59


def _ensure_eastern(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is timezone-aware US/Eastern.

    yfinance returns UTC for intraday and tz-naive or exchange-tz for daily.
    This normalises everything to US/Eastern.
    """
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yf.download and standardise names.

    Handles both single-ticker (flat columns) and multi-ticker (MultiIndex)
    results, returning a DataFrame with columns [Open, High, Low, Close, Volume].
    """
    if isinstance(df.columns, pd.MultiIndex):
        # yf.download with a single ticker still returns MultiIndex when
        # called with a list; collapse to flat columns.
        df = df.droplevel(level=1, axis=1) if df.columns.nlevels == 2 else df

    # Rename yfinance's "Price" level if present
    rename_map = {}
    for col in df.columns:
        lower = str(col).lower()
        for canonical in ("Open", "High", "Low", "Close", "Volume"):
            if lower == canonical.lower():
                rename_map[col] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)

    target_cols = ["Open", "High", "Low", "Close", "Volume"]
    present = [c for c in target_cols if c in df.columns]
    return df[present].copy()


# ---------------------------------------------------------------------------
# Daily data
# ---------------------------------------------------------------------------

def fetch_daily(
    ticker: str,
    start: str | datetime,
    end: str | datetime,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch daily OHLCV for *ticker* between *start* and *end* (inclusive).

    Results are cached to ``data/backtest/{ticker}/daily.parquet``.
    Set *force=True* to bypass the cache and re-download.

    Returns a DataFrame indexed by timezone-aware datetime (US/Eastern)
    with columns [Open, High, Low, Close, Volume].
    """
    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d")

    cache_dir = BACKTEST_DATA_DIR / ticker
    cache_path = cache_dir / "daily.parquet"

    if not force and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached = _ensure_eastern(cached)
        mask = (cached.index >= pd.Timestamp(start_str, tz="US/Eastern")) & (
            cached.index <= pd.Timestamp(end_str, tz="US/Eastern") + pd.Timedelta(days=1)
        )
        subset = cached.loc[mask]
        if not subset.empty:
            logger.debug("Daily cache hit for %s (%d rows)", ticker, len(subset))
            return subset

    logger.info("Fetching daily data for %s (%s → %s)", ticker, start_str, end_str)
    try:
        # yf.download end is exclusive, add 1 day
        end_dl = (pd.Timestamp(end_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start_str, end=end_dl, progress=False, threads=False)
    except Exception as e:
        logger.error("yfinance daily download failed for %s: %s", ticker, e)
        return pd.DataFrame()

    if df.empty:
        logger.warning("No daily data returned for %s", ticker)
        return pd.DataFrame()

    df = _normalize_ohlcv(df)
    df = _ensure_eastern(df)

    # Merge with any existing cache to grow the file over time
    if cache_path.exists():
        try:
            existing = pd.read_parquet(cache_path)
            existing = _ensure_eastern(existing)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        except Exception:
            pass  # overwrite on error

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

    mask = (df.index >= pd.Timestamp(start_str, tz="US/Eastern")) & (
        df.index <= pd.Timestamp(end_str, tz="US/Eastern") + pd.Timedelta(days=1)
    )
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Intraday data
# ---------------------------------------------------------------------------

def _fetch_intraday_chunk(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: str,
) -> pd.DataFrame:
    """Download a single intraday chunk (must be ≤60 days)."""
    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        df = yf.download(
            ticker, start=start_str, end=end_str,
            interval=interval, progress=False, threads=False,
        )
    except Exception as e:
        logger.warning("Intraday chunk failed for %s (%s→%s): %s", ticker, start_str, end_str, e)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    return _normalize_ohlcv(df)


def fetch_intraday(
    ticker: str,
    start: str | datetime,
    end: str | datetime,
    interval: str = "5m",
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch intraday OHLCV for *ticker*, chunking around yfinance's 60-day limit.

    Results are cached per date in ``data/backtest/{ticker}/intraday/{date}.parquet``.
    Set *force=True* to bypass the cache.

    Returns a DataFrame indexed by timezone-aware datetime (US/Eastern)
    with columns [Open, High, Low, Close, Volume].
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    cache_dir = BACKTEST_DATA_DIR / ticker / "intraday"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which trading dates we need
    all_dates = pd.bdate_range(start_ts, end_ts)
    frames: list[pd.DataFrame] = []
    dates_to_fetch: list[pd.Timestamp] = []

    # Check per-date caches first
    for dt in all_dates:
        date_str = dt.strftime("%Y-%m-%d")
        parquet_path = cache_dir / f"{date_str}.parquet"
        if not force and parquet_path.exists():
            try:
                cached = pd.read_parquet(parquet_path)
                cached = _ensure_eastern(cached)
                if not cached.empty:
                    frames.append(cached)
                    continue
            except Exception:
                pass
        dates_to_fetch.append(dt)

    if dates_to_fetch:
        # Group into ≤59-day chunks
        chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        chunk_start = dates_to_fetch[0]
        for i, dt in enumerate(dates_to_fetch):
            days_span = (dt - chunk_start).days
            if days_span >= _YF_INTRADAY_MAX_DAYS:
                chunks.append((chunk_start, dates_to_fetch[i - 1]))
                chunk_start = dt
        chunks.append((chunk_start, dates_to_fetch[-1]))

        for c_start, c_end in chunks:
            logger.info(
                "Fetching intraday %s for %s (%s → %s)",
                interval, ticker,
                c_start.strftime("%Y-%m-%d"), c_end.strftime("%Y-%m-%d"),
            )
            chunk_df = _fetch_intraday_chunk(ticker, c_start, c_end, interval)
            if chunk_df.empty:
                continue

            chunk_df = _ensure_eastern(chunk_df)

            # Cache each date individually
            for date_str, day_df in chunk_df.groupby(chunk_df.index.date):
                pq_path = cache_dir / f"{date_str}.parquet"
                day_df.to_parquet(pq_path)

            frames.append(chunk_df)

    if not frames:
        logger.warning("No intraday data for %s", ticker)
        return pd.DataFrame()

    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result


# ---------------------------------------------------------------------------
# Multi-ticker convenience
# ---------------------------------------------------------------------------

def fetch_multi(
    tickers: list[str],
    start: str | datetime,
    end: str | datetime,
    *,
    daily: bool = True,
    interval: str = "5m",
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        start: Start date.
        end: End date.
        daily: If True, fetch daily data; if False, fetch intraday.
        interval: Intraday interval (only used when daily=False).
        force: Bypass disk cache.

    Returns:
        dict mapping ticker → DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        if daily:
            df = fetch_daily(ticker, start, end, force=force)
        else:
            df = fetch_intraday(ticker, start, end, interval=interval, force=force)
        if not df.empty:
            results[ticker] = df
        else:
            logger.warning("No data returned for %s", ticker)
    return results


# ---------------------------------------------------------------------------
# ORB-strategy data integration
# ---------------------------------------------------------------------------

def _load_orb_date(ticker: str, date_str: str) -> pd.DataFrame | None:
    """Load a single date's intraday data from the orb-strategy cache."""
    path = ORB_INTRADAY_DIR / ticker / f"{date_str}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df = _ensure_eastern(df)
        # Standardise column names — orb-strategy uses title-case already
        df = _normalize_ohlcv(df)
        return df if not df.empty else None
    except Exception as e:
        logger.debug("Failed to read orb data %s: %s", path, e)
        return None


def load_orb_data(
    tickers: list[str],
    start: str | datetime,
    end: str | datetime,
) -> dict[str, pd.DataFrame]:
    """Load intraday data from orb-strategy's cache, falling back to yfinance.

    The orb-strategy project stores 5-min OHLCV parquet files at
    ``C:/Users/vrike/orb-strategy/data/intraday/{ticker}/{date}.parquet``.
    For any dates not available there, this function downloads from yfinance
    via :func:`fetch_intraday`.

    Returns:
        dict mapping ticker → DataFrame with timezone-aware US/Eastern index.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    trading_dates = pd.bdate_range(start_ts, end_ts)

    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        frames: list[pd.DataFrame] = []
        missing_dates: list[pd.Timestamp] = []

        for dt in trading_dates:
            date_str = dt.strftime("%Y-%m-%d")
            orb_df = _load_orb_date(ticker, date_str)
            if orb_df is not None:
                frames.append(orb_df)
            else:
                missing_dates.append(dt)

        # Fetch missing dates from yfinance
        if missing_dates:
            logger.info(
                "ORB cache miss for %s on %d dates, falling back to yfinance",
                ticker, len(missing_dates),
            )
            # Group consecutive missing dates into ranges for efficient fetching
            ranges = _group_consecutive_dates(missing_dates)
            for r_start, r_end in ranges:
                yf_df = fetch_intraday(ticker, r_start, r_end, interval="5m")
                if not yf_df.empty:
                    frames.append(yf_df)

        if frames:
            combined = pd.concat(frames).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            results[ticker] = combined
        else:
            logger.warning("No intraday data for %s", ticker)

    return results


def _group_consecutive_dates(
    dates: list[pd.Timestamp],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group a sorted list of dates into consecutive ranges.

    Returns list of (start, end) tuples for efficient batch downloading.
    """
    if not dates:
        return []

    dates = sorted(dates)
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    range_start = dates[0]
    prev = dates[0]

    for dt in dates[1:]:
        # Allow up to 3-day gaps (weekends) to still count as consecutive
        if (dt - prev).days > 3:
            ranges.append((range_start, prev))
            range_start = dt
        prev = dt

    ranges.append((range_start, prev))
    return ranges
