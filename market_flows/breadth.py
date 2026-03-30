"""S&P 500 market breadth: % of constituents above key moving averages."""

import json
import logging
from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

from .config import DATA_DIR

logger = logging.getLogger(__name__)


def _cache_path(filename):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / filename


def fetch_sp500_tickers():
    """Scrape current S&P 500 tickers from Wikipedia.

    Caches to data/sp500_tickers.json; refreshes if cache is >7 days old.
    """
    cache = _cache_path("sp500_tickers.json")

    if cache.exists():
        try:
            cached = json.loads(cache.read_text())
            cached_date = datetime.fromisoformat(cached["date"])
            age_days = (datetime.now(UTC) - cached_date).days
            if age_days <= 7:
                return cached["tickers"]
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.debug("S&P 500 ticker cache invalid, refreshing")

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

        cache.write_text(json.dumps({
            "date": datetime.now(UTC).isoformat(),
            "tickers": tickers,
        }))
        return tickers

    except Exception as e:
        logger.warning("S&P 500 ticker scrape failed: %s", e)
        if cache.exists():
            try:
                return json.loads(cache.read_text())["tickers"]
            except (json.JSONDecodeError, KeyError):
                pass
        return None


def fetch_market_breadth():
    """Compute S&P 500 breadth: % above 50 DMA and 200 DMA.

    Downloads 1 year of daily closes for all S&P 500 constituents,
    computes per-ticker moving average crossovers, and aggregates.
    Caches to data/breadth.json (same-day check avoids redundant downloads).
    """
    cache = _cache_path("breadth.json")

    # Same-day check
    if cache.exists():
        try:
            cached = json.loads(cache.read_text())
            if cached.get("date") == datetime.now(UTC).strftime("%Y-%m-%d"):
                return cached
        except (json.JSONDecodeError, KeyError):
            pass

    tickers = fetch_sp500_tickers()
    if not tickers:
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    try:
        # Download in chunks of 50 to avoid yfinance limits
        all_closes = pd.DataFrame()
        for i in range(0, len(tickers), 50):
            chunk = tickers[i:i + 50]
            try:
                data = yf.download(chunk, period="1y", progress=False, threads=True)
                if "Close" in data.columns.get_level_values(0):
                    closes = data["Close"]
                elif len(chunk) == 1:
                    closes = data[["Close"]]
                    closes.columns = chunk
                else:
                    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
                all_closes = pd.concat([all_closes, closes], axis=1)
            except Exception as e:
                logger.warning("Breadth chunk %d-%d failed: %s", i, i + len(chunk), e)
                continue

        if all_closes.empty:
            return None

        # Compute moving averages
        ma50 = all_closes.rolling(50, min_periods=50).mean()
        ma200 = all_closes.rolling(200, min_periods=200).mean()

        above_50 = (all_closes > ma50).sum(axis=1)
        above_200 = (all_closes > ma200).sum(axis=1)
        total = all_closes.notna().sum(axis=1)

        pct_50 = (above_50 / total * 100).dropna()
        pct_200 = (above_200 / total * 100).dropna()

        # Advance/decline ratio for the latest day
        if len(all_closes) >= 2:
            daily_returns = all_closes.iloc[-1] / all_closes.iloc[-2] - 1
            advancers = (daily_returns > 0).sum()
            decliners = (daily_returns < 0).sum()
            ad_ratio = round(advancers / max(decliners, 1), 2)
        else:
            ad_ratio = None

        # Build time series for charting (sample to weekly to keep size small)
        pct_50_weekly = pct_50.resample("W-FRI").last().dropna()
        pct_200_weekly = pct_200.resample("W-FRI").last().dropna()

        result = {
            "date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "pct_above_50dma": {
                "dates": [d.strftime("%Y-%m-%d") for d in pct_50_weekly.index],
                "values": [round(v, 1) for v in pct_50_weekly.values],
            },
            "pct_above_200dma": {
                "dates": [d.strftime("%Y-%m-%d") for d in pct_200_weekly.index],
                "values": [round(v, 1) for v in pct_200_weekly.values],
            },
            "current_50": round(float(pct_50.iloc[-1]), 1) if len(pct_50) > 0 else None,
            "current_200": round(float(pct_200.iloc[-1]), 1) if len(pct_200) > 0 else None,
            "ad_ratio": ad_ratio,
            "total_tickers": int(total.iloc[-1]) if len(total) > 0 else 0,
        }

        cache.write_text(json.dumps(result))
        return result

    except Exception as e:
        logger.warning("Market breadth fetch failed: %s", e)
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return None
