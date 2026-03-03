"""External data sources (non-yfinance): FINRA, FRED, AAII, CBOE."""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR, FRED_SERIES


def _cache_path(filename):
    """Return path to a cache file in the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / filename


def fetch_margin_debt():
    """Fetch FINRA margin debt statistics from their public Excel file.

    Returns dict with dates, debit_balances, current values, and YoY change.
    Caches to data/margin_debt.json.
    """
    cache = _cache_path("margin_debt.json")
    try:
        url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        # Write to temp file for pandas to read
        tmp = cache.with_suffix(".xlsx")
        tmp.write_bytes(resp.content)

        df = pd.read_excel(tmp, engine="openpyxl")
        tmp.unlink(missing_ok=True)

        # Find the debit balances column (varies by sheet format)
        # Common column names: "Debit Balances in Customers' Securities Margin Accounts"
        debit_col = None
        date_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if "debit" in col_lower and "balance" in col_lower:
                debit_col = col
            if "date" in col_lower or "month" in col_lower or "year" in col_lower:
                date_col = col

        # If no obvious column names, use positional heuristic
        if date_col is None:
            date_col = df.columns[0]
        if debit_col is None:
            # Try second column as debit balances
            for col in df.columns[1:]:
                if df[col].dtype in ("float64", "int64") or pd.api.types.is_numeric_dtype(df[col]):
                    debit_col = col
                    break

        if debit_col is None:
            return None

        df = df[[date_col, debit_col]].dropna()
        df.columns = ["date", "debit"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna().sort_values("date")

        # Convert to millions if values look like they're in thousands
        if df["debit"].median() > 1e6:
            df["debit"] = df["debit"] / 1e6  # to billions
            unit = "B"
        elif df["debit"].median() > 1e3:
            df["debit"] = df["debit"] / 1e3  # to millions from thousands
            unit = "M"
        else:
            unit = "M"

        dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
        values = df["debit"].tolist()

        # YoY change
        yoy_change_pct = None
        if len(df) >= 12:
            current = values[-1]
            year_ago = values[-12] if values[-12] != 0 else values[-13]
            if year_ago:
                yoy_change_pct = round((current / year_ago - 1) * 100, 1)

        result = {
            "dates": dates,
            "debit_balances": values,
            "current_debit": round(values[-1], 1) if values else None,
            "current_date": dates[-1] if dates else None,
            "yoy_change_pct": yoy_change_pct,
            "unit": unit,
        }

        cache.write_text(json.dumps(result, default=str))
        return result

    except Exception as e:
        print(f"  Margin debt fetch failed: {e}")
        # Try loading from cache
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except Exception:
                pass
        return None


def fetch_fred_fund_flows():
    """Fetch fund flow data from FRED API.

    Requires FRED_API_KEY environment variable (free registration).
    Returns dict with series data. Caches to data/fred_flows.json.
    """
    cache = _cache_path("fred_flows.json")
    api_key = os.environ.get("FRED_API_KEY")

    if not api_key:
        print("  FRED_API_KEY not set, skipping fund flows")
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except Exception:
                pass
        return None

    try:
        series_list = []
        for key, (series_id, name) in FRED_SERIES.items():
            try:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={series_id}&api_key={api_key}"
                    f"&file_type=json&observation_start=2010-01-01"
                )
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                observations = data.get("observations", [])
                dates = []
                values = []
                for obs in observations:
                    if obs["value"] != ".":
                        dates.append(obs["date"])
                        values.append(float(obs["value"]))

                if dates:
                    series_list.append({
                        "id": series_id,
                        "key": key,
                        "name": name,
                        "dates": dates,
                        "values": values,
                    })
            except Exception as e:
                print(f"    FRED series {series_id} failed: {e}")
                continue

        result = {
            "has_data": len(series_list) > 0,
            "series": series_list,
        }

        cache.write_text(json.dumps(result))
        return result

    except Exception as e:
        print(f"  FRED fund flows fetch failed: {e}")
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except Exception:
                pass
        return None


def fetch_aaii_sentiment():
    """Fetch AAII Investor Sentiment Survey from Nasdaq Data Link.

    Requires NASDAQ_DATA_LINK_API_KEY environment variable (free registration).
    Returns dict with bullish/bearish/neutral percentages. Caches to data/aaii_sentiment.json.
    """
    cache = _cache_path("aaii_sentiment.json")
    api_key = os.environ.get("NASDAQ_DATA_LINK_API_KEY")

    if not api_key:
        print("  NASDAQ_DATA_LINK_API_KEY not set, skipping AAII sentiment")
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except Exception:
                pass
        return None

    try:
        url = (
            f"https://data.nasdaq.com/api/v3/datasets/AAII/AAII_SENTIMENT.json"
            f"?api_key={api_key}&rows=200"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        dataset = data.get("dataset", {})
        columns = dataset.get("column_names", [])
        rows = dataset.get("data", [])

        if not rows:
            return None

        # Map column indices
        col_map = {c.lower(): i for i, c in enumerate(columns)}
        date_idx = col_map.get("date", 0)
        bull_idx = col_map.get("bullish", 1)
        neutral_idx = col_map.get("neutral", 2)
        bear_idx = col_map.get("bearish", 3)

        # Data comes newest-first; reverse for chronological order
        rows = list(reversed(rows))

        dates = [r[date_idx] for r in rows]
        bullish = [r[bull_idx] * 100 if r[bull_idx] is not None else None for r in rows]
        neutral = [r[neutral_idx] * 100 if r[neutral_idx] is not None else None for r in rows]
        bearish = [r[bear_idx] * 100 if r[bear_idx] is not None else None for r in rows]

        # Current values (last row = most recent after reversal)
        current = {
            "bullish": round(bullish[-1], 1) if bullish[-1] is not None else None,
            "neutral": round(neutral[-1], 1) if neutral[-1] is not None else None,
            "bearish": round(bearish[-1], 1) if bearish[-1] is not None else None,
            "date": dates[-1],
        }

        bull_bear_spread = None
        if current["bullish"] is not None and current["bearish"] is not None:
            bull_bear_spread = round(current["bullish"] - current["bearish"], 1)

        result = {
            "dates": dates,
            "bullish": bullish,
            "neutral": neutral,
            "bearish": bearish,
            "current": current,
            "bull_bear_spread": bull_bear_spread,
        }

        cache.write_text(json.dumps(result))
        return result

    except Exception as e:
        print(f"  AAII sentiment fetch failed: {e}")
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except Exception:
                pass
        return None


def fetch_putcall_ratio():
    """Fetch CBOE equity put/call ratio historical data.

    Downloads historical CSV (2006-2019) and computes 20-day MA.
    Returns dict with dates, ratios, moving average. Caches to data/putcall_history.json.
    Note: historical data only (ends ~2019). Still useful for pattern context.
    """
    cache = _cache_path("putcall_history.json")
    try:
        url = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Find date and ratio columns
        date_col = None
        ratio_col = None
        for col in df.columns:
            if "date" in col or "trade" in col:
                date_col = col
            if "p/c" in col or "put" in col or "ratio" in col:
                ratio_col = col

        if date_col is None:
            date_col = df.columns[0]
        if ratio_col is None:
            ratio_col = df.columns[-1]

        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["ratio"] = pd.to_numeric(df[ratio_col], errors="coerce")
        df = df[["date", "ratio"]].dropna().sort_values("date")

        # Compute 20-day moving average
        df["ma_20"] = df["ratio"].rolling(window=20, min_periods=1).mean()

        dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
        ratios = df["ratio"].round(3).tolist()
        ma_20 = df["ma_20"].round(3).tolist()

        result = {
            "dates": dates,
            "ratios": ratios,
            "ma_20": ma_20,
            "current_ratio": ratios[-1] if ratios else None,
            "current_date": dates[-1] if dates else None,
        }

        cache.write_text(json.dumps(result))
        return result

    except Exception as e:
        print(f"  Put/call ratio fetch failed: {e}")
        if cache.exists():
            try:
                return json.loads(cache.read_text())
            except Exception:
                pass
        return None
