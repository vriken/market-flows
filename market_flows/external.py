"""External data sources (non-yfinance): FINRA, FRED, AAII, CNN, Yahoo."""

import contextlib
import json
import logging
import os

import pandas as pd
import requests
import yfinance as yf

from .config import (
    CREDIT_SPREAD_SERIES,
    DATA_DIR,
    FED_LIQUIDITY_SERIES,
    FRED_SERIES,
    JOBLESS_CLAIMS_SERIES,
    NFCI_SERIES,
    REAL_YIELDS_SERIES,
)
from .http import get_session

logger = logging.getLogger(__name__)


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
        resp = get_session().get(url, timeout=30)
        resp.raise_for_status()

        tmp = cache.with_suffix(".xlsx")
        tmp.write_bytes(resp.content)

        df = pd.read_excel(tmp, engine="openpyxl")
        tmp.unlink(missing_ok=True)

        debit_col = None
        date_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if "debit" in col_lower and "balance" in col_lower:
                debit_col = col
            if "date" in col_lower or "month" in col_lower or "year" in col_lower:
                date_col = col

        if date_col is None:
            date_col = df.columns[0]
        if debit_col is None:
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

        if df["debit"].median() > 1e6:
            df["debit"] = df["debit"] / 1e6
            unit = "B"
        elif df["debit"].median() > 1e3:
            df["debit"] = df["debit"] / 1e3
            unit = "M"
        else:
            unit = "M"

        dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
        values = df["debit"].tolist()

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

    except requests.RequestException as e:
        logger.warning("Margin debt fetch failed: %s", e)
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
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
        logger.debug("FRED_API_KEY not set, skipping fund flows")
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    try:
        series_list = []
        for key, (series_id, name) in FRED_SERIES.items():
            try:
                resp = get_session().get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": api_key,
                        "file_type": "json",
                        "observation_start": "2010-01-01",
                    },
                    timeout=30,
                )
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
            except requests.RequestException as e:
                logger.warning("FRED series %s failed: %s", series_id, e)
                continue

        result = {
            "has_data": len(series_list) > 0,
            "series": series_list,
        }

        cache.write_text(json.dumps(result))
        return result

    except requests.RequestException as e:
        logger.warning("FRED fund flows fetch failed: %s", e)
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
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
        logger.debug("NASDAQ_DATA_LINK_API_KEY not set, skipping AAII sentiment")
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    try:
        resp = get_session().get(
            "https://data.nasdaq.com/api/v3/datasets/AAII/AAII_SENTIMENT.json",
            params={"api_key": api_key, "rows": "200"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        dataset = data.get("dataset", {})
        columns = dataset.get("column_names", [])
        rows = dataset.get("data", [])

        if not rows:
            return None

        col_map = {c.lower(): i for i, c in enumerate(columns)}
        date_idx = col_map.get("date", 0)
        bull_idx = col_map.get("bullish", 1)
        neutral_idx = col_map.get("neutral", 2)
        bear_idx = col_map.get("bearish", 3)

        rows = list(reversed(rows))

        dates = [r[date_idx] for r in rows]
        bullish = [r[bull_idx] * 100 if r[bull_idx] is not None else None for r in rows]
        neutral = [r[neutral_idx] * 100 if r[neutral_idx] is not None else None for r in rows]
        bearish = [r[bear_idx] * 100 if r[bear_idx] is not None else None for r in rows]

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

    except requests.RequestException as e:
        logger.warning("AAII sentiment fetch failed: %s", e)
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return None


def _fetch_fred_series(series_id, api_key, start_date="2019-01-01"):
    """Fetch a single FRED series. Returns list of (date_str, float) tuples."""
    resp = get_session().get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
        },
        timeout=30,
    )
    resp.raise_for_status()
    observations = resp.json().get("observations", [])
    return [
        (obs["date"], float(obs["value"]))
        for obs in observations
        if obs["value"] != "."
    ]


def fetch_credit_spreads():
    """Fetch ICE BofA credit spread indices from FRED.

    Returns dict with HY/IG OAS time series, current values, and 5Y percentiles.
    Caches to data/credit_spreads.json.
    """
    cache = _cache_path("credit_spreads.json")
    api_key = os.environ.get("FRED_API_KEY")

    if not api_key:
        logger.debug("FRED_API_KEY not set, skipping credit spreads")
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    try:
        series_data = {}
        for label, series_id in CREDIT_SPREAD_SERIES.items():
            try:
                obs = _fetch_fred_series(series_id, api_key, start_date="2019-01-01")
                if obs:
                    series_data[label] = obs
            except requests.RequestException as e:
                logger.warning("FRED %s failed: %s", series_id, e)

        if not series_data:
            return None

        result = {}

        if "HY OAS" in series_data:
            hy = series_data["HY OAS"]
            hy_values = [v for _, v in hy]
            current_hy = hy_values[-1]
            hy_pctile = sum(1 for v in hy_values if v <= current_hy) / len(hy_values) * 100
            result["hy_oas"] = {"dates": [d for d, _ in hy], "values": hy_values}
            result["current_hy"] = round(current_hy, 0)
            result["hy_percentile"] = round(hy_pctile, 0)

        if "IG OAS" in series_data:
            ig = series_data["IG OAS"]
            ig_values = [v for _, v in ig]
            current_ig = ig_values[-1]
            ig_pctile = sum(1 for v in ig_values if v <= current_ig) / len(ig_values) * 100
            result["ig_oas"] = {"dates": [d for d, _ in ig], "values": ig_values}
            result["current_ig"] = round(current_ig, 0)
            result["ig_percentile"] = round(ig_pctile, 0)

        cache.write_text(json.dumps(result))
        return result

    except requests.RequestException as e:
        logger.warning("Credit spreads fetch failed: %s", e)
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return None


def fetch_fed_liquidity():
    """Fetch Fed balance sheet, RRP, and TGA from FRED to compute Net Liquidity.

    Net Liquidity = WALCL - RRP - TGA (aligned via forward-fill).
    Returns dict with component time series and net liquidity.
    Caches to data/fed_liquidity.json.
    """
    cache = _cache_path("fed_liquidity.json")
    api_key = os.environ.get("FRED_API_KEY")

    if not api_key:
        logger.debug("FRED_API_KEY not set, skipping Fed liquidity")
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    try:
        raw = {}
        for label, series_id in FED_LIQUIDITY_SERIES.items():
            try:
                obs = _fetch_fred_series(series_id, api_key, start_date="2024-01-01")
                if obs:
                    raw[label] = obs
            except requests.RequestException as e:
                logger.warning("FRED %s failed: %s", series_id, e)

        if "Fed Balance Sheet" not in raw:
            return None

        frames = {}
        for label, obs in raw.items():
            s = pd.Series(
                [v for _, v in obs],
                index=pd.to_datetime([d for d, _ in obs]),
                name=label,
            )
            frames[label] = s

        df = pd.DataFrame(frames)
        df = df.sort_index().ffill().dropna()

        if df.empty:
            return None

        df["net_liquidity"] = df["Fed Balance Sheet"]
        if "Reverse Repo" in df.columns:
            df["net_liquidity"] -= df["Reverse Repo"]
        if "Treasury General Account" in df.columns:
            df["net_liquidity"] -= df["Treasury General Account"]

        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        net_current = df["net_liquidity"].iloc[-1]

        net_change_4w = 0.0
        net_change_4w_pct = 0.0
        idx_4w = max(0, len(df) - 20)
        if idx_4w < len(df) - 1:
            old_val = df["net_liquidity"].iloc[idx_4w]
            net_change_4w = net_current - old_val
            if old_val != 0:
                net_change_4w_pct = round((net_change_4w / old_val) * 100, 2)

        result = {
            "dates": dates,
            "walcl": df["Fed Balance Sheet"].tolist(),
            "net_liquidity": df["net_liquidity"].tolist(),
            "net_current": net_current,
            "net_change_4w": net_change_4w,
            "net_change_4w_pct": net_change_4w_pct,
        }
        if "Reverse Repo" in df.columns:
            result["rrp"] = df["Reverse Repo"].tolist()
        if "Treasury General Account" in df.columns:
            result["tga"] = df["Treasury General Account"].tolist()

        cache.write_text(json.dumps(result, default=str))
        return result

    except requests.RequestException as e:
        logger.warning("Fed liquidity fetch failed: %s", e)
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return None


def fetch_fear_greed():
    """Fetch CNN Fear & Greed Index current value.

    Returns dict with current value, description, and accumulated history.
    History is built up over time from daily snapshots in data/fear_greed.json.
    """
    cache = _cache_path("fear_greed.json")

    # Load existing history
    existing = {"dates": [], "values": [], "descriptions": []}
    if cache.exists():
        with contextlib.suppress(json.JSONDecodeError, KeyError):
            existing = json.loads(cache.read_text())

    try:
        import fear_and_greed

        fg = fear_and_greed.get()
        value = round(fg.value, 1)
        description = fg.description
        update_date = fg.last_update.strftime("%Y-%m-%d")

        # Append to history if new date
        if not existing["dates"] or existing["dates"][-1] != update_date:
            existing["dates"].append(update_date)
            existing["values"].append(value)
            existing["descriptions"].append(description)
        else:
            # Update today's value
            existing["dates"][-1] = update_date
            existing["values"][-1] = value
            existing["descriptions"][-1] = description

        result = {
            "dates": existing["dates"],
            "values": existing["values"],
            "descriptions": existing["descriptions"],
            "current_value": value,
            "current_description": description,
            "current_date": update_date,
        }

        cache.write_text(json.dumps(result))
        return result

    except Exception as e:
        logger.warning("Fear & Greed fetch failed: %s", e)
        if existing.get("values"):
            existing["current_value"] = existing["values"][-1]
            existing["current_description"] = existing["descriptions"][-1]
            existing["current_date"] = existing["dates"][-1]
            existing["_stale"] = True
            return existing
        return None


def fetch_real_yields():
    """Fetch real yields and inflation breakevens from FRED.

    Returns dict with time series for 10Y real yield, 10Y/5Y breakevens,
    and 5Y5Y forward inflation expectations. Caches to data/real_yields.json.
    """
    cache = _cache_path("real_yields.json")
    api_key = os.environ.get("FRED_API_KEY")

    if not api_key:
        logger.debug("FRED_API_KEY not set, skipping real yields")
        if cache.exists():
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
        return None

    try:
        series_data = {}
        for label, series_id in REAL_YIELDS_SERIES.items():
            try:
                obs = _fetch_fred_series(series_id, api_key, start_date="2019-01-01")
                if obs:
                    series_data[label] = {
                        "dates": [d for d, _ in obs],
                        "values": [v for _, v in obs],
                    }
            except requests.RequestException as e:
                logger.warning("FRED %s failed: %s", series_id, e)

        if not series_data:
            return None

        # Current values
        result = {"series": series_data}
        for label, data in series_data.items():
            if data["values"]:
                result[f"current_{label}"] = round(data["values"][-1], 3)

        cache.write_text(json.dumps(result))
        return result

    except requests.RequestException as e:
        logger.warning("Real yields fetch failed: %s", e)
        if cache.exists():
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
        return None


def fetch_jobless_claims():
    """Fetch initial and continued jobless claims from FRED.

    Returns dict with weekly claims, 4-week moving average, and trend.
    Caches to data/jobless_claims.json.
    """
    cache = _cache_path("jobless_claims.json")
    api_key = os.environ.get("FRED_API_KEY")

    if not api_key:
        logger.debug("FRED_API_KEY not set, skipping jobless claims")
        if cache.exists():
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
        return None

    try:
        series_data = {}
        for label, series_id in JOBLESS_CLAIMS_SERIES.items():
            try:
                obs = _fetch_fred_series(series_id, api_key, start_date="2019-01-01")
                if obs:
                    series_data[label] = {
                        "dates": [d for d, _ in obs],
                        "values": [v for _, v in obs],
                    }
            except requests.RequestException as e:
                logger.warning("FRED %s failed: %s", series_id, e)

        if "Initial Claims" not in series_data:
            return None

        ic = series_data["Initial Claims"]
        ic_vals = ic["values"]

        # 4-week moving average
        ma4 = []
        for i in range(len(ic_vals)):
            window = ic_vals[max(0, i - 3):i + 1]
            ma4.append(round(sum(window) / len(window)))

        # Trend: compare latest 4W MA to 8 weeks ago
        trend = "flat"
        if len(ma4) >= 8:
            if ma4[-1] > ma4[-8] * 1.05:
                trend = "rising"
            elif ma4[-1] < ma4[-8] * 0.95:
                trend = "falling"

        result = {
            "series": series_data,
            "ma4": ma4,
            "current_claims": ic_vals[-1] if ic_vals else None,
            "current_ma4": ma4[-1] if ma4 else None,
            "current_date": ic["dates"][-1] if ic["dates"] else None,
            "trend": trend,
        }

        cache.write_text(json.dumps(result))
        return result

    except requests.RequestException as e:
        logger.warning("Jobless claims fetch failed: %s", e)
        if cache.exists():
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
        return None


def fetch_nfci():
    """Fetch Chicago Fed National Financial Conditions Index from FRED.

    NFCI > 0 = tighter than average, NFCI < 0 = looser than average.
    Caches to data/nfci.json.
    """
    cache = _cache_path("nfci.json")
    api_key = os.environ.get("FRED_API_KEY")

    if not api_key:
        logger.debug("FRED_API_KEY not set, skipping NFCI")
        if cache.exists():
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
        return None

    try:
        obs = _fetch_fred_series(NFCI_SERIES, api_key, start_date="2019-01-01")
        if not obs:
            return None

        dates = [d for d, _ in obs]
        values = [v for _, v in obs]
        current = values[-1]

        # Classify conditions
        if current > 0:
            condition = "Tightening"
        elif current < -0.5:
            condition = "Very Loose"
        else:
            condition = "Loose"

        result = {
            "dates": dates,
            "values": values,
            "current": round(current, 3),
            "current_date": dates[-1],
            "condition": condition,
        }

        cache.write_text(json.dumps(result))
        return result

    except requests.RequestException as e:
        logger.warning("NFCI fetch failed: %s", e)
        if cache.exists():
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
        return None


def fetch_move_skew_dxy():
    """Fetch MOVE index, SKEW index, and DXY from Yahoo Finance.

    MOVE = ICE BofA bond volatility (^MOVE)
    SKEW = CBOE tail-risk pricing (^SKEW)
    DXY = US Dollar Index (DX-Y.NYB)

    Returns dict with current values and 1Y history for each.
    Caches to data/move_skew_dxy.json.
    """
    cache = _cache_path("move_skew_dxy.json")

    tickers = {"MOVE": "^MOVE", "SKEW": "^SKEW", "DXY": "DX-Y.NYB"}

    try:
        ticker_list = list(tickers.values())
        data = yf.download(ticker_list, period="1y", progress=False, threads=True)

        if data.empty:
            raise ValueError("No data returned from yfinance")

        result = {}
        for label, ticker in tickers.items():
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    series = data["Close"][ticker].dropna()
                else:
                    series = data["Close"].dropna()

                if series.empty:
                    continue

                dates = [d.strftime("%Y-%m-%d") for d in series.index]
                values = series.round(2).tolist()
                current = values[-1]

                entry = {
                    "dates": dates,
                    "values": values,
                    "current": current,
                }

                # DXY: add 50/200 DMA
                if label == "DXY" and len(series) >= 50:
                    ma50 = series.rolling(50).mean().dropna()
                    ma200 = series.rolling(200).mean().dropna() if len(series) >= 200 else pd.Series()
                    entry["ma50"] = ma50.round(2).tolist()
                    entry["ma50_dates"] = [d.strftime("%Y-%m-%d") for d in ma50.index]
                    if not ma200.empty:
                        entry["ma200"] = ma200.round(2).tolist()
                        entry["ma200_dates"] = [d.strftime("%Y-%m-%d") for d in ma200.index]
                    entry["above_ma50"] = current > ma50.iloc[-1]
                    entry["trend"] = "Strong" if not ma200.empty and current > ma200.iloc[-1] else "Weak"

                # MOVE: classify regime
                if label == "MOVE":
                    if current > 140:
                        entry["regime"] = "Crisis"
                    elif current > 120:
                        entry["regime"] = "Elevated"
                    elif current > 100:
                        entry["regime"] = "Normal"
                    else:
                        entry["regime"] = "Low"

                # SKEW: classify
                if label == "SKEW":
                    if current > 150:
                        entry["signal"] = "Elevated tail risk"
                    elif current > 130:
                        entry["signal"] = "Moderate tail risk"
                    else:
                        entry["signal"] = "Low tail risk"

                result[label] = entry
            except (KeyError, IndexError) as e:
                logger.debug("Failed to process %s: %s", ticker, e)

        if not result:
            return None

        cache.write_text(json.dumps(result))
        return result

    except Exception as e:
        logger.warning("MOVE/SKEW/DXY fetch failed: %s", e)
        if cache.exists():
            with contextlib.suppress(json.JSONDecodeError, KeyError):
                data = json.loads(cache.read_text())
                data["_stale"] = True
                return data
        return None


def fetch_equity_putcall():
    """Compute equity put/call ratio from SPY options volume.

    Sums put and call volume across near-term expirations (next 4 available)
    to produce a live put/call ratio. Accumulates daily history in
    data/putcall_live.json for charting.
    """
    cache = _cache_path("putcall_live.json")

    existing = {"dates": [], "ratios": []}
    if cache.exists():
        with contextlib.suppress(json.JSONDecodeError, KeyError):
            existing = json.loads(cache.read_text())

    try:
        spy = yf.Ticker("SPY")
        expirations = spy.options
        if not expirations:
            raise ValueError("No SPY options expirations available")

        total_put_vol = 0
        total_call_vol = 0

        # Use up to 4 nearest expirations for a robust ratio
        for exp in expirations[:4]:
            chain = spy.option_chain(exp)
            call_vol = chain.calls["volume"].sum()
            put_vol = chain.puts["volume"].sum()
            if pd.notna(call_vol):
                total_call_vol += int(call_vol)
            if pd.notna(put_vol):
                total_put_vol += int(put_vol)

        if total_call_vol == 0:
            raise ValueError("No call volume available")

        ratio = round(total_put_vol / total_call_vol, 3)
        today = pd.Timestamp.now().strftime("%Y-%m-%d")

        # Append or update today
        if not existing["dates"] or existing["dates"][-1] != today:
            existing["dates"].append(today)
            existing["ratios"].append(ratio)
        else:
            existing["dates"][-1] = today
            existing["ratios"][-1] = ratio

        # Cap history
        if len(existing["dates"]) > 250:
            existing["dates"] = existing["dates"][-250:]
            existing["ratios"] = existing["ratios"][-250:]

        # 20-day MA
        ratios = existing["ratios"]
        ma20 = []
        for i in range(len(ratios)):
            window = ratios[max(0, i - 19):i + 1]
            ma20.append(round(sum(window) / len(window), 3))

        result = {
            "dates": existing["dates"],
            "ratios": existing["ratios"],
            "ma20": ma20,
            "current_ratio": ratio,
            "current_date": today,
            "put_volume": total_put_vol,
            "call_volume": total_call_vol,
        }

        cache.write_text(json.dumps(result))
        return result

    except Exception as e:
        logger.warning("Equity put/call fetch failed: %s", e)
        if existing.get("ratios"):
            return {
                **existing,
                "ma20": existing["ratios"],
                "current_ratio": existing["ratios"][-1],
                "current_date": existing["dates"][-1],
                "_stale": True,
            }
        return None
