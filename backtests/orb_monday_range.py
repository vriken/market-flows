"""
ORB + Monday Range Breakout Strategy Backtest

Signal condition: candle opens AND closes beyond the level, with NO wick
crossing back (entire candle beyond the level).

Strategy variants:
  A. ORB Breakout/Breakdown (intraday, 60d @ 5m + 8d @ 1m)
  B. Monday Range Breakout/Breakdown (daily, 5 years)
  C. ORB + Monday Confluence (intraday, 60d)
  D. ORB Window Size Comparison (5m/15m/30m opening range definition)
  E. VWAP Band Confirmation (intraday, 60d)
  F. SMA Trend Filter (trending vs non-trending days)
  G. Monthly Hold (conviction trades held 20 trading days)

Signal timeframes tested: 1m, 5m, 10m, 15m, 30m, 1h
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ──────────────────────────────────────────────────────────────────

TICKERS_SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLB", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC"]
TICKERS_STOCKS = ["JPM", "INTC", "NFLX", "META", "AMZN"]
TICKERS_COMMODITIES = ["GLD", "SLV", "CPER", "PPLT"]

# Top 5 holdings per sector ETF (for individual stock testing)
SECTOR_HOLDINGS = {
    "XLK": ["NVDA", "AAPL", "MSFT", "AVGO", "MU"],
    "XLF": ["BRK-B", "V", "MA", "BAC"],  # JPM already in TICKERS_STOCKS
    "XLE": ["XOM", "CVX", "COP", "SLB", "WMB"],
    "XLV": ["LLY", "JNJ", "ABBV", "MRK", "UNH"],
    "XLB": ["LIN", "NEM", "FCX", "CRH", "SHW"],
    "XLI": ["GE", "CAT", "RTX", "GEV", "BA"],
    "XLY": ["TSLA", "HD", "MCD", "TJX"],  # AMZN already in TICKERS_STOCKS
    "XLP": ["WMT", "COST", "PG", "KO", "PM"],
    "XLU": ["NEE", "SO", "DUK", "CEG", "AEP"],
    "XLRE": ["WELL", "PLD", "AMT", "EQIX", "SPG"],
    "XLC": ["GOOGL", "GOOG", "VZ"],  # META, NFLX already in TICKERS_STOCKS
}
TICKERS_HOLDINGS = sorted({t for tickers in SECTOR_HOLDINGS.values() for t in tickers})

ALL_TICKERS = TICKERS_SECTORS + TICKERS_STOCKS + TICKERS_COMMODITIES + TICKERS_HOLDINGS

CATEGORY_MAP = {}
for t in TICKERS_SECTORS:
    CATEGORY_MAP[t] = "Sector ETF"
for t in TICKERS_STOCKS:
    CATEGORY_MAP[t] = "Stock"
for t in TICKERS_COMMODITIES:
    CATEGORY_MAP[t] = "Commodity"
for t in TICKERS_HOLDINGS:
    # Map each holding to its parent sector
    for sector, holdings in SECTOR_HOLDINGS.items():
        if t in holdings:
            CATEGORY_MAP[t] = f"{sector} holding"
            break

# Signal candle timeframes (resample source data to these)
SIGNAL_TIMEFRAMES_5M = ["5min", "10min", "15min", "30min", "60min"]
SIGNAL_TIMEFRAMES_1M = ["1min", "5min", "10min", "15min", "30min", "60min"]
ORB_WINDOWS = [5, 15, 30]  # minutes for ORB definition

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MARKET_OPEN = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()


# ── Data Fetching ───────────────────────────────────────────────────────────

def _split_download(tickers: list[str], **kwargs) -> dict[str, pd.DataFrame]:
    """Download data and split into per-ticker DataFrames."""
    data = yf.download(tickers, group_by="ticker", progress=False, **kwargs)
    result = {}
    for ticker in tickers:
        try:
            df = data.copy() if len(tickers) == 1 else data[ticker].copy()
            df = df.dropna(subset=["Close"])
            if not df.empty:
                result[ticker] = df
        except (KeyError, TypeError):
            pass
    return result


def fetch_intraday_1m(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download ~8 days of 1-min data."""
    print(f"Fetching 1-min data (~8 days) for {len(tickers)} tickers...")
    result = _split_download(tickers, period="5d", interval="1m")
    print(f"  Got data for {len(result)}/{len(tickers)} tickers")
    return result


def fetch_intraday_5m(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download ~60 days of 5-min data."""
    print(f"Fetching 5-min data (~60 days) for {len(tickers)} tickers...")
    result = _split_download(tickers, period="60d", interval="5m")
    print(f"  Got data for {len(result)}/{len(tickers)} tickers")
    return result


def fetch_daily(tickers: list[str], period: str = "5y") -> dict[str, pd.DataFrame]:
    """Download daily data."""
    print(f"Fetching daily data ({period}) for {len(tickers)} tickers...")
    result = _split_download(tickers, period=period, interval="1d")
    print(f"  Got data for {len(result)}/{len(tickers)} tickers")
    return result


# ── Timezone Helpers ────────────────────────────────────────────────────────

def _to_ny(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with index in Eastern time."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    return df


def _market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to market hours only. Assumes Eastern-time index."""
    times = df.index.time
    return df[(times >= MARKET_OPEN) & (times < MARKET_CLOSE)]


# ── ORB Computation ─────────────────────────────────────────────────────────

def compute_orb(df: pd.DataFrame, window_minutes: int = 5,
                source_interval_minutes: int = 5) -> pd.DataFrame:
    """Opening range high/low from first N minutes of each trading day.

    source_interval_minutes: the bar size of df (1 or 5).
    """
    df = _market_hours(_to_ny(df))
    df_dates = df.index.date
    candles_needed = max(1, window_minutes // source_interval_minutes)

    orb_rows = []
    for date in np.unique(df_dates):
        day = df[df_dates == date]
        opening = day.iloc[:candles_needed]
        if opening.empty:
            continue
        orb_rows.append({
            "date": date,
            "orb_open": opening["Open"].iloc[0],
            "orb_high": opening["High"].max(),
            "orb_low": opening["Low"].min(),
        })
    return pd.DataFrame(orb_rows)


# ── Monday Range ────────────────────────────────────────────────────────────

def _build_mon_lookup(mr: pd.DataFrame) -> dict:
    """Map every weekday (Mon-Fri) of each ISO week to (mon_high, mon_low).

    week_start may be Tuesday+ when Monday is a holiday. We derive the ISO
    Monday and map the full Mon-Fri regardless of which day we actually have data for.
    """
    lookup = {}
    for _, mon in mr.iterrows():
        ws = mon["week_start"]
        if isinstance(ws, pd.Timestamp):
            ws = ws.date()
        # Derive ISO Monday: weekday 0 = Mon
        iso_monday = ws - pd.Timedelta(days=ws.weekday())
        for d in range(5):
            lookup[iso_monday + pd.Timedelta(days=d)] = (mon["mon_high"], mon["mon_low"])
    return lookup

def compute_monday_range(daily_df: pd.DataFrame) -> pd.DataFrame:
    """First trading day's high/low as the weekly reference range.

    Uses Monday when available; falls back to Tuesday (or later) when
    Monday is a holiday.
    """
    df = daily_df.copy()
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert("America/New_York")

    # Group by ISO year-week and pick the first trading day of each week
    dates = pd.Series(idx.date, index=df.index)
    iso_weeks = pd.Series(
        [d.isocalendar()[:2] for d in idx],
        index=df.index,
    )
    rows = []
    seen_weeks = set()
    for i, row in df.iterrows():
        wk = iso_weeks[i]
        if wk in seen_weeks:
            continue
        seen_weeks.add(wk)
        d = i.date() if hasattr(i, "date") else i
        rows.append({
            "week_start": d,
            "mon_high": row["High"],
            "mon_low": row["Low"],
        })
    return pd.DataFrame(rows)


# ── VWAP Bands ──────────────────────────────────────────────────────────────

def compute_vwap_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Intraday VWAP with +/- 1 SD bands. Adds columns to a copy of df."""
    df = _market_hours(_to_ny(df)).copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol = tp * df["Volume"]
    dates = df.index.date

    df["vwap"] = np.nan
    df["vwap_upper_1sd"] = np.nan
    df["vwap_lower_1sd"] = np.nan

    for date in np.unique(dates):
        mask = dates == date
        idx = df.index[mask]
        cum_vol = df.loc[idx, "Volume"].cumsum().replace(0, np.nan)
        vwap = tp_vol[idx].cumsum() / cum_vol
        variance = ((tp[idx] ** 2 * df.loc[idx, "Volume"]).cumsum() / cum_vol) - vwap**2
        sd = np.sqrt(variance.clip(lower=0))
        df.loc[idx, "vwap"] = vwap
        df.loc[idx, "vwap_upper_1sd"] = vwap + sd
        df.loc[idx, "vwap_lower_1sd"] = vwap - sd

    return df


# ── SMA Computation ─────────────────────────────────────────────────────────

SMA_PERIODS = [20, 50, 150, 200, 250, 300, 400, 500, 600]


def compute_sma_trend(daily_df: pd.DataFrame) -> dict:
    """Compute SMAs (20-600) on daily Close. Returns dict: date -> trend info.

    Trend categories:
      'strong_up' = Close above ALL SMAs
      'up'        = Close above >= 6 of 9 SMAs
      'down'      = Close below >= 6 of 9 SMAs
      'strong_down' = Close below ALL SMAs
      'none'      = mixed / insufficient data
    """
    df = daily_df.copy()
    for p in SMA_PERIODS:
        df[f"sma{p}"] = df["Close"].rolling(p).mean()

    trend = {}
    for i, row in df.iterrows():
        d = i.date() if hasattr(i, "date") else i
        sma_vals = [row[f"sma{p}"] for p in SMA_PERIODS]
        if any(pd.isna(v) for v in sma_vals):
            # Use available SMAs only
            available = [(p, row[f"sma{p}"]) for p in SMA_PERIODS if not pd.isna(row[f"sma{p}"])]
            if len(available) < 2:
                trend[d] = {"label": "none", "above_count": 0, "total_smas": 0}
                continue
            above = sum(1 for _, v in available if row["Close"] > v)
            total = len(available)
        else:
            above = sum(1 for v in sma_vals if row["Close"] > v)
            total = len(sma_vals)

        below = total - above
        if above == total:
            label = "strong_up"
        elif above >= total * 2 / 3:
            label = "up"
        elif below == total:
            label = "strong_down"
        elif below >= total * 2 / 3:
            label = "down"
        else:
            label = "none"

        trend[d] = {"label": label, "above_count": above, "total_smas": total}
    return trend


def compute_sma_gradient(daily_df: pd.DataFrame, sma_period: int = 20,
                         lookback: int = 5) -> dict:
    """Compute SMA slope as % change over lookback trading days.

    gradient_pct = (SMA_today - SMA_{lookback}_ago) / SMA_{lookback}_ago * 100
    Positive = SMA rising, negative = SMA falling.

    Returns dict: date -> {"gradient_pct": float, "sma_value": float}
    """
    df = daily_df.copy()
    sma = df["Close"].rolling(sma_period).mean()
    gradient = (sma - sma.shift(lookback)) / sma.shift(lookback) * 100

    result = {}
    for i in df.index:
        d = i.date() if hasattr(i, "date") else i
        g = gradient.loc[i]
        s = sma.loc[i]
        if pd.notna(g) and pd.notna(s):
            result[d] = {"gradient_pct": float(g), "sma_value": float(s)}
    return result


# ── Resampling ──────────────────────────────────────────────────────────────

def resample_intraday(df: pd.DataFrame, interval: str,
                      source_interval: str = "5min") -> pd.DataFrame:
    """Resample intraday data to a wider interval. No-op if same interval."""
    if interval == source_interval:
        return _market_hours(_to_ny(df))

    df = _market_hours(_to_ny(df))
    resampled = df.resample(interval, closed="left", label="left").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna(subset=["Close"])

    times = resampled.index.time
    return resampled[(times >= MARKET_OPEN) & (times < MARKET_CLOSE)]


# ── Signal Detection ────────────────────────────────────────────────────────

def detect_orb_signals(source_df: pd.DataFrame, orb: pd.DataFrame,
                       signal_interval: str, source_interval: str = "5min",
                       orb_window_minutes: int = 5) -> list[dict]:
    """Detect ORB breakout/breakdown signals with NO-WICK condition.

    Bullish:  Low > orb_high  (entire candle above ORB high)
    Bearish:  High < orb_low  (entire candle below ORB low)

    Only the first signal per day is recorded.
    Each signal includes volume data:
      - signal_volume: volume of the signal candle
      - vol_ratio: signal candle volume / rolling 20-period SMA of volume
    """
    df = resample_intraday(source_df, signal_interval, source_interval)
    dates = df.index.date
    orb_lookup = {row["date"]: row for _, row in orb.iterrows()}

    orb_end = (pd.Timestamp("09:30") + pd.Timedelta(minutes=orb_window_minutes)).time()

    # Rolling volume average (20-period SMA) for spike detection
    vol_sma = df["Volume"].rolling(20, min_periods=1).mean()

    signals = []
    seen_dates = set()

    for ts, row in df.iterrows():
        date = dates[df.index.get_loc(ts)]
        if date in seen_dates:
            continue
        orb_data = orb_lookup.get(date)
        if orb_data is None:
            continue
        if ts.time() < orb_end:
            continue

        orb_high = orb_data["orb_high"]
        orb_low = orb_data["orb_low"]

        sig_vol = row["Volume"]
        avg_vol = vol_sma.get(ts, sig_vol) if sig_vol > 0 else 1.0
        vr = sig_vol / avg_vol if avg_vol > 0 else 1.0

        vol_data = {"signal_volume": sig_vol, "vol_ratio": round(vr, 2)}

        # No-wick breakout: entire candle above ORB high
        if row["Low"] > orb_high:
            signals.append({
                "date": date, "time": ts.time(), "direction": "long",
                "entry_price": row["Close"],
                "orb_high": orb_high, "orb_low": orb_low,
                **vol_data,
            })
            seen_dates.add(date)
        # No-wick breakdown: entire candle below ORB low
        elif row["High"] < orb_low:
            signals.append({
                "date": date, "time": ts.time(), "direction": "short",
                "entry_price": row["Close"],
                "orb_high": orb_high, "orb_low": orb_low,
                **vol_data,
            })
            seen_dates.add(date)

    return signals


# ── Daily Close Extraction ──────────────────────────────────────────────────

def get_daily_closes(df: pd.DataFrame) -> dict:
    """Extract daily close price from intraday data."""
    df = _market_hours(_to_ny(df))
    dates = df.index.date
    closes = {}
    for date in np.unique(dates):
        day = df[dates == date]
        closes[date] = day["Close"].iloc[-1]
    return closes


def get_daily_close_series(daily_df: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by date with Close prices."""
    s = daily_df["Close"].copy()
    s.index = pd.to_datetime(s.index).date if not hasattr(s.index[0], "date") else [i.date() if hasattr(i, "date") else i for i in s.index]
    return s


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_orb_signals(signals: list[dict], daily_closes: dict) -> pd.DataFrame:
    """Did daily close finish beyond the ORB level?"""
    results = []
    for sig in signals:
        dc = daily_closes.get(sig["date"])
        if dc is None:
            continue
        win = dc > sig["orb_high"] if sig["direction"] == "long" else dc < sig["orb_low"]
        results.append({**sig, "daily_close": dc, "win": win})
    return pd.DataFrame(results)


def evaluate_monthly_hold(signals: list[dict], daily_close_series: pd.Series) -> pd.DataFrame:
    """Did the close 20 trading days later stay beyond the ORB level?"""
    dates_sorted = sorted(daily_close_series.index)
    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}

    results = []
    for sig in signals:
        idx = date_to_idx.get(sig["date"])
        if idx is None:
            continue
        target_idx = idx + 20
        if target_idx >= len(dates_sorted):
            continue
        target_date = dates_sorted[target_idx]
        target_close = daily_close_series[target_date]
        win = target_close > sig["orb_high"] if sig["direction"] == "long" else target_close < sig["orb_low"]
        results.append({
            **sig,
            "monthly_close": target_close,
            "monthly_close_date": target_date,
            "win": win,
        })
    return pd.DataFrame(results)


# ── Strategy Runners ────────────────────────────────────────────────────────

def run_strategy_a(data_5m: dict, data_1m: dict) -> pd.DataFrame:
    """A. ORB Breakout/Breakdown across all tickers and signal timeframes."""
    print("\n=== Strategy A: ORB Breakout/Breakdown (no-wick) ===")
    all_results = []

    for ticker, df in data_5m.items():
        orb = compute_orb(df, window_minutes=5, source_interval_minutes=5)
        closes = get_daily_closes(df)
        for tf in SIGNAL_TIMEFRAMES_5M:
            sigs = detect_orb_signals(df, orb, tf, source_interval="5min")
            ev = evaluate_orb_signals(sigs, closes)
            if not ev.empty:
                ev["ticker"] = ticker
                ev["signal_tf"] = tf
                ev["category"] = CATEGORY_MAP[ticker]
                all_results.append(ev)

    # 1-min signals from 1-min data
    for ticker, df in data_1m.items():
        orb = compute_orb(df, window_minutes=5, source_interval_minutes=1)
        closes = get_daily_closes(df)
        sigs = detect_orb_signals(df, orb, "1min", source_interval="1min")
        ev = evaluate_orb_signals(sigs, closes)
        if not ev.empty:
            ev["ticker"] = ticker
            ev["signal_tf"] = "1min"
            ev["category"] = CATEGORY_MAP[ticker]
            all_results.append(ev)

    if not all_results:
        print("  No signals found.")
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def run_strategy_b(daily_data: dict) -> pd.DataFrame:
    """B. Monday Range Breakout/Breakdown (daily, 5Y)."""
    print("\n=== Strategy B: Monday Range Breakout/Breakdown ===")
    all_results = []

    for ticker, df in daily_data.items():
        mr = compute_monday_range(df)
        if mr.empty:
            continue

        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert("America/New_York")
        df = df.copy()
        df["_weekday"] = idx.weekday
        df["_date"] = idx.date

        for _, mon in mr.iterrows():
            ws = mon["week_start"]
            # Derive ISO Monday so week_end is always Friday
            if isinstance(ws, pd.Timestamp):
                ws_date = ws.date()
            else:
                ws_date = ws
            iso_monday = ws_date - pd.Timedelta(days=ws_date.weekday())
            week_end = iso_monday + pd.Timedelta(days=4)  # Friday
            week_days = df[(df["_date"] > ws) & (df["_date"] <= week_end)]
            if week_days.empty:
                continue

            weekly_close = week_days["Close"].iloc[-1]
            weekly_close_date = week_days["_date"].iloc[-1]

            for _, day_row in week_days.iterrows():
                if day_row["Close"] < mon["mon_low"]:
                    all_results.append({
                        "week_start": ws, "signal_date": day_row["_date"],
                        "direction": "short",
                        "mon_high": mon["mon_high"], "mon_low": mon["mon_low"],
                        "signal_close": day_row["Close"],
                        "weekly_close": weekly_close,
                        "weekly_close_date": weekly_close_date,
                        "win": weekly_close < mon["mon_low"],
                        "ticker": ticker, "category": CATEGORY_MAP[ticker],
                    })
                    break
                elif day_row["Close"] > mon["mon_high"]:
                    all_results.append({
                        "week_start": ws, "signal_date": day_row["_date"],
                        "direction": "long",
                        "mon_high": mon["mon_high"], "mon_low": mon["mon_low"],
                        "signal_close": day_row["Close"],
                        "weekly_close": weekly_close,
                        "weekly_close_date": weekly_close_date,
                        "win": weekly_close > mon["mon_high"],
                        "ticker": ticker, "category": CATEGORY_MAP[ticker],
                    })
                    break

    if not all_results:
        print("  No signals found.")
        return pd.DataFrame()
    return pd.DataFrame(all_results)


def run_strategy_c(data_5m: dict, daily_data: dict) -> pd.DataFrame:
    """C. ORB + Monday Confluence — signal only when both levels break."""
    print("\n=== Strategy C: ORB + Monday Confluence ===")
    all_results = []

    for ticker, df in data_5m.items():
        daily_df = daily_data.get(ticker)
        if daily_df is None:
            continue

        orb = compute_orb(df, window_minutes=5, source_interval_minutes=5)
        closes = get_daily_closes(df)
        mr = compute_monday_range(daily_df)
        if mr.empty:
            continue

        mon_lookup = _build_mon_lookup(mr)

        for tf in SIGNAL_TIMEFRAMES_5M:
            sigs = detect_orb_signals(df, orb, tf, source_interval="5min")
            for sig in sigs:
                mon = mon_lookup.get(sig["date"])
                if mon is None:
                    continue
                mon_high, mon_low = mon

                confluence = False
                if sig["direction"] == "long" and sig["entry_price"] > mon_high:
                    confluence = True
                elif sig["direction"] == "short" and sig["entry_price"] < mon_low:
                    confluence = True

                dc = closes.get(sig["date"])
                if dc is None:
                    continue
                win = dc > sig["orb_high"] if sig["direction"] == "long" else dc < sig["orb_low"]

                all_results.append({
                    **sig, "daily_close": dc, "win": win, "confluence": confluence,
                    "mon_high": mon_high, "mon_low": mon_low,
                    "ticker": ticker, "signal_tf": tf, "category": CATEGORY_MAP[ticker],
                })

    if not all_results:
        print("  No signals found.")
        return pd.DataFrame()
    return pd.DataFrame(all_results)


def run_strategy_d(data_5m: dict) -> pd.DataFrame:
    """D. ORB Window Size Comparison (5m/15m/30m ORB definition)."""
    print("\n=== Strategy D: ORB Window Size Comparison ===")
    all_results = []

    for ticker, df in data_5m.items():
        closes = get_daily_closes(df)
        for orb_win in ORB_WINDOWS:
            orb = compute_orb(df, window_minutes=orb_win, source_interval_minutes=5)
            # Use 5min signal TF for fair comparison across ORB sizes
            sigs = detect_orb_signals(df, orb, "5min", source_interval="5min",
                                      orb_window_minutes=orb_win)
            ev = evaluate_orb_signals(sigs, closes)
            if not ev.empty:
                ev["ticker"] = ticker
                ev["orb_window"] = orb_win
                ev["category"] = CATEGORY_MAP[ticker]
                all_results.append(ev)

    if not all_results:
        print("  No signals found.")
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def run_strategy_e(data_5m: dict) -> pd.DataFrame:
    """E. VWAP Band Confirmation."""
    print("\n=== Strategy E: VWAP Band Confirmation ===")
    all_results = []

    for ticker, df in data_5m.items():
        orb = compute_orb(df, window_minutes=5, source_interval_minutes=5)
        closes = get_daily_closes(df)
        df_vwap = compute_vwap_bands(df)

        # Build lookup: (date, time) -> vwap data
        vwap_lookup = {}
        vwap_dates = df_vwap.index.date
        for ts, row in df_vwap.iterrows():
            d = vwap_dates[df_vwap.index.get_loc(ts)]
            vwap_lookup[(d, ts.time())] = {
                "vwap": row["vwap"],
                "upper": row["vwap_upper_1sd"],
                "lower": row["vwap_lower_1sd"],
            }

        for tf in SIGNAL_TIMEFRAMES_5M:
            sigs = detect_orb_signals(df, orb, tf, source_interval="5min")
            for sig in sigs:
                dc = closes.get(sig["date"])
                if dc is None:
                    continue

                vd = vwap_lookup.get((sig["date"], sig["time"]))
                confirmed = False
                if vd and vd["vwap"] is not None and not np.isnan(vd["vwap"]):
                    if sig["direction"] == "long" and sig["entry_price"] > vd["upper"]:
                        confirmed = True
                    elif sig["direction"] == "short" and sig["entry_price"] < vd["lower"]:
                        confirmed = True

                win = dc > sig["orb_high"] if sig["direction"] == "long" else dc < sig["orb_low"]
                all_results.append({
                    **sig, "daily_close": dc, "win": win,
                    "vwap_confirmed": confirmed,
                    "ticker": ticker, "signal_tf": tf, "category": CATEGORY_MAP[ticker],
                })

    if not all_results:
        print("  No signals found.")
        return pd.DataFrame()
    return pd.DataFrame(all_results)


def run_strategy_f(data_5m: dict, daily_data: dict) -> pd.DataFrame:
    """F. SMA Trend Filter — compare win rates on trending vs non-trending days."""
    print("\n=== Strategy F: SMA Trend Filter ===")
    all_results = []

    for ticker, df in data_5m.items():
        daily_df = daily_data.get(ticker)
        if daily_df is None:
            continue

        orb = compute_orb(df, window_minutes=5, source_interval_minutes=5)
        closes = get_daily_closes(df)
        trend = compute_sma_trend(daily_df)

        for tf in SIGNAL_TIMEFRAMES_5M:
            sigs = detect_orb_signals(df, orb, tf, source_interval="5min")
            for sig in sigs:
                dc = closes.get(sig["date"])
                if dc is None:
                    continue
                win = dc > sig["orb_high"] if sig["direction"] == "long" else dc < sig["orb_low"]
                t = trend.get(sig["date"], {"label": "none", "above_count": 0, "total_smas": 0})
                day_trend = t["label"]

                # Aligned = long in uptrend/strong_up or short in downtrend/strong_down
                aligned = (sig["direction"] == "long" and day_trend in ("up", "strong_up")) or \
                          (sig["direction"] == "short" and day_trend in ("down", "strong_down"))

                all_results.append({
                    **sig, "daily_close": dc, "win": win,
                    "trend": day_trend, "trend_aligned": aligned,
                    "smas_above": t["above_count"], "smas_total": t["total_smas"],
                    "ticker": ticker, "signal_tf": tf, "category": CATEGORY_MAP[ticker],
                })

    if not all_results:
        print("  No signals found.")
        return pd.DataFrame()
    return pd.DataFrame(all_results)


def run_strategy_g(data_5m: dict, daily_data: dict) -> pd.DataFrame:
    """G. Monthly Hold — ORB signals evaluated after 20 trading days."""
    print("\n=== Strategy G: Monthly Hold ===")
    all_results = []

    for ticker, df in data_5m.items():
        daily_df = daily_data.get(ticker)
        if daily_df is None:
            continue

        orb = compute_orb(df, window_minutes=5, source_interval_minutes=5)
        close_series = get_daily_close_series(daily_df)

        # Use 30min signal TF (best performer from Strategy A)
        sigs = detect_orb_signals(df, orb, "30min", source_interval="5min")
        ev = evaluate_monthly_hold(sigs, close_series)
        if not ev.empty:
            ev["ticker"] = ticker
            ev["category"] = CATEGORY_MAP[ticker]
            all_results.append(ev)

    if not all_results:
        print("  No signals found.")
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


# ── S/R Zones (replicates Pine Script logic) ────────────────────────────────

SR_PIVOT_LEN = 15
SR_ZONE_ATR_MULT = 0.5
SR_BREAK_ATR_MULT = 0.3
SR_MAX_LEVELS = 5
SR_VOL_SMA_LEN = 20


def compute_sr_levels(df_5m: pd.DataFrame) -> dict:
    """Compute S/R levels per day using pivot detection on 5m data.

    Replicates the Pine Script S/R indicator logic:
    - Pivot highs = resistance, pivot lows = support
    - Zones have width = ATR(14) * 0.5
    - Breaks: close > level + ATR * 0.3 (and low > level) for R break
    - Broken S/R flips polarity (once)

    Returns dict: date -> {"support": [(level, vol_ratio), ...],
                           "resistance": [(level, vol_ratio), ...]}
    """
    df = _market_hours(_to_ny(df_5m)).copy()
    if df.empty:
        return {}

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values
    dates = df.index.date
    n = len(df)

    # ATR(14) - simple true range average
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    # Volume SMA
    vol_sma = pd.Series(volume).rolling(SR_VOL_SMA_LEN, min_periods=1).mean().values

    # Detect pivots
    pl = SR_PIVOT_LEN
    res_levels = []  # (level, vol_ratio, flipped)
    sup_levels = []

    def level_exists(levels, price, threshold):
        return any(abs(l[0] - price) < threshold for l in levels)

    # Track S/R state over time, snapshot per day
    daily_sr = {}

    for i in range(pl, n - pl):
        # Pivot high detection (lookback = pl, lookahead = pl)
        if i + pl < n:
            window_high = high[i - pl:i + pl + 1]
            if high[i] == window_high.max() and np.sum(window_high == high[i]) == 1:
                threshold = atr[i] * 0.8
                if not level_exists(res_levels, high[i], threshold) and \
                   not level_exists(sup_levels, high[i], threshold):
                    vr = volume[i] / vol_sma[i] if vol_sma[i] > 0 else 1.0
                    res_levels.append((high[i], vr, False))
                    if len(res_levels) > SR_MAX_LEVELS:
                        res_levels.pop(0)

            # Pivot low detection
            window_low = low[i - pl:i + pl + 1]
            if low[i] == window_low.min() and np.sum(window_low == low[i]) == 1:
                threshold = atr[i] * 0.8
                if not level_exists(res_levels, low[i], threshold) and \
                   not level_exists(sup_levels, low[i], threshold):
                    vr = volume[i] / vol_sma[i] if vol_sma[i] > 0 else 1.0
                    sup_levels.append((low[i], vr, False))
                    if len(sup_levels) > SR_MAX_LEVELS:
                        sup_levels.pop(0)

        # Check for breaks
        break_threshold = atr[i] * SR_BREAK_ATR_MULT

        # Resistance breaks (close above level + threshold, low > level)
        new_res = []
        for lvl, vr, flipped in res_levels:
            if close[i] > lvl + break_threshold and low[i] > lvl:
                # Broken resistance flips to support (if not already flipped)
                if not flipped:
                    if not level_exists(sup_levels, lvl, atr[i] * 0.5):
                        sup_levels.append((lvl, vr, True))
                        if len(sup_levels) > SR_MAX_LEVELS:
                            sup_levels.pop(0)
            else:
                new_res.append((lvl, vr, flipped))
        res_levels = new_res

        # Support breaks (close below level - threshold, high < level)
        new_sup = []
        for lvl, vr, flipped in sup_levels:
            if close[i] < lvl - break_threshold and high[i] < lvl:
                # Broken support flips to resistance (if not already flipped)
                if not flipped:
                    if not level_exists(res_levels, lvl, atr[i] * 0.5):
                        res_levels.append((lvl, vr, True))
                        if len(res_levels) > SR_MAX_LEVELS:
                            res_levels.pop(0)
            else:
                new_sup.append((lvl, vr, flipped))
        sup_levels = new_sup

        # Snapshot at end of day (will be used as levels for the NEXT day)
        date = dates[i]
        daily_sr[date] = {
            "support": [(l, v) for l, v, _ in sup_levels],
            "resistance": [(l, v) for l, v, _ in res_levels],
            "atr": atr[i],
            "zone_width": atr[i] * SR_ZONE_ATR_MULT,
        }

    # Shift: use previous day's closing S/R state for next day's signals
    # This avoids look-ahead bias
    sorted_dates = sorted(daily_sr.keys())
    shifted = {}
    for i in range(1, len(sorted_dates)):
        shifted[sorted_dates[i]] = daily_sr[sorted_dates[i - 1]]
    return shifted


def run_strategy_h(data_5m: dict) -> pd.DataFrame:
    """H. S/R Zone Confirmation — ORB signals with/without S/R confluence.

    Tests whether ORB breakout through resistance (or breakdown through
    support) improves win rate vs ORB signals away from S/R zones.
    """
    print("\n=== Strategy H: S/R Zone Confirmation ===")
    all_results = []

    for ticker, df in data_5m.items():
        orb = compute_orb(df, window_minutes=5, source_interval_minutes=5)
        closes = get_daily_closes(df)
        daily_sr = compute_sr_levels(df)

        for tf in SIGNAL_TIMEFRAMES_5M:
            sigs = detect_orb_signals(df, orb, tf, source_interval="5min")
            for sig in sigs:
                dc = closes.get(sig["date"])
                if dc is None:
                    continue
                sr = daily_sr.get(sig["date"])
                if sr is None:
                    continue

                win = dc > sig["orb_high"] if sig["direction"] == "long" else dc < sig["orb_low"]
                price = sig["entry_price"]
                zone_w = sr["zone_width"]

                # Check if signal breaks through an S/R level
                sr_break = False
                sr_near = False  # within zone width of a level

                if sig["direction"] == "long":
                    # Bullish: check if price just broke above a resistance
                    for lvl, vr in sr["resistance"]:
                        if price > lvl and price < lvl + zone_w * 2:
                            sr_break = True
                            break
                    # Or near support (bounce)
                    for lvl, vr in sr["support"]:
                        if abs(price - lvl) < zone_w:
                            sr_near = True
                            break
                else:
                    # Bearish: check if price just broke below a support
                    for lvl, vr in sr["support"]:
                        if price < lvl and price > lvl - zone_w * 2:
                            sr_break = True
                            break
                    # Or near resistance (rejection)
                    for lvl, vr in sr["resistance"]:
                        if abs(price - lvl) < zone_w:
                            sr_near = True
                            break

                sr_type = "sr_break" if sr_break else ("sr_near" if sr_near else "no_sr")

                all_results.append({
                    **sig, "daily_close": dc, "win": win,
                    "sr_type": sr_type,
                    "ticker": ticker, "signal_tf": tf, "category": CATEGORY_MAP[ticker],
                })

    if not all_results:
        print("  No signals found.")
        return pd.DataFrame()
    return pd.DataFrame(all_results)


def run_strategy_i(data_5m: dict, daily_data: dict) -> pd.DataFrame:
    """I. ORB gap vs Monday range — does the 5m ORB open below/above Monday's range?

    Tests whether the day's open relative to Monday's weekly range
    affects the win rate of ORB signals.
    """
    print("\n=== Strategy I: ORB Open vs Monday Range ===")
    all_results = []

    for ticker, df in data_5m.items():
        daily_df = daily_data.get(ticker)
        if daily_df is None:
            continue

        orb = compute_orb(df, window_minutes=5, source_interval_minutes=5)
        closes = get_daily_closes(df)
        mr = compute_monday_range(daily_df)
        if mr.empty:
            continue

        mon_lookup = _build_mon_lookup(mr)

        orb_lookup = {row["date"]: row for _, row in orb.iterrows()}

        for tf in SIGNAL_TIMEFRAMES_5M:
            sigs = detect_orb_signals(df, orb, tf, source_interval="5min")
            for sig in sigs:
                dc = closes.get(sig["date"])
                if dc is None:
                    continue
                mon = mon_lookup.get(sig["date"])
                if mon is None:
                    continue
                orb_data = orb_lookup.get(sig["date"])
                if orb_data is None:
                    continue

                mon_high, mon_low = mon
                orb_open = orb_data["orb_open"]

                if orb_open < mon_low:
                    gap_type = "below_monday"
                elif orb_open > mon_high:
                    gap_type = "above_monday"
                else:
                    gap_type = "inside_monday"

                win = dc > sig["orb_high"] if sig["direction"] == "long" else dc < sig["orb_low"]

                all_results.append({
                    **sig, "daily_close": dc, "win": win,
                    "gap_type": gap_type, "orb_open": orb_open,
                    "mon_high": mon_high, "mon_low": mon_low,
                    "ticker": ticker, "signal_tf": tf, "category": CATEGORY_MAP[ticker],
                })

    if not all_results:
        print("  No signals found.")
        return pd.DataFrame()
    return pd.DataFrame(all_results)


# ── Reporting ───────────────────────────────────────────────────────────────

def win_rate_stats(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if group_cols:
        grouped = df.groupby(group_cols)
    else:
        grouped = df.groupby(lambda _: "Overall")

    rows = []
    for name, grp in grouped:
        total = len(grp)
        wins = int(grp["win"].sum())
        longs = grp[grp["direction"] == "long"]
        shorts = grp[grp["direction"] == "short"]
        key = (dict(zip(group_cols, name)) if group_cols and isinstance(name, tuple)
               else {group_cols[0]: name} if group_cols else {"group": name})
        rows.append({
            **key,
            "trades": total,
            "wins": wins,
            "WR": f"{wins/total:.1%}" if total else "N/A",
            "L_trades": len(longs),
            "L_wins": int(longs["win"].sum()) if len(longs) else 0,
            "L_WR": f"{longs['win'].sum()/len(longs):.1%}" if len(longs) else "N/A",
            "S_trades": len(shorts),
            "S_wins": int(shorts["win"].sum()) if len(shorts) else 0,
            "S_WR": f"{shorts['win'].sum()/len(shorts):.1%}" if len(shorts) else "N/A",
        })
    return pd.DataFrame(rows)


def prt(title: str, df: pd.DataFrame):
    print(f"\n{'─'*90}")
    print(f"  {title}")
    print(f"{'─'*90}")
    if df.empty:
        print("  No data.")
    else:
        print(df.to_string(index=False))
    print()


# ── Unified Signal Builder ──────────────────────────────────────────────────

def build_unified_signals(data_5m: dict, daily_data: dict,
                          data_1m: dict | None = None) -> pd.DataFrame:
    """Build a single DataFrame where every ORB signal is tagged with ALL factors:
    confluence, vwap, trend, s/r, gap_type. Enables cross-analysis of every combination.
    """
    print("\n=== Building unified signal table (all factors per signal) ===")
    all_rows = []

    for ticker, df in data_5m.items():
        daily_df = daily_data.get(ticker)
        if daily_df is None:
            continue

        # Pre-compute all factors for this ticker
        orb = compute_orb(df, window_minutes=5, source_interval_minutes=5)
        closes = get_daily_closes(df)
        orb_lookup = {row["date"]: row for _, row in orb.iterrows()}

        mr = compute_monday_range(daily_df)
        mon_lookup = _build_mon_lookup(mr) if not mr.empty else {}

        df_vwap = compute_vwap_bands(df)
        vwap_lookup = {}
        vwap_dates = df_vwap.index.date
        for ts, row in df_vwap.iterrows():
            d = vwap_dates[df_vwap.index.get_loc(ts)]
            vwap_lookup[(d, ts.time())] = {
                "vwap": row["vwap"],
                "upper": row["vwap_upper_1sd"],
                "lower": row["vwap_lower_1sd"],
            }

        trend = compute_sma_trend(daily_df)
        daily_sr = compute_sr_levels(df)

        for tf in SIGNAL_TIMEFRAMES_5M:
            sigs = detect_orb_signals(df, orb, tf, source_interval="5min")
            for sig in sigs:
                dc = closes.get(sig["date"])
                if dc is None:
                    continue

                price = sig["entry_price"]
                win = dc > sig["orb_high"] if sig["direction"] == "long" else dc < sig["orb_low"]

                # Monday confluence
                mon = mon_lookup.get(sig["date"])
                confluence = False
                gap_type = "no_monday_data"
                if mon:
                    mon_high, mon_low = mon
                    if sig["direction"] == "long" and price > mon_high:
                        confluence = True
                    elif sig["direction"] == "short" and price < mon_low:
                        confluence = True
                    orb_data = orb_lookup.get(sig["date"])
                    if orb_data is not None:
                        orb_open = orb_data["orb_open"]
                        if orb_open < mon_low:
                            gap_type = "below_monday"
                        elif orb_open > mon_high:
                            gap_type = "above_monday"
                        else:
                            gap_type = "inside_monday"

                # VWAP
                vd = vwap_lookup.get((sig["date"], sig["time"]))
                vwap_confirmed = False
                if vd and vd["vwap"] is not None and not np.isnan(vd["vwap"]):
                    if sig["direction"] == "long" and price > vd["upper"]:
                        vwap_confirmed = True
                    elif sig["direction"] == "short" and price < vd["lower"]:
                        vwap_confirmed = True

                # SMA trend
                t = trend.get(sig["date"], {"label": "none", "above_count": 0, "total_smas": 0})
                day_trend = t["label"]
                trend_aligned = (sig["direction"] == "long" and day_trend in ("up", "strong_up")) or \
                                (sig["direction"] == "short" and day_trend in ("down", "strong_down"))

                # S/R zones
                sr = daily_sr.get(sig["date"])
                sr_type = "no_sr"
                if sr:
                    zone_w = sr["zone_width"]
                    if sig["direction"] == "long":
                        for lvl, vr in sr["resistance"]:
                            if price > lvl and price < lvl + zone_w * 2:
                                sr_type = "sr_break"
                                break
                        if sr_type == "no_sr":
                            for lvl, vr in sr["support"]:
                                if abs(price - lvl) < zone_w:
                                    sr_type = "sr_near"
                                    break
                    else:
                        for lvl, vr in sr["support"]:
                            if price < lvl and price > lvl - zone_w * 2:
                                sr_type = "sr_break"
                                break
                        if sr_type == "no_sr":
                            for lvl, vr in sr["resistance"]:
                                if abs(price - lvl) < zone_w:
                                    sr_type = "sr_near"
                                    break

                # Volume spike: signal candle volume ≥ 2× rolling average
                vr = sig.get("vol_ratio", 1.0)
                vol_spike = vr >= 2.0

                all_rows.append({
                    **sig, "daily_close": dc, "win": win,
                    "ticker": ticker, "signal_tf": tf, "category": CATEGORY_MAP[ticker],
                    "confluence": confluence,
                    "vwap_confirmed": vwap_confirmed,
                    "trend": day_trend, "trend_aligned": trend_aligned,
                    "sr_type": sr_type,
                    "gap_type": gap_type,
                    "vol_spike": vol_spike,
                })

    # Also build 1-min signals if 1-min data is available
    if data_1m:
        n_before = len(all_rows)
        for ticker, df_1m in data_1m.items():
            daily_df = daily_data.get(ticker)
            if daily_df is None:
                continue

            orb_1m = compute_orb(df_1m, window_minutes=5, source_interval_minutes=1)
            closes_1m = get_daily_closes(df_1m)

            mr = compute_monday_range(daily_df)
            mon_lookup_1m = _build_mon_lookup(mr) if not mr.empty else {}

            df_vwap_1m = compute_vwap_bands(df_1m)
            vwap_lookup_1m = {}
            vwap_dates_1m = df_vwap_1m.index.date
            for ts, row in df_vwap_1m.iterrows():
                d = vwap_dates_1m[df_vwap_1m.index.get_loc(ts)]
                vwap_lookup_1m[(d, ts.time())] = {
                    "vwap": row["vwap"], "upper": row["vwap_upper_1sd"],
                    "lower": row["vwap_lower_1sd"],
                }

            trend = compute_sma_trend(daily_df)
            daily_sr = compute_sr_levels(df_1m)
            orb_lookup_1m = {row["date"]: row for _, row in orb_1m.iterrows()}

            # Only generate 1-min signal candles
            sigs = detect_orb_signals(df_1m, orb_1m, "1min", source_interval="1min")
            for sig in sigs:
                dc = closes_1m.get(sig["date"])
                if dc is None:
                    continue

                price = sig["entry_price"]
                win = dc > sig["orb_high"] if sig["direction"] == "long" else dc < sig["orb_low"]

                mon = mon_lookup_1m.get(sig["date"])
                confluence = False
                gap_type = "no_monday_data"
                if mon:
                    mon_high, mon_low = mon
                    if sig["direction"] == "long" and price > mon_high:
                        confluence = True
                    elif sig["direction"] == "short" and price < mon_low:
                        confluence = True
                    orb_data = orb_lookup_1m.get(sig["date"])
                    if orb_data is not None:
                        orb_open = orb_data["orb_open"]
                        if orb_open < mon_low:
                            gap_type = "below_monday"
                        elif orb_open > mon_high:
                            gap_type = "above_monday"
                        else:
                            gap_type = "inside_monday"

                vd = vwap_lookup_1m.get((sig["date"], sig["time"]))
                vwap_confirmed = False
                if vd and vd["vwap"] is not None and not np.isnan(vd["vwap"]):
                    if sig["direction"] == "long" and price > vd["upper"]:
                        vwap_confirmed = True
                    elif sig["direction"] == "short" and price < vd["lower"]:
                        vwap_confirmed = True

                t = trend.get(sig["date"], {"label": "none", "above_count": 0, "total_smas": 0})
                day_trend = t["label"]
                trend_aligned = (sig["direction"] == "long" and day_trend in ("up", "strong_up")) or \
                                (sig["direction"] == "short" and day_trend in ("down", "strong_down"))

                sr = daily_sr.get(sig["date"])
                sr_type = "no_sr"
                if sr:
                    zone_w = sr["zone_width"]
                    if sig["direction"] == "long":
                        for lvl, vr in sr["resistance"]:
                            if price > lvl and price < lvl + zone_w * 2:
                                sr_type = "sr_break"
                                break
                        if sr_type == "no_sr":
                            for lvl, vr in sr["support"]:
                                if abs(price - lvl) < zone_w:
                                    sr_type = "sr_near"
                                    break
                    else:
                        for lvl, vr in sr["support"]:
                            if price < lvl and price > lvl - zone_w * 2:
                                sr_type = "sr_break"
                                break
                        if sr_type == "no_sr":
                            for lvl, vr in sr["resistance"]:
                                if abs(price - lvl) < zone_w:
                                    sr_type = "sr_near"
                                    break

                vr = sig.get("vol_ratio", 1.0)
                vol_spike = vr >= 2.0

                all_rows.append({
                    **sig, "daily_close": dc, "win": win,
                    "ticker": ticker, "signal_tf": "1min", "category": CATEGORY_MAP[ticker],
                    "confluence": confluence, "vwap_confirmed": vwap_confirmed,
                    "trend": day_trend, "trend_aligned": trend_aligned,
                    "sr_type": sr_type, "gap_type": gap_type,
                    "vol_spike": vol_spike,
                })

        print(f"  Added {len(all_rows) - n_before} signals from 1-min data")

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    print(f"  Built {len(df)} signals with all factor tags")
    return df


# ── Let-It-Run Simulation (no-wick SL, no TP cap) ──────────────────────────

def simulate_let_it_run(unified: pd.DataFrame, data_5m: dict) -> pd.DataFrame:
    """Simulate trades with NO profit cap — only exit is full-reversal stop.

    Entry: the signal candle (no-wick breakout beyond ORB).
    Stop: candle CLOSES through the entire ORB range (full reversal).
      LONG stop: close < orb_low  |  SHORT stop: close > orb_high
    If no stop triggers, holds to daily close.

    Measures R-multiples: actual_move / risk, where risk = |entry - opposite ORB level|.
    Also tracks Max Favorable Excursion (MFE) in R-multiples.
    """
    print("\n  Simulating let-it-run (no TP cap, full-reversal SL)...")
    results = []

    for ticker, ticker_signals in unified.groupby("ticker"):
        df_raw = data_5m.get(ticker)
        if df_raw is None:
            continue

        df = _market_hours(_to_ny(df_raw))

        for _, sig in ticker_signals.iterrows():
            direction = sig["direction"]
            entry = sig["entry_price"]
            orb_high = sig["orb_high"]
            orb_low = sig["orb_low"]
            trade_date = sig["date"]
            signal_time = sig["time"]
            signal_tf = sig["signal_tf"]

            # Risk = distance from entry to opposite ORB level (full-reversal stop)
            if direction == "long":
                risk = entry - orb_low
            else:
                risk = orb_high - entry

            if risk <= 0:
                continue

            # Get day's data after signal
            if isinstance(trade_date, str):
                trade_date_obj = pd.Timestamp(trade_date).date()
            else:
                trade_date_obj = trade_date

            day_data = df[df.index.date == trade_date_obj]
            if isinstance(signal_time, str):
                time_filter = signal_time
            else:
                time_filter = str(signal_time)

            after = day_data[day_data.index.strftime("%H:%M:%S") > time_filter]
            if len(after) == 0:
                continue

            # Resample to match signal TF for exit candle check
            tf_minutes = {"1min": 1, "5min": 5, "10min": 10, "15min": 15,
                          "30min": 30, "60min": 60}.get(signal_tf, 5)
            if tf_minutes > 5:
                after = after.resample(f"{tf_minutes}min").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum",
                }).dropna()

            outcome = "hold"  # held to close (no stop triggered)
            exit_price = float(day_data.iloc[-1]["Close"]) if len(day_data) > 0 else entry

            # Track MFE (max favorable excursion) in price
            mfe_price = entry

            for _, candle in after.iterrows():
                if direction == "long":
                    mfe_price = max(mfe_price, float(candle["High"]))
                    # SL: candle CLOSES below ORB low (full reversal through range)
                    if candle["Close"] < orb_low:
                        outcome = "stopped"
                        exit_price = float(candle["Close"])
                        break
                else:
                    mfe_price = min(mfe_price, float(candle["Low"]))
                    # SL: candle CLOSES above ORB high (full reversal through range)
                    if candle["Close"] > orb_high:
                        outcome = "stopped"
                        exit_price = float(candle["Close"])
                        break

            # Calculate P&L and R-multiples
            if direction == "long":
                pnl = (exit_price - entry) / entry * 100
                r_multiple = (exit_price - entry) / risk
                mfe_r = (mfe_price - entry) / risk
            else:
                pnl = (entry - exit_price) / entry * 100
                r_multiple = (entry - exit_price) / risk
                mfe_r = (entry - mfe_price) / risk

            results.append({
                "ticker": ticker, "date": str(sig["date"]),
                "signal_tf": signal_tf, "direction": direction,
                "entry_price": entry, "exit_price": exit_price,
                "orb_high": orb_high, "orb_low": orb_low,
                "risk_pct": risk / entry * 100,
                "pnl_pct": pnl,
                "r_multiple": r_multiple,
                "mfe_r": mfe_r,
                "outcome": outcome,
                "confluence": sig["confluence"],
                "vwap_confirmed": sig["vwap_confirmed"],
                "trend_aligned": sig["trend_aligned"],
                "sr_type": sig["sr_type"],
                "gap_type": sig["gap_type"],
                "category": sig["category"],
                "vol_ratio": sig.get("vol_ratio", 1.0),
                "vol_spike": sig.get("vol_spike", False),
            })

    df = pd.DataFrame(results)
    print(f"  Simulated {len(df)} trades (no TP cap, no-wick reversal SL)")
    return df


def print_let_it_run_summary(lir: pd.DataFrame):
    """Print let-it-run simulation results with R-multiple distribution."""

    def _dist_line(label, sub):
        if len(sub) < 10:
            return
        n = len(sub)
        stopped = (sub["outcome"] == "stopped").sum()
        held = (sub["outcome"] == "hold").sum()
        avg_r = sub["r_multiple"].mean()
        med_r = sub["r_multiple"].median()
        avg_mfe = sub["mfe_r"].mean()

        # R-multiple buckets
        neg = (sub["r_multiple"] < 0).sum()
        r0_1 = ((sub["r_multiple"] >= 0) & (sub["r_multiple"] < 1)).sum()
        r1_2 = ((sub["r_multiple"] >= 1) & (sub["r_multiple"] < 2)).sum()
        r2_3 = ((sub["r_multiple"] >= 2) & (sub["r_multiple"] < 3)).sum()
        r3_5 = ((sub["r_multiple"] >= 3) & (sub["r_multiple"] < 5)).sum()
        r5_10 = ((sub["r_multiple"] >= 5) & (sub["r_multiple"] < 10)).sum()
        r10 = (sub["r_multiple"] >= 10).sum()

        print(f"  {label:30s} | n={n:4d} | stop={stopped/n*100:4.1f}% hold={held/n*100:4.1f}% | "
              f"avgR={avg_r:+.2f} medR={med_r:+.2f} | mfeR={avg_mfe:.2f} | "
              f"<0:{neg} 0-1R:{r0_1} 1-2R:{r1_2} 2-3R:{r2_3} 3-5R:{r3_5} 5-10R:{r5_10} 10R+:{r10}")

    print("\n── By Signal TF ──")
    for tf in ["5min", "10min", "15min", "30min", "60min"]:
        _dist_line(tf, lir[lir["signal_tf"] == tf])

    print("\n── By Direction ──")
    _dist_line("Long", lir[lir["direction"] == "long"])
    _dist_line("Short", lir[lir["direction"] == "short"])

    print("\n── By Factor (5min signals only) ──")
    lir5 = lir[lir["signal_tf"] == "5min"]
    _dist_line("Baseline (5min)", lir5)
    _dist_line("+ Confluence", lir5[lir5["confluence"]])
    _dist_line("+ Trend-aligned", lir5[lir5["trend_aligned"]])
    _dist_line("+ Confl + Trend", lir5[lir5["confluence"] & lir5["trend_aligned"]])
    _dist_line("+ Confl + VWAP + Trend", lir5[lir5["confluence"] & lir5["vwap_confirmed"] & lir5["trend_aligned"]])
    _dist_line("Inside Monday", lir5[lir5["gap_type"] == "inside_monday"])
    if "vol_spike" in lir5.columns:
        _dist_line("+ Vol Spike (≥2x)", lir5[lir5["vol_spike"]])
        _dist_line("+ Vol Spike + Trend", lir5[lir5["vol_spike"] & lir5["trend_aligned"]])

    # The "reverse-engineered" filter: ORB range > 0.8%, risk < 0.25%, trend-aligned
    big_orb = lir5[(lir5["orb_high"] - lir5["orb_low"]) / lir5["entry_price"] * 100 > 0.8]
    tight_risk = big_orb[big_orb["risk_pct"] < 0.25]
    filtered = tight_risk[tight_risk["trend_aligned"]]
    _dist_line("ORBrng>0.8%+risk<0.25%+trend", filtered)

    print("\n── By Category (5min signals) ──")
    for cat in ["Sector ETF", "Stock", "Commodity"]:
        _dist_line(cat, lir5[lir5["category"] == cat])

    # Summary stats
    print(f"\n  {'='*80}")
    profitable = (lir["r_multiple"] > 0).sum()
    total = len(lir)
    big_winners = (lir["r_multiple"] >= 3).sum()
    print(f"  Profitable: {profitable}/{total} ({profitable/total*100:.1f}%)")
    print(f"  3R+ winners: {big_winners}/{total} ({big_winners/total*100:.1f}%)")
    print(f"  Average R: {lir['r_multiple'].mean():+.2f}")
    print(f"  Average MFE: {lir['mfe_r'].mean():.2f}R")
    print(f"  Total R earned: {lir['r_multiple'].sum():+.1f}R over {total} trades")


# ── Turbo Simulation ───────────────────────────────────────────────────────

def simulate_turbo(unified: pd.DataFrame, data_5m: dict,
                   ko_buffer_pct: float = 1.5, position_sek: float = 500.0,
                   n_trades: int = 0, signal_tf_filter: str = "5min",
                   quiet: bool = False, data_1m: dict | None = None,
                   hold_days: int = 0) -> pd.DataFrame:
    """Simulate turbo/knock-out certificate trading on ORB signals.

    Turbo mechanics:
    - Long turbo: KO level = entry × (1 - ko_buffer/100). Leverage ≈ 100/ko_buffer.
    - If underlying Low ever touches KO → knocked out, lose entire position.
    - Otherwise exit at daily close. Turbo P&L = position × (underlying_return / ko_buffer).

    Exit conditions (whichever comes first):
    1. KO hit (any candle wick touches KO level) → -position_sek
    2. No-wick reversal past ORB (manual exit) → turbo P&L at that candle's close
    3. Daily close → turbo P&L at close

    hold_days: 0 = exit at same-day close. 1 = hold overnight, exit next day close.
               2+ = hold multiple nights. KO can still trigger on subsequent days.
    """
    # Filter to desired signal TF and sort chronologically
    if signal_tf_filter:
        signals = unified[unified["signal_tf"] == signal_tf_filter].copy()
    else:
        signals = unified.copy()
    signals = signals.sort_values(["date", "time"])
    if n_trades > 0:
        signals = signals.head(n_trades)

    hold_label = "same-day" if hold_days == 0 else f"+{hold_days}d hold"
    if not quiet:
        print(f"\n  Simulating turbo: {len(signals)} trades @ {ko_buffer_pct}% KO buffer, "
              f"{position_sek:.0f} SEK, {hold_label}...")

    # Pre-build sorted trading dates per ticker for multi-day holds
    ticker_dates = {}
    if hold_days > 0:
        for ticker, df_raw in data_5m.items():
            df_full = _market_hours(_to_ny(df_raw))
            ticker_dates[ticker] = sorted(set(df_full.index.date))

    results = []
    for _, sig in signals.iterrows():
        direction = sig["direction"]
        entry = sig["entry_price"]
        orb_high = sig["orb_high"]
        orb_low = sig["orb_low"]
        trade_date = sig["date"]
        signal_time = sig["time"]

        # KO level
        buffer_frac = ko_buffer_pct / 100
        if direction == "long":
            ko_level = entry * (1 - buffer_frac)
        else:
            ko_level = entry * (1 + buffer_frac)

        if isinstance(trade_date, str):
            trade_date_obj = pd.Timestamp(trade_date).date()
        else:
            trade_date_obj = trade_date

        ticker = sig["ticker"]

        # Build list of days to simulate (signal day + hold_days)
        if hold_days > 0 and ticker in ticker_dates:
            all_dates = ticker_dates[ticker]
            try:
                start_idx = all_dates.index(trade_date_obj)
            except ValueError:
                start_idx = -1
            if start_idx >= 0:
                sim_dates = all_dates[start_idx:start_idx + 1 + hold_days]
            else:
                sim_dates = [trade_date_obj]
        else:
            sim_dates = [trade_date_obj]

        # Collect all candles across sim_dates
        all_candles = []
        for sim_date in sim_dates:
            day_data = pd.DataFrame()
            if data_1m is not None and ticker in data_1m:
                df_1m_raw = _market_hours(_to_ny(data_1m[ticker]))
                day_data = df_1m_raw[df_1m_raw.index.date == sim_date]
            if len(day_data) == 0:
                df_5m_raw = data_5m.get(ticker)
                if df_5m_raw is None:
                    continue
                df_5m = _market_hours(_to_ny(df_5m_raw))
                day_data = df_5m[df_5m.index.date == sim_date]

            if sim_date == trade_date_obj:
                # First day: only candles after signal time
                if isinstance(signal_time, str):
                    time_filter = signal_time
                else:
                    time_filter = str(signal_time)
                day_data = day_data[day_data.index.strftime("%H:%M:%S") > time_filter]

            all_candles.append(day_data)

        if not all_candles:
            continue
        candles_df = pd.concat(all_candles)
        if len(candles_df) == 0:
            continue

        outcome = "close"  # held to close of last day
        exit_price = float(candles_df.iloc[-1]["Close"])
        exit_time = ""

        for ts, candle in candles_df.iterrows():
            if direction == "long":
                if candle["Low"] <= ko_level:
                    outcome = "ko"
                    exit_price = ko_level
                    exit_time = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                    break
                # Only check manual exit on signal day (ORB levels are intraday)
                # Full-reversal stop: close through entire ORB range
                if ts.date() == trade_date_obj and candle["Close"] < orb_low:
                    outcome = "manual_exit"
                    exit_price = float(candle["Close"])
                    exit_time = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                    break
            else:
                if candle["High"] >= ko_level:
                    outcome = "ko"
                    exit_price = ko_level
                    exit_time = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                    break
                # Full-reversal stop: close through entire ORB range
                if ts.date() == trade_date_obj and candle["Close"] > orb_high:
                    outcome = "manual_exit"
                    exit_price = float(candle["Close"])
                    exit_time = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                    break

        # Calculate turbo P&L
        if outcome == "ko":
            pnl_sek = -position_sek
        else:
            if direction == "long":
                underlying_return = (exit_price - entry) / entry
            else:
                underlying_return = (entry - exit_price) / entry
            turbo_return = underlying_return / buffer_frac
            turbo_return = max(turbo_return, -1.0)
            pnl_sek = position_sek * turbo_return

        results.append({
            "ticker": sig["ticker"], "date": str(trade_date),
            "time": signal_time, "direction": direction,
            "entry_price": entry, "exit_price": exit_price,
            "ko_level": ko_level, "orb_high": orb_high, "orb_low": orb_low,
            "outcome": outcome, "exit_time": exit_time,
            "underlying_move_pct": (exit_price - entry) / entry * 100 * (1 if direction == "long" else -1),
            "turbo_pnl_sek": pnl_sek,
            "confluence": sig["confluence"],
            "vwap_confirmed": sig["vwap_confirmed"],
            "trend_aligned": sig["trend_aligned"],
            "category": sig["category"],
            "vol_ratio": sig.get("vol_ratio", 1.0),
            "vol_spike": sig.get("vol_spike", False),
        })

    return pd.DataFrame(results)


def simulate_bull_bear(unified: pd.DataFrame, data_5m: dict,
                       leverage: float = 5.0, position_sek: float = 500.0,
                       daily_funding_pct: float = 0.07,
                       n_trades: int = 0, signal_tf_filter: str = "5min",
                       quiet: bool = False, hold_days: int = 0) -> pd.DataFrame:
    """Simulate bull/bear certificate trading on ORB signals.

    Bull/bear cert mechanics (different from turbos):
    - NO knockout level — can hold through any drawdown
    - Fixed leverage (e.g. 2x, 5x, 10x)
    - Daily compounding: each day's return is leveraged independently
    - Daily funding cost deducted from cert value (~0.05-0.10% per day)
    - Can hold overnight without risk of total loss

    Exit conditions:
    1. No-wick reversal past ORB (manual exit, signal day only)
    2. Close of last holding day

    hold_days: 0 = exit at same-day close. 1+ = hold additional days.
    """
    if signal_tf_filter:
        signals = unified[unified["signal_tf"] == signal_tf_filter].copy()
    else:
        signals = unified.copy()
    signals = signals.sort_values(["date", "time"])
    if n_trades > 0:
        signals = signals.head(n_trades)

    hold_label = "same-day" if hold_days == 0 else f"+{hold_days}d hold"
    if not quiet:
        print(f"\n  Simulating bull/bear cert: {len(signals)} trades @ {leverage}x leverage, "
              f"{position_sek:.0f} SEK, {hold_label}, funding {daily_funding_pct}%/day...")

    # Pre-build sorted trading dates per ticker for multi-day holds
    ticker_dates = {}
    for ticker, df_raw in data_5m.items():
        df_full = _market_hours(_to_ny(df_raw))
        ticker_dates[ticker] = sorted(set(df_full.index.date))

    results = []
    for _, sig in signals.iterrows():
        direction = sig["direction"]
        entry = sig["entry_price"]
        orb_high = sig["orb_high"]
        orb_low = sig["orb_low"]
        trade_date = sig["date"]
        signal_time = sig["time"]

        if isinstance(trade_date, str):
            trade_date_obj = pd.Timestamp(trade_date).date()
        else:
            trade_date_obj = trade_date

        ticker = sig["ticker"]

        # Build list of days to simulate
        if hold_days > 0 and ticker in ticker_dates:
            all_dates = ticker_dates[ticker]
            try:
                start_idx = all_dates.index(trade_date_obj)
            except ValueError:
                start_idx = -1
            if start_idx >= 0:
                sim_dates = all_dates[start_idx:start_idx + 1 + hold_days]
            else:
                sim_dates = [trade_date_obj]
        else:
            sim_dates = [trade_date_obj]

        # Track daily closes for compounding
        day_closes = []
        outcome = "close"
        exit_price = entry
        exit_time = ""
        manual_exit_day = False

        for day_idx, sim_date in enumerate(sim_dates):
            df_5m_raw = data_5m.get(ticker)
            if df_5m_raw is None:
                continue
            df_5m = _market_hours(_to_ny(df_5m_raw))
            day_data = df_5m[df_5m.index.date == sim_date]
            if len(day_data) == 0:
                continue

            if sim_date == trade_date_obj:
                # First day: only candles after signal
                if isinstance(signal_time, str):
                    time_filter = signal_time
                else:
                    time_filter = str(signal_time)
                after = day_data[day_data.index.strftime("%H:%M:%S") > time_filter]
                if len(after) == 0:
                    continue

                # Check for manual exit on signal day
                # Full-reversal stop: close through entire ORB range
                for ts, candle in after.iterrows():
                    if direction == "long" and candle["Close"] < orb_low:
                        outcome = "manual_exit"
                        exit_price = float(candle["Close"])
                        exit_time = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                        manual_exit_day = True
                        break
                    elif direction == "short" and candle["Close"] > orb_high:
                        outcome = "manual_exit"
                        exit_price = float(candle["Close"])
                        exit_time = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                        manual_exit_day = True
                        break

                if manual_exit_day:
                    day_closes.append(exit_price)
                    break
                else:
                    day_closes.append(float(day_data.iloc[-1]["Close"]))
            else:
                # Subsequent days: hold to close
                day_closes.append(float(day_data.iloc[-1]["Close"]))

        if not day_closes:
            continue

        # Calculate bull/bear cert P&L with daily compounding
        # Each day: cert_return = leverage × daily_underlying_return - funding_cost
        cert_value = position_sek
        prev_price = entry
        n_overnight = 0

        for i, dc in enumerate(day_closes):
            if direction == "long":
                daily_return = (dc - prev_price) / prev_price
            else:
                daily_return = (prev_price - dc) / prev_price

            cert_daily_return = leverage * daily_return
            cert_value *= (1 + cert_daily_return)

            # Apply funding cost for overnight holds (not on entry day)
            if i > 0:
                cert_value *= (1 - daily_funding_pct / 100)
                n_overnight += 1

            prev_price = dc

        # Cert value can't go below 0
        cert_value = max(cert_value, 0.0)
        pnl_sek = cert_value - position_sek
        exit_price = day_closes[-1]
        if not manual_exit_day:
            outcome = "close"

        results.append({
            "ticker": sig["ticker"], "date": str(trade_date),
            "time": signal_time, "direction": direction,
            "entry_price": entry, "exit_price": exit_price,
            "orb_high": orb_high, "orb_low": orb_low,
            "outcome": outcome, "exit_time": exit_time,
            "underlying_move_pct": (exit_price - entry) / entry * 100 * (1 if direction == "long" else -1),
            "cert_pnl_sek": pnl_sek,
            "cert_value": cert_value,
            "n_overnight": n_overnight,
            "confluence": sig["confluence"],
            "vwap_confirmed": sig["vwap_confirmed"],
            "trend_aligned": sig["trend_aligned"],
            "category": sig["category"],
            "vol_ratio": sig.get("vol_ratio", 1.0),
            "vol_spike": sig.get("vol_spike", False),
        })

    return pd.DataFrame(results)


def print_turbo_summary(turbo: pd.DataFrame, ko_buffer: float, position: float):
    """Print turbo simulation results."""
    if turbo.empty:
        return

    n = len(turbo)
    n_ko = (turbo["outcome"] == "ko").sum()
    n_manual = (turbo["outcome"] == "manual_exit").sum()
    n_close = (turbo["outcome"] == "close").sum()
    total_pnl = turbo["turbo_pnl_sek"].sum()
    total_invested = n * position
    avg_pnl = turbo["turbo_pnl_sek"].mean()
    winners = (turbo["turbo_pnl_sek"] > 0).sum()
    big_winners = (turbo["turbo_pnl_sek"] > position).sum()  # doubled+

    leverage = round(100 / ko_buffer)
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  TURBO SIMULATION  |  {ko_buffer}% buffer  |  ~{leverage}x leverage     │")
    print(f"  │  {n} trades × {position:.0f} SEK = {total_invested:,.0f} SEK invested          │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Knocked out:    {n_ko:4d} ({n_ko/n*100:5.1f}%)  → lost {n_ko * position:>10,.0f} SEK │")
    print(f"  │  Manual exit:    {n_manual:4d} ({n_manual/n*100:5.1f}%)                        │")
    print(f"  │  Held to close:  {n_close:4d} ({n_close/n*100:5.1f}%)                        │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Winners:        {winners:4d} ({winners/n*100:5.1f}%)                        │")
    print(f"  │  Doubled+:       {big_winners:4d} ({big_winners/n*100:5.1f}%)  (>100% return)     │")
    print(f"  │  Avg P&L/trade:  {avg_pnl:>+10,.0f} SEK                       │")
    print(f"  │  ─────────────────────────────────────────────────────  │")
    print(f"  │  TOTAL P&L:      {total_pnl:>+10,.0f} SEK                       │")
    print(f"  │  ROI:            {total_pnl/total_invested*100:>+9.1f}%                        │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # Breakdown by direction
    for d in ["long", "short"]:
        sub = turbo[turbo["direction"] == d]
        if len(sub) < 5:
            continue
        ko_r = (sub["outcome"] == "ko").sum() / len(sub) * 100
        wr = (sub["turbo_pnl_sek"] > 0).sum() / len(sub) * 100
        pnl = sub["turbo_pnl_sek"].sum()
        print(f"    {d.upper():5s}: n={len(sub):3d} | KO={ko_r:4.1f}% | WR={wr:4.1f}% | P&L={pnl:>+10,.0f} SEK")

    # Breakdown by factor
    print()
    factor_checks = [
        ("Confluence", turbo["confluence"]),
        ("Trend-aligned", turbo["trend_aligned"]),
        ("Confl+Trend", turbo["confluence"] & turbo["trend_aligned"]),
    ]
    if "vol_spike" in turbo.columns:
        factor_checks.append(("Vol Spike (≥2x)", turbo["vol_spike"]))
        factor_checks.append(("Vol Spike+Trend", turbo["vol_spike"] & turbo["trend_aligned"]))
    for label, mask in factor_checks:
        sub = turbo[mask]
        if len(sub) < 10:
            continue
        ko_r = (sub["outcome"] == "ko").sum() / len(sub) * 100
        wr = (sub["turbo_pnl_sek"] > 0).sum() / len(sub) * 100
        pnl = sub["turbo_pnl_sek"].sum()
        print(f"    {label:15s}: n={len(sub):3d} | KO={ko_r:4.1f}% | WR={wr:4.1f}% | P&L={pnl:>+10,.0f} SEK")

    # Top 5 winners and worst 5
    print(f"\n    Top 5 winners:")
    top = turbo.nlargest(5, "turbo_pnl_sek")
    for _, r in top.iterrows():
        print(f"      {r['ticker']:5s} {r['date']} {r['direction']:5s} | "
              f"underlying {r['underlying_move_pct']:+.2f}% | "
              f"P&L {r['turbo_pnl_sek']:>+,.0f} SEK ({r['outcome']})")

    print(f"\n    Worst 5:")
    bot = turbo.nsmallest(5, "turbo_pnl_sek")
    for _, r in bot.iterrows():
        print(f"      {r['ticker']:5s} {r['date']} {r['direction']:5s} | "
              f"underlying {r['underlying_move_pct']:+.2f}% | "
              f"P&L {r['turbo_pnl_sek']:>+,.0f} SEK ({r['outcome']})")

    # Running P&L curve data
    turbo_sorted = turbo.sort_values(["date", "time"])
    running = turbo_sorted["turbo_pnl_sek"].cumsum()
    peak = running.cummax()
    max_dd = (running - peak).min()
    print(f"\n    Max drawdown: {max_dd:>+,.0f} SEK")
    print(f"    Peak P&L:     {peak.max():>+,.0f} SEK")


# ── Turbo Runner Simulation (SMA-gradient-based overnight hold) ────────────

def simulate_turbo_runner(unified: pd.DataFrame, data_5m: dict,
                          daily_data: dict,
                          ko_buffer_pct: float = 1.0,
                          position_sek: float = 500.0,
                          sma_period: int = 20,
                          gradient_lookback: int = 5,
                          gradient_threshold: float = 0.3,
                          max_hold_days: int = 5,
                          signal_tf_filter: str = "5min",
                          quiet: bool = False,
                          data_1m: dict | None = None) -> pd.DataFrame:
    """Turbo simulation with SMA-gradient-based runner holding.

    Instead of always exiting at EOD:
    1. Enter on signal candle (same as simulate_turbo)
    2. KO and full-reversal stop apply on signal day
    3. At EOD: check SMA gradient for the ticker
       - LONG: gradient > +threshold → hold overnight
       - SHORT: gradient < -threshold → hold overnight
       - Otherwise: exit at close
    4. On subsequent days: KO can still trigger, but no ORB-based stop
    5. At each EOD: re-check gradient → hold or exit
    6. Max hold = max_hold_days total
    """
    if signal_tf_filter:
        signals = unified[unified["signal_tf"] == signal_tf_filter].copy()
    else:
        signals = unified.copy()
    signals = signals.sort_values(["date", "time"])

    if not quiet:
        print(f"\n  Simulating turbo runner: {len(signals)} trades @ {ko_buffer_pct}% KO, "
              f"SMA({sma_period}) gradient>{gradient_threshold}%, max {max_hold_days}d hold...")

    # Pre-compute SMA gradient per ticker
    ticker_gradients = {}
    ticker_dates = {}
    for ticker, df_raw in data_5m.items():
        daily_df = daily_data.get(ticker)
        if daily_df is not None:
            ticker_gradients[ticker] = compute_sma_gradient(
                daily_df, sma_period=sma_period, lookback=gradient_lookback)
        df_full = _market_hours(_to_ny(df_raw))
        ticker_dates[ticker] = sorted(set(df_full.index.date))

    results = []
    for _, sig in signals.iterrows():
        direction = sig["direction"]
        entry = sig["entry_price"]
        orb_high = sig["orb_high"]
        orb_low = sig["orb_low"]
        trade_date = sig["date"]
        signal_time = sig["time"]
        ticker = sig["ticker"]

        buffer_frac = ko_buffer_pct / 100
        if direction == "long":
            ko_level = entry * (1 - buffer_frac)
        else:
            ko_level = entry * (1 + buffer_frac)

        if isinstance(trade_date, str):
            trade_date_obj = pd.Timestamp(trade_date).date()
        else:
            trade_date_obj = trade_date

        # Get available trading dates for this ticker
        all_dates = ticker_dates.get(ticker, [])
        try:
            start_idx = all_dates.index(trade_date_obj)
        except ValueError:
            continue

        gradients = ticker_gradients.get(ticker, {})

        # Simulate day by day
        outcome = "close"
        exit_price = entry
        exit_time = ""
        days_held = 0
        hold_reason = ""
        done = False

        for day_offset in range(max_hold_days + 1):
            if done:
                break
            if start_idx + day_offset >= len(all_dates):
                break

            sim_date = all_dates[start_idx + day_offset]
            days_held = day_offset + 1

            # Get intraday data for this date
            day_data = pd.DataFrame()
            if data_1m is not None and ticker in data_1m:
                df_1m_raw = _market_hours(_to_ny(data_1m[ticker]))
                day_data = df_1m_raw[df_1m_raw.index.date == sim_date]
            if len(day_data) == 0:
                df_5m_raw = data_5m.get(ticker)
                if df_5m_raw is None:
                    break
                df_5m = _market_hours(_to_ny(df_5m_raw))
                day_data = df_5m[df_5m.index.date == sim_date]

            if len(day_data) == 0:
                break

            # On signal day: start after signal time
            if sim_date == trade_date_obj:
                if isinstance(signal_time, str):
                    time_filter = signal_time
                else:
                    time_filter = str(signal_time)
                day_data = day_data[day_data.index.strftime("%H:%M:%S") > time_filter]
                if len(day_data) == 0:
                    break

            # Check each candle for KO and (signal-day-only) full-reversal stop
            for ts, candle in day_data.iterrows():
                if direction == "long":
                    if candle["Low"] <= ko_level:
                        outcome = "ko"
                        exit_price = ko_level
                        exit_time = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                        done = True
                        break
                    # Full-reversal stop only on signal day
                    if sim_date == trade_date_obj and candle["Close"] < orb_low:
                        outcome = "manual_exit"
                        exit_price = float(candle["Close"])
                        exit_time = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                        done = True
                        break
                else:
                    if candle["High"] >= ko_level:
                        outcome = "ko"
                        exit_price = ko_level
                        exit_time = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                        done = True
                        break
                    if sim_date == trade_date_obj and candle["Close"] > orb_high:
                        outcome = "manual_exit"
                        exit_price = float(candle["Close"])
                        exit_time = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                        done = True
                        break

            if done:
                break

            # EOD decision: check gradient to decide hold vs exit
            eod_price = float(day_data.iloc[-1]["Close"])

            # Last possible day → must exit
            if day_offset >= max_hold_days:
                exit_price = eod_price
                outcome = "close"
                hold_reason = "max_days"
                break

            # Check SMA gradient for hold decision
            grad_info = gradients.get(sim_date)
            if grad_info is None:
                # No gradient data → exit at close
                exit_price = eod_price
                outcome = "close"
                hold_reason = "no_gradient_data"
                break

            grad = grad_info["gradient_pct"]
            favorable = (direction == "long" and grad > gradient_threshold) or \
                        (direction == "short" and grad < -gradient_threshold)

            if not favorable:
                # Gradient not strong enough → exit at close
                exit_price = eod_price
                outcome = "close"
                hold_reason = f"grad={grad:+.2f}%"
                break
            # else: gradient is favorable → continue to next day (hold overnight)

        # If we exhausted all days without exiting, close at last available price
        if not done and exit_price == entry:
            exit_price = entry  # shouldn't happen, but safety

        # Calculate turbo P&L
        if outcome == "ko":
            pnl_sek = -position_sek
        else:
            if direction == "long":
                underlying_return = (exit_price - entry) / entry
            else:
                underlying_return = (entry - exit_price) / entry
            turbo_return = underlying_return / buffer_frac
            turbo_return = max(turbo_return, -1.0)
            pnl_sek = position_sek * turbo_return

        results.append({
            "ticker": ticker, "date": str(trade_date),
            "time": signal_time, "direction": direction,
            "entry_price": entry, "exit_price": exit_price,
            "ko_level": ko_level, "orb_high": orb_high, "orb_low": orb_low,
            "outcome": outcome, "exit_time": exit_time,
            "days_held": days_held,
            "hold_reason": hold_reason,
            "underlying_move_pct": (exit_price - entry) / entry * 100 * (1 if direction == "long" else -1),
            "turbo_pnl_sek": pnl_sek,
            "confluence": sig["confluence"],
            "vwap_confirmed": sig["vwap_confirmed"],
            "trend_aligned": sig["trend_aligned"],
            "category": sig["category"],
            "vol_ratio": sig.get("vol_ratio", 1.0),
            "vol_spike": sig.get("vol_spike", False),
        })

    return pd.DataFrame(results)


def print_runner_summary(runner: pd.DataFrame, ko_buffer: float, position: float,
                         label: str = ""):
    """Print turbo runner simulation results."""
    if runner.empty:
        return

    n = len(runner)
    n_ko = (runner["outcome"] == "ko").sum()
    n_manual = (runner["outcome"] == "manual_exit").sum()
    n_close = (runner["outcome"] == "close").sum()
    total_pnl = runner["turbo_pnl_sek"].sum()
    total_invested = n * position
    avg_pnl = runner["turbo_pnl_sek"].mean()
    winners = (runner["turbo_pnl_sek"] > 0).sum()
    avg_days = runner["days_held"].mean()

    # Trades that actually held overnight
    held_overnight = (runner["days_held"] > 1).sum()

    leverage = round(100 / ko_buffer)
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  TURBO RUNNER  |  {label:40s} │")
    print(f"  │  {ko_buffer}% buffer (~{leverage}x)  |  {n} trades × {position:.0f} SEK           │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Knocked out:    {n_ko:4d} ({n_ko/n*100:5.1f}%)                        │")
    print(f"  │  Manual exit:    {n_manual:4d} ({n_manual/n*100:5.1f}%)                        │")
    print(f"  │  Held to close:  {n_close:4d} ({n_close/n*100:5.1f}%)                        │")
    print(f"  │  Held overnight: {held_overnight:4d} ({held_overnight/n*100:5.1f}%)                        │")
    print(f"  │  Avg days held:  {avg_days:5.2f}                                   │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │  Winners:        {winners:4d} ({winners/n*100:5.1f}%)                        │")
    print(f"  │  Avg P&L/trade:  {avg_pnl:>+10,.0f} SEK                       │")
    print(f"  │  TOTAL P&L:      {total_pnl:>+10,.0f} SEK                       │")
    print(f"  │  ROI:            {total_pnl/total_invested*100:>+9.1f}%                        │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # Breakdown: held overnight vs same-day exit
    same_day = runner[runner["days_held"] == 1]
    multi_day = runner[runner["days_held"] > 1]
    print(f"\n    Same-day exit: n={len(same_day):4d} | "
          f"WR={((same_day['turbo_pnl_sek'] > 0).sum() / max(len(same_day), 1) * 100):5.1f}% | "
          f"P&L={same_day['turbo_pnl_sek'].sum():>+10,.0f} SEK")
    if len(multi_day) > 0:
        print(f"    Held overnight: n={len(multi_day):4d} | "
              f"WR={((multi_day['turbo_pnl_sek'] > 0).sum() / len(multi_day) * 100):5.1f}% | "
              f"P&L={multi_day['turbo_pnl_sek'].sum():>+10,.0f} SEK | "
              f"avg {multi_day['days_held'].mean():.1f}d")
        # Days held distribution
        for d in sorted(multi_day["days_held"].unique()):
            sub = multi_day[multi_day["days_held"] == d]
            ko_r = (sub["outcome"] == "ko").sum() / len(sub) * 100
            wr = (sub["turbo_pnl_sek"] > 0).sum() / len(sub) * 100
            print(f"      {d}d held: n={len(sub):3d} | KO={ko_r:4.1f}% | WR={wr:4.1f}% | "
                  f"P&L={sub['turbo_pnl_sek'].sum():>+8,.0f} SEK")

    # Running P&L
    sorted_df = runner.sort_values(["date", "time"])
    running = sorted_df["turbo_pnl_sek"].cumsum()
    peak = running.cummax()
    max_dd = (running - peak).min()
    print(f"\n    Max drawdown: {max_dd:>+,.0f} SEK")
    print(f"    Peak P&L:     {peak.max():>+,.0f} SEK")


# ── R:R Exit Simulation ─────────────────────────────────────────────────────

def simulate_rr_exit(unified: pd.DataFrame, data_5m: dict,
                     rr_target: float = 1.5) -> pd.DataFrame:
    """Simulate exits with TP = rr_target * risk and SL = full-reversal through ORB.

    For each signal timeframe, the exit candle size matches the signal candle size.
    Risk = |entry_price - opposite ORB level| (distance to full-reversal stop).
    TP triggers on a price touch (wick is fine).
    SL triggers when a candle CLOSES through the entire ORB range (full reversal).
    """
    print("\n  Simulating R:R exits...")
    results = []

    # Group signals by ticker for efficient data access
    for ticker, ticker_signals in unified.groupby("ticker"):
        df_raw = data_5m.get(ticker)
        if df_raw is None:
            continue

        df = _market_hours(_to_ny(df_raw))

        for _, sig in ticker_signals.iterrows():
            direction = sig["direction"]
            entry = sig["entry_price"]
            orb_high = sig["orb_high"]
            orb_low = sig["orb_low"]
            trade_date = sig["date"]
            signal_time = sig["time"]
            signal_tf = sig["signal_tf"]

            # Determine risk (distance to full-reversal stop)
            if direction == "long":
                risk = entry - orb_low
                tp_price = entry + rr_target * risk
            else:
                risk = orb_high - entry
                tp_price = entry - rr_target * risk

            if risk <= 0:
                continue

            # Get day's data after signal
            if isinstance(trade_date, str):
                trade_date_obj = pd.Timestamp(trade_date).date()
            else:
                trade_date_obj = trade_date

            day_data = df[df.index.date == trade_date_obj]
            if isinstance(signal_time, str):
                time_filter = signal_time
            else:
                time_filter = str(signal_time)

            after = day_data[day_data.index.strftime("%H:%M:%S") > time_filter]
            if len(after) == 0:
                continue

            # Resample to match signal TF for the exit candle check
            tf_minutes = {"1min": 1, "5min": 5, "10min": 10, "15min": 15,
                          "30min": 30, "60min": 60}.get(signal_tf, 5)
            if tf_minutes > 5:
                after = after.resample(f"{tf_minutes}min").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum",
                }).dropna()

            outcome = "hold"
            exit_price = float(day_data.iloc[-1]["Close"]) if len(day_data) > 0 else entry

            for _, candle in after.iterrows():
                if direction == "long":
                    # TP: price touch
                    if candle["High"] >= tp_price:
                        outcome = "tp"
                        exit_price = tp_price
                        break
                    # SL: candle CLOSES below ORB low (full reversal through range)
                    if candle["Close"] < orb_low:
                        outcome = "sl"
                        exit_price = float(candle["Close"])
                        break
                else:
                    if candle["Low"] <= tp_price:
                        outcome = "tp"
                        exit_price = tp_price
                        break
                    # SL: candle CLOSES above ORB high (full reversal through range)
                    if candle["Close"] > orb_high:
                        outcome = "sl"
                        exit_price = float(candle["Close"])
                        break

            pnl = (exit_price - entry) / entry * 100
            if direction == "short":
                pnl *= -1

            results.append({
                "ticker": ticker, "date": str(sig["date"]),
                "signal_tf": signal_tf, "direction": direction,
                "entry_price": entry, "exit_price": exit_price,
                "orb_high": orb_high, "orb_low": orb_low,
                "risk_pct": risk / entry * 100,
                "pnl_pct": pnl,
                "outcome": outcome,
                "confluence": sig["confluence"],
                "vwap_confirmed": sig["vwap_confirmed"],
                "trend_aligned": sig["trend_aligned"],
                "sr_type": sig["sr_type"],
                "gap_type": sig["gap_type"],
                "category": sig["category"],
            })

    df = pd.DataFrame(results)
    print(f"  Simulated {len(df)} trades with {rr_target}:1 R:R exits")
    return df


def print_rr_summary(rr: pd.DataFrame):
    """Print R:R simulation results."""
    def _rr_line(label, sub):
        if len(sub) < 10:
            return
        n = len(sub)
        n_tp = (sub["outcome"] == "tp").sum()
        n_sl = (sub["outcome"] == "sl").sum()
        n_hold = (sub["outcome"] == "hold").sum()
        exp = sub["pnl_pct"].mean()
        total = sub["pnl_pct"].sum()
        print(f"  {label:30s} | n={n:5d} | TP={n_tp/n*100:5.1f}% | "
              f"SL={n_sl/n*100:5.1f}% | Hold={n_hold/n*100:5.1f}% | "
              f"expect={exp:+.4f}% | total={total:+.1f}%")

    print("\n── By Signal TF ──")
    for tf in ["5min", "10min", "15min", "30min", "60min"]:
        _rr_line(tf, rr[rr["signal_tf"] == tf])

    print("\n── By Direction ──")
    _rr_line("Long", rr[rr["direction"] == "long"])
    _rr_line("Short", rr[rr["direction"] == "short"])

    print("\n── By Factor (5min signals only) ──")
    rr5 = rr[rr["signal_tf"] == "5min"]
    _rr_line("Baseline (5min)", rr5)
    _rr_line("+ Confluence", rr5[rr5["confluence"]])
    _rr_line("+ Trend-aligned", rr5[rr5["trend_aligned"]])
    _rr_line("+ Confluence + Trend", rr5[rr5["confluence"] & rr5["trend_aligned"]])
    _rr_line("+ Confl + VWAP + Trend", rr5[rr5["confluence"] & rr5["vwap_confirmed"] & rr5["trend_aligned"]])
    _rr_line("Inside Monday", rr5[rr5["gap_type"] == "inside_monday"])
    _rr_line("Long + Confl + Trend", rr5[(rr5["direction"] == "long") & rr5["confluence"] & rr5["trend_aligned"]])

    print("\n── By Category (5min signals) ──")
    for cat in ["Sector ETF", "Stock", "Commodity"]:
        _rr_line(cat, rr5[rr5["category"] == cat])


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  ORB + Monday Range Backtest  |  No-Wick Signal Condition")
    print("  Signal TFs: 1m, 5m, 10m, 15m, 30m, 1h")
    print("=" * 90)

    data_5m = fetch_intraday_5m(ALL_TICKERS)
    data_1m = fetch_intraday_1m(ALL_TICKERS)
    daily_data = fetch_daily(ALL_TICKERS, period="5y")

    sample_5m = next(iter(data_5m.values()))
    sample_1m = next(iter(data_1m.values()))
    days_5m = len(np.unique(_market_hours(_to_ny(sample_5m)).index.date))
    days_1m = len(np.unique(_market_hours(_to_ny(sample_1m)).index.date))
    print(f"\n  5-min data: ~{days_5m} trading days  |  1-min data: ~{days_1m} trading days")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 1: Standalone strategies (A-D use different data/windows)
    # ═══════════════════════════════════════════════════════════════════════════

    # A. ORB across all timeframes (includes 1m from separate data)
    res_a = run_strategy_a(data_5m, data_1m)
    if not res_a.empty:
        prt("A. ORB (no-wick) — Overall", win_rate_stats(res_a))
        prt("A. ORB — Per Signal Timeframe", win_rate_stats(res_a, ["signal_tf"]))
        prt("A. ORB — Per Ticker", win_rate_stats(res_a, ["ticker"]))
        prt("A. ORB — Per Category", win_rate_stats(res_a, ["category"]))
        res_a.to_csv(RESULTS_DIR / "strategy_a_orb.csv", index=False)

    # B. Monday Range (daily, 5Y — standalone)
    res_b = run_strategy_b(daily_data)
    if not res_b.empty:
        prt("B. Monday Range — Overall", win_rate_stats(res_b))
        prt("B. Monday Range — Per Ticker", win_rate_stats(res_b, ["ticker"]))
        prt("B. Monday Range — Per Category", win_rate_stats(res_b, ["category"]))
        res_b.to_csv(RESULTS_DIR / "strategy_b_monday_range.csv", index=False)

    # D. ORB Window Size Comparison (different ORB definitions)
    res_d = run_strategy_d(data_5m)
    if not res_d.empty:
        prt("D. ORB Window — Per Window Size", win_rate_stats(res_d, ["orb_window"]))
        res_d.to_csv(RESULTS_DIR / "strategy_d_orb_window.csv", index=False)

    # G. Monthly Hold
    res_g = run_strategy_g(data_5m, daily_data)
    if not res_g.empty:
        prt("G. Monthly Hold (20 trading days) — Overall", win_rate_stats(res_g))
        prt("G. Monthly Hold — Per Ticker", win_rate_stats(res_g, ["ticker"]))
        prt("G. Monthly Hold — Per Category", win_rate_stats(res_g, ["category"]))
        res_g.to_csv(RESULTS_DIR / "strategy_g_monthly_hold.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 2: Unified signal table with ALL factor tags + cross-analysis
    # ═══════════════════════════════════════════════════════════════════════════

    unified = build_unified_signals(data_5m, daily_data, data_1m=data_1m)
    if unified.empty:
        print("No unified signals to analyze.")
        return

    unified.to_csv(RESULTS_DIR / "unified_signals.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3: Let-It-Run simulation (no TP cap, no-wick reversal SL only)
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("  LET-IT-RUN SIMULATION  |  No profit cap")
    print("  SL = no-wick candle retreats past ORB level  |  Risk = entry - ORB level")
    print("  Exit: stopped out OR held to daily close")
    print("=" * 90)

    lir_results = simulate_let_it_run(unified, data_5m)
    if not lir_results.empty:
        lir_results.to_csv(RESULTS_DIR / "let_it_run.csv", index=False)
        print_let_it_run_summary(lir_results)

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3b: Turbo simulation (500 SEK × 500 trades, 1%/1.5%/2% KO buffer)
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("  TURBO SIMULATION  |  500 SEK per trade × 500 trades")
    print("  KO = any wick touch  |  Manual exit = close through entire ORB range")
    print("  Exit at daily close if neither triggers")
    print("=" * 90)

    for buffer in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        turbo = simulate_turbo(unified, data_5m, ko_buffer_pct=buffer,
                               position_sek=500, n_trades=500,
                               signal_tf_filter="5min", data_1m=data_1m)
        if not turbo.empty:
            turbo.to_csv(RESULTS_DIR / f"turbo_{buffer:.1f}pct.csv", index=False)
            print_turbo_summary(turbo, buffer, 500)

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3c: Turbo × Strategy filter comparison
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("  TURBO × STRATEGY FILTERS  |  500 SEK per trade  |  All 5-min signals")
    print("  How does each filter affect turbo P&L?")
    print("=" * 90)

    strategy_filters_rows = []
    for sig_tf_label, sig_tf_vals in [("5min", ["5min"]), ("1min", ["1min"])]:
        uni_tf = unified[unified["signal_tf"].isin(sig_tf_vals)]
        if len(uni_tf) < 10:
            print(f"\n  Skipping {sig_tf_label} — only {len(uni_tf)} signals")
            continue

        has_vol = "vol_spike" in uni_tf.columns
        strategy_filters = [
            ("Baseline", uni_tf),
            ("Inside Monday", uni_tf[uni_tf["gap_type"] == "inside_monday"]),
            ("Trend-aligned", uni_tf[uni_tf["trend_aligned"]]),
            ("Inside Mon + Trend", uni_tf[(uni_tf["gap_type"] == "inside_monday") & uni_tf["trend_aligned"]]),
            ("Vol Spike (≥2x)", uni_tf[uni_tf["vol_spike"]] if has_vol else uni_tf.head(0)),
            ("Vol Spike + Trend", uni_tf[uni_tf["vol_spike"] & uni_tf["trend_aligned"]] if has_vol else uni_tf.head(0)),
            ("Inside Mon + Vol", uni_tf[(uni_tf["gap_type"] == "inside_monday") & uni_tf["vol_spike"]] if has_vol else uni_tf.head(0)),
        ]

        for buffer in [0.5, 1.0, 2.0]:
            print(f"\n  ── {sig_tf_label} signals | Buffer {buffer}% (~{round(100/buffer)}x leverage) ──")
            print(f"  {'Strategy':30s} | {'n':>5s} | {'KO%':>5s} | {'WR%':>5s} | {'Avg P&L':>10s} | {'Total P&L':>12s} | {'ROI':>7s}")
            print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*7}")

            for label, filtered in strategy_filters:
                if len(filtered) < 5:
                    continue
                result = simulate_turbo(filtered, data_5m, ko_buffer_pct=buffer,
                                        position_sek=500, signal_tf_filter="",
                                        quiet=True, data_1m=data_1m)
                if result.empty:
                    continue
                n = len(result)
                ko_rate = (result["outcome"] == "ko").sum() / n * 100
                win_rate = (result["turbo_pnl_sek"] > 0).sum() / n * 100
                avg_pnl = result["turbo_pnl_sek"].mean()
                total_pnl = result["turbo_pnl_sek"].sum()
                roi = total_pnl / (n * 500) * 100

                print(f"  {label:30s} | {n:5d} | {ko_rate:5.1f} | {win_rate:5.1f} | "
                      f"{avg_pnl:>+10,.0f} | {total_pnl:>+12,.0f} | {roi:>+6.1f}%")

                strategy_filters_rows.append({
                    "signal_tf": sig_tf_label, "strategy": label,
                    "buffer": buffer, "n_trades": n,
                    "ko_rate": ko_rate, "win_rate": win_rate,
                    "avg_pnl": avg_pnl, "total_pnl": total_pnl, "roi": roi,
                })

    if strategy_filters_rows:
        pd.DataFrame(strategy_filters_rows).to_csv(
            RESULTS_DIR / "turbo_strategy_comparison.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3e: OVERNIGHT HOLD COMPARISON — same-day exit vs holding 1-3 nights
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("  OVERNIGHT HOLD: TURBO vs BULL/BEAR CERTIFICATES")
    print("  Turbo: KO on wick touch, high leverage | Bull/Bear: no KO, daily compounding + funding")
    print("=" * 90)

    overnight_rows = []

    # ── TURBO overnight comparison ──
    print("\n  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │  TURBO CERTIFICATES  (KO on wick touch)                            │")
    print("  └─────────────────────────────────────────────────────────────────────┘")

    for buffer in [0.5, 1.0, 2.0]:
        lev = round(100 / buffer)
        print(f"\n  ── Turbo {buffer}% buffer (~{lev}x leverage) ──")
        print(f"  {'Hold':15s} | {'n':>5s} | {'KO%':>5s} | {'WR%':>5s} | {'Avg P&L':>10s} | {'Total P&L':>12s} | {'ROI':>7s}")
        print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*7}")

        for hd in [0, 1, 2, 3]:
            label = "Same-day" if hd == 0 else f"+{hd} night{'s' if hd > 1 else ''}"
            result = simulate_turbo(unified, data_5m, ko_buffer_pct=buffer,
                                    position_sek=500, signal_tf_filter="5min",
                                    quiet=True, data_1m=data_1m, hold_days=hd)
            if result.empty:
                continue
            n = len(result)
            ko_rate = (result["outcome"] == "ko").sum() / n * 100
            win_rate = (result["turbo_pnl_sek"] > 0).sum() / n * 100
            avg_pnl = result["turbo_pnl_sek"].mean()
            total_pnl = result["turbo_pnl_sek"].sum()
            roi = total_pnl / (n * 500) * 100

            print(f"  {label:15s} | {n:5d} | {ko_rate:5.1f} | {win_rate:5.1f} | "
                  f"{avg_pnl:>+10,.0f} | {total_pnl:>+12,.0f} | {roi:>+6.1f}%")

            overnight_rows.append({
                "product": "turbo", "leverage": lev,
                "buffer": buffer, "hold_days": hd, "label": label,
                "n_trades": n, "ko_rate": ko_rate, "win_rate": win_rate,
                "avg_pnl": avg_pnl, "total_pnl": total_pnl, "roi": roi,
            })

    # ── BULL/BEAR CERT overnight comparison ──
    print("\n  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │  BULL/BEAR CERTIFICATES  (no KO, daily compounding, 0.07%/day fee) │")
    print("  └─────────────────────────────────────────────────────────────────────┘")

    for lev in [3, 5, 10, 15]:
        print(f"\n  ── Bull/Bear {lev}x leverage ──")
        print(f"  {'Hold':15s} | {'n':>5s} | {'WR%':>5s} | {'Avg P&L':>10s} | {'Total P&L':>12s} | {'ROI':>7s} | {'MaxDD':>10s}")
        print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*7}-+-{'-'*10}")

        for hd in [0, 1, 2, 3, 5]:
            label = "Same-day" if hd == 0 else f"+{hd} night{'s' if hd > 1 else ''}"
            result = simulate_bull_bear(unified, data_5m, leverage=lev,
                                        position_sek=500, daily_funding_pct=0.07,
                                        signal_tf_filter="5min", quiet=True,
                                        hold_days=hd)
            if result.empty:
                continue
            n = len(result)
            win_rate = (result["cert_pnl_sek"] > 0).sum() / n * 100
            avg_pnl = result["cert_pnl_sek"].mean()
            total_pnl = result["cert_pnl_sek"].sum()
            roi = total_pnl / (n * 500) * 100
            # Max drawdown
            cum_pnl = result["cert_pnl_sek"].cumsum()
            peak = cum_pnl.expanding().max()
            maxdd = (cum_pnl - peak).min()

            print(f"  {label:15s} | {n:5d} | {win_rate:5.1f} | "
                  f"{avg_pnl:>+10,.0f} | {total_pnl:>+12,.0f} | {roi:>+6.1f}% | {maxdd:>+10,.0f}")

            overnight_rows.append({
                "product": "bull_bear", "leverage": lev,
                "buffer": 0, "hold_days": hd, "label": label,
                "n_trades": n, "ko_rate": 0, "win_rate": win_rate,
                "avg_pnl": avg_pnl, "total_pnl": total_pnl, "roi": roi,
            })

    if overnight_rows:
        pd.DataFrame(overnight_rows).to_csv(
            RESULTS_DIR / "overnight_comparison.csv", index=False)

    # ── Head-to-head: Turbo vs Bull/Bear at similar leverage, trend-aligned ──
    print("\n  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │  HEAD-TO-HEAD: Turbo vs Bull/Bear (trend-aligned, ~100x leverage)  │")
    print("  └─────────────────────────────────────────────────────────────────────┘")

    trend_uni = unified[unified["trend_aligned"]]
    print(f"\n  {'Product':20s} | {'Hold':10s} | {'n':>5s} | {'KO%':>5s} | {'WR%':>5s} | {'Avg P&L':>10s} | {'Total P&L':>12s} | {'ROI':>7s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*7}")
    for hd in [0, 1, 2]:
        hold_label = "Same-day" if hd == 0 else f"+{hd} night{'s' if hd > 1 else ''}"

        # Turbo 1% buffer (~100x)
        t = simulate_turbo(trend_uni, data_5m, ko_buffer_pct=1.0,
                           position_sek=500, signal_tf_filter="5min",
                           quiet=True, data_1m=data_1m, hold_days=hd)
        if not t.empty:
            n = len(t)
            ko_r = (t["outcome"] == "ko").sum() / n * 100
            wr = (t["turbo_pnl_sek"] > 0).sum() / n * 100
            roi = t["turbo_pnl_sek"].sum() / (n * 500) * 100
            print(f"  {'Turbo 1% (~100x)':20s} | {hold_label:10s} | {n:5d} | {ko_r:5.1f} | {wr:5.1f} | "
                  f"{t['turbo_pnl_sek'].mean():>+10,.0f} | {t['turbo_pnl_sek'].sum():>+12,.0f} | {roi:>+6.1f}%")

        # Bull/Bear 10x
        b = simulate_bull_bear(trend_uni, data_5m, leverage=10,
                               position_sek=500, daily_funding_pct=0.07,
                               signal_tf_filter="5min", quiet=True, hold_days=hd)
        if not b.empty:
            n = len(b)
            wr = (b["cert_pnl_sek"] > 0).sum() / n * 100
            roi = b["cert_pnl_sek"].sum() / (n * 500) * 100
            print(f"  {'Bull/Bear 10x':20s} | {hold_label:10s} | {n:5d} |   n/a | {wr:5.1f} | "
                  f"{b['cert_pnl_sek'].mean():>+10,.0f} | {b['cert_pnl_sek'].sum():>+12,.0f} | {roi:>+6.1f}%")

        # Bull/Bear 15x
        b15 = simulate_bull_bear(trend_uni, data_5m, leverage=15,
                                 position_sek=500, daily_funding_pct=0.07,
                                 signal_tf_filter="5min", quiet=True, hold_days=hd)
        if not b15.empty:
            n = len(b15)
            wr = (b15["cert_pnl_sek"] > 0).sum() / n * 100
            roi = b15["cert_pnl_sek"].sum() / (n * 500) * 100
            print(f"  {'Bull/Bear 15x':20s} | {hold_label:10s} | {n:5d} |   n/a | {wr:5.1f} | "
                  f"{b15['cert_pnl_sek'].mean():>+10,.0f} | {b15['cert_pnl_sek'].sum():>+12,.0f} | {roi:>+6.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3f: SMA-GRADIENT RUNNER — hold overnight when trend is strong
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("  SMA-GRADIENT RUNNER  |  Hold overnight when SMA slope is strong")
    print("  At EOD: check SMA gradient → if steep enough in trade direction, hold")
    print("  KO still applies on all days | Full-reversal stop on signal day only")
    print("=" * 90)

    runner_rows = []

    # Test different SMA periods and gradient thresholds
    # Focus on trend-aligned signals (the user's preferred filter)
    trend_uni = unified[unified["trend_aligned"]]

    for buffer in [1.0, 2.0]:
        lev = round(100 / buffer)
        print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │  Turbo {buffer}% buffer (~{lev}x) — Trend-aligned signals              │")
        print(f"  └─────────────────────────────────────────────────────────────────┘")

        # Baseline: same-day exit (hold_days=0)
        baseline = simulate_turbo(trend_uni, data_5m, ko_buffer_pct=buffer,
                                  position_sek=500, signal_tf_filter="5min",
                                  quiet=True, data_1m=data_1m, hold_days=0)
        if not baseline.empty:
            n = len(baseline)
            ko_r = (baseline["outcome"] == "ko").sum() / n * 100
            wr = (baseline["turbo_pnl_sek"] > 0).sum() / n * 100
            roi = baseline["turbo_pnl_sek"].sum() / (n * 500) * 100
            print(f"\n  BASELINE (always exit at EOD):")
            print(f"    n={n} | KO={ko_r:.1f}% | WR={wr:.1f}% | ROI={roi:+.1f}%")

        # Always hold +1 night (for comparison)
        always_hold = simulate_turbo(trend_uni, data_5m, ko_buffer_pct=buffer,
                                     position_sek=500, signal_tf_filter="5min",
                                     quiet=True, data_1m=data_1m, hold_days=1)
        if not always_hold.empty:
            n = len(always_hold)
            ko_r = (always_hold["outcome"] == "ko").sum() / n * 100
            wr = (always_hold["turbo_pnl_sek"] > 0).sum() / n * 100
            roi = always_hold["turbo_pnl_sek"].sum() / (n * 500) * 100
            print(f"\n  ALWAYS HOLD +1 NIGHT:")
            print(f"    n={n} | KO={ko_r:.1f}% | WR={wr:.1f}% | ROI={roi:+.1f}%")

        # Runner: SMA gradient-based hold
        print(f"\n  {'SMA':>5s} | {'Grad%':>5s} | {'MaxD':>4s} | {'n':>5s} | {'KO%':>5s} | {'WR%':>5s} | "
              f"{'Avg P&L':>10s} | {'Total P&L':>12s} | {'ROI':>7s} | {'Held ON':>7s} | {'Avg Days':>8s}")
        print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-"
              f"{'-'*10}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

        for sma_p in [10, 20, 50]:
            for grad_thresh in [0.1, 0.3, 0.5, 1.0]:
                for max_d in [3, 5]:
                    result = simulate_turbo_runner(
                        trend_uni, data_5m, daily_data,
                        ko_buffer_pct=buffer, position_sek=500,
                        sma_period=sma_p, gradient_lookback=5,
                        gradient_threshold=grad_thresh,
                        max_hold_days=max_d,
                        signal_tf_filter="5min", quiet=True,
                        data_1m=data_1m)
                    if result.empty:
                        continue
                    n = len(result)
                    ko_r = (result["outcome"] == "ko").sum() / n * 100
                    wr = (result["turbo_pnl_sek"] > 0).sum() / n * 100
                    avg_pnl = result["turbo_pnl_sek"].mean()
                    total_pnl = result["turbo_pnl_sek"].sum()
                    roi = total_pnl / (n * 500) * 100
                    held_on = (result["days_held"] > 1).sum()
                    avg_days = result["days_held"].mean()

                    print(f"  {sma_p:5d} | {grad_thresh:5.1f} | {max_d:4d} | {n:5d} | {ko_r:5.1f} | {wr:5.1f} | "
                          f"{avg_pnl:>+10,.0f} | {total_pnl:>+12,.0f} | {roi:>+6.1f}% | "
                          f"{held_on:4d}({held_on/n*100:4.1f}%) | {avg_days:6.2f}")

                    runner_rows.append({
                        "buffer": buffer, "sma_period": sma_p,
                        "gradient_threshold": grad_thresh, "max_hold": max_d,
                        "n_trades": n, "ko_rate": ko_r, "win_rate": wr,
                        "avg_pnl": avg_pnl, "total_pnl": total_pnl, "roi": roi,
                        "held_overnight": held_on, "avg_days": avg_days,
                    })

    if runner_rows:
        pd.DataFrame(runner_rows).to_csv(
            RESULTS_DIR / "runner_comparison.csv", index=False)

    # Print detailed summary for the best-looking runner config
    if runner_rows:
        best = max(runner_rows, key=lambda r: r["roi"])
        print(f"\n  Best runner config: SMA({best['sma_period']}) gradient>{best['gradient_threshold']}% "
              f"max {best['max_hold']}d @ {best['buffer']}% buffer → ROI={best['roi']:+.1f}%")

        # Run detailed summary for best config
        best_result = simulate_turbo_runner(
            trend_uni, data_5m, daily_data,
            ko_buffer_pct=best["buffer"], position_sek=500,
            sma_period=best["sma_period"], gradient_lookback=5,
            gradient_threshold=best["gradient_threshold"],
            max_hold_days=best["max_hold"],
            signal_tf_filter="5min", quiet=True, data_1m=data_1m)
        if not best_result.empty:
            print_runner_summary(best_result, best["buffer"], 500,
                                f"SMA({best['sma_period']}) grad>{best['gradient_threshold']}%")
            best_result.to_csv(RESULTS_DIR / "runner_best.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3d: 1.5:1 R:R simulation (kept for comparison)
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 90)
    print("  R:R EXIT SIMULATION  |  TP = 1.5x risk (price touch)")
    print("  SL = no-wick candle retreats past ORB level  |  Risk = entry - ORB level")
    print("=" * 90)

    rr_results = simulate_rr_exit(unified, data_5m, rr_target=1.5)
    if not rr_results.empty:
        rr_results.to_csv(RESULTS_DIR / "rr_simulation.csv", index=False)
        print_rr_summary(rr_results)

    # ── Individual factors ──
    prt("UNIFIED — Overall", win_rate_stats(unified))
    prt("UNIFIED — Per Signal TF", win_rate_stats(unified, ["signal_tf"]))
    prt("UNIFIED — Per Category", win_rate_stats(unified, ["category"]))

    prt("FACTOR: Monday Confluence", win_rate_stats(unified, ["confluence"]))
    prt("FACTOR: VWAP Confirmed (>1 SD)", win_rate_stats(unified, ["vwap_confirmed"]))
    prt("FACTOR: SMA Trend", win_rate_stats(unified, ["trend"]))
    prt("FACTOR: Trend-Aligned", win_rate_stats(unified, ["trend_aligned"]))
    prt("FACTOR: S/R Zone", win_rate_stats(unified, ["sr_type"]))
    prt("FACTOR: ORB Gap vs Monday", win_rate_stats(unified, ["gap_type"]))

    # ── Cross-factor combinations ──
    print("\n" + "=" * 90)
    print("  CROSS-FACTOR ANALYSIS (every combination)")
    print("=" * 90)

    # Confluence x VWAP
    prt("Confluence x VWAP", win_rate_stats(unified, ["confluence", "vwap_confirmed"]))

    # Confluence x Trend-aligned
    prt("Confluence x Trend-Aligned", win_rate_stats(unified, ["confluence", "trend_aligned"]))

    # Confluence x S/R
    prt("Confluence x S/R", win_rate_stats(unified, ["confluence", "sr_type"]))

    # VWAP x Trend-aligned
    prt("VWAP x Trend-Aligned", win_rate_stats(unified, ["vwap_confirmed", "trend_aligned"]))

    # VWAP x S/R
    prt("VWAP x S/R", win_rate_stats(unified, ["vwap_confirmed", "sr_type"]))

    # Trend x S/R
    prt("Trend-Aligned x S/R", win_rate_stats(unified, ["trend_aligned", "sr_type"]))

    # Gap type x Confluence
    prt("Gap Type x Confluence", win_rate_stats(unified, ["gap_type", "confluence"]))

    # Gap type x Trend-aligned
    prt("Gap Type x Trend-Aligned", win_rate_stats(unified, ["gap_type", "trend_aligned"]))

    # Triple combos (most interesting stacks)
    prt("Confluence x VWAP x Trend-Aligned",
        win_rate_stats(unified, ["confluence", "vwap_confirmed", "trend_aligned"]))

    prt("Confluence x VWAP x S/R",
        win_rate_stats(unified, ["confluence", "vwap_confirmed", "sr_type"]))

    # ── Best signal TF for each factor combination ──
    print("\n" + "=" * 90)
    print("  BEST SIGNAL TF PER FACTOR COMBO")
    print("=" * 90)

    # Key combos by signal TF
    for label, mask in [
        ("Confluence + Trend-Aligned", unified["confluence"] & unified["trend_aligned"]),
        ("Confluence + VWAP", unified["confluence"] & unified["vwap_confirmed"]),
        ("Confluence + VWAP + Trend", unified["confluence"] & unified["vwap_confirmed"] & unified["trend_aligned"]),
        ("All factors (confl+VWAP+trend+S/R break)", unified["confluence"] & unified["vwap_confirmed"] & unified["trend_aligned"] & (unified["sr_type"] == "sr_break")),
    ]:
        subset = unified[mask]
        if not subset.empty:
            prt(f"{label} — Per Signal TF", win_rate_stats(subset, ["signal_tf"]))

    # ── Cross-strategy summary ──
    print("\n" + "=" * 90)
    print("  CROSS-STRATEGY SUMMARY")
    print("=" * 90)

    def _sr(label, df):
        if df.empty:
            return None
        total = len(df)
        wins = int(df["win"].sum())
        longs = df[df["direction"] == "long"]
        shorts = df[df["direction"] == "short"]
        return {
            "strategy": label, "trades": total,
            "WR": f"{wins/total:.1%}",
            "L_WR": f"{longs['win'].sum()/len(longs):.1%}" if len(longs) else "N/A",
            "S_WR": f"{shorts['win'].sum()/len(shorts):.1%}" if len(shorts) else "N/A",
        }

    summary = []
    for label, df in [
        ("Baseline ORB (all TFs)", res_a),
        ("Monday Range (5Y daily)", res_b),
        ("ORB 5m window", res_d[res_d["orb_window"] == 5] if not res_d.empty else pd.DataFrame()),
        ("ORB 15m window", res_d[res_d["orb_window"] == 15] if not res_d.empty else pd.DataFrame()),
        ("ORB 30m window", res_d[res_d["orb_window"] == 30] if not res_d.empty else pd.DataFrame()),
        ("Monthly hold", res_g),
        ("──────", pd.DataFrame()),
        ("+ Monday confluence", unified[unified["confluence"]]),
        ("+ VWAP confirmed", unified[unified["vwap_confirmed"]]),
        ("+ Trend-aligned", unified[unified["trend_aligned"]]),
        ("+ S/R break", unified[unified["sr_type"] == "sr_break"]),
        ("+ Open inside Monday", unified[unified["gap_type"] == "inside_monday"]),
        ("──────", pd.DataFrame()),
        ("Confluence + VWAP", unified[unified["confluence"] & unified["vwap_confirmed"]]),
        ("Confluence + Trend", unified[unified["confluence"] & unified["trend_aligned"]]),
        ("Confluence + VWAP + Trend", unified[unified["confluence"] & unified["vwap_confirmed"] & unified["trend_aligned"]]),
        ("All 4 factors", unified[unified["confluence"] & unified["vwap_confirmed"] & unified["trend_aligned"] & (unified["sr_type"] == "sr_break")]),
    ]:
        if label == "──────":
            summary.append({"strategy": "──────", "trades": "", "WR": "", "L_WR": "", "S_WR": ""})
            continue
        r = _sr(label, df)
        if r:
            summary.append(r)

    if summary:
        print(pd.DataFrame(summary).to_string(index=False))
    print()
    print(f"\nResults saved to: {RESULTS_DIR.resolve()}")
    print(f"Unified signal CSV: {RESULTS_DIR / 'unified_signals.csv'}")


if __name__ == "__main__":
    main()
