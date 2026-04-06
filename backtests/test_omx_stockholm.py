"""
ORB Strategy Backtest for OMX Stockholm Top 20 Stocks

Tests the same ORB strategy as the main backtest, but adapted for Swedish market hours:
- Market opens at 09:00 CET/CEST (vs 09:30 ET for US)
- Market closes at 17:30 CET/CEST (vs 16:00 ET for US)
- ORB = first 5-minute candle after market open (09:00-09:05)
- All tickers use .ST suffix for Yahoo Finance

Strategy variants:
  A. ORB Breakout (5min signal timeframe, 60d history)
     - Turbo simulation at 1% and 2% KO buffer
     - Full reversal stop: long stopped when close < ORB low, short when close > ORB high
  B. SMA Gradient Runner (max 5 days hold, SMA20 gradient > 0.3%)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ──────────────────────────────────────────────────────────────────

# Top 20 OMXS30 stocks (Yahoo Finance format with .ST suffix)
TICKERS = [
    "VOLV-B.ST",    # Volvo
    "ERIC-B.ST",    # Ericsson
    "ASSA-B.ST",    # Assa Abloy
    "ATCO-A.ST",    # Atlas Copco A
    "ATCO-B.ST",    # Atlas Copco B
    "SEB-A.ST",     # SEB
    "SWED-A.ST",    # Swedbank
    "SHB-A.ST",     # Handelsbanken
    "INVE-B.ST",    # Investor B
    "ABB.ST",       # ABB
    "HEXA-B.ST",    # Hexagon B
    "SAND.ST",      # Sandvik
    "ALFA.ST",      # Alfa Laval
    "ESSITY-B.ST",  # Essity
    "TELIA.ST",     # Telia
    "HM-B.ST",      # H&M
    "SINCH.ST",     # Sinch
    "ELUX-B.ST",    # Electrolux
    "SAAB-B.ST",    # Saab
    "EVO.ST",       # Evolution
]

# Stockholm market hours (local time CET/CEST)
MARKET_OPEN = pd.Timestamp("09:00").time()
MARKET_CLOSE = pd.Timestamp("17:30").time()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# US benchmark results for comparison (from main backtest)
US_BENCHMARK = {
    "turbo_1pct_roi": 1.7,
    "turbo_2pct_roi": -3.2,
    "runner_roi": 16.7,
}


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


def fetch_intraday_5m(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download ~60 days of 5-min data."""
    print(f"Fetching 5-min data (~60 days) for {len(tickers)} Swedish stocks...")
    result = _split_download(tickers, period="60d", interval="5m")
    print(f"  Got data for {len(result)}/{len(tickers)} tickers")
    return result


def fetch_daily(tickers: list[str], period: str = "5y") -> dict[str, pd.DataFrame]:
    """Download daily data."""
    print(f"Fetching daily data ({period}) for {len(tickers)} Swedish stocks...")
    result = _split_download(tickers, period=period, interval="1d")
    print(f"  Got data for {len(result)}/{len(tickers)} tickers")
    return result


# ── Timezone Helpers ────────────────────────────────────────────────────────

def _to_stockholm(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with index in Stockholm time."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Europe/Stockholm")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("Europe/Stockholm")
    return df


def _market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Stockholm market hours only. Assumes Stockholm-time index."""
    times = df.index.time
    return df[(times >= MARKET_OPEN) & (times <= MARKET_CLOSE)]


# ── ORB Computation ─────────────────────────────────────────────────────────

def compute_orb(df: pd.DataFrame, window_minutes: int = 5) -> pd.DataFrame:
    """Opening range high/low from first 5 minutes of each trading day.

    For Stockholm: 09:00-09:05 CET/CEST
    """
    df = _market_hours(_to_stockholm(df))
    df_dates = df.index.date

    orb_rows = []
    for date in np.unique(df_dates):
        day = df[df_dates == date]
        # First candle (09:00-09:05)
        opening = day.iloc[:1]
        if opening.empty:
            continue
        orb_rows.append({
            "date": date,
            "orb_open": opening["Open"].iloc[0],
            "orb_high": opening["High"].max(),
            "orb_low": opening["Low"].min(),
        })
    return pd.DataFrame(orb_rows)


# ── SMA Gradient ────────────────────────────────────────────────────────────

def compute_sma_gradient(daily_df: pd.DataFrame, sma_period: int = 20,
                         lookback: int = 5) -> dict:
    """Compute SMA slope as % change over lookback trading days.

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


# ── Signal Detection ────────────────────────────────────────────────────────

def compute_monday_range(daily_df: pd.DataFrame) -> dict:
    """Compute Monday's high/low as weekly reference levels.

    Returns dict: date -> {"mon_high": float, "mon_low": float}
    Maps each Tue-Fri to that week's Monday range.
    """
    df = daily_df.copy()
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        pass  # already tz-aware
    result = {}
    current_mon_high = None
    current_mon_low = None

    for i in range(len(df)):
        dt = df.index[i]
        d = dt.date() if hasattr(dt, 'date') else dt
        dow = dt.weekday() if hasattr(dt, 'weekday') else pd.Timestamp(dt).weekday()

        if dow == 0:  # Monday
            current_mon_high = float(df["High"].iloc[i])
            current_mon_low = float(df["Low"].iloc[i])
        elif current_mon_high is not None:
            result[d] = {"mon_high": current_mon_high, "mon_low": current_mon_low}

    return result


def detect_orb_signals(df: pd.DataFrame, orb: pd.DataFrame,
                       daily_df: pd.DataFrame = None,
                       vol_sma_len: int = 20, vol_spike_mult: float = 2.0,
                       trend_sma_len: int = 50) -> list[dict]:
    """Detect ORB breakout/breakdown signals with NO-WICK condition and quality scoring.

    Bullish:  Low > orb_high  (entire candle above ORB high)
    Bearish:  High < orb_low  (entire candle below ORB low)

    Quality score (0-4):
      T = trend-aligned (close > SMA50 for long, close < SMA50 for short)
      M = inside Monday range (Tue-Fri only, open between Monday high/low)
      B = breakout momentum >= 1%
      V = volume spike >= 2x volume SMA(20)

    Only the first signal per day is recorded.
    """
    df_stk = _market_hours(_to_stockholm(df))
    dates = df_stk.index.date
    orb_lookup = {row["date"]: row for _, row in orb.iterrows()}

    # Pre-compute volume SMA on intraday data
    vol_sma = df_stk["Volume"].rolling(vol_sma_len).mean()

    # Pre-compute daily SMA for trend and Monday range
    trend_sma = {}
    monday_range = {}
    if daily_df is not None:
        sma = daily_df["Close"].rolling(trend_sma_len).mean()
        for i in range(len(daily_df)):
            dt = daily_df.index[i]
            d = dt.date() if hasattr(dt, 'date') else dt
            if pd.notna(sma.iloc[i]):
                trend_sma[d] = float(sma.iloc[i])
        monday_range = compute_monday_range(daily_df)

    # ORB ends at 09:05
    orb_end = pd.Timestamp("09:05").time()

    signals = []
    seen_dates = set()

    for idx_pos in range(len(df_stk)):
        ts = df_stk.index[idx_pos]
        row = df_stk.iloc[idx_pos]
        date = dates[idx_pos]

        if date in seen_dates:
            continue
        orb_data = orb_lookup.get(date)
        if orb_data is None:
            continue
        if ts.time() <= orb_end:
            continue

        orb_high = orb_data["orb_high"]
        orb_low = orb_data["orb_low"]

        direction = None
        # No-wick breakout: entire candle above ORB high
        if row["Low"] > orb_high:
            direction = "long"
        # No-wick breakdown: entire candle below ORB low
        elif row["High"] < orb_low:
            direction = "short"

        if direction is None:
            continue

        entry = row["Close"]

        # Quality scoring
        # T: trend-aligned
        sma_val = trend_sma.get(date)
        if sma_val is not None:
            trend_aligned = (direction == "long" and entry > sma_val) or \
                           (direction == "short" and entry < sma_val)
        else:
            trend_aligned = False

        # M: inside Monday range (skip Mondays)
        dow = ts.weekday()
        mon_data = monday_range.get(date)
        if mon_data is not None and dow != 0:
            # Check if day's open is inside Monday's range
            day_data = df_stk[dates == date]
            day_open = day_data.iloc[0]["Open"] if len(day_data) > 0 else entry
            inside_monday = mon_data["mon_low"] <= day_open <= mon_data["mon_high"]
        else:
            inside_monday = False

        # B: breakout momentum >= 1%
        breakout_pct = (entry - orb_high) / entry * 100 if direction == "long" else (orb_low - entry) / entry * 100
        is_breakout_momentum = breakout_pct >= 1.0

        # V: volume spike
        vs = vol_sma.iloc[idx_pos] if idx_pos < len(vol_sma) else None
        vol_ratio = row["Volume"] / vs if vs and vs > 0 else 0
        is_vol_spike = vol_ratio >= vol_spike_mult

        quality = sum([trend_aligned, inside_monday, is_breakout_momentum, is_vol_spike])

        signals.append({
            "date": date,
            "time": ts.time(),
            "direction": direction,
            "entry_price": entry,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "quality": quality,
            "trend_aligned": trend_aligned,
            "inside_monday": inside_monday,
            "breakout_pct": breakout_pct,
            "is_breakout_momentum": is_breakout_momentum,
            "vol_ratio": vol_ratio,
            "is_vol_spike": is_vol_spike,
        })
        seen_dates.add(date)

    return signals


# ── Daily Close Extraction ──────────────────────────────────────────────────

def get_daily_closes(df: pd.DataFrame) -> dict:
    """Extract daily close price from intraday data."""
    df = _market_hours(_to_stockholm(df))
    dates = df.index.date
    closes = {}
    for date in np.unique(dates):
        day = df[dates == date]
        closes[date] = day["Close"].iloc[-1]
    return closes


# ── Turbo Simulation ────────────────────────────────────────────────────────

def simulate_turbo(signals: list[dict], df_5m: pd.DataFrame,
                   ko_buffer_pct: float = 1.0, position_sek: float = 500.0) -> list[dict]:
    """Simulate turbo/knock-out certificate trading on ORB signals.

    Turbo mechanics:
    - Long turbo: KO level = entry × (1 - ko_buffer/100)
    - Short turbo: KO level = entry × (1 + ko_buffer/100)
    - If underlying touches KO → lose entire position
    - Otherwise exit at daily close

    Exit conditions (whichever comes first):
    1. KO hit (any candle wick touches KO level) → -position_sek
    2. Full reversal past ORB (manual exit) → turbo P&L at that candle's close
    3. Daily close → turbo P&L at close
    """
    df = _market_hours(_to_stockholm(df_5m))
    dates = df.index.date
    buffer_frac = ko_buffer_pct / 100

    results = []
    for sig in signals:
        direction = sig["direction"]
        entry = sig["entry_price"]
        orb_high = sig["orb_high"]
        orb_low = sig["orb_low"]
        trade_date = sig["date"]
        signal_time = sig["time"]

        # KO level
        ko_level = entry * (1 - buffer_frac) if direction == "long" else entry * (1 + buffer_frac)

        # Get all candles after signal time on signal day
        day_data = df[dates == trade_date]
        time_filter = signal_time if isinstance(signal_time, str) else str(signal_time)
        candles = day_data[day_data.index.strftime("%H:%M:%S") > time_filter]

        if len(candles) == 0:
            continue

        outcome = "close"
        exit_price = float(candles.iloc[-1]["Close"])

        # Check for KO or manual exit
        for _ts, candle in candles.iterrows():
            if direction == "long":
                if candle["Low"] <= ko_level:
                    outcome = "ko"
                    exit_price = ko_level
                    break
                # Full-reversal stop: close below ORB low
                if candle["Close"] < orb_low:
                    outcome = "manual_exit"
                    exit_price = float(candle["Close"])
                    break
            else:
                if candle["High"] >= ko_level:
                    outcome = "ko"
                    exit_price = ko_level
                    break
                # Full-reversal stop: close above ORB high
                if candle["Close"] > orb_high:
                    outcome = "manual_exit"
                    exit_price = float(candle["Close"])
                    break

        # Calculate turbo P&L
        if outcome == "ko":
            pnl_sek = -position_sek
        else:
            underlying_return = (exit_price - entry) / entry if direction == "long" else (entry - exit_price) / entry
            turbo_return = underlying_return / buffer_frac
            turbo_return = max(turbo_return, -1.0)
            pnl_sek = position_sek * turbo_return

        results.append({
            **sig,
            "exit_price": exit_price,
            "ko_level": ko_level,
            "outcome": outcome,
            "turbo_pnl_sek": pnl_sek,
        })

    return results


# ── SMA Gradient Runner Simulation ─────────────────────────────────────────

def simulate_runner(signals: list[dict], df_5m: pd.DataFrame,
                   gradients: dict, ticker_dates: list,
                   ko_buffer_pct: float = 1.0, position_sek: float = 500.0,
                   gradient_threshold: float = 0.3, max_hold_days: int = 5) -> list[dict]:
    """Turbo simulation with SMA-gradient-based runner holding.

    Logic:
    1. Enter on signal candle
    2. KO and full-reversal stop apply on signal day
    3. At EOD: check SMA gradient
       - LONG: gradient > +threshold → hold overnight
       - SHORT: gradient < -threshold → hold overnight
       - Otherwise: exit at close
    4. On subsequent days: KO can still trigger, but no ORB-based stop
    5. Max hold = max_hold_days
    """
    df = _market_hours(_to_stockholm(df_5m))
    dates = df.index.date
    buffer_frac = ko_buffer_pct / 100

    results = []
    for sig in signals:
        direction = sig["direction"]
        entry = sig["entry_price"]
        orb_high = sig["orb_high"]
        orb_low = sig["orb_low"]
        trade_date = sig["date"]
        signal_time = sig["time"]

        # KO level
        ko_level = entry * (1 - buffer_frac) if direction == "long" else entry * (1 + buffer_frac)

        # Find trade date index
        try:
            start_idx = ticker_dates.index(trade_date)
        except ValueError:
            continue

        # Simulate day by day
        outcome = "close"
        exit_price = entry
        days_held = 0
        done = False

        for day_offset in range(max_hold_days + 1):
            if done or start_idx + day_offset >= len(ticker_dates):
                break

            sim_date = ticker_dates[start_idx + day_offset]
            days_held = day_offset + 1

            # Get intraday data for this date
            day_data = df[dates == sim_date]
            if len(day_data) == 0:
                break

            # On signal day: start after signal time
            if sim_date == trade_date:
                time_filter = signal_time if isinstance(signal_time, str) else str(signal_time)
                day_data = day_data[day_data.index.strftime("%H:%M:%S") > time_filter]
                if len(day_data) == 0:
                    break

            # Check each candle for KO and (signal-day-only) full-reversal stop
            for _ts, candle in day_data.iterrows():
                if direction == "long":
                    if candle["Low"] <= ko_level:
                        outcome = "ko"
                        exit_price = ko_level
                        done = True
                        break
                    # Full-reversal stop only on signal day
                    if sim_date == trade_date and candle["Close"] < orb_low:
                        outcome = "manual_exit"
                        exit_price = float(candle["Close"])
                        done = True
                        break
                else:
                    if candle["High"] >= ko_level:
                        outcome = "ko"
                        exit_price = ko_level
                        done = True
                        break
                    if sim_date == trade_date and candle["Close"] > orb_high:
                        outcome = "manual_exit"
                        exit_price = float(candle["Close"])
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
                break

            # Check SMA gradient for hold decision
            grad_info = gradients.get(sim_date)
            if grad_info is None:
                # No gradient data → exit at close
                exit_price = eod_price
                outcome = "close"
                break

            grad = grad_info["gradient_pct"]
            favorable = (direction == "long" and grad > gradient_threshold) or \
                       (direction == "short" and grad < -gradient_threshold)

            if not favorable:
                # Gradient not strong enough → exit at close
                exit_price = eod_price
                outcome = "close"
                break
            # else: gradient is favorable → continue to next day

        # Calculate turbo P&L
        if outcome == "ko":
            pnl_sek = -position_sek
        else:
            underlying_return = (exit_price - entry) / entry if direction == "long" else (entry - exit_price) / entry
            turbo_return = underlying_return / buffer_frac
            turbo_return = max(turbo_return, -1.0)
            pnl_sek = position_sek * turbo_return

        results.append({
            **sig,
            "exit_price": exit_price,
            "ko_level": ko_level,
            "outcome": outcome,
            "days_held": days_held,
            "turbo_pnl_sek": pnl_sek,
        })

    return results


# ── Reporting ───────────────────────────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict:
    """Compute overall statistics from simulation results."""
    if not results:
        return {
            "trades": 0, "ko_pct": 0, "wr_pct": 0,
            "total_pnl": 0, "roi_pct": 0,
            "long_trades": 0, "long_wins": 0, "long_wr_pct": 0,
            "short_trades": 0, "short_wins": 0, "short_wr_pct": 0,
        }

    df = pd.DataFrame(results)
    total_trades = len(df)
    ko_count = len(df[df["outcome"] == "ko"])

    # Win = made profit (pnl > 0)
    wins = len(df[df["turbo_pnl_sek"] > 0])

    # Long/Short breakdown
    longs = df[df["direction"] == "long"]
    shorts = df[df["direction"] == "short"]
    long_wins = len(longs[longs["turbo_pnl_sek"] > 0]) if len(longs) else 0
    short_wins = len(shorts[shorts["turbo_pnl_sek"] > 0]) if len(shorts) else 0

    # Total P&L and ROI
    total_pnl = df["turbo_pnl_sek"].sum()
    total_invested = total_trades * 500.0  # 500 SEK per trade
    roi_pct = (total_pnl / total_invested * 100) if total_invested else 0

    return {
        "trades": total_trades,
        "ko_pct": ko_count / total_trades * 100 if total_trades else 0,
        "wr_pct": wins / total_trades * 100 if total_trades else 0,
        "total_pnl": total_pnl,
        "roi_pct": roi_pct,
        "long_trades": len(longs),
        "long_wins": long_wins,
        "long_wr_pct": long_wins / len(longs) * 100 if len(longs) else 0,
        "short_trades": len(shorts),
        "short_wins": short_wins,
        "short_wr_pct": short_wins / len(shorts) * 100 if len(shorts) else 0,
    }


def print_stats_table(stats: dict, label: str):
    """Print statistics as formatted table."""
    print(f"\n{label}")
    print("─" * 90)
    print(f"  Trades: {stats['trades']}")
    print(f"  KO Rate: {stats['ko_pct']:.1f}%")
    print(f"  Win Rate: {stats['wr_pct']:.1f}%")
    print(f"  Total P&L: {stats['total_pnl']:+.0f} SEK")
    print(f"  ROI: {stats['roi_pct']:+.1f}%")
    print(f"\n  Long:  {stats['long_trades']} trades, {stats['long_wins']} wins ({stats['long_wr_pct']:.1f}%)")
    print(f"  Short: {stats['short_trades']} trades, {stats['short_wins']} wins ({stats['short_wr_pct']:.1f}%)")
    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 90)
    print("  ORB Strategy Backtest - OMX Stockholm Top 20")
    print("=" * 90)

    # Fetch data
    data_5m = fetch_intraday_5m(TICKERS)
    daily_data = fetch_daily(TICKERS)

    if not data_5m:
        print("\nERROR: No 5-min data available. Exiting.")
        return

    # Process each ticker
    all_turbo_1pct = []
    all_turbo_2pct = []
    all_runner = []
    ticker_stats = []

    for ticker in sorted(data_5m.keys()):
        print(f"\n{'─'*90}")
        print(f"  Processing {ticker}")
        print(f"{'─'*90}")

        df_5m = data_5m[ticker]
        df_daily = daily_data.get(ticker)

        # Compute ORB
        orb = compute_orb(df_5m, window_minutes=5)
        if orb.empty:
            print(f"  No ORB data for {ticker}")
            continue

        # Detect signals (with quality scoring)
        signals = detect_orb_signals(df_5m, orb, daily_df=df_daily)
        if not signals:
            print(f"  No signals for {ticker}")
            continue

        print(f"  Found {len(signals)} ORB signals")

        # Turbo 1%
        turbo_1 = simulate_turbo(signals, df_5m, ko_buffer_pct=1.0)
        for t in turbo_1:
            t["ticker"] = ticker
        all_turbo_1pct.extend(turbo_1)

        # Turbo 2%
        turbo_2 = simulate_turbo(signals, df_5m, ko_buffer_pct=2.0)
        for t in turbo_2:
            t["ticker"] = ticker
        all_turbo_2pct.extend(turbo_2)

        # Runner (only if daily data available)
        if df_daily is not None:
            gradients = compute_sma_gradient(df_daily, sma_period=20, lookback=5)
            ticker_dates = sorted(set(_market_hours(_to_stockholm(df_5m)).index.date))
            runner = simulate_runner(signals, df_5m, gradients, ticker_dates,
                                    ko_buffer_pct=1.0, gradient_threshold=0.3,
                                    max_hold_days=5)
            for r in runner:
                r["ticker"] = ticker
            all_runner.extend(runner)
        else:
            runner = []

        # Per-ticker stats
        t1_stats = compute_stats(turbo_1)
        t2_stats = compute_stats(turbo_2)
        r_stats = compute_stats(runner)

        ticker_stats.append({
            "ticker": ticker,
            "signals": len(signals),
            "t1_roi": t1_stats["roi_pct"],
            "t2_roi": t2_stats["roi_pct"],
            "r_roi": r_stats["roi_pct"],
            "t1_wr": t1_stats["wr_pct"],
            "t2_wr": t2_stats["wr_pct"],
            "r_wr": r_stats["wr_pct"],
        })

    # Quality distribution
    all_signals = []
    for t in all_turbo_1pct:
        all_signals.append(t)

    print("\n" + "=" * 90)
    print("  QUALITY SCORE DISTRIBUTION")
    print("=" * 90)
    for q in range(5):
        q_sigs = [s for s in all_signals if s.get("quality", 0) == q]
        n = len(q_sigs)
        pct = n / len(all_signals) * 100 if all_signals else 0
        print(f"  Quality {q}: {n:>5} signals ({pct:>5.1f}%)")
    ta_sigs = [s for s in all_signals if s.get("trend_aligned", False)]
    print(f"  Trend-aligned: {len(ta_sigs)} / {len(all_signals)} ({len(ta_sigs)/len(all_signals)*100:.1f}%)")

    # Results by quality threshold
    print("\n" + "=" * 90)
    print("  RESULTS BY QUALITY THRESHOLD")
    print("=" * 90)

    quality_levels = [
        ("All signals (no filter)", lambda s: True),
        ("Trend-aligned (T=1, quality >= 1)", lambda s: s.get("trend_aligned", False)),
        ("Quality >= 2", lambda s: s.get("quality", 0) >= 2),
        ("Quality >= 3", lambda s: s.get("quality", 0) >= 3),
    ]

    print(f"\n  {'Filter':<35} {'Mode':<12} {'Trades':>7} {'KO%':>7} {'WR%':>7} {'ROI%':>8} {'Long WR':>8} {'Short WR':>9}")
    print("  " + "─" * 95)

    for label, filter_fn in quality_levels:
        # Filter turbo 1%
        t1_filtered = [t for t in all_turbo_1pct if filter_fn(t)]
        t2_filtered = [t for t in all_turbo_2pct if filter_fn(t)]
        r_filtered = [t for t in all_runner if filter_fn(t)]

        for mode, data in [("turbo 1%", t1_filtered), ("turbo 2%", t2_filtered), ("runner", r_filtered)]:
            s = compute_stats(data)
            print(f"  {label:<35} {mode:<12} {s['trades']:>7} {s['ko_pct']:>6.1f}% {s['wr_pct']:>6.1f}% {s['roi_pct']:>+7.1f}% {s['long_wr_pct']:>7.1f}% {s['short_wr_pct']:>8.1f}%")
        print()

    # Per-ticker breakdown (trend-aligned only)
    if ticker_stats:
        print("\n" + "=" * 90)
        print("  PER-TICKER BREAKDOWN (all signals)")
        print("=" * 90)
        df_tickers = pd.DataFrame(ticker_stats)
        df_tickers = df_tickers.sort_values("t1_roi", ascending=False)
        print(df_tickers.to_string(index=False, float_format=lambda x: f"{x:+.1f}" if abs(x) > 0.01 else "0.0"))
        print()

    # Comparison with US market
    stats_1pct = compute_stats(all_turbo_1pct)
    stats_runner = compute_stats(all_runner)
    ta_t1 = compute_stats([t for t in all_turbo_1pct if t.get("trend_aligned", False)])
    ta_runner = compute_stats([t for t in all_runner if t.get("trend_aligned", False)])

    print("\n" + "=" * 90)
    print("  COMPARISON: OMX Stockholm vs US Market")
    print("=" * 90)
    print(f"\n  {'Strategy':<40} {'OMX (all)':<15} {'OMX (T=1)':<15} {'US (T=1)':<15}")
    print("  " + "─" * 85)
    print(f"  {'Turbo 1% ROI':<40} {stats_1pct['roi_pct']:+.1f}%{'':<10} {ta_t1['roi_pct']:+.1f}%{'':<10} {US_BENCHMARK['turbo_1pct_roi']:+.1f}%")
    print(f"  {'Runner ROI':<40} {stats_runner['roi_pct']:+.1f}%{'':<10} {ta_runner['roi_pct']:+.1f}%{'':<10} {US_BENCHMARK['runner_roi']:+.1f}%")
    print()

    # Save results
    output_file = RESULTS_DIR / "omx_stockholm.csv"

    all_results = []
    for r in all_turbo_1pct:
        all_results.append({**r, "strategy": "turbo_1pct"})
    for r in all_turbo_2pct:
        all_results.append({**r, "strategy": "turbo_2pct"})
    for r in all_runner:
        all_results.append({**r, "strategy": "runner"})

    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

    # Summary stats
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"  Tickers analyzed: {len(data_5m)}/{len(TICKERS)}")
    print(f"  Total ORB signals: {len(all_turbo_1pct)}")
    print("  Date range: ~60 days intraday + 5 years daily")
    print(f"  Market hours: {MARKET_OPEN.strftime('%H:%M')} - {MARKET_CLOSE.strftime('%H:%M')} CET/CEST")
    print("  ORB window: 09:00-09:05 (first 5-min candle)")
    print()


if __name__ == "__main__":
    main()
