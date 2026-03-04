"""
Re-Breakout Entry Analysis

Tests whether re-entering after a full-reversal stop is worthwhile.

Current strategy: only takes the FIRST breakout signal per day.
This test allows multiple entries per day when price breaks out,
gets stopped (close through full ORB range), and breaks out again.

Stop logic:
  LONG stop  = close < orb_LOW  (full reversal through entire ORB range)
  SHORT stop = close > orb_HIGH (full reversal through entire ORB range)
"""

import sys
from datetime import time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from orb_monday_range import (
    ALL_TICKERS,
    CATEGORY_MAP,
    RESULTS_DIR,
    _market_hours,
    _to_ny,
    compute_orb,
    fetch_daily,
    fetch_intraday_5m,
)

POSITION_SEK = 500.0
ORB_END = dtime(9, 35)  # first 5-min candle ends at 9:35


# ── Re-breakout signal detection ─────────────────────────────────────────────

def detect_rebreakout_signals(
    data_5m: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Walk through each day candle-by-candle, allowing multiple entries.

    State machine per day:
      idle -> in_trade (on breakout) -> stopped (on full reversal close) -> idle

    Returns DataFrame with one row per trade attempt.
    """
    all_trades: list[dict] = []

    for ticker, df_raw in data_5m.items():
        df = _market_hours(_to_ny(df_raw))
        if df.empty:
            continue

        orb = compute_orb(df_raw, window_minutes=5, source_interval_minutes=5)
        orb_lookup = {row["date"]: row for _, row in orb.iterrows()}

        # Rolling 20-bar volume SMA for the ticker (across all days)
        vol_sma = df["Volume"].rolling(20, min_periods=1).mean()

        dates = sorted(set(df.index.date))

        for date in dates:
            orb_data = orb_lookup.get(date)
            if orb_data is None:
                continue

            orb_high = orb_data["orb_high"]
            orb_low = orb_data["orb_low"]
            orb_range = orb_high - orb_low
            if orb_range <= 0:
                continue

            day_df = df[df.index.date == date]
            # Skip ORB formation candle(s) — only look after 9:35
            day_df = day_df[day_df.index.time >= ORB_END]
            if day_df.empty:
                continue

            # State machine
            state = "idle"  # idle | in_trade
            attempt_number = 0
            entry_price = None
            entry_time = None
            direction = None

            for ts, candle in day_df.iterrows():
                candle_vol = candle["Volume"]
                avg_vol = vol_sma.get(ts, candle_vol) if candle_vol > 0 else 1.0
                vol_ratio = candle_vol / avg_vol if avg_vol > 0 else 1.0

                if state == "idle":
                    # Check for breakout (no-wick condition)
                    # LONG: entire candle above orb_high
                    if candle["Low"] > orb_high:
                        attempt_number += 1
                        state = "in_trade"
                        direction = "long"
                        entry_price = candle["Close"]
                        entry_time = ts
                    # SHORT: entire candle below orb_low
                    elif candle["High"] < orb_low:
                        attempt_number += 1
                        state = "in_trade"
                        direction = "short"
                        entry_price = candle["Close"]
                        entry_time = ts

                elif state == "in_trade":
                    # Check for full-reversal stop
                    if direction == "long" and candle["Close"] < orb_low:
                        # Stopped out — record trade
                        exit_price = candle["Close"]
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                        all_trades.append({
                            "ticker": ticker,
                            "category": CATEGORY_MAP.get(ticker, "Unknown"),
                            "date": date,
                            "direction": direction,
                            "attempt_number": attempt_number,
                            "entry_time": entry_time.time(),
                            "exit_time": ts.time(),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "exit_type": "stop",
                            "pnl_pct": pnl_pct,
                            "entry_vol_ratio": round(vol_ratio, 2),
                            "breakout_beyond_pct": round(
                                (entry_price - orb_high) / orb_high * 100, 3
                            ),
                        })
                        state = "idle"
                        direction = None
                        entry_price = None
                        entry_time = None

                    elif direction == "short" and candle["Close"] > orb_high:
                        exit_price = candle["Close"]
                        pnl_pct = (entry_price - exit_price) / entry_price * 100
                        all_trades.append({
                            "ticker": ticker,
                            "category": CATEGORY_MAP.get(ticker, "Unknown"),
                            "date": date,
                            "direction": direction,
                            "attempt_number": attempt_number,
                            "entry_time": entry_time.time(),
                            "exit_time": ts.time(),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "exit_type": "stop",
                            "pnl_pct": pnl_pct,
                            "entry_vol_ratio": round(vol_ratio, 2),
                            "breakout_beyond_pct": round(
                                (orb_low - entry_price) / orb_low * 100, 3
                            ),
                        })
                        state = "idle"
                        direction = None
                        entry_price = None
                        entry_time = None

            # EOD: if still in trade, close at last candle
            if state == "in_trade" and entry_price is not None:
                last = day_df.iloc[-1]
                exit_price = last["Close"]
                if direction == "long":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                # vol ratio at entry time (approximate — use last candle as fallback)
                all_trades.append({
                    "ticker": ticker,
                    "category": CATEGORY_MAP.get(ticker, "Unknown"),
                    "date": date,
                    "direction": direction,
                    "attempt_number": attempt_number,
                    "entry_time": entry_time.time(),
                    "exit_time": day_df.index[-1].time(),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "exit_type": "close",
                    "pnl_pct": pnl_pct,
                    "entry_vol_ratio": 1.0,
                    "breakout_beyond_pct": round(
                        abs(entry_price - (orb_high if direction == "long" else orb_low))
                        / (orb_high if direction == "long" else orb_low) * 100,
                        3,
                    ),
                })

    if not all_trades:
        return pd.DataFrame()
    return pd.DataFrame(all_trades)


# ── Turbo simulation for re-breakout signals ─────────────────────────────────

def simulate_turbo_rebreakout(
    signals_df: pd.DataFrame,
    data_5m: dict[str, pd.DataFrame],
    ko_buffer_pct: float = 1.0,
    position_sek: float = 500.0,
) -> pd.DataFrame:
    """Simulate turbo certificates for each signal (including re-entries).

    - KO checked on wicks (High/Low)
    - Full-reversal stop checked on Close
    - Exit at EOD if neither triggers
    """
    results: list[dict] = []

    # Pre-process 5m data per ticker
    processed: dict[str, pd.DataFrame] = {}
    for ticker, df_raw in data_5m.items():
        processed[ticker] = _market_hours(_to_ny(df_raw))

    buffer_frac = ko_buffer_pct / 100

    for _, sig in signals_df.iterrows():
        ticker = sig["ticker"]
        direction = sig["direction"]
        entry = sig["entry_price"]
        orb_high = sig["orb_high"]
        orb_low = sig["orb_low"]
        trade_date = sig["date"]
        signal_time = sig["entry_time"]

        df = processed.get(ticker)
        if df is None:
            continue

        # KO level
        if direction == "long":
            ko_level = entry * (1 - buffer_frac)
        else:
            ko_level = entry * (1 + buffer_frac)

        # Get candles after signal time on the same day
        day_df = df[df.index.date == trade_date]
        if isinstance(signal_time, str):
            time_filter = signal_time
        else:
            time_filter = str(signal_time)
        day_df = day_df[day_df.index.strftime("%H:%M:%S") > time_filter]

        if day_df.empty:
            continue

        outcome = "close"
        exit_price = float(day_df.iloc[-1]["Close"])
        exit_time = day_df.index[-1].time()

        for ts, candle in day_df.iterrows():
            if direction == "long":
                # KO on wick
                if candle["Low"] <= ko_level:
                    outcome = "ko"
                    exit_price = ko_level
                    exit_time = ts.time()
                    break
                # Full-reversal stop on close
                if candle["Close"] < orb_low:
                    outcome = "manual_exit"
                    exit_price = float(candle["Close"])
                    exit_time = ts.time()
                    break
            else:
                if candle["High"] >= ko_level:
                    outcome = "ko"
                    exit_price = ko_level
                    exit_time = ts.time()
                    break
                if candle["Close"] > orb_high:
                    outcome = "manual_exit"
                    exit_price = float(candle["Close"])
                    exit_time = ts.time()
                    break

        # Turbo P&L
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

        underlying_move = (exit_price - entry) / entry * 100
        if direction == "short":
            underlying_move = -underlying_move

        results.append({
            "ticker": ticker,
            "category": sig["category"],
            "date": str(trade_date),
            "direction": direction,
            "attempt_number": sig["attempt_number"],
            "entry_time": str(signal_time),
            "exit_time": str(exit_time),
            "entry_price": entry,
            "exit_price": exit_price,
            "ko_level": ko_level,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "outcome": outcome,
            "underlying_move_pct": underlying_move,
            "turbo_pnl_sek": pnl_sek,
            "ko_buffer_pct": ko_buffer_pct,
            "entry_vol_ratio": sig.get("entry_vol_ratio", 1.0),
            "breakout_beyond_pct": sig.get("breakout_beyond_pct", 0.0),
        })

    return pd.DataFrame(results)


# ── Analysis helpers ─────────────────────────────────────────────────────────

def turbo_stats(df: pd.DataFrame) -> dict:
    """Compute summary stats from a turbo sim DataFrame."""
    if df.empty:
        return {"n": 0, "ko_pct": np.nan, "wr_pct": np.nan,
                "avg_pnl": np.nan, "total_roi_pct": np.nan}
    n = len(df)
    n_ko = (df["outcome"] == "ko").sum()
    n_win = (df["turbo_pnl_sek"] > 0).sum()
    total_pnl = df["turbo_pnl_sek"].sum()
    invested = n * POSITION_SEK
    return {
        "n": n,
        "ko_pct": round(n_ko / n * 100, 1),
        "wr_pct": round(n_win / n * 100, 1),
        "avg_pnl": round(total_pnl / n, 1),
        "total_roi_pct": round(total_pnl / invested * 100, 1),
    }


def print_table(title: str, rows: list[dict], columns: list[str]):
    """Print a formatted table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Compute column widths
    widths = {}
    for col in columns:
        widths[col] = max(len(col), max(len(str(r.get(col, ""))) for r in rows))

    # Header
    header = "  ".join(f"{col:>{widths[col]}}" for col in columns)
    print(f"  {header}")
    print(f"  {'  '.join('-' * widths[col] for col in columns)}")

    # Rows
    for row in rows:
        line = "  ".join(f"{str(row.get(col, '')):>{widths[col]}}" for col in columns)
        print(f"  {line}")
    print()


# ── Analysis 1: By attempt number ────────────────────────────────────────────

def analysis_by_attempt(turbo_df: pd.DataFrame):
    """How does 1st attempt compare to 2nd, 3rd, etc.?"""
    rows = []
    for attempt in sorted(turbo_df["attempt_number"].unique()):
        sub = turbo_df[turbo_df["attempt_number"] == attempt]
        stats = turbo_stats(sub)
        rows.append({"attempt": int(attempt), **stats})

    # Also show totals
    stats = turbo_stats(turbo_df)
    rows.append({"attempt": "ALL", **stats})

    print_table(
        "ANALYSIS 1: Performance by Attempt Number",
        rows,
        ["attempt", "n", "ko_pct", "wr_pct", "avg_pnl", "total_roi_pct"],
    )


# ── Analysis 2: Max re-entries comparison ────────────────────────────────────

def analysis_max_reentries(turbo_df: pd.DataFrame):
    """Compare strategies with different max entry limits."""
    strategies = [
        ("A: First only (max=1)", 1),
        ("B: 1 re-entry (max=2)", 2),
        ("C: 2 re-entries (max=3)", 3),
        ("D: Unlimited", 999),
    ]

    rows = []
    for label, max_entries in strategies:
        sub = turbo_df[turbo_df["attempt_number"] <= max_entries]
        stats = turbo_stats(sub)
        total_pnl = sub["turbo_pnl_sek"].sum() if not sub.empty else 0
        rows.append({"strategy": label, **stats, "total_pnl_sek": round(total_pnl, 0)})

    print_table(
        "ANALYSIS 2: Max Re-entries Comparison",
        rows,
        ["strategy", "n", "ko_pct", "wr_pct", "avg_pnl", "total_roi_pct", "total_pnl_sek"],
    )


# ── Analysis 3: Conditional re-entry ─────────────────────────────────────────

def analysis_conditional_reentry(turbo_df: pd.DataFrame):
    """Compare conditional vs unconditional re-entry filters."""
    # Only re-entries (attempt >= 2)
    reentries = turbo_df[turbo_df["attempt_number"] >= 2].copy()
    first_only = turbo_df[turbo_df["attempt_number"] == 1].copy()

    if reentries.empty:
        print("\n  No re-entries found — cannot run conditional analysis.")
        return

    # Condition A: Volume >= 1.5x SMA
    vol_filter = reentries[reentries["entry_vol_ratio"] >= 1.5]

    # Condition B: Breakout >= 0.5% beyond ORB level
    strength_filter = reentries[reentries["breakout_beyond_pct"] >= 0.5]

    # Condition C: Both
    both_filter = reentries[
        (reentries["entry_vol_ratio"] >= 1.5) & (reentries["breakout_beyond_pct"] >= 0.5)
    ]

    configs = [
        ("First signals only", first_only),
        ("All re-entries (unconditional)", reentries),
        ("Re-entry: vol >= 1.5x SMA", vol_filter),
        ("Re-entry: breakout >= 0.5%", strength_filter),
        ("Re-entry: vol + breakout", both_filter),
    ]

    rows = []
    for label, sub in configs:
        stats = turbo_stats(sub)
        rows.append({"filter": label, **stats})

    # Also compute "first + conditional re-entry" combos
    print_table(
        "ANALYSIS 3: Conditional Re-entry Filters (re-entries only)",
        rows,
        ["filter", "n", "ko_pct", "wr_pct", "avg_pnl", "total_roi_pct"],
    )

    # Combined strategies (first + filtered re-entries)
    combined_rows = []
    for label, sub in configs[1:]:  # skip "first signals only"
        combined = pd.concat([first_only, sub], ignore_index=True)
        stats = turbo_stats(combined)
        total_pnl = combined["turbo_pnl_sek"].sum()
        combined_rows.append({
            "strategy": f"First + {label}",
            **stats,
            "total_pnl_sek": round(total_pnl, 0),
        })

    # Also show first-only as baseline
    stats = turbo_stats(first_only)
    combined_rows.insert(0, {
        "strategy": "First only (baseline)",
        **stats,
        "total_pnl_sek": round(first_only["turbo_pnl_sek"].sum() if not first_only.empty else 0, 0),
    })

    print_table(
        "ANALYSIS 3b: Combined Strategies (First + Filtered Re-entries)",
        combined_rows,
        ["strategy", "n", "ko_pct", "wr_pct", "avg_pnl", "total_roi_pct", "total_pnl_sek"],
    )


# ── Analysis 4: Time gap between stop and re-entry ──────────────────────────

def analysis_time_gap(signals_df: pd.DataFrame, turbo_df: pd.DataFrame):
    """How long between stop and re-entry? Does a minimum gap help?"""
    # Find re-entries and compute gap from previous stop
    reentries = signals_df[signals_df["attempt_number"] >= 2].copy()
    if reentries.empty:
        print("\n  No re-entries found — cannot analyze time gaps.")
        return

    # For each re-entry, find the previous stop's exit_time
    gaps_minutes: list[float] = []
    gap_data: list[dict] = []

    for _, row in reentries.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        attempt = row["attempt_number"]
        entry_t = row["entry_time"]

        # Find previous attempt for same ticker/date
        prev = signals_df[
            (signals_df["ticker"] == ticker)
            & (signals_df["date"] == date)
            & (signals_df["attempt_number"] == attempt - 1)
        ]
        if prev.empty:
            continue

        prev_exit_t = prev.iloc[0]["exit_time"]

        # Compute gap in minutes
        dt_entry = pd.Timestamp(f"2000-01-01 {entry_t}")
        dt_exit = pd.Timestamp(f"2000-01-01 {prev_exit_t}")
        gap_min = (dt_entry - dt_exit).total_seconds() / 60.0

        gaps_minutes.append(gap_min)
        gap_data.append({
            "ticker": ticker,
            "date": date,
            "attempt": attempt,
            "gap_minutes": gap_min,
        })

    if not gaps_minutes:
        print("\n  Could not compute time gaps.")
        return

    gaps = np.array(gaps_minutes)
    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS 4: Time Gap Between Stop and Re-entry")
    print(f"{'=' * 80}")
    print(f"  Re-entries analyzed: {len(gaps)}")
    print(f"  Mean gap:   {np.mean(gaps):6.1f} minutes")
    print(f"  Median gap: {np.median(gaps):6.1f} minutes")
    print(f"  Min gap:    {np.min(gaps):6.1f} minutes")
    print(f"  Max gap:    {np.max(gaps):6.1f} minutes")
    print()

    # Test minimum gap filters
    gap_filters = [0, 15, 30, 45, 60]
    turbo_reentries = turbo_df[turbo_df["attempt_number"] >= 2].copy()
    turbo_first = turbo_df[turbo_df["attempt_number"] == 1]

    # Merge gap data into turbo_reentries
    gap_df = pd.DataFrame(gap_data)
    if not gap_df.empty and not turbo_reentries.empty:
        turbo_reentries = turbo_reentries.merge(
            gap_df[["ticker", "date", "attempt", "gap_minutes"]],
            left_on=["ticker", "date", "attempt_number"],
            right_on=["ticker", "date", "attempt"],
            how="left",
        )

        rows = []
        for min_gap in gap_filters:
            if min_gap == 0:
                filtered = turbo_reentries
            else:
                filtered = turbo_reentries[turbo_reentries["gap_minutes"] >= min_gap]

            combined = pd.concat([turbo_first, filtered], ignore_index=True)
            stats = turbo_stats(combined)
            re_count = len(filtered)
            rows.append({
                "min_gap_min": min_gap,
                "re_entries": re_count,
                **stats,
            })

        print_table(
            "Minimum Gap Filter (First + Gap-Filtered Re-entries)",
            rows,
            ["min_gap_min", "re_entries", "n", "ko_pct", "wr_pct", "avg_pnl", "total_roi_pct"],
        )


# ── Analysis 5: Direction consistency ────────────────────────────────────────

def analysis_direction_consistency(signals_df: pd.DataFrame, turbo_df: pd.DataFrame):
    """Does re-breakout tend to be same or opposite direction?"""
    reentries = signals_df[signals_df["attempt_number"] >= 2].copy()
    if reentries.empty:
        print("\n  No re-entries found — cannot analyze direction consistency.")
        return

    same_dir = 0
    opp_dir = 0
    direction_data: list[dict] = []

    for _, row in reentries.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        attempt = row["attempt_number"]
        current_dir = row["direction"]

        prev = signals_df[
            (signals_df["ticker"] == ticker)
            & (signals_df["date"] == date)
            & (signals_df["attempt_number"] == attempt - 1)
        ]
        if prev.empty:
            continue

        prev_dir = prev.iloc[0]["direction"]
        if current_dir == prev_dir:
            same_dir += 1
            consistency = "same"
        else:
            opp_dir += 1
            consistency = "opposite"

        direction_data.append({
            "ticker": ticker,
            "date": date,
            "attempt": attempt,
            "consistency": consistency,
        })

    total = same_dir + opp_dir
    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS 5: Direction Consistency of Re-entries")
    print(f"{'=' * 80}")
    if total > 0:
        print(f"  Total re-entries:     {total}")
        print(f"  Same direction:       {same_dir} ({same_dir/total*100:.1f}%)")
        print(f"  Opposite direction:   {opp_dir} ({opp_dir/total*100:.1f}%)")
    else:
        print(f"  No direction data available.")
        return

    # Performance by consistency
    if direction_data and not turbo_df.empty:
        cons_df = pd.DataFrame(direction_data)
        turbo_re = turbo_df[turbo_df["attempt_number"] >= 2].copy()

        turbo_re = turbo_re.merge(
            cons_df[["ticker", "date", "attempt", "consistency"]],
            left_on=["ticker", "date", "attempt_number"],
            right_on=["ticker", "date", "attempt"],
            how="left",
        )

        rows = []
        for cons_type in ["same", "opposite"]:
            sub = turbo_re[turbo_re["consistency"] == cons_type]
            stats = turbo_stats(sub)
            rows.append({"re_entry_dir": cons_type, **stats})

        print_table(
            "Re-entry Performance by Direction Consistency",
            rows,
            ["re_entry_dir", "n", "ko_pct", "wr_pct", "avg_pnl", "total_roi_pct"],
        )

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  RE-BREAKOUT ENTRY ANALYSIS")
    print("  Testing whether re-entries after full-reversal stops are worthwhile")
    print("=" * 80)

    # ── Fetch data ────────────────────────────────────────────────────────
    data_5m = fetch_intraday_5m(ALL_TICKERS)
    # daily_data not needed for core analysis (no confluence/trend here)

    # ── Detect re-breakout signals ────────────────────────────────────────
    print("\nDetecting re-breakout signals...")
    signals_df = detect_rebreakout_signals(data_5m)

    if signals_df.empty:
        print("No signals detected. Exiting.")
        return

    n_total = len(signals_df)
    n_first = (signals_df["attempt_number"] == 1).sum()
    n_re = n_total - n_first
    max_attempt = signals_df["attempt_number"].max()

    print(f"\n  Total trade attempts:  {n_total}")
    print(f"  First attempts:        {n_first}")
    print(f"  Re-entries:            {n_re}")
    print(f"  Max attempt number:    {max_attempt}")
    print(f"  Tickers with signals:  {signals_df['ticker'].nunique()}")
    print(f"  Days with signals:     {signals_df.groupby('ticker')['date'].nunique().sum()}")

    # Distribution of attempts
    print("\n  Attempt distribution:")
    for attempt in sorted(signals_df["attempt_number"].unique()):
        count = (signals_df["attempt_number"] == attempt).sum()
        print(f"    Attempt {attempt}: {count} trades")

    # ── Raw signal performance (no turbo) ─────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  RAW SIGNAL PERFORMANCE (underlying % move, no turbo)")
    print(f"{'=' * 80}")
    for attempt in sorted(signals_df["attempt_number"].unique()):
        sub = signals_df[signals_df["attempt_number"] == attempt]
        n = len(sub)
        wins = (sub["pnl_pct"] > 0).sum()
        wr = wins / n * 100 if n > 0 else 0
        avg_pnl = sub["pnl_pct"].mean()
        n_stopped = (sub["exit_type"] == "stop").sum()
        n_closed = (sub["exit_type"] == "close").sum()
        print(f"  Attempt {attempt}: n={n:4d} | WR={wr:5.1f}% | "
              f"avg_pnl={avg_pnl:+6.3f}% | stopped={n_stopped} | held_to_close={n_closed}")

    # ── Turbo simulations ─────────────────────────────────────────────────
    ko_buffers = [1.0, 2.0]
    all_turbo: dict[float, pd.DataFrame] = {}

    for ko_buf in ko_buffers:
        print(f"\n  Simulating turbo @ {ko_buf}% KO buffer for {n_total} signals...")
        turbo = simulate_turbo_rebreakout(signals_df, data_5m, ko_buffer_pct=ko_buf)
        all_turbo[ko_buf] = turbo

        if turbo.empty:
            print(f"    No turbo results for {ko_buf}% buffer.")
            continue

        leverage = round(100 / ko_buf)
        n = len(turbo)
        total_pnl = turbo["turbo_pnl_sek"].sum()
        invested = n * POSITION_SEK
        n_ko = (turbo["outcome"] == "ko").sum()
        n_win = (turbo["turbo_pnl_sek"] > 0).sum()
        print(f"    ~{leverage}x leverage | {n} trades | "
              f"KO={n_ko/n*100:.1f}% | WR={n_win/n*100:.1f}% | "
              f"Total P&L={total_pnl:+,.0f} SEK | ROI={total_pnl/invested*100:+.1f}%")

    # ── Run all analyses for each KO buffer ───────────────────────────────
    for ko_buf in ko_buffers:
        turbo = all_turbo.get(ko_buf)
        if turbo is None or turbo.empty:
            continue

        leverage = round(100 / ko_buf)
        print(f"\n{'#' * 80}")
        print(f"  TURBO @ {ko_buf}% KO BUFFER (~{leverage}x leverage)")
        print(f"{'#' * 80}")

        analysis_by_attempt(turbo)
        analysis_max_reentries(turbo)
        analysis_conditional_reentry(turbo)
        analysis_time_gap(signals_df, turbo)
        analysis_direction_consistency(signals_df, turbo)

    # ── Save results ──────────────────────────────────────────────────────
    save_rows: list[dict] = []
    for ko_buf, turbo in all_turbo.items():
        if turbo.empty:
            continue
        for _, row in turbo.iterrows():
            save_rows.append({**row.to_dict(), "ko_buffer": ko_buf})

    if save_rows:
        out_path = RESULTS_DIR / "rebreakout_analysis.csv"
        out_df = pd.DataFrame(save_rows)
        out_df.to_csv(out_path, index=False)
        print(f"\n  Results saved to {out_path}")

    # Also save signals
    sig_path = RESULTS_DIR / "rebreakout_signals.csv"
    signals_df.to_csv(sig_path, index=False)
    print(f"  Signals saved to {sig_path}")

    print(f"\n{'=' * 80}")
    print(f"  DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
