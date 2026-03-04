"""
Time-of-Day Analysis for ORB Breakout Signals

Analyzes whether the time of day when a breakout signal fires affects
signal quality, win rate, and turbo simulation performance.

Time buckets:
  09:35-10:00  First 30 min after ORB
  10:00-10:30  Early morning
  10:30-11:00  Late morning
  11:00-12:00  Midday
  12:00-14:00  Early afternoon
  14:00-16:00  Late afternoon
"""

import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the backtests directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

from orb_monday_range import (
    ALL_TICKERS,
    RESULTS_DIR,
    build_unified_signals,
    fetch_daily,
    fetch_intraday_5m,
    simulate_turbo,
    simulate_turbo_runner,
)

# ── Time bucket definitions ──────────────────────────────────────────────────

TIME_BUCKETS = [
    ("09:35-10:00", dtime(9, 35), dtime(10, 0)),
    ("10:00-10:30", dtime(10, 0), dtime(10, 30)),
    ("10:30-11:00", dtime(10, 30), dtime(11, 0)),
    ("11:00-12:00", dtime(11, 0), dtime(12, 0)),
    ("12:00-14:00", dtime(12, 0), dtime(14, 0)),
    ("14:00-16:00", dtime(14, 0), dtime(16, 0)),
]

BUCKET_ORDER = [b[0] for b in TIME_BUCKETS]


def assign_time_bucket(t) -> str:
    """Assign a time value to one of the defined buckets."""
    if isinstance(t, str):
        parts = t.split(":")
        t = dtime(int(parts[0]), int(parts[1]))
    for label, start, end in TIME_BUCKETS:
        if start <= t < end:
            return label
    return "other"


# ── Quality score computation ────────────────────────────────────────────────

def calc_breakout_pct(row) -> float:
    """Breakout distance: how far entry is beyond the ORB level, in %."""
    if row["direction"] == "long":
        return (row["entry_price"] - row["orb_high"]) / row["orb_high"] * 100
    else:
        return (row["orb_low"] - row["entry_price"]) / row["orb_low"] * 100


def compute_quality(row) -> int:
    """Quality score: T + M + B + V.

    T = trend_aligned
    M = gap_type == 'inside_monday' AND weekday != Monday
    B = breakout >= 1%
    V = vol_spike
    """
    score = 0
    if row.get("trend_aligned", False):
        score += 1
    # M: inside monday gap AND not Monday
    if row.get("gap_type") == "inside_monday":
        date = row.get("date")
        if date is not None:
            if isinstance(date, str):
                date = pd.Timestamp(date).date()
            if hasattr(date, "weekday") and date.weekday() != 0:
                score += 1
    if row.get("breakout_pct", 0) >= 1.0:
        score += 1
    if row.get("vol_spike", False):
        score += 1
    return score


# ── Turbo stats helper ───────────────────────────────────────────────────────

def turbo_stats(turbo_df: pd.DataFrame) -> dict:
    """Compute summary stats from a turbo simulation DataFrame."""
    if turbo_df.empty:
        return {"n": 0, "ko_pct": np.nan, "wr_pct": np.nan, "roi_pct": np.nan}
    n = len(turbo_df)
    n_ko = (turbo_df["outcome"] == "ko").sum()
    n_win = (turbo_df["turbo_pnl_sek"] > 0).sum()
    total_pnl = turbo_df["turbo_pnl_sek"].sum()
    invested = n * 500.0
    return {
        "n": n,
        "ko_pct": round(n_ko / n * 100, 1),
        "wr_pct": round(n_win / n * 100, 1),
        "roi_pct": round(total_pnl / invested * 100, 1) if invested > 0 else np.nan,
    }


def _tag_turbo_bucket(turbo_df: pd.DataFrame) -> pd.DataFrame:
    """Add time_bucket column to turbo results."""
    if turbo_df.empty:
        turbo_df["time_bucket"] = pd.Series(dtype=str)
        return turbo_df
    turbo_df = turbo_df.copy()
    turbo_df["time_bucket"] = turbo_df["time"].apply(assign_time_bucket)
    return turbo_df


def _build_result_row(bucket: str, filt: str, direction: str,
                      subset: pd.DataFrame,
                      t1_bucket: pd.DataFrame,
                      t2_bucket: pd.DataFrame,
                      r_bucket: pd.DataFrame) -> dict:
    """Build a single result row with all metrics for a given bucket."""
    n = len(subset)
    wr = subset["win"].mean() * 100 if n > 0 else np.nan
    avg_breakout = subset["breakout_pct"].mean() if n > 0 else np.nan

    ts1 = turbo_stats(t1_bucket)
    ts2 = turbo_stats(t2_bucket)
    rs = turbo_stats(r_bucket)

    return {
        "time_bucket": bucket,
        "filter": filt,
        "direction": direction,
        "n_signals": n,
        "win_rate_pct": round(wr, 1) if not np.isnan(wr) else np.nan,
        "avg_breakout_pct": round(avg_breakout, 2) if not np.isnan(avg_breakout) else np.nan,
        "turbo_1pct_ko": ts1["ko_pct"],
        "turbo_1pct_wr": ts1["wr_pct"],
        "turbo_1pct_roi": ts1["roi_pct"],
        "turbo_2pct_ko": ts2["ko_pct"],
        "turbo_2pct_wr": ts2["wr_pct"],
        "turbo_2pct_roi": ts2["roi_pct"],
        "runner_ko": rs["ko_pct"],
        "runner_wr": rs["wr_pct"],
        "runner_roi": rs["roi_pct"],
    }


# ── Printing ─────────────────────────────────────────────────────────────────

def _print_table(title: str, rows: list[dict]):
    """Print a formatted table from a list of dicts."""
    if not rows:
        print(f"\n  {title}: no data")
        return

    df = pd.DataFrame(rows)
    print(f"\n  {title}")
    print("  " + "-" * 140)

    # Build header
    cols = list(df.columns)
    header = ""
    for col in cols:
        if col == "time_bucket":
            header += f"{'Time Bucket':>14s} "
        elif col == "filter":
            header += f"{'Filter':>12s} "
        elif col == "direction":
            header += f"{'Dir':>6s} "
        elif col == "n_signals":
            header += f"{'N':>5s} "
        else:
            header += f"{col:>13s} "
    print(f"  {header}")
    print("  " + "-" * 140)

    for _, row in df.iterrows():
        line = ""
        for col in cols:
            val = row[col]
            if col == "time_bucket":
                line += f"{val:>14s} "
            elif col == "filter":
                line += f"{val:>12s} "
            elif col == "direction":
                line += f"{val:>6s} "
            elif col == "n_signals":
                line += f"{int(val):>5d} "
            elif pd.isna(val):
                line += f"{'n/a':>13s} "
            elif isinstance(val, float):
                line += f"{val:>13.1f} "
            else:
                line += f"{str(val):>13s} "
        print(f"  {line}")

    print()


# ── Main analysis ────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("TIME-OF-DAY ANALYSIS FOR ORB BREAKOUT SIGNALS")
    print("=" * 80)

    # ── 1. Fetch data ────────────────────────────────────────────────────────
    data_5m = fetch_intraday_5m(ALL_TICKERS)
    daily_data = fetch_daily(ALL_TICKERS)

    # ── 2. Build unified signals ─────────────────────────────────────────────
    unified = build_unified_signals(data_5m, daily_data)

    # ── 3. Prepare 5-min signal subset with derived columns ──────────────────
    sig5 = unified[unified["signal_tf"] == "5min"].copy()
    sig5["breakout_pct"] = sig5.apply(calc_breakout_pct, axis=1)
    sig5["quality"] = sig5.apply(compute_quality, axis=1)
    sig5["time_bucket"] = sig5["time"].apply(assign_time_bucket)
    sig5 = sig5[sig5["time_bucket"] != "other"].copy()
    print(f"\nTotal 5-min signals: {len(sig5)}")

    # ── 4. Run turbo simulations ONCE on full 5-min set, tag with bucket ─────
    print("\nRunning turbo simulations (all 5-min signals)...")

    u5 = unified[unified["signal_tf"] == "5min"].copy()

    turbo_1_all = _tag_turbo_bucket(
        simulate_turbo(u5, data_5m, ko_buffer_pct=1.0,
                       signal_tf_filter=None, quiet=True))
    turbo_2_all = _tag_turbo_bucket(
        simulate_turbo(u5, data_5m, ko_buffer_pct=2.0,
                       signal_tf_filter=None, quiet=True))
    runner_all = _tag_turbo_bucket(
        simulate_turbo_runner(u5, data_5m, daily_data,
                              ko_buffer_pct=1.0, sma_period=20,
                              gradient_threshold=0.3, max_hold_days=5,
                              signal_tf_filter=None, quiet=True))

    print(f"  Turbo 1%: {len(turbo_1_all)} trades | "
          f"Turbo 2%: {len(turbo_2_all)} trades | "
          f"Runner: {len(runner_all)} trades")

    # ════════════════════════════════════════════════════════════════════════
    # PART 1: Overall performance by time bucket
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 1: OVERALL PERFORMANCE BY TIME BUCKET")
    print("=" * 80)

    part1_results = []
    for bucket in BUCKET_ORDER:
        subset = sig5[sig5["time_bucket"] == bucket]
        if subset.empty:
            continue
        part1_results.append(_build_result_row(
            bucket, "all", "all", subset,
            turbo_1_all[turbo_1_all["time_bucket"] == bucket],
            turbo_2_all[turbo_2_all["time_bucket"] == bucket],
            runner_all[runner_all["time_bucket"] == bucket],
        ))

    _print_table("Overall by Time Bucket", part1_results)
    all_results = list(part1_results)

    # ════════════════════════════════════════════════════════════════════════
    # PART 2: Quality subsets by time bucket
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 2: QUALITY SUBSETS BY TIME BUCKET")
    print("=" * 80)

    quality_results = []

    for q_min in [2, 3]:
        q_label = f"quality>={q_min}"

        # Prepare quality-filtered unified for turbo sims
        q_unified = unified.copy()
        q_unified["breakout_pct"] = q_unified.apply(calc_breakout_pct, axis=1)
        q_unified["quality"] = q_unified.apply(compute_quality, axis=1)
        q_unified = q_unified[q_unified["quality"] >= q_min]
        q_u5 = q_unified[q_unified["signal_tf"] == "5min"].copy()

        # Run turbo sims once for this quality level
        qt1 = _tag_turbo_bucket(
            simulate_turbo(q_u5, data_5m, ko_buffer_pct=1.0,
                           signal_tf_filter=None, quiet=True))
        qt2 = _tag_turbo_bucket(
            simulate_turbo(q_u5, data_5m, ko_buffer_pct=2.0,
                           signal_tf_filter=None, quiet=True))
        qr = _tag_turbo_bucket(
            simulate_turbo_runner(q_u5, data_5m, daily_data,
                                  ko_buffer_pct=1.0, sma_period=20,
                                  gradient_threshold=0.3, max_hold_days=5,
                                  signal_tf_filter=None, quiet=True))

        q_sig5 = sig5[sig5["quality"] >= q_min]

        for bucket in BUCKET_ORDER:
            subset = q_sig5[q_sig5["time_bucket"] == bucket]
            if subset.empty:
                continue
            quality_results.append(_build_result_row(
                bucket, q_label, "all", subset,
                qt1[qt1["time_bucket"] == bucket],
                qt2[qt2["time_bucket"] == bucket],
                qr[qr["time_bucket"] == bucket],
            ))

    _print_table("Quality Subsets by Time Bucket", quality_results)
    all_results.extend(quality_results)

    # ════════════════════════════════════════════════════════════════════════
    # PART 3: Long vs Short by time bucket
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PART 3: LONG vs SHORT BY TIME BUCKET")
    print("=" * 80)

    dir_results = []

    for direction in ["long", "short"]:
        d_u5 = u5[u5["direction"] == direction].copy()

        dt1 = _tag_turbo_bucket(
            simulate_turbo(d_u5, data_5m, ko_buffer_pct=1.0,
                           signal_tf_filter=None, quiet=True))
        dt2 = _tag_turbo_bucket(
            simulate_turbo(d_u5, data_5m, ko_buffer_pct=2.0,
                           signal_tf_filter=None, quiet=True))
        dr = _tag_turbo_bucket(
            simulate_turbo_runner(d_u5, data_5m, daily_data,
                                  ko_buffer_pct=1.0, sma_period=20,
                                  gradient_threshold=0.3, max_hold_days=5,
                                  signal_tf_filter=None, quiet=True))

        d_sig5 = sig5[sig5["direction"] == direction]

        for bucket in BUCKET_ORDER:
            subset = d_sig5[d_sig5["time_bucket"] == bucket]
            if subset.empty:
                continue
            dir_results.append(_build_result_row(
                bucket, "all", direction, subset,
                dt1[dt1["time_bucket"] == bucket],
                dt2[dt2["time_bucket"] == bucket],
                dr[dr["time_bucket"] == bucket],
            ))

    _print_table("Long vs Short by Time Bucket", dir_results)
    all_results.extend(dir_results)

    # ── Save all results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "time_of_day.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
