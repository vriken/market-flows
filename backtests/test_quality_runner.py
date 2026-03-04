"""
Quality Score Runner Backtest

Tests whether filtering runners by QUALITY SCORE improves performance.
Instead of holding overnight on ALL trend-aligned signals with strong gradient,
only hold on quality >= N signals for fewer but higher-conviction runners.

Quality factors (0-4 score):
  T  trend_aligned     - price direction matches SMA trend
  M  inside_monday     - gap_type == "inside_monday" AND not a Monday
  B  breakout_momentum - entry breakout >= 1% beyond ORB level
  V  vol_spike         - volume >= 2x rolling average

Compares: all trend-aligned (baseline), quality>=2, quality>=3, quality==4
across turbo buffers 1.0% and 2.0%, with gradient threshold 0.3%, max 5 days.
Also tests same-day-exit-only (no runner) at each quality level.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent dir so we can import from backtests
sys.path.insert(0, str(Path(__file__).resolve().parent))

from orb_monday_range import (
    ALL_TICKERS,
    RESULTS_DIR,
    fetch_daily,
    fetch_intraday_1m,
    fetch_intraday_5m,
    build_unified_signals,
    simulate_turbo_runner,
    compute_sma_gradient,
)


# ── Quality Score Computation ───────────────────────────────────────────────

def add_quality_score(unified: pd.DataFrame) -> pd.DataFrame:
    """Add quality score (0-4) to unified signals DataFrame.

    Factors:
      T (trend_aligned):      already in unified
      M (inside_monday):      gap_type == "inside_monday" AND weekday != Monday
      B (breakout_momentum):  entry >= 1% beyond ORB level in trade direction
      V (vol_spike):          already in unified
    """
    df = unified.copy()

    # T: trend_aligned (boolean)
    t_score = df["trend_aligned"].astype(int)

    # M: inside_monday AND not a Monday
    dates = pd.to_datetime(df["date"])
    is_monday = dates.dt.weekday == 0  # Monday = 0
    is_inside_monday = df["gap_type"] == "inside_monday"
    m_score = (is_inside_monday & ~is_monday).astype(int)

    # B: breakout momentum >= 1%
    b_score = pd.Series(0, index=df.index)
    long_mask = df["direction"] == "long"
    short_mask = df["direction"] == "short"

    # Long: (entry_price - orb_high) / entry_price >= 0.01
    long_momentum = (df["entry_price"] - df["orb_high"]) / df["entry_price"]
    b_score.loc[long_mask] = (long_momentum[long_mask] >= 0.01).astype(int)

    # Short: (orb_low - entry_price) / entry_price >= 0.01
    short_momentum = (df["orb_low"] - df["entry_price"]) / df["entry_price"]
    b_score.loc[short_mask] = (short_momentum[short_mask] >= 0.01).astype(int)

    # V: vol_spike (boolean)
    v_score = df["vol_spike"].astype(int)

    df["q_trend"] = t_score
    df["q_monday"] = m_score
    df["q_breakout"] = b_score
    df["q_volspike"] = v_score
    df["quality"] = t_score + m_score + b_score + v_score

    return df


# ── Same-Day Exit Simulation (no runner) ────────────────────────────────────

def simulate_same_day_exit(unified: pd.DataFrame, data_5m: dict,
                           daily_data: dict,
                           ko_buffer_pct: float = 1.0,
                           position_sek: float = 500.0,
                           signal_tf_filter: str = "5min",
                           data_1m: dict | None = None) -> pd.DataFrame:
    """Like turbo runner but forces exit at EOD (no gradient check, no overnight).

    Uses simulate_turbo_runner with gradient_threshold=999 so the gradient
    check never passes and every trade exits at the end of signal day.
    """
    return simulate_turbo_runner(
        unified, data_5m, daily_data,
        ko_buffer_pct=ko_buffer_pct,
        position_sek=position_sek,
        sma_period=20,
        gradient_lookback=5,
        gradient_threshold=999.0,  # impossibly high -> never holds overnight
        max_hold_days=0,           # force same-day exit
        signal_tf_filter=signal_tf_filter,
        quiet=True,
        data_1m=data_1m,
    )


# ── Summary Metrics ─────────────────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame, position_sek: float = 500.0) -> dict:
    """Compute summary metrics for a set of turbo runner results."""
    if results.empty:
        return {
            "n_trades": 0, "ko_pct": 0, "wr_pct": 0, "roi_pct": 0,
            "held_overnight_pct": 0, "avg_days": 0, "total_pnl": 0,
            "avg_pnl": 0,
        }
    n = len(results)
    n_ko = (results["outcome"] == "ko").sum()
    winners = (results["turbo_pnl_sek"] > 0).sum()
    total_pnl = results["turbo_pnl_sek"].sum()
    total_invested = n * position_sek
    avg_pnl = results["turbo_pnl_sek"].mean()
    avg_days = results["days_held"].mean()
    held_overnight = (results["days_held"] > 1).sum()

    return {
        "n_trades": n,
        "ko_pct": n_ko / n * 100 if n else 0,
        "wr_pct": winners / n * 100 if n else 0,
        "roi_pct": total_pnl / total_invested * 100 if total_invested else 0,
        "held_overnight_pct": held_overnight / n * 100 if n else 0,
        "avg_days": avg_days,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  QUALITY SCORE RUNNER BACKTEST")
    print("  Test: does filtering by quality score improve runner performance?")
    print("=" * 90)

    # ── Fetch data ──────────────────────────────────────────────────────────
    data_5m = fetch_intraday_5m(ALL_TICKERS)
    daily_data = fetch_daily(ALL_TICKERS)
    data_1m = fetch_intraday_1m(ALL_TICKERS)

    # ── Build unified signals ───────────────────────────────────────────────
    unified = build_unified_signals(data_5m, daily_data, data_1m=data_1m)
    if unified.empty:
        print("ERROR: No signals generated. Exiting.")
        return

    # ── Add quality scores ──────────────────────────────────────────────────
    unified = add_quality_score(unified)

    print("\n── Quality Score Distribution ──")
    q_dist = unified["quality"].value_counts().sort_index()
    for q_val, count in q_dist.items():
        pct = count / len(unified) * 100
        factors = []
        sub = unified[unified["quality"] == q_val]
        if len(sub) > 0:
            factors.append(f"T={sub['q_trend'].mean():.0%}")
            factors.append(f"M={sub['q_monday'].mean():.0%}")
            factors.append(f"B={sub['q_breakout'].mean():.0%}")
            factors.append(f"V={sub['q_volspike'].mean():.0%}")
        print(f"  Quality {q_val}: {count:5d} signals ({pct:5.1f}%)  [{', '.join(factors)}]")

    # Show trend-aligned subset quality distribution
    ta = unified[unified["trend_aligned"]]
    print(f"\n  Trend-aligned signals: {len(ta)} / {len(unified)} ({len(ta)/len(unified)*100:.1f}%)")
    ta_dist = ta["quality"].value_counts().sort_index()
    for q_val, count in ta_dist.items():
        print(f"    Quality {q_val}: {count:5d} ({count/len(ta)*100:.1f}%)")

    # ── Test configurations ─────────────────────────────────────────────────
    buffers = [1.0, 2.0]
    gradient_threshold = 0.3
    max_hold_days = 5
    position_sek = 500.0

    quality_filters = {
        "All trend-aligned": lambda df: df[df["trend_aligned"]],
        "Quality >= 2": lambda df: df[df["trend_aligned"] & (df["quality"] >= 2)],
        "Quality >= 3": lambda df: df[df["trend_aligned"] & (df["quality"] >= 3)],
        "Quality == 4": lambda df: df[df["trend_aligned"] & (df["quality"] == 4)],
    }

    all_results = []

    for buffer in buffers:
        print(f"\n{'=' * 90}")
        print(f"  TURBO BUFFER: {buffer}% (~{round(100/buffer)}x leverage)")
        print(f"  Gradient threshold: {gradient_threshold}% | Max hold: {max_hold_days} days")
        print(f"{'=' * 90}")

        for q_label, q_filter in quality_filters.items():
            filtered = q_filter(unified)
            n_filtered = len(filtered)

            if n_filtered == 0:
                print(f"\n  {q_label}: 0 signals — skipping")
                all_results.append({
                    "filter": q_label, "mode": "runner",
                    "buffer_pct": buffer, **compute_metrics(pd.DataFrame()),
                })
                all_results.append({
                    "filter": q_label, "mode": "same_day",
                    "buffer_pct": buffer, **compute_metrics(pd.DataFrame()),
                })
                continue

            # Runner (hold overnight when gradient is favorable)
            runner = simulate_turbo_runner(
                filtered, data_5m, daily_data,
                ko_buffer_pct=buffer,
                position_sek=position_sek,
                sma_period=20,
                gradient_lookback=5,
                gradient_threshold=gradient_threshold,
                max_hold_days=max_hold_days,
                signal_tf_filter="5min",
                quiet=True,
                data_1m=data_1m,
            )

            runner_metrics = compute_metrics(runner, position_sek)
            all_results.append({
                "filter": q_label, "mode": "runner",
                "buffer_pct": buffer, **runner_metrics,
            })

            # Same-day exit (no runner, for comparison)
            same_day = simulate_same_day_exit(
                filtered, data_5m, daily_data,
                ko_buffer_pct=buffer,
                position_sek=position_sek,
                signal_tf_filter="5min",
                data_1m=data_1m,
            )

            sd_metrics = compute_metrics(same_day, position_sek)
            all_results.append({
                "filter": q_label, "mode": "same_day",
                "buffer_pct": buffer, **sd_metrics,
            })

        # ── Print comparison table for this buffer ──────────────────────────
        print(f"\n  ┌─{'─' * 88}─┐")
        print(f"  │  {'Buffer ' + str(buffer) + '%':88s} │")
        print(f"  ├─{'─' * 15}─┬─{'─' * 9}─┬─{'─' * 7}─┬─{'─' * 7}─┬─{'─' * 7}─┬─{'─' * 8}─┬─{'─' * 9}─┬─{'─' * 10}─┤")
        print(f"  │ {'Filter':15s} │ {'Mode':9s} │ {'Trades':>7s} │ {'KO%':>7s} │ {'WR%':>7s} │ {'ROI%':>8s} │ {'Ovrnght%':>9s} │ {'Avg Days':>10s} │")
        print(f"  ├─{'─' * 15}─┼─{'─' * 9}─┼─{'─' * 7}─┼─{'─' * 7}─┼─{'─' * 7}─┼─{'─' * 8}─┼─{'─' * 9}─┼─{'─' * 10}─┤")

        for r in all_results:
            if r["buffer_pct"] != buffer:
                continue
            print(f"  │ {r['filter']:15s} │ {r['mode']:9s} │ "
                  f"{r['n_trades']:7d} │ {r['ko_pct']:6.1f}% │ {r['wr_pct']:6.1f}% │ "
                  f"{r['roi_pct']:+7.1f}% │ {r['held_overnight_pct']:8.1f}% │ "
                  f"{r['avg_days']:10.2f} │")

        print(f"  └─{'─' * 15}─┴─{'─' * 9}─┴─{'─' * 7}─┴─{'─' * 7}─┴─{'─' * 7}─┴─{'─' * 8}─┴─{'─' * 9}─┴─{'─' * 10}─┘")

    # ── Combined comparison ─────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 90)
    print("  COMBINED RESULTS: Runner vs Same-Day by Quality Filter")
    print("=" * 90)

    # Pivot: for each filter+buffer, show runner ROI vs same-day ROI
    for buffer in buffers:
        print(f"\n  Buffer {buffer}% (~{round(100/buffer)}x):")
        print(f"  {'Filter':20s} │ {'Runner ROI':>10s} │ {'Same-Day ROI':>12s} │ {'Delta':>8s} │ {'Runner Trades':>13s} │ {'Runner WR%':>10s}")
        print(f"  {'─' * 20}─┼─{'─' * 10}─┼─{'─' * 12}─┼─{'─' * 8}─┼─{'─' * 13}─┼─{'─' * 10}")

        buf_data = results_df[results_df["buffer_pct"] == buffer]
        for q_label in quality_filters:
            runner_row = buf_data[(buf_data["filter"] == q_label) & (buf_data["mode"] == "runner")]
            sd_row = buf_data[(buf_data["filter"] == q_label) & (buf_data["mode"] == "same_day")]
            if runner_row.empty or sd_row.empty:
                continue
            r = runner_row.iloc[0]
            s = sd_row.iloc[0]
            delta = r["roi_pct"] - s["roi_pct"]
            print(f"  {q_label:20s} │ {r['roi_pct']:+9.1f}% │ {s['roi_pct']:+11.1f}% │ "
                  f"{delta:+7.1f}% │ {r['n_trades']:13d} │ {r['wr_pct']:9.1f}%")

    # ── Quality factor breakdown (which factors matter most?) ───────────────
    print("\n" + "=" * 90)
    print("  QUALITY FACTOR BREAKDOWN (trend-aligned, 5min, runner @ 1.0% buffer)")
    print("=" * 90)

    ta_5min = unified[(unified["trend_aligned"]) & (unified["signal_tf"] == "5min")]
    if len(ta_5min) > 0:
        factor_names = [
            ("q_trend", "T (Trend-aligned)"),
            ("q_monday", "M (Inside Monday)"),
            ("q_breakout", "B (Breakout >=1%)"),
            ("q_volspike", "V (Vol Spike)"),
        ]

        print(f"\n  {'Factor':25s} │ {'Signals':>8s} │ {'% of TA':>8s}")
        print(f"  {'─' * 25}─┼─{'─' * 8}─┼─{'─' * 8}")
        for col, label in factor_names:
            n_with = ta_5min[col].sum()
            print(f"  {label:25s} │ {int(n_with):8d} │ {n_with/len(ta_5min)*100:7.1f}%")

        # Factor combination analysis
        print(f"\n  {'Combination':35s} │ {'Count':>6s} │ {'% of TA':>8s}")
        print(f"  {'─' * 35}─┼─{'─' * 6}─┼─{'─' * 8}")

        combos = [
            ("T only", ta_5min["q_trend"] & ~ta_5min["q_monday"].astype(bool) & ~ta_5min["q_breakout"].astype(bool) & ~ta_5min["q_volspike"].astype(bool)),
            ("T + M", ta_5min["q_trend"].astype(bool) & ta_5min["q_monday"].astype(bool)),
            ("T + B", ta_5min["q_trend"].astype(bool) & ta_5min["q_breakout"].astype(bool)),
            ("T + V", ta_5min["q_trend"].astype(bool) & ta_5min["q_volspike"].astype(bool)),
            ("T + M + B", ta_5min["q_trend"].astype(bool) & ta_5min["q_monday"].astype(bool) & ta_5min["q_breakout"].astype(bool)),
            ("T + M + V", ta_5min["q_trend"].astype(bool) & ta_5min["q_monday"].astype(bool) & ta_5min["q_volspike"].astype(bool)),
            ("T + B + V", ta_5min["q_trend"].astype(bool) & ta_5min["q_breakout"].astype(bool) & ta_5min["q_volspike"].astype(bool)),
            ("T + M + B + V (quality=4)", ta_5min["quality"] == 4),
        ]
        for label, mask in combos:
            n = mask.sum()
            if n > 0:
                print(f"  {label:35s} │ {n:6d} │ {n/len(ta_5min)*100:7.1f}%")

    # ── Save results ────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "quality_runner.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")

    print("\n" + "=" * 90)
    print("  DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
