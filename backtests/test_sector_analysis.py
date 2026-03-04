"""
Sector & Ticker Performance Analysis

Breaks down ORB strategy performance by sector, category, and individual ticker.
Identifies which sectors/tickers work best for turbo trading.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import everything from the main backtest module ─────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from orb_monday_range import (
    ALL_TICKERS,
    CATEGORY_MAP,
    RESULTS_DIR,
    SECTOR_HOLDINGS,
    TICKERS_COMMODITIES,
    TICKERS_SECTORS,
    TICKERS_STOCKS,
    build_unified_signals,
    fetch_daily,
    fetch_intraday_5m,
    simulate_turbo,
    simulate_turbo_runner,
)

POSITION_SEK = 500.0


# ── Helpers ─────────────────────────────────────────────────────────────────

def _turbo_stats(turbo_df: pd.DataFrame) -> dict:
    """Compute KO%, WR%, ROI from a turbo simulation DataFrame."""
    if turbo_df.empty:
        return {"n": 0, "ko_pct": np.nan, "wr_pct": np.nan, "roi_pct": np.nan,
                "avg_pnl": np.nan, "total_pnl": np.nan}
    n = len(turbo_df)
    ko = (turbo_df["outcome"] == "ko").sum()
    winners = (turbo_df["turbo_pnl_sek"] > 0).sum()
    total_pnl = turbo_df["turbo_pnl_sek"].sum()
    total_invested = n * POSITION_SEK
    return {
        "n": n,
        "ko_pct": ko / n * 100,
        "wr_pct": winners / n * 100,
        "roi_pct": total_pnl / total_invested * 100 if total_invested > 0 else 0,
        "avg_pnl": turbo_df["turbo_pnl_sek"].mean(),
        "total_pnl": total_pnl,
    }


def _signal_stats(signals_df: pd.DataFrame) -> dict:
    """Basic signal stats: n, win_rate, avg breakout distance."""
    if signals_df.empty:
        return {"n_signals": 0, "signal_wr_pct": np.nan, "avg_breakout_dist_pct": np.nan}
    n = len(signals_df)
    wr = signals_df["win"].sum() / n * 100

    # Breakout distance = how far entry is past the ORB level, in %
    dists = []
    for _, row in signals_df.iterrows():
        if row["direction"] == "long":
            dist = (row["entry_price"] - row["orb_high"]) / row["orb_high"] * 100
        else:
            dist = (row["orb_low"] - row["entry_price"]) / row["orb_low"] * 100
        dists.append(dist)
    return {
        "n_signals": n,
        "signal_wr_pct": wr,
        "avg_breakout_dist_pct": np.mean(dists),
    }


def _direction_stats(signals_df: pd.DataFrame) -> dict:
    """Win rate by direction."""
    result = {}
    for d in ["long", "short"]:
        sub = signals_df[signals_df["direction"] == d]
        if len(sub) == 0:
            result[f"{d}_n"] = 0
            result[f"{d}_wr_pct"] = np.nan
        else:
            result[f"{d}_n"] = len(sub)
            result[f"{d}_wr_pct"] = sub["win"].sum() / len(sub) * 100
    return result


def print_table(title: str, rows: list[dict], columns: list[str],
                col_widths: dict | None = None, fmt: dict | None = None):
    """Pretty-print a table."""
    if not rows:
        print(f"\n{title}: No data.")
        return

    default_width = 10
    widths = {c: col_widths.get(c, default_width) if col_widths else default_width
              for c in columns}
    # Make name column wider
    if "name" in widths:
        widths["name"] = max(widths["name"], 22)
    if "ticker" in widths:
        widths["ticker"] = max(widths["ticker"], 8)

    fmt = fmt or {}

    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

    # Header
    header = ""
    for c in columns:
        w = widths[c]
        header += f"  {c:>{w}}" if c != columns[0] else f"  {c:<{w}}"
    print(header)
    print("  " + "-" * (sum(widths[c] for c in columns) + 2 * len(columns)))

    # Rows
    for row in rows:
        line = ""
        for i, c in enumerate(columns):
            w = widths[c]
            val = row.get(c, "")
            f = fmt.get(c, None)
            if f and val is not None and not (isinstance(val, float) and np.isnan(val)):
                formatted = f.format(val)
            elif isinstance(val, float):
                if np.isnan(val):
                    formatted = "-"
                else:
                    formatted = f"{val:.1f}"
            else:
                formatted = str(val)
            if i == 0:
                line += f"  {formatted:<{w}}"
            else:
                line += f"  {formatted:>{w}}"
        print(line)
    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Fetch data and build unified signals ────────────────────────────
    print("=" * 80)
    print("  SECTOR & TICKER PERFORMANCE ANALYSIS")
    print("=" * 80)

    data_5m = fetch_intraday_5m(ALL_TICKERS)
    daily_data = fetch_daily(ALL_TICKERS, period="2y")
    unified = build_unified_signals(data_5m, daily_data)

    # Filter to 5-min signals only
    u5 = unified[unified["signal_tf"] == "5min"].copy()
    print(f"\n  Total 5-min signals: {len(u5)}")

    # ── 2. Run turbo simulations ───────────────────────────────────────────
    print("\n  Running turbo simulations...")

    turbo_1pct = simulate_turbo(
        unified, data_5m, ko_buffer_pct=1.0, position_sek=POSITION_SEK,
        signal_tf_filter="5min", quiet=True)

    turbo_2pct = simulate_turbo(
        unified, data_5m, ko_buffer_pct=2.0, position_sek=POSITION_SEK,
        signal_tf_filter="5min", quiet=True)

    turbo_runner = simulate_turbo_runner(
        unified, data_5m, daily_data, ko_buffer_pct=1.0,
        position_sek=POSITION_SEK, sma_period=20, gradient_threshold=0.3,
        max_hold_days=5, signal_tf_filter="5min", quiet=True)

    print(f"  Turbo 1% trades: {len(turbo_1pct)}")
    print(f"  Turbo 2% trades: {len(turbo_2pct)}")
    print(f"  Turbo runner trades: {len(turbo_runner)}")

    # Count total trading days available per ticker (from 5m data)
    from orb_monday_range import _to_ny, _market_hours
    trading_days_per_ticker = {}
    for ticker, df_raw in data_5m.items():
        df_mh = _market_hours(_to_ny(df_raw))
        trading_days_per_ticker[ticker] = len(set(df_mh.index.date))

    # ── Build reverse map: ticker -> parent sector ETF ─────────────────────
    ticker_to_sector = {}
    for sector_etf, holdings in SECTOR_HOLDINGS.items():
        for h in holdings:
            ticker_to_sector[h] = sector_etf

    # ====================================================================
    # ANALYSIS 1: By Category
    # ====================================================================
    print("\n\n" + "#" * 80)
    print("  ANALYSIS 1: Performance by Category")
    print("#" * 80)

    categories = ["Sector ETF", "Stock", "Commodity"]
    # Also add per-sector-holding groups
    for sector_etf in TICKERS_SECTORS:
        if sector_etf in SECTOR_HOLDINGS:
            categories.append(f"{sector_etf} holding")

    cat_rows = []
    for cat in categories:
        u_cat = u5[u5["category"] == cat]
        t1_cat = turbo_1pct[turbo_1pct["category"] == cat] if not turbo_1pct.empty else pd.DataFrame()
        t2_cat = turbo_2pct[turbo_2pct["category"] == cat] if not turbo_2pct.empty else pd.DataFrame()
        tr_cat = turbo_runner[turbo_runner["category"] == cat] if not turbo_runner.empty else pd.DataFrame()

        sig = _signal_stats(u_cat)
        t1 = _turbo_stats(t1_cat)
        t2 = _turbo_stats(t2_cat)
        tr = _turbo_stats(tr_cat)

        cat_rows.append({
            "name": cat,
            "n_signals": sig["n_signals"],
            "signal_wr": sig["signal_wr_pct"],
            "avg_bo_dist": sig["avg_breakout_dist_pct"],
            "t1_n": t1["n"], "t1_ko": t1["ko_pct"], "t1_wr": t1["wr_pct"], "t1_roi": t1["roi_pct"],
            "tr_n": tr["n"], "tr_ko": tr["ko_pct"], "tr_wr": tr["wr_pct"], "tr_roi": tr["roi_pct"],
            "t2_n": t2["n"], "t2_ko": t2["ko_pct"], "t2_wr": t2["wr_pct"], "t2_roi": t2["roi_pct"],
        })

    cols = ["name", "n_signals", "signal_wr", "avg_bo_dist",
            "t1_n", "t1_ko", "t1_wr", "t1_roi",
            "tr_n", "tr_ko", "tr_wr", "tr_roi",
            "t2_n", "t2_ko", "t2_wr", "t2_roi"]
    fmts = {
        "n_signals": "{:d}", "signal_wr": "{:.1f}%", "avg_bo_dist": "{:.3f}%",
        "t1_n": "{:d}", "t1_ko": "{:.1f}%", "t1_wr": "{:.1f}%", "t1_roi": "{:+.1f}%",
        "tr_n": "{:d}", "tr_ko": "{:.1f}%", "tr_wr": "{:.1f}%", "tr_roi": "{:+.1f}%",
        "t2_n": "{:d}", "t2_ko": "{:.1f}%", "t2_wr": "{:.1f}%", "t2_roi": "{:+.1f}%",
    }
    widths = {c: 10 for c in cols}
    widths["name"] = 22

    print_table(
        "Category Breakdown  |  Turbo 1% same-day  |  Runner SMA(20) grad>0.3%  |  Turbo 2% same-day",
        cat_rows, cols, widths, fmts)

    # ====================================================================
    # ANALYSIS 2: By Individual Ticker (Top 20 / Bottom 20)
    # ====================================================================
    print("\n\n" + "#" * 80)
    print("  ANALYSIS 2: Performance by Individual Ticker")
    print("#" * 80)

    ticker_rows = []
    for ticker in sorted(u5["ticker"].unique()):
        u_tick = u5[u5["ticker"] == ticker]
        t1_tick = turbo_1pct[turbo_1pct["ticker"] == ticker] if not turbo_1pct.empty else pd.DataFrame()
        t2_tick = turbo_2pct[turbo_2pct["ticker"] == ticker] if not turbo_2pct.empty else pd.DataFrame()
        tr_tick = turbo_runner[turbo_runner["ticker"] == ticker] if not turbo_runner.empty else pd.DataFrame()

        sig = _signal_stats(u_tick)
        t1 = _turbo_stats(t1_tick)
        t2 = _turbo_stats(t2_tick)
        tr = _turbo_stats(tr_tick)

        ticker_rows.append({
            "ticker": ticker,
            "category": CATEGORY_MAP.get(ticker, "?"),
            "n_signals": sig["n_signals"],
            "signal_wr": sig["signal_wr_pct"],
            "avg_bo_dist": sig["avg_breakout_dist_pct"],
            "t1_n": t1["n"], "t1_ko": t1["ko_pct"], "t1_wr": t1["wr_pct"], "t1_roi": t1["roi_pct"],
            "tr_n": tr["n"], "tr_ko": tr["ko_pct"], "tr_wr": tr["wr_pct"], "tr_roi": tr["roi_pct"],
            "t2_n": t2["n"], "t2_ko": t2["ko_pct"], "t2_wr": t2["wr_pct"], "t2_roi": t2["roi_pct"],
            "t1_total_pnl": t1["total_pnl"],
        })

    # Sort by turbo 1% ROI descending
    ticker_rows.sort(key=lambda r: r.get("t1_roi", float("-inf"))
                     if not (isinstance(r.get("t1_roi"), float) and np.isnan(r.get("t1_roi", 0)))
                     else float("-inf"), reverse=True)

    tcols = ["ticker", "category", "n_signals", "signal_wr", "avg_bo_dist",
             "t1_n", "t1_ko", "t1_wr", "t1_roi",
             "tr_n", "tr_ko", "tr_wr", "tr_roi",
             "t2_n", "t2_ko", "t2_wr", "t2_roi"]
    tfmts = {
        "n_signals": "{:d}", "signal_wr": "{:.1f}%", "avg_bo_dist": "{:.3f}%",
        "t1_n": "{:d}", "t1_ko": "{:.1f}%", "t1_wr": "{:.1f}%", "t1_roi": "{:+.1f}%",
        "tr_n": "{:d}", "tr_ko": "{:.1f}%", "tr_wr": "{:.1f}%", "tr_roi": "{:+.1f}%",
        "t2_n": "{:d}", "t2_ko": "{:.1f}%", "t2_wr": "{:.1f}%", "t2_roi": "{:+.1f}%",
    }
    twidths = {c: 10 for c in tcols}
    twidths["ticker"] = 8
    twidths["category"] = 16

    top20 = ticker_rows[:20]
    bottom20 = ticker_rows[-20:] if len(ticker_rows) > 20 else []

    print_table(
        "TOP 20 TICKERS by Turbo 1% ROI",
        top20, tcols, twidths, tfmts)

    if bottom20:
        print_table(
            "BOTTOM 20 TICKERS by Turbo 1% ROI",
            bottom20, tcols, twidths, tfmts)

    # ====================================================================
    # ANALYSIS 3: Sector ETF vs Its Holdings
    # ====================================================================
    print("\n\n" + "#" * 80)
    print("  ANALYSIS 3: Sector ETF vs Top Holdings")
    print("#" * 80)

    for sector_etf in TICKERS_SECTORS:
        if sector_etf not in SECTOR_HOLDINGS:
            continue

        holdings = SECTOR_HOLDINGS[sector_etf]
        print(f"\n  --- {sector_etf} vs {', '.join(holdings)} ---")

        cmp_rows = []

        # The ETF itself
        u_etf = u5[u5["ticker"] == sector_etf]
        t1_etf = turbo_1pct[turbo_1pct["ticker"] == sector_etf] if not turbo_1pct.empty else pd.DataFrame()
        tr_etf = turbo_runner[turbo_runner["ticker"] == sector_etf] if not turbo_runner.empty else pd.DataFrame()
        sig_etf = _signal_stats(u_etf)
        t1_e = _turbo_stats(t1_etf)
        tr_e = _turbo_stats(tr_etf)
        cmp_rows.append({
            "ticker": f"{sector_etf} (ETF)",
            "n_signals": sig_etf["n_signals"],
            "signal_wr": sig_etf["signal_wr_pct"],
            "t1_ko": t1_e["ko_pct"], "t1_wr": t1_e["wr_pct"], "t1_roi": t1_e["roi_pct"],
            "tr_ko": tr_e["ko_pct"], "tr_wr": tr_e["wr_pct"], "tr_roi": tr_e["roi_pct"],
        })

        # Each holding
        for h in holdings:
            u_h = u5[u5["ticker"] == h]
            t1_h = turbo_1pct[turbo_1pct["ticker"] == h] if not turbo_1pct.empty else pd.DataFrame()
            tr_h = turbo_runner[turbo_runner["ticker"] == h] if not turbo_runner.empty else pd.DataFrame()
            sig_h = _signal_stats(u_h)
            t1_hh = _turbo_stats(t1_h)
            tr_hh = _turbo_stats(tr_h)
            cmp_rows.append({
                "ticker": h,
                "n_signals": sig_h["n_signals"],
                "signal_wr": sig_h["signal_wr_pct"],
                "t1_ko": t1_hh["ko_pct"], "t1_wr": t1_hh["wr_pct"], "t1_roi": t1_hh["roi_pct"],
                "tr_ko": tr_hh["ko_pct"], "tr_wr": tr_hh["wr_pct"], "tr_roi": tr_hh["roi_pct"],
            })

        # Combined holdings
        all_h = u5[u5["ticker"].isin(holdings)]
        t1_all = turbo_1pct[turbo_1pct["ticker"].isin(holdings)] if not turbo_1pct.empty else pd.DataFrame()
        tr_all = turbo_runner[turbo_runner["ticker"].isin(holdings)] if not turbo_runner.empty else pd.DataFrame()
        sig_all = _signal_stats(all_h)
        t1_a = _turbo_stats(t1_all)
        tr_a = _turbo_stats(tr_all)
        cmp_rows.append({
            "ticker": "-- ALL HOLDINGS --",
            "n_signals": sig_all["n_signals"],
            "signal_wr": sig_all["signal_wr_pct"],
            "t1_ko": t1_a["ko_pct"], "t1_wr": t1_a["wr_pct"], "t1_roi": t1_a["roi_pct"],
            "tr_ko": tr_a["ko_pct"], "tr_wr": tr_a["wr_pct"], "tr_roi": tr_a["roi_pct"],
        })

        ccols = ["ticker", "n_signals", "signal_wr",
                 "t1_ko", "t1_wr", "t1_roi", "tr_ko", "tr_wr", "tr_roi"]
        cfmts = {
            "n_signals": "{:d}", "signal_wr": "{:.1f}%",
            "t1_ko": "{:.1f}%", "t1_wr": "{:.1f}%", "t1_roi": "{:+.1f}%",
            "tr_ko": "{:.1f}%", "tr_wr": "{:.1f}%", "tr_roi": "{:+.1f}%",
        }
        cwidths = {c: 10 for c in ccols}
        cwidths["ticker"] = 20

        print_table(
            f"{sector_etf}  |  Turbo 1% same-day  |  Runner SMA(20)",
            cmp_rows, ccols, cwidths, cfmts)

    # ====================================================================
    # ANALYSIS 4: Direction Bias by Sector
    # ====================================================================
    print("\n\n" + "#" * 80)
    print("  ANALYSIS 4: Direction Bias by Sector/Category")
    print("#" * 80)

    dir_rows = []
    # By main categories
    for cat in ["Sector ETF", "Stock", "Commodity"]:
        u_cat = u5[u5["category"] == cat]
        ds = _direction_stats(u_cat)

        # Also get turbo direction stats
        t1_cat = turbo_1pct[turbo_1pct["category"] == cat] if not turbo_1pct.empty else pd.DataFrame()
        long_t1 = t1_cat[t1_cat["direction"] == "long"] if not t1_cat.empty else pd.DataFrame()
        short_t1 = t1_cat[t1_cat["direction"] == "short"] if not t1_cat.empty else pd.DataFrame()
        t1_long = _turbo_stats(long_t1)
        t1_short = _turbo_stats(short_t1)

        dir_rows.append({
            "name": cat,
            "long_n": ds["long_n"], "long_wr": ds["long_wr_pct"],
            "short_n": ds["short_n"], "short_wr": ds["short_wr_pct"],
            "t1_long_wr": t1_long["wr_pct"], "t1_long_roi": t1_long["roi_pct"],
            "t1_short_wr": t1_short["wr_pct"], "t1_short_roi": t1_short["roi_pct"],
        })

    # By individual sector ETFs
    for sector_etf in TICKERS_SECTORS:
        u_sec = u5[u5["ticker"] == sector_etf]
        ds = _direction_stats(u_sec)
        t1_sec = turbo_1pct[turbo_1pct["ticker"] == sector_etf] if not turbo_1pct.empty else pd.DataFrame()
        long_t1 = t1_sec[t1_sec["direction"] == "long"] if not t1_sec.empty else pd.DataFrame()
        short_t1 = t1_sec[t1_sec["direction"] == "short"] if not t1_sec.empty else pd.DataFrame()
        t1_long = _turbo_stats(long_t1)
        t1_short = _turbo_stats(short_t1)

        dir_rows.append({
            "name": f"  {sector_etf}",
            "long_n": ds["long_n"], "long_wr": ds["long_wr_pct"],
            "short_n": ds["short_n"], "short_wr": ds["short_wr_pct"],
            "t1_long_wr": t1_long["wr_pct"], "t1_long_roi": t1_long["roi_pct"],
            "t1_short_wr": t1_short["wr_pct"], "t1_short_roi": t1_short["roi_pct"],
        })

    dcols = ["name", "long_n", "long_wr", "short_n", "short_wr",
             "t1_long_wr", "t1_long_roi", "t1_short_wr", "t1_short_roi"]
    dfmts = {
        "long_n": "{:d}", "long_wr": "{:.1f}%",
        "short_n": "{:d}", "short_wr": "{:.1f}%",
        "t1_long_wr": "{:.1f}%", "t1_long_roi": "{:+.1f}%",
        "t1_short_wr": "{:.1f}%", "t1_short_roi": "{:+.1f}%",
    }
    dwidths = {c: 12 for c in dcols}
    dwidths["name"] = 22

    print_table(
        "Direction Bias  |  Signal WR  |  Turbo 1% WR & ROI by direction",
        dir_rows, dcols, dwidths, dfmts)

    # ====================================================================
    # ANALYSIS 5: Consistency Check (Signal Frequency)
    # ====================================================================
    print("\n\n" + "#" * 80)
    print("  ANALYSIS 5: Signal Frequency & Consistency")
    print("#" * 80)

    freq_rows = []
    for ticker in sorted(u5["ticker"].unique()):
        u_tick = u5[u5["ticker"] == ticker]
        total_days = trading_days_per_ticker.get(ticker, 0)
        signal_days = u_tick["date"].nunique()
        freq_pct = signal_days / total_days * 100 if total_days > 0 else 0

        # Get turbo 1% ROI for context
        t1_tick = turbo_1pct[turbo_1pct["ticker"] == ticker] if not turbo_1pct.empty else pd.DataFrame()
        t1 = _turbo_stats(t1_tick)

        freq_rows.append({
            "ticker": ticker,
            "category": CATEGORY_MAP.get(ticker, "?"),
            "total_days": total_days,
            "signal_days": signal_days,
            "signal_freq": freq_pct,
            "signals_per_wk": signal_days / max(total_days / 5, 1),
            "t1_roi": t1["roi_pct"],
            "t1_wr": t1["wr_pct"],
        })

    # Sort by signal frequency descending
    freq_rows.sort(key=lambda r: r["signal_freq"], reverse=True)

    fcols = ["ticker", "category", "total_days", "signal_days", "signal_freq",
             "signals_per_wk", "t1_wr", "t1_roi"]
    ffmts = {
        "total_days": "{:d}", "signal_days": "{:d}", "signal_freq": "{:.1f}%",
        "signals_per_wk": "{:.2f}", "t1_wr": "{:.1f}%", "t1_roi": "{:+.1f}%",
    }
    fwidths = {c: 12 for c in fcols}
    fwidths["ticker"] = 8
    fwidths["category"] = 16

    print_table(
        "Signal Frequency  |  How often does each ticker produce a 5-min ORB signal?",
        freq_rows, fcols, fwidths, ffmts)

    # Highlight top profitable + frequent tickers
    print("\n  --- PRACTICAL PICKS: tickers with both decent frequency AND positive ROI ---")
    practical = [r for r in freq_rows
                 if r["signal_freq"] > 15
                 and isinstance(r["t1_roi"], (int, float))
                 and not np.isnan(r["t1_roi"])
                 and r["t1_roi"] > 0]
    practical.sort(key=lambda r: r["t1_roi"], reverse=True)
    if practical:
        print_table(
            "Frequent (>15% of days) + Profitable (ROI>0) Tickers",
            practical, fcols, fwidths, ffmts)
    else:
        print("  No tickers meet both criteria (>15% frequency AND positive ROI).")
        # Relax to 10%
        practical2 = [r for r in freq_rows
                      if r["signal_freq"] > 10
                      and isinstance(r["t1_roi"], (int, float))
                      and not np.isnan(r["t1_roi"])
                      and r["t1_roi"] > 0]
        practical2.sort(key=lambda r: r["t1_roi"], reverse=True)
        if practical2:
            print_table(
                "Frequent (>10% of days) + Profitable (ROI>0) Tickers",
                practical2, fcols, fwidths, ffmts)

    # ====================================================================
    # Save CSVs
    # ====================================================================
    print("\n\n" + "#" * 80)
    print("  SAVING RESULTS")
    print("#" * 80)

    # sector_analysis.csv — the category breakdown
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(RESULTS_DIR / "sector_analysis.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'sector_analysis.csv'}")

    # ticker_ranking.csv — all tickers ranked by turbo 1% ROI
    ticker_df = pd.DataFrame(ticker_rows)
    ticker_df.to_csv(RESULTS_DIR / "ticker_ranking.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'ticker_ranking.csv'}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
