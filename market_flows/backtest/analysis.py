"""Trade analysis tools for backtest results.

Provides diagnostic functions to identify what's working and what isn't
across strategies, regimes, time periods, and trade characteristics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def exit_reason_decomposition(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Break down exit reasons by strategy.

    Returns DataFrame with columns: strategy, outcome, count, pct, avg_pnl
    Shows what % of trades exit by KO vs stop vs target vs close vs max_hold.
    """
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for strategy, grp in trades_df.groupby("strategy"):
        total = len(grp)
        for outcome, sub in grp.groupby("outcome"):
            rows.append({
                "strategy": strategy,
                "outcome": outcome,
                "count": len(sub),
                "pct": round(100 * len(sub) / total, 1),
                "avg_pnl": round(sub["pnl"].mean(), 2),
            })
    return pd.DataFrame(rows)


def pnl_by_stop_width(trades_df: pd.DataFrame, ko_buffer: float = 0.02, buckets: int = 5) -> pd.DataFrame:
    """Analyze PnL by stop distance bucket.

    Computes stop_width_pct = abs(entry_price - stop_price) / entry_price
    and groups into quantile buckets.
    """
    if trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    df["stop_width_pct"] = (df["entry_price"] - df["stop_price"]).abs() / df["entry_price"]
    df["stop_bucket"] = pd.qcut(df["stop_width_pct"], q=buckets, duplicates="drop")

    rows = []
    for bucket, grp in df.groupby("stop_bucket", observed=True):
        winners = grp[grp["pnl"] > 0]
        rows.append({
            "stop_range": str(bucket),
            "count": len(grp),
            "avg_pnl": round(grp["pnl"].mean(), 2),
            "win_rate": round(100 * len(winners) / len(grp), 1) if len(grp) > 0 else 0,
            "total_pnl": round(grp["pnl"].sum(), 2),
            "avg_stop_width": round(grp["stop_width_pct"].mean() * 100, 2),
            "ko_pct": round(100 * len(grp[grp["outcome"] == "ko"]) / len(grp), 1),
        })
    return pd.DataFrame(rows)


def mfe_mae_analysis(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Approximate MFE/MAE from entry, exit, stop, and target prices.

    MFE (Max Favorable Excursion) approximated as:
    - For winners hitting target: target - entry (longs)
    - For losers: small favorable move before reversal (estimated)

    MAE (Max Adverse Excursion) approximated as:
    - For KO trades: full KO distance
    - For stopped trades: stop distance

    Returns per-strategy summary with avg MFE, avg MAE, efficiency ratio.
    """
    if trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()

    # Approximate MFE and MAE in percentage terms
    is_long = df["direction"] == "long"

    # MFE: favorable price movement (what we captured or could have captured)
    df["realized_r"] = np.where(
        is_long,
        (df["exit_price"] - df["entry_price"]) / df["entry_price"],
        (df["entry_price"] - df["exit_price"]) / df["entry_price"],
    )

    # For target exits, MFE = realized (we hit our target)
    # For other exits, MFE >= realized (price may have moved further before exit)
    df["mfe_approx_pct"] = np.where(
        df["pnl"] > 0,
        df["realized_r"] * 100,  # at least what we realized
        df["realized_r"].clip(lower=0) * 100,  # losers may have had brief favorable move
    )

    # MAE: adverse movement experienced
    df["stop_dist_pct"] = (df["entry_price"] - df["stop_price"]).abs() / df["entry_price"] * 100
    df["mae_approx_pct"] = np.where(
        df["outcome"] == "ko",
        df["stop_dist_pct"].clip(upper=10),  # KO = worst case
        np.where(
            df["pnl"] < 0,
            abs(df["realized_r"]) * 100,  # loss = at least what we lost
            0,  # winners had no adverse close
        ),
    )

    rows = []
    for strategy, grp in df.groupby("strategy"):
        winners = grp[grp["pnl"] > 0]
        losers = grp[grp["pnl"] <= 0]
        rows.append({
            "strategy": strategy,
            "trades": len(grp),
            "avg_winner_pnl": round(winners["pnl"].mean(), 2) if len(winners) > 0 else 0,
            "avg_loser_pnl": round(losers["pnl"].mean(), 2) if len(losers) > 0 else 0,
            "avg_mfe_pct": round(grp["mfe_approx_pct"].mean(), 3),
            "avg_mae_pct": round(grp["mae_approx_pct"].mean(), 3),
            "edge_ratio": round(
                grp["mfe_approx_pct"].mean() / grp["mae_approx_pct"].mean(), 2
            ) if grp["mae_approx_pct"].mean() > 0 else float("inf"),
        })
    return pd.DataFrame(rows)


def time_of_day_heatmap(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze PnL by entry hour.

    Groups trades by entry_time hour bucket.
    Returns DataFrame with hour, count, avg_pnl, win_rate, total_pnl.
    """
    if trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()

    # Parse entry_time to extract hour
    def _extract_hour(t):
        if pd.isna(t) or t is None or t == "":
            return -1  # daily strategies with no time
        try:
            if hasattr(t, "hour"):
                return t.hour
            parts = str(t).split(":")
            return int(parts[0])
        except (ValueError, IndexError):
            return -1

    df["entry_hour"] = df["entry_time"].apply(_extract_hour)

    rows = []
    for hour, grp in df.groupby("entry_hour"):
        winners = grp[grp["pnl"] > 0]
        label = "daily" if hour == -1 else f"{hour:02d}:00"
        rows.append({
            "hour": label,
            "count": len(grp),
            "avg_pnl": round(grp["pnl"].mean(), 2),
            "win_rate": round(100 * len(winners) / len(grp), 1),
            "total_pnl": round(grp["pnl"].sum(), 2),
        })
    return pd.DataFrame(rows)


def pnl_by_regime(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table: strategy x regime_composite.

    Returns DataFrame with strategy, regime, count, avg_pnl, win_rate, total_pnl.
    """
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for (strategy, regime), grp in trades_df.groupby(["strategy", "regime_composite"]):
        if not regime:
            regime = "unknown"
        winners = grp[grp["pnl"] > 0]
        rows.append({
            "strategy": strategy,
            "regime": regime,
            "count": len(grp),
            "avg_pnl": round(grp["pnl"].mean(), 2),
            "win_rate": round(100 * len(winners) / len(grp), 1),
            "total_pnl": round(grp["pnl"].sum(), 2),
        })
    return pd.DataFrame(rows)


def print_analysis(trades_df: pd.DataFrame, ko_buffer: float = 0.02) -> None:
    """Print all analysis tables to stdout."""
    print("\n" + "=" * 60)
    print("  TRADE ANALYSIS")
    print("=" * 60)

    print("\n── Exit Reason Decomposition ──\n")
    exit_df = exit_reason_decomposition(trades_df)
    if not exit_df.empty:
        print(exit_df.to_string(index=False))

    print("\n── PnL by Stop Width ──\n")
    stop_df = pnl_by_stop_width(trades_df, ko_buffer=ko_buffer)
    if not stop_df.empty:
        print(stop_df.to_string(index=False))

    print("\n── MFE / MAE Analysis ──\n")
    mfe_df = mfe_mae_analysis(trades_df)
    if not mfe_df.empty:
        print(mfe_df.to_string(index=False))

    print("\n── Time of Day ──\n")
    tod_df = time_of_day_heatmap(trades_df)
    if not tod_df.empty:
        print(tod_df.to_string(index=False))

    print("\n── PnL by Regime ──\n")
    regime_df = pnl_by_regime(trades_df)
    if not regime_df.empty:
        print(regime_df.to_string(index=False))

    print()
