"""
Hybrid Product Runner Strategy Backtest

Concept: Use turbos for intraday (Day 1), but when the SMA gradient says
"hold overnight", sell the turbo at EOD and buy a bull/bear certificate
instead (no KO risk overnight). This avoids the ~46.5% KO rate that kills
overnight turbo holds.

Simulation:
  Day 1 (signal day):
    - Trade as turbo with KO buffer. Check for KO and full-reversal stop.
    - At EOD: check SMA(20) gradient over 5 days.
      - If gradient < threshold -> exit turbo at EOD close (done)
      - If gradient >= threshold -> note EOD close price, "switch to bull/bear cert"
  Day 2+:
    - Simulate as bull/bear certificate (Nx leverage, 0.07%/day funding, NO KO).
    - At each subsequent EOD, re-check gradient -> hold or exit.
    - Max 5 days total.
  P&L = turbo P&L for day 1 + bull/bear cert P&L for days 2+

Compare against:
  - Pure turbo runner (same-day + overnight with turbo, from simulate_turbo with hold_days=1)
  - Baseline (turbo same-day exit only)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the backtests directory is importable
sys.path.insert(0, str(Path(__file__).parent))

from orb_monday_range import (
    ALL_TICKERS,
    RESULTS_DIR,
    fetch_intraday_5m,
    fetch_intraday_1m,
    fetch_daily,
    build_unified_signals,
    compute_sma_gradient,
    simulate_turbo,
    _market_hours,
    _to_ny,
)

POSITION_SEK = 500.0
SMA_PERIOD = 20
GRADIENT_LOOKBACK = 5
MAX_HOLD_DAYS = 5
SIGNAL_TF = "5min"


def simulate_hybrid_runner(
    unified: pd.DataFrame,
    data_5m: dict,
    daily_data: dict,
    ko_buffer_pct: float = 1.0,
    cert_leverage: float = 5.0,
    cert_daily_funding_pct: float = 0.07,
    position_sek: float = 500.0,
    sma_period: int = 20,
    gradient_lookback: int = 5,
    gradient_threshold: float = 0.3,
    max_hold_days: int = 5,
    signal_tf_filter: str = "5min",
    quiet: bool = False,
    data_1m: dict | None = None,
) -> pd.DataFrame:
    """Hybrid turbo + bull/bear cert runner.

    Day 1: trade as turbo (with KO risk).
    Day 2+: if SMA gradient is favorable, switch to bull/bear cert (no KO).
    """
    if signal_tf_filter:
        signals = unified[unified["signal_tf"] == signal_tf_filter].copy()
    else:
        signals = unified.copy()
    signals = signals.sort_values(["date", "time"])

    if not quiet:
        print(
            f"\n  Simulating hybrid runner: {len(signals)} signals @ "
            f"{ko_buffer_pct}% turbo KO, {cert_leverage}x cert, "
            f"grad>{gradient_threshold}%, max {max_hold_days}d..."
        )

    # Pre-compute SMA gradient per ticker
    ticker_gradients: dict[str, dict] = {}
    ticker_dates: dict[str, list] = {}
    for ticker, df_raw in data_5m.items():
        daily_df = daily_data.get(ticker)
        if daily_df is not None:
            ticker_gradients[ticker] = compute_sma_gradient(
                daily_df, sma_period=sma_period, lookback=gradient_lookback
            )
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

        all_dates = ticker_dates.get(ticker, [])
        try:
            start_idx = all_dates.index(trade_date_obj)
        except ValueError:
            continue

        gradients = ticker_gradients.get(ticker, {})

        # ── Day 1: Turbo simulation ──────────────────────────────────────
        day1_date = all_dates[start_idx]

        # Get intraday data for Day 1
        day1_data = pd.DataFrame()
        if data_1m is not None and ticker in data_1m:
            df_1m_raw = _market_hours(_to_ny(data_1m[ticker]))
            day1_data = df_1m_raw[df_1m_raw.index.date == day1_date]
        if len(day1_data) == 0:
            df_5m_raw = data_5m.get(ticker)
            if df_5m_raw is None:
                continue
            df_5m = _market_hours(_to_ny(df_5m_raw))
            day1_data = df_5m[df_5m.index.date == day1_date]

        if len(day1_data) == 0:
            continue

        # Filter to candles after signal time
        if isinstance(signal_time, str):
            time_filter = signal_time
        else:
            time_filter = str(signal_time)
        day1_data = day1_data[day1_data.index.strftime("%H:%M:%S") > time_filter]
        if len(day1_data) == 0:
            continue

        # Check for KO or full-reversal stop on Day 1
        day1_outcome = "close"
        day1_exit_price = float(day1_data.iloc[-1]["Close"])
        day1_exit_time = ""
        done = False

        for ts, candle in day1_data.iterrows():
            if direction == "long":
                if candle["Low"] <= ko_level:
                    day1_outcome = "ko"
                    day1_exit_price = ko_level
                    day1_exit_time = (
                        ts.strftime("%Y-%m-%d %H:%M")
                        if hasattr(ts, "strftime")
                        else str(ts)
                    )
                    done = True
                    break
                if candle["Close"] < orb_low:
                    day1_outcome = "manual_exit"
                    day1_exit_price = float(candle["Close"])
                    day1_exit_time = (
                        ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                    )
                    done = True
                    break
            else:
                if candle["High"] >= ko_level:
                    day1_outcome = "ko"
                    day1_exit_price = ko_level
                    day1_exit_time = (
                        ts.strftime("%Y-%m-%d %H:%M")
                        if hasattr(ts, "strftime")
                        else str(ts)
                    )
                    done = True
                    break
                if candle["Close"] > orb_high:
                    day1_outcome = "manual_exit"
                    day1_exit_price = float(candle["Close"])
                    day1_exit_time = (
                        ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
                    )
                    done = True
                    break

        # Day 1 turbo P&L
        if day1_outcome == "ko":
            turbo_pnl = -position_sek
        else:
            if direction == "long":
                underlying_return = (day1_exit_price - entry) / entry
            else:
                underlying_return = (entry - day1_exit_price) / entry
            turbo_return = underlying_return / buffer_frac
            turbo_return = max(turbo_return, -1.0)
            turbo_pnl = position_sek * turbo_return

        # If Day 1 ended early (KO or manual exit), no overnight decision
        if done:
            results.append(
                _build_result(
                    sig,
                    entry=entry,
                    exit_price=day1_exit_price,
                    ko_level=ko_level,
                    outcome=day1_outcome,
                    exit_time=day1_exit_time,
                    days_held=1,
                    turbo_pnl=turbo_pnl,
                    cert_pnl=0.0,
                    total_pnl=turbo_pnl,
                    hold_reason="day1_exit",
                    switched_to_cert=False,
                    position_sek=position_sek,
                )
            )
            continue

        # ── EOD Day 1: Gradient check for overnight hold ─────────────────
        grad_info = gradients.get(day1_date)
        if grad_info is None:
            # No gradient data -> exit at close, turbo only
            results.append(
                _build_result(
                    sig,
                    entry=entry,
                    exit_price=day1_exit_price,
                    ko_level=ko_level,
                    outcome="close",
                    exit_time="",
                    days_held=1,
                    turbo_pnl=turbo_pnl,
                    cert_pnl=0.0,
                    total_pnl=turbo_pnl,
                    hold_reason="no_gradient_data",
                    switched_to_cert=False,
                    position_sek=position_sek,
                )
            )
            continue

        grad = grad_info["gradient_pct"]
        favorable = (direction == "long" and grad > gradient_threshold) or (
            direction == "short" and grad < -gradient_threshold
        )

        if not favorable:
            # Gradient not strong enough -> exit turbo at close, done
            results.append(
                _build_result(
                    sig,
                    entry=entry,
                    exit_price=day1_exit_price,
                    ko_level=ko_level,
                    outcome="close",
                    exit_time="",
                    days_held=1,
                    turbo_pnl=turbo_pnl,
                    cert_pnl=0.0,
                    total_pnl=turbo_pnl,
                    hold_reason=f"grad={grad:+.2f}%",
                    switched_to_cert=False,
                    position_sek=position_sek,
                )
            )
            continue

        # ── Day 2+: Switch to bull/bear certificate ──────────────────────
        # "Sell turbo at EOD, buy cert at same EOD price"
        cert_entry_price = day1_exit_price
        cert_value = position_sek + turbo_pnl  # reinvest Day 1 proceeds
        if cert_value <= 0:
            # Nothing left to reinvest
            results.append(
                _build_result(
                    sig,
                    entry=entry,
                    exit_price=day1_exit_price,
                    ko_level=ko_level,
                    outcome="close",
                    exit_time="",
                    days_held=1,
                    turbo_pnl=turbo_pnl,
                    cert_pnl=0.0,
                    total_pnl=turbo_pnl,
                    hold_reason="no_capital_for_cert",
                    switched_to_cert=False,
                    position_sek=position_sek,
                )
            )
            continue

        cert_initial = cert_value
        prev_price = cert_entry_price
        cert_exit_price = cert_entry_price
        total_days_held = 1
        cert_outcome = "close"
        cert_exit_time = ""
        cert_done = False

        for day_offset in range(1, max_hold_days):
            if start_idx + day_offset >= len(all_dates):
                break

            sim_date = all_dates[start_idx + day_offset]
            total_days_held = day_offset + 1

            # Get intraday data
            day_data = pd.DataFrame()
            df_5m_raw = data_5m.get(ticker)
            if df_5m_raw is not None:
                df_5m = _market_hours(_to_ny(df_5m_raw))
                day_data = df_5m[df_5m.index.date == sim_date]

            if len(day_data) == 0:
                break

            # Bull/bear cert: no KO. Just track daily return + funding.
            eod_price = float(day_data.iloc[-1]["Close"])

            if direction == "long":
                daily_return = (eod_price - prev_price) / prev_price
            else:
                daily_return = (prev_price - eod_price) / prev_price

            cert_daily_return = cert_leverage * daily_return
            cert_value *= 1 + cert_daily_return
            # Apply daily funding cost
            cert_value *= 1 - cert_daily_funding_pct / 100
            cert_value = max(cert_value, 0.0)

            prev_price = eod_price
            cert_exit_price = eod_price

            # Last possible day -> must exit
            if day_offset >= max_hold_days - 1:
                cert_outcome = "close"
                break

            # Re-check gradient for next overnight decision
            grad_info_next = gradients.get(sim_date)
            if grad_info_next is None:
                cert_outcome = "close"
                break

            grad_next = grad_info_next["gradient_pct"]
            still_favorable = (
                direction == "long" and grad_next > gradient_threshold
            ) or (direction == "short" and grad_next < -gradient_threshold)

            if not still_favorable:
                cert_outcome = "close"
                break
            # else: hold another night

        cert_pnl = cert_value - cert_initial
        total_pnl = turbo_pnl + cert_pnl

        results.append(
            _build_result(
                sig,
                entry=entry,
                exit_price=cert_exit_price,
                ko_level=ko_level,
                outcome=cert_outcome,
                exit_time=cert_exit_time,
                days_held=total_days_held,
                turbo_pnl=turbo_pnl,
                cert_pnl=cert_pnl,
                total_pnl=total_pnl,
                hold_reason=f"cert_{total_days_held - 1}d",
                switched_to_cert=True,
                position_sek=position_sek,
            )
        )

    return pd.DataFrame(results)


def _build_result(
    sig,
    entry: float,
    exit_price: float,
    ko_level: float,
    outcome: str,
    exit_time: str,
    days_held: int,
    turbo_pnl: float,
    cert_pnl: float,
    total_pnl: float,
    hold_reason: str,
    switched_to_cert: bool,
    position_sek: float,
) -> dict:
    direction = sig["direction"]
    return {
        "ticker": sig["ticker"],
        "date": str(sig["date"]),
        "time": sig["time"],
        "direction": direction,
        "entry_price": entry,
        "exit_price": exit_price,
        "ko_level": ko_level,
        "orb_high": sig["orb_high"],
        "orb_low": sig["orb_low"],
        "outcome": outcome,
        "exit_time": exit_time,
        "days_held": days_held,
        "hold_reason": hold_reason,
        "switched_to_cert": switched_to_cert,
        "underlying_move_pct": (exit_price - entry) / entry * 100 * (1 if direction == "long" else -1),
        "turbo_pnl_day1": turbo_pnl,
        "cert_pnl_day2plus": cert_pnl,
        "total_pnl_sek": total_pnl,
        "confluence": sig["confluence"],
        "vwap_confirmed": sig["vwap_confirmed"],
        "trend_aligned": sig["trend_aligned"],
        "category": sig["category"],
        "vol_ratio": sig.get("vol_ratio", 1.0),
        "vol_spike": sig.get("vol_spike", False),
    }


def summarize(df: pd.DataFrame, label: str, pnl_col: str, position: float) -> dict:
    """Compute summary stats for a results DataFrame."""
    if df.empty:
        return {
            "config": label,
            "n_trades": 0,
            "KO%": 0.0,
            "WR%": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "ROI%": 0.0,
        }
    n = len(df)
    n_ko = (df["outcome"] == "ko").sum() if "outcome" in df.columns else 0
    winners = (df[pnl_col] > 0).sum()
    total_pnl = df[pnl_col].sum()
    total_invested = n * position
    return {
        "config": label,
        "n_trades": n,
        "KO%": round(n_ko / n * 100, 1) if n > 0 else 0.0,
        "WR%": round(winners / n * 100, 1) if n > 0 else 0.0,
        "avg_pnl": round(total_pnl / n, 1) if n > 0 else 0.0,
        "total_pnl": round(total_pnl, 0),
        "ROI%": round(total_pnl / total_invested * 100, 2) if total_invested > 0 else 0.0,
    }


def main():
    print("=" * 72)
    print("  HYBRID PRODUCT RUNNER BACKTEST")
    print("  Turbo Day 1 + Bull/Bear Cert Day 2+ (no KO overnight)")
    print("=" * 72)

    # ── Fetch data ───────────────────────────────────────────────────────
    data_5m = fetch_intraday_5m(ALL_TICKERS)
    daily_data = fetch_daily(ALL_TICKERS)
    data_1m = fetch_intraday_1m(ALL_TICKERS)

    # ── Build unified signals ────────────────────────────────────────────
    unified = build_unified_signals(data_5m, daily_data, data_1m=data_1m)
    print(f"\n  Total unified signals: {len(unified)}")

    # Filter to trend-aligned 5-min signals only
    trend_signals = unified[
        (unified["signal_tf"] == SIGNAL_TF) & (unified["trend_aligned"])
    ].copy()
    print(f"  Trend-aligned {SIGNAL_TF} signals: {len(trend_signals)}")

    # ── Test grid ────────────────────────────────────────────────────────
    turbo_buffers = [1.0, 2.0]
    cert_leverages = [5, 10]
    gradient_thresholds = [0.1, 0.3, 0.5, 1.0]

    all_summaries = []

    # ── Baselines ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  BASELINES")
    print("=" * 72)

    for buf in turbo_buffers:
        # Baseline: same-day exit only (hold_days=0)
        baseline = simulate_turbo(
            trend_signals, data_5m, ko_buffer_pct=buf,
            position_sek=POSITION_SEK, signal_tf_filter=SIGNAL_TF,
            quiet=True, data_1m=data_1m, hold_days=0,
        )
        label = f"Baseline turbo {buf}% (same-day)"
        s = summarize(baseline, label, "turbo_pnl_sek", POSITION_SEK)
        all_summaries.append(s)
        print(f"  {label}: n={s['n_trades']} KO={s['KO%']}% WR={s['WR%']}% "
              f"avg={s['avg_pnl']:+.0f} total={s['total_pnl']:+,.0f} ROI={s['ROI%']:+.2f}%")

        # Pure turbo runner (hold overnight WITH turbo -> KO risk)
        pure_runner = simulate_turbo(
            trend_signals, data_5m, ko_buffer_pct=buf,
            position_sek=POSITION_SEK, signal_tf_filter=SIGNAL_TF,
            quiet=True, data_1m=data_1m, hold_days=1,
        )
        label = f"Pure turbo runner {buf}% (+1d hold)"
        s = summarize(pure_runner, label, "turbo_pnl_sek", POSITION_SEK)
        all_summaries.append(s)
        print(f"  {label}: n={s['n_trades']} KO={s['KO%']}% WR={s['WR%']}% "
              f"avg={s['avg_pnl']:+.0f} total={s['total_pnl']:+,.0f} ROI={s['ROI%']:+.2f}%")

    # ── Hybrid runner grid ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  HYBRID RUNNER GRID")
    print("=" * 72)

    for buf in turbo_buffers:
        for lev in cert_leverages:
            for thresh in gradient_thresholds:
                hybrid = simulate_hybrid_runner(
                    trend_signals,
                    data_5m,
                    daily_data,
                    ko_buffer_pct=buf,
                    cert_leverage=float(lev),
                    cert_daily_funding_pct=0.07,
                    position_sek=POSITION_SEK,
                    sma_period=SMA_PERIOD,
                    gradient_lookback=GRADIENT_LOOKBACK,
                    gradient_threshold=thresh,
                    max_hold_days=MAX_HOLD_DAYS,
                    signal_tf_filter=SIGNAL_TF,
                    quiet=True,
                    data_1m=data_1m,
                )
                label = f"Hybrid {buf}%KO / {lev}x cert / grad>{thresh}%"
                s = summarize(hybrid, label, "total_pnl_sek", POSITION_SEK)

                # Extra hybrid-specific stats
                if not hybrid.empty:
                    n_switched = hybrid["switched_to_cert"].sum()
                    s["n_switched_to_cert"] = int(n_switched)
                    s["switch_rate%"] = round(n_switched / len(hybrid) * 100, 1)
                    cert_only = hybrid[hybrid["switched_to_cert"]]
                    if len(cert_only) > 0:
                        s["cert_avg_pnl"] = round(cert_only["cert_pnl_day2plus"].mean(), 1)
                        s["cert_total_pnl"] = round(cert_only["cert_pnl_day2plus"].sum(), 0)
                    else:
                        s["cert_avg_pnl"] = 0.0
                        s["cert_total_pnl"] = 0.0
                    s["avg_days_held"] = round(hybrid["days_held"].mean(), 2)
                else:
                    s["n_switched_to_cert"] = 0
                    s["switch_rate%"] = 0.0
                    s["cert_avg_pnl"] = 0.0
                    s["cert_total_pnl"] = 0.0
                    s["avg_days_held"] = 0.0

                all_summaries.append(s)
                print(
                    f"  {label}: n={s['n_trades']} KO={s['KO%']}% WR={s['WR%']}% "
                    f"avg={s['avg_pnl']:+.0f} total={s['total_pnl']:+,.0f} ROI={s['ROI%']:+.2f}% "
                    f"| switched={s['n_switched_to_cert']} ({s['switch_rate%']}%) "
                    f"cert_pnl={s['cert_total_pnl']:+,.0f} avg_days={s['avg_days_held']:.1f}"
                )

    # ── Comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  COMPARISON TABLE")
    print("=" * 72)

    header = (
        f"  {'Config':<45s} {'n':>5s} {'KO%':>6s} {'WR%':>6s} "
        f"{'avg_pnl':>9s} {'total_pnl':>11s} {'ROI%':>8s}"
    )
    print(header)
    print("  " + "-" * 95)

    for s in all_summaries:
        print(
            f"  {s['config']:<45s} {s['n_trades']:>5d} {s['KO%']:>5.1f}% "
            f"{s['WR%']:>5.1f}% {s['avg_pnl']:>+9.0f} {s['total_pnl']:>+11,.0f} "
            f"{s['ROI%']:>+7.2f}%"
        )

    # ── Save results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_summaries)
    out_path = RESULTS_DIR / "hybrid_runner.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
