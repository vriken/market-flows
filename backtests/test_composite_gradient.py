"""
Composite SMA Gradient Backtest

Tests whether using multiple SMA periods with weighted gradients
improves runner performance vs single SMA(20) gradient.

Approach:
- Compute gradient for each SMA period (20, 50, 100, 150, 200, 250, 300, 400, 500, 600)
- Weight shorter-term SMAs more heavily (inverse of period)
- Create composite score = weighted average of all gradients
- Compare runner performance: single SMA(20) vs composite
"""

import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import pandas as pd

# Import from main backtest
from orb_monday_range import (
    ALL_TICKERS,
    RESULTS_DIR,
    _market_hours,
    _to_ny,
    compute_orb,
    fetch_daily,
    fetch_intraday_5m,
)

SMA_PERIODS = [20, 50, 100, 150, 200, 250, 300, 400, 500, 600]
LOOKBACK = 5  # gradient lookback in trading days


def compute_composite_gradient(daily_df: pd.DataFrame,
                                sma_periods: list[int] = SMA_PERIODS,
                                lookback: int = LOOKBACK,
                                weight_scheme: str = "inverse") -> dict:
    """Compute weighted composite gradient across multiple SMA periods.

    Weight schemes:
    - "inverse": weight = 1/period (shorter = more weight)
    - "inverse_sqrt": weight = 1/sqrt(period)
    - "equal": all weights equal
    - "exponential": weight = exp(-period/100)

    Returns dict: date -> {
        "composite_gradient": float,  # weighted average
        "gradients": dict[int, float],  # per-period gradient
        "agreement_pct": float,  # % of SMAs pointing same direction as composite
        "all_positive": bool,
        "all_negative": bool,
    }
    """
    df = daily_df.copy()

    # Compute all SMAs and their gradients
    sma_data = {}
    for period in sma_periods:
        if len(df) < period + lookback:
            continue
        sma = df["Close"].rolling(period).mean()
        gradient = (sma - sma.shift(lookback)) / sma.shift(lookback) * 100
        sma_data[period] = gradient

    if not sma_data:
        return {}

    # Compute weights
    available_periods = sorted(sma_data.keys())
    if weight_scheme == "inverse":
        weights = {p: 1.0 / p for p in available_periods}
    elif weight_scheme == "inverse_sqrt":
        weights = {p: 1.0 / np.sqrt(p) for p in available_periods}
    elif weight_scheme == "equal":
        weights = {p: 1.0 for p in available_periods}
    elif weight_scheme == "exponential":
        weights = {p: np.exp(-p / 100) for p in available_periods}
    else:
        weights = {p: 1.0 / p for p in available_periods}

    # Normalize weights
    total_w = sum(weights.values())
    weights = {p: w / total_w for p, w in weights.items()}

    result = {}
    for i in df.index:
        d = i.date() if hasattr(i, "date") else i

        gradients = {}
        for period in available_periods:
            g = sma_data[period].loc[i]
            if pd.notna(g):
                gradients[period] = float(g)
            else:
                pass

        if not gradients:
            continue

        # Weighted composite
        composite = sum(gradients[p] * weights[p] for p in gradients if p in weights)
        # Renormalize if some periods are missing
        actual_weight = sum(weights[p] for p in gradients if p in weights)
        if actual_weight > 0:
            composite = composite / actual_weight

        # Agreement: what % of SMAs agree with composite direction
        if composite > 0:
            agree = sum(1 for g in gradients.values() if g > 0)
        elif composite < 0:
            agree = sum(1 for g in gradients.values() if g < 0)
        else:
            agree = 0
        agreement_pct = agree / len(gradients) * 100 if gradients else 0

        result[d] = {
            "composite_gradient": composite,
            "gradients": gradients,
            "agreement_pct": agreement_pct,
            "all_positive": all(g > 0 for g in gradients.values()),
            "all_negative": all(g < 0 for g in gradients.values()),
            "n_smas": len(gradients),
        }

    return result


def simulate_runner_composite(signals_df, data_5m, daily_data,
                               gradient_fn, threshold, max_hold=5,
                               ko_buffer_pct=1.0, position_sek=500):
    """Simulate turbo runner using a custom gradient function.

    gradient_fn: callable(ticker, date) -> (gradient_value, extra_info)
    """
    buffer_frac = ko_buffer_pct / 100
    results = []

    for _, sig in signals_df.iterrows():
        ticker = sig["ticker"]
        direction = sig["direction"]
        entry = sig["entry_price"]
        orb_high = sig["orb_high"]
        orb_low = sig["orb_low"]
        trade_date = sig["date"]

        if ticker not in data_5m:
            continue

        df = _market_hours(_to_ny(data_5m[ticker]))
        dates_available = sorted(set(df.index.date))

        if trade_date not in dates_available:
            continue
        start_idx = dates_available.index(trade_date)

        # KO level
        ko_level = entry * (1 - buffer_frac) if direction == "long" else entry * (1 + buffer_frac)

        outcome = "close"
        exit_price = entry
        days_held = 0

        for day_offset in range(max_hold + 1):
            if start_idx + day_offset >= len(dates_available):
                break

            sim_date = dates_available[start_idx + day_offset]
            days_held = day_offset + 1
            day_data = df[df.index.date == sim_date]

            if len(day_data) == 0:
                break

            # On signal day, start after signal
            if sim_date == trade_date:
                sig_time = sig.get("time")
                if sig_time:
                    if isinstance(sig_time, str):
                        day_data = day_data[day_data.index.strftime("%H:%M:%S") > sig_time]
                    else:
                        day_data = day_data[day_data.index.time > sig_time]
                if len(day_data) == 0:
                    break

            # Check KO and (signal-day) full reversal
            done = False
            for _, candle in day_data.iterrows():
                if direction == "long":
                    if candle["Low"] <= ko_level:
                        outcome = "ko"
                        exit_price = ko_level
                        done = True
                        break
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

            eod_price = float(day_data.iloc[-1]["Close"])

            if day_offset >= max_hold:
                exit_price = eod_price
                break

            # Gradient check
            grad_val, extra = gradient_fn(ticker, sim_date)
            if grad_val is None:
                exit_price = eod_price
                break

            favorable = (direction == "long" and grad_val > threshold) or \
                       (direction == "short" and grad_val < -threshold)

            if not favorable:
                exit_price = eod_price
                break

        # P&L
        if outcome == "ko":
            pnl = -position_sek
        else:
            ret = (exit_price - entry) / entry if direction == "long" else (entry - exit_price) / entry
            turbo_ret = max(ret / buffer_frac, -1.0)
            pnl = position_sek * turbo_ret

        results.append({
            "ticker": ticker, "date": trade_date, "direction": direction,
            "entry_price": entry, "exit_price": exit_price,
            "outcome": outcome, "days_held": days_held,
            "pnl": pnl,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 90)
    print("  COMPOSITE SMA GRADIENT BACKTEST")
    print("=" * 90)

    data_5m = fetch_intraday_5m(ALL_TICKERS)
    daily_data = fetch_daily(ALL_TICKERS)

    # Build signals (trend-aligned, 5min)
    print("\nBuilding signals...")
    all_signals = []
    for ticker in sorted(data_5m.keys()):
        df = data_5m[ticker]
        orb = compute_orb(df)
        if orb.empty:
            continue

        df_mkt = _market_hours(_to_ny(df))
        dates = df_mkt.index.date
        orb_lookup = {row["date"]: row for _, row in orb.iterrows()}

        # Daily SMA50 for trend
        dd = daily_data.get(ticker)
        if dd is None:
            continue
        sma50 = dd["Close"].rolling(50).mean()
        sma50_lookup = {}
        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            if pd.notna(sma50.iloc[i]):
                sma50_lookup[d] = float(sma50.iloc[i])

        seen = set()
        for idx in range(len(df_mkt)):
            ts = df_mkt.index[idx]
            row = df_mkt.iloc[idx]
            date = dates[idx]
            if date in seen:
                continue
            orb_data = orb_lookup.get(date)
            if orb_data is None:
                continue

            orb_high = orb_data["orb_high"]
            orb_low = orb_data["orb_low"]

            # Check after ORB
            if ts.time() <= pd.Timestamp("09:35").time():
                continue

            direction = None
            if row["Low"] > orb_high:
                direction = "long"
            elif row["High"] < orb_low:
                direction = "short"

            if direction is None:
                continue

            entry = row["Close"]

            # Trend alignment check
            sma_val = sma50_lookup.get(date)
            if sma_val is None:
                continue
            trend_aligned = (direction == "long" and entry > sma_val) or \
                           (direction == "short" and entry < sma_val)
            if not trend_aligned:
                continue

            all_signals.append({
                "ticker": ticker, "date": date, "time": ts.strftime("%H:%M:%S"),
                "direction": direction, "entry_price": entry,
                "orb_high": orb_high, "orb_low": orb_low,
            })
            seen.add(date)

    signals_df = pd.DataFrame(all_signals)
    print(f"  Total trend-aligned signals: {len(signals_df)}")

    # Pre-compute gradients for all tickers
    print("\nComputing gradients...")

    # 1. Single SMA(20) gradient (baseline)
    single_grads = {}
    for ticker in daily_data:
        dd = daily_data[ticker]
        sma = dd["Close"].rolling(20).mean()
        grad = (sma - sma.shift(LOOKBACK)) / sma.shift(LOOKBACK) * 100
        single_grads[ticker] = {}
        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            if pd.notna(grad.iloc[i]):
                single_grads[ticker][d] = float(grad.iloc[i])

    # 2. Composite gradients with different weight schemes
    composite_grads = {}
    for scheme in ["inverse", "inverse_sqrt", "equal", "exponential"]:
        composite_grads[scheme] = {}
        for ticker in daily_data:
            composite_grads[scheme][ticker] = compute_composite_gradient(
                daily_data[ticker], weight_scheme=scheme
            )

    # Define gradient functions
    def make_single_fn(threshold_sma_period=20):
        grads = {}
        for ticker in daily_data:
            dd = daily_data[ticker]
            sma = dd["Close"].rolling(threshold_sma_period).mean()
            grad = (sma - sma.shift(LOOKBACK)) / sma.shift(LOOKBACK) * 100
            grads[ticker] = {}
            for i in range(len(dd)):
                d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
                if pd.notna(grad.iloc[i]):
                    grads[ticker][d] = float(grad.iloc[i])

        def fn(ticker, date):
            g = grads.get(ticker, {}).get(date)
            return (g, {}) if g is not None else (None, {})
        return fn

    def make_composite_fn(scheme, min_agreement=0):
        def fn(ticker, date):
            data = composite_grads[scheme].get(ticker, {}).get(date)
            if data is None:
                return (None, {})
            if min_agreement > 0 and data["agreement_pct"] < min_agreement:
                return (None, {})  # not enough agreement → exit
            return (data["composite_gradient"], data)
        return fn

    def make_all_agree_fn(scheme):
        """Only hold if ALL SMAs agree on direction."""
        def fn(ticker, date):
            data = composite_grads[scheme].get(ticker, {}).get(date)
            if data is None:
                return (None, {})
            g = data["composite_gradient"]
            if g > 0 and not data["all_positive"]:
                return (0.0, data)  # not all agree → return 0 (won't pass threshold)
            if g < 0 and not data["all_negative"]:
                return (0.0, data)
            return (g, data)
        return fn

    # Run simulations
    configs = [
        ("Baseline: SMA(20) only", make_single_fn(20), 0.3),
        ("Single SMA(50)", make_single_fn(50), 0.3),
        ("Single SMA(100)", make_single_fn(100), 0.3),
        ("Composite (inverse weight)", make_composite_fn("inverse"), 0.3),
        ("Composite (inverse sqrt)", make_composite_fn("inverse_sqrt"), 0.3),
        ("Composite (equal weight)", make_composite_fn("equal"), 0.3),
        ("Composite (exponential)", make_composite_fn("exponential"), 0.3),
        ("Composite inverse, thresh 0.1", make_composite_fn("inverse"), 0.1),
        ("Composite inverse, thresh 0.5", make_composite_fn("inverse"), 0.5),
        ("Composite inverse, agree>70%", make_composite_fn("inverse", min_agreement=70), 0.3),
        ("Composite inverse, agree>90%", make_composite_fn("inverse", min_agreement=90), 0.3),
        ("All SMAs agree (inverse)", make_all_agree_fn("inverse"), 0.3),
    ]

    # Also test same-day (no runner) as baseline
    print("\n" + "=" * 90)
    print("  RUNNER COMPARISON: Single vs Composite Gradient")
    print("  (trend-aligned signals, 1% KO buffer, max 5 day hold)")
    print("=" * 90)

    print(f"\n  {'Config':<40} {'Trades':>7} {'KO%':>7} {'WR%':>7} {'ROI%':>8} {'Avg PnL':>9} {'Overnight%':>11} {'Avg Days':>9}")
    print("  " + "─" * 100)

    # Same-day baseline
    def same_day_fn(ticker, date):
        return (0.0, {})  # always return 0 → never holds overnight

    res_sd = simulate_runner_composite(signals_df, data_5m, daily_data, same_day_fn, 0.3)
    if len(res_sd) > 0:
        n = len(res_sd)
        ko = (res_sd["outcome"] == "ko").mean() * 100
        wr = (res_sd["pnl"] > 0).mean() * 100
        roi = res_sd["pnl"].sum() / (n * 500) * 100
        avg = res_sd["pnl"].mean()
        ovr = (res_sd["days_held"] > 1).mean() * 100
        ad = res_sd["days_held"].mean()
        print(f"  {'Same-day exit (no runner)':<40} {n:>7} {ko:>6.1f}% {wr:>6.1f}% {roi:>+7.1f}% {avg:>+8.0f} {ovr:>10.1f}% {ad:>8.1f}")

    print()

    all_results = []
    for label, grad_fn, threshold in configs:
        res = simulate_runner_composite(signals_df, data_5m, daily_data, grad_fn, threshold)
        if len(res) == 0:
            continue

        n = len(res)
        ko = (res["outcome"] == "ko").mean() * 100
        wr = (res["pnl"] > 0).mean() * 100
        roi = res["pnl"].sum() / (n * 500) * 100
        avg = res["pnl"].mean()
        ovr = (res["days_held"] > 1).mean() * 100
        ad = res["days_held"].mean()

        print(f"  {label:<40} {n:>7} {ko:>6.1f}% {wr:>6.1f}% {roi:>+7.1f}% {avg:>+8.0f} {ovr:>10.1f}% {ad:>8.1f}")

        all_results.append({
            "config": label, "threshold": threshold,
            "trades": n, "ko_pct": ko, "wr_pct": wr, "roi_pct": roi,
            "avg_pnl": avg, "overnight_pct": ovr, "avg_days": ad,
        })

    # Show what the composite gradient looks like for a sample ticker
    print("\n" + "=" * 90)
    print("  SAMPLE: Composite vs Single Gradient (first available ticker, last 20 days)")
    print("=" * 90)

    sample_ticker = sorted(daily_data.keys())[0]
    comp = composite_grads["inverse"].get(sample_ticker, {})
    single = single_grads.get(sample_ticker, {})

    dates_both = sorted(set(comp.keys()) & set(single.keys()))
    if dates_both:
        print(f"\n  Ticker: {sample_ticker}")
        print(f"  {'Date':<12} {'SMA20 Grad':>11} {'Composite':>11} {'Agreement':>11} {'All Agree':>10}")
        print("  " + "─" * 60)
        for d in dates_both[-20:]:
            sg = single[d]
            cg = comp[d]["composite_gradient"]
            ag = comp[d]["agreement_pct"]
            aa = "YES" if (comp[d]["all_positive"] or comp[d]["all_negative"]) else "no"
            print(f"  {str(d):<12} {sg:>+10.2f}% {cg:>+10.2f}% {ag:>10.0f}% {aa:>10}")

    # Save results
    if all_results:
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / "composite_gradient.csv", index=False)
        print(f"\n  Results saved to {RESULTS_DIR / 'composite_gradient.csv'}")

    print("\n" + "=" * 90)
    print("  DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
