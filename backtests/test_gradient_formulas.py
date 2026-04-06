"""
Test different formulas for combining multi-SMA gradients.
Same parameters (SMA 20,50,100,150,200,250,300,400,500,600), different equations.
"""
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import numpy as np
import pandas as pd
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
LOOKBACK = 5


def precompute_all_gradients(daily_data):
    """Precompute per-period gradients for all tickers."""
    all_grads = {}  # ticker -> {date -> {period -> gradient}}
    all_sma_vals = {}  # ticker -> {date -> {period -> sma_value}}

    for ticker, dd in daily_data.items():
        grads_by_date = {}
        sma_by_date = {}
        sma_series = {}
        grad_series = {}

        for p in SMA_PERIODS:
            if len(dd) < p + LOOKBACK:
                continue
            sma = dd["Close"].rolling(p).mean()
            grad = (sma - sma.shift(LOOKBACK)) / sma.shift(LOOKBACK) * 100
            sma_series[p] = sma
            grad_series[p] = grad

        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            close = float(dd["Close"].iloc[i])

            g_dict = {}
            s_dict = {}
            for p in sma_series:
                g = grad_series[p].iloc[i]
                s = sma_series[p].iloc[i]
                if pd.notna(g) and pd.notna(s):
                    g_dict[p] = float(g)
                    s_dict[p] = float(s)

            if g_dict:
                grads_by_date[d] = g_dict
                s_dict["close"] = close
                sma_by_date[d] = s_dict

        all_grads[ticker] = grads_by_date
        all_sma_vals[ticker] = sma_by_date

    return all_grads, all_sma_vals


def formula_weighted_avg(grads, sma_vals, weight_fn):
    """Weighted average of gradients."""
    weights = {p: weight_fn(p) for p in grads}
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    return sum(grads[p] * weights[p] for p in grads) / total_w


def formula_trend_strength_index(grads, sma_vals):
    """TSI = sum(grad_i * w_i) / sum(|grad_i| * w_i). Range [-1, +1].
    +1 = all SMAs strongly trending up, -1 = all down."""
    weights = {p: 1.0 / p for p in grads}
    numerator = sum(grads[p] * weights[p] for p in grads)
    denominator = sum(abs(grads[p]) * weights[p] for p in grads)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def formula_sma_stack_score(grads, sma_vals):
    """Score based on SMA ordering. Perfect bullish stack: SMA20 > SMA50 > SMA100 > ...
    Score = fraction of correctly ordered pairs, signed by direction.
    Range [-1, +1]."""
    periods = sorted([p for p in sma_vals if p != "close"])
    if len(periods) < 2:
        return 0.0

    correct_bull = 0  # shorter SMA above longer SMA
    correct_bear = 0  # shorter SMA below longer SMA
    total_pairs = 0

    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            p_short = periods[i]  # shorter period
            p_long = periods[j]   # longer period
            total_pairs += 1
            if sma_vals[p_short] > sma_vals[p_long]:
                correct_bull += 1
            else:
                correct_bear += 1

    if total_pairs == 0:
        return 0.0

    # Net score: positive if bullish stack, negative if bearish
    return (correct_bull - correct_bear) / total_pairs


def formula_price_vs_smas(grads, sma_vals):
    """Score = weighted fraction of SMAs that price is above/below.
    Price above SMA → +1, below → -1. Weight by inverse period.
    Range [-1, +1]."""
    close = sma_vals.get("close")
    if close is None:
        return 0.0

    periods = [p for p in sma_vals if p != "close"]
    if not periods:
        return 0.0

    weights = {p: 1.0 / p for p in periods}
    total_w = sum(weights.values())

    score = 0.0
    for p in periods:
        if close > sma_vals[p]:
            score += weights[p]
        else:
            score -= weights[p]

    return score / total_w if total_w > 0 else 0.0


def formula_momentum_acceleration(grads, sma_vals):
    """Check if shorter-term gradients are steeper than longer-term.
    Positive = momentum accelerating (short gradients > long gradients).
    Weighted by pair distance."""
    periods = sorted(grads.keys())
    if len(periods) < 2:
        return 0.0

    score = 0.0
    total_w = 0.0
    for i in range(len(periods)):
        for j in range(i + 1, len(periods)):
            p_short = periods[i]
            p_long = periods[j]
            # Shorter SMA gradient should be steeper in trend direction
            diff = grads[p_short] - grads[p_long]
            w = 1.0 / p_short  # weight by shorter period
            score += diff * w
            total_w += abs(diff) * w if diff != 0 else w

    if total_w == 0:
        return 0.0
    return score / total_w


def formula_combined(grads, sma_vals):
    """Combine TSI + stack score + price position. Each [-1,+1], average them."""
    tsi = formula_trend_strength_index(grads, sma_vals)
    stack = formula_sma_stack_score(grads, sma_vals)
    price_pos = formula_price_vs_smas(grads, sma_vals)
    return (tsi + stack + price_pos) / 3.0


def simulate_runner(signals_df, data_5m, grad_fn, threshold, max_hold=5,
                    ko_buffer_pct=1.0, position_sek=500):
    """Generic runner simulation."""
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

            if sim_date == trade_date:
                sig_time = sig.get("time")
                if sig_time:
                    if isinstance(sig_time, str):
                        day_data = day_data[day_data.index.strftime("%H:%M:%S") > sig_time]
                    else:
                        day_data = day_data[day_data.index.time > sig_time]
                if len(day_data) == 0:
                    break

            done = False
            for _, candle in day_data.iterrows():
                if direction == "long":
                    if candle["Low"] <= ko_level:
                        outcome, exit_price, done = "ko", ko_level, True
                        break
                    if sim_date == trade_date and candle["Close"] < orb_low:
                        outcome, exit_price, done = "manual_exit", float(candle["Close"]), True
                        break
                else:
                    if candle["High"] >= ko_level:
                        outcome, exit_price, done = "ko", ko_level, True
                        break
                    if sim_date == trade_date and candle["Close"] > orb_high:
                        outcome, exit_price, done = "manual_exit", float(candle["Close"]), True
                        break

            if done:
                break

            eod_price = float(day_data.iloc[-1]["Close"])

            if day_offset >= max_hold:
                exit_price = eod_price
                break

            grad_val = grad_fn(ticker, sim_date)
            if grad_val is None:
                exit_price = eod_price
                break

            favorable = (direction == "long" and grad_val > threshold) or \
                       (direction == "short" and grad_val < -threshold)

            if not favorable:
                exit_price = eod_price
                break

        if outcome == "ko":
            pnl = -position_sek
        else:
            ret = (exit_price - entry) / entry if direction == "long" else (entry - exit_price) / entry
            pnl = position_sek * max(ret / buffer_frac, -1.0)

        results.append({"ticker": ticker, "date": trade_date, "direction": direction,
                        "outcome": outcome, "days_held": days_held, "pnl": pnl})

    return pd.DataFrame(results)


def main():
    print("=" * 100)
    print("  GRADIENT FORMULA COMPARISON")
    print("  Same SMA periods (20-600), different equations")
    print("=" * 100)

    data_5m = fetch_intraday_5m(ALL_TICKERS)
    daily_data = fetch_daily(ALL_TICKERS)

    # Build trend-aligned signals
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
            if ts.time() <= pd.Timestamp("09:35").time():
                continue

            orb_high, orb_low = orb_data["orb_high"], orb_data["orb_low"]
            direction = None
            if row["Low"] > orb_high:
                direction = "long"
            elif row["High"] < orb_low:
                direction = "short"
            if direction is None:
                continue

            entry = row["Close"]
            sma_val = sma50_lookup.get(date)
            if sma_val is None:
                continue
            if not ((direction == "long" and entry > sma_val) or (direction == "short" and entry < sma_val)):
                continue

            all_signals.append({"ticker": ticker, "date": date, "time": ts.strftime("%H:%M:%S"),
                                "direction": direction, "entry_price": entry,
                                "orb_high": orb_high, "orb_low": orb_low})
            seen.add(date)

    signals_df = pd.DataFrame(all_signals)
    print(f"  Total trend-aligned signals: {len(signals_df)}")

    # Precompute all gradients
    print("  Computing multi-SMA gradients...")
    all_grads, all_sma_vals = precompute_all_gradients(daily_data)

    # Define formula configs
    formulas = {
        "1. Baseline SMA(20)": {
            "fn": lambda t, d: all_grads.get(t, {}).get(d, {}).get(20),
            "thresholds": [0.3],
        },
        "2. Weighted avg (1/period)": {
            "fn": lambda t, d: formula_weighted_avg(
                all_grads.get(t, {}).get(d, {}), all_sma_vals.get(t, {}).get(d, {}),
                lambda p: 1.0/p
            ) if all_grads.get(t, {}).get(d) else None,
            "thresholds": [0.1, 0.2, 0.3],
        },
        "3. Weighted avg (1/sqrt)": {
            "fn": lambda t, d: formula_weighted_avg(
                all_grads.get(t, {}).get(d, {}), all_sma_vals.get(t, {}).get(d, {}),
                lambda p: 1.0/np.sqrt(p)
            ) if all_grads.get(t, {}).get(d) else None,
            "thresholds": [0.1, 0.2, 0.3],
        },
        "4. Trend Strength Index": {
            "fn": lambda t, d: formula_trend_strength_index(
                all_grads.get(t, {}).get(d, {}), all_sma_vals.get(t, {}).get(d, {})
            ) if all_grads.get(t, {}).get(d) else None,
            "thresholds": [0.3, 0.5, 0.7, 0.9],
        },
        "5. SMA Stack Order": {
            "fn": lambda t, d: formula_sma_stack_score(
                all_grads.get(t, {}).get(d, {}), all_sma_vals.get(t, {}).get(d, {})
            ) if all_sma_vals.get(t, {}).get(d) else None,
            "thresholds": [0.2, 0.4, 0.6, 0.8],
        },
        "6. Price vs SMAs": {
            "fn": lambda t, d: formula_price_vs_smas(
                all_grads.get(t, {}).get(d, {}), all_sma_vals.get(t, {}).get(d, {})
            ) if all_sma_vals.get(t, {}).get(d) else None,
            "thresholds": [0.2, 0.4, 0.6, 0.8],
        },
        "7. Momentum Acceleration": {
            "fn": lambda t, d: formula_momentum_acceleration(
                all_grads.get(t, {}).get(d, {}), all_sma_vals.get(t, {}).get(d, {})
            ) if all_grads.get(t, {}).get(d) else None,
            "thresholds": [0.1, 0.3, 0.5],
        },
        "8. Combined (TSI+Stack+Price)": {
            "fn": lambda t, d: formula_combined(
                all_grads.get(t, {}).get(d, {}), all_sma_vals.get(t, {}).get(d, {})
            ) if all_grads.get(t, {}).get(d) and all_sma_vals.get(t, {}).get(d) else None,
            "thresholds": [0.2, 0.4, 0.6],
        },
    }

    # Same-day baseline
    print("\n" + "=" * 100)
    print(f"  {'Formula + Threshold':<45} {'Trades':>7} {'KO%':>7} {'WR%':>7} {'ROI%':>8} {'PnL':>10} {'O/N%':>7} {'AvgD':>6}")
    print("  " + "─" * 97)

    res_sd = simulate_runner(signals_df, data_5m, lambda t, d: 0.0, 0.3)
    n = len(res_sd)
    print(f"  {'Same-day (no runner)':<45} {n:>7} {(res_sd['outcome']=='ko').mean()*100:>6.1f}% {(res_sd['pnl']>0).mean()*100:>6.1f}% {res_sd['pnl'].sum()/(n*500)*100:>+7.1f}% {res_sd['pnl'].sum():>+9.0f} {(res_sd['days_held']>1).mean()*100:>6.1f}% {res_sd['days_held'].mean():>5.1f}")
    print()

    all_results = []
    for name, config in formulas.items():
        for thresh in config["thresholds"]:
            label = f"{name} > {thresh}"
            res = simulate_runner(signals_df, data_5m, config["fn"], thresh)
            if len(res) == 0:
                continue

            n = len(res)
            ko = (res["outcome"] == "ko").mean() * 100
            wr = (res["pnl"] > 0).mean() * 100
            roi = res["pnl"].sum() / (n * 500) * 100
            total_pnl = res["pnl"].sum()
            ovr = (res["days_held"] > 1).mean() * 100
            ad = res["days_held"].mean()

            print(f"  {label:<45} {n:>7} {ko:>6.1f}% {wr:>6.1f}% {roi:>+7.1f}% {total_pnl:>+9.0f} {ovr:>6.1f}% {ad:>5.1f}")

            all_results.append({"formula": name, "threshold": thresh,
                                "trades": n, "ko_pct": ko, "wr_pct": wr,
                                "roi_pct": roi, "total_pnl": total_pnl,
                                "overnight_pct": ovr, "avg_days": ad})
        print()

    # Top 5 by ROI
    if all_results:
        print("\n" + "=" * 100)
        print("  TOP 5 CONFIGURATIONS BY ROI")
        print("=" * 100)
        df_res = pd.DataFrame(all_results).sort_values("roi_pct", ascending=False)
        for i, row in df_res.head(5).iterrows():
            print(f"  {i+1}. {row['formula']} > {row['threshold']} → ROI={row['roi_pct']:+.1f}% | KO={row['ko_pct']:.1f}% | O/N={row['overnight_pct']:.1f}% | PnL={row['total_pnl']:+.0f}")

        df_res.to_csv(RESULTS_DIR / "gradient_formulas.csv", index=False)
        print(f"\n  Saved to {RESULTS_DIR / 'gradient_formulas.csv'}")


if __name__ == "__main__":
    main()
