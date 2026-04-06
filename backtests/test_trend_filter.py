"""
Test whether tightening the T (trend-aligned) quality factor improves results.

Current: T=1 when price > SMA(50) for longs, price < SMA(50) for shorts
Proposed: T=1 when price > SMA(50) AND combinedGrad > 0 for longs (and inverse for shorts)

This prevents taking longs during short-term bounces when major SMAs all slope down.
"""
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import pandas as pd
from orb_monday_range import (
    ALL_TICKERS,
    _market_hours,
    _to_ny,
    compute_monday_range,
    compute_orb,
    fetch_daily,
    fetch_intraday_5m,
)

SMA_PERIODS = [20, 50, 100, 150, 200, 250, 300, 400, 500, 600]
LOOKBACK = 5


def precompute_combined_grad(daily_data):
    """Precompute combined gradient (TSI + Stack + PricePos) per ticker per date."""
    results = {}  # ticker -> {date -> combined_grad}

    for ticker, dd in daily_data.items():
        sma_series = {}
        grad_series = {}

        for p in SMA_PERIODS:
            if len(dd) < p + LOOKBACK:
                continue
            sma = dd["Close"].rolling(p).mean()
            grad = (sma - sma.shift(LOOKBACK)) / sma.shift(LOOKBACK) * 100
            sma_series[p] = sma
            grad_series[p] = grad

        by_date = {}
        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            close_val = float(dd["Close"].iloc[i])

            grads = {}
            sma_vals = {}
            for p in sma_series:
                g = grad_series[p].iloc[i]
                s = sma_series[p].iloc[i]
                if pd.notna(g) and pd.notna(s):
                    grads[p] = float(g)
                    sma_vals[p] = float(s)

            if not grads:
                continue

            # TSI
            tsi_num = sum(grads[p] * (1.0 / p) for p in grads)
            tsi_den = sum(abs(grads[p]) * (1.0 / p) for p in grads)
            tsi = tsi_num / tsi_den if tsi_den > 0 else 0.0

            # Stack order
            periods = sorted(sma_vals.keys())
            bull = bear = total = 0
            for a in range(len(periods)):
                for b in range(a + 1, len(periods)):
                    total += 1
                    if sma_vals[periods[a]] > sma_vals[periods[b]]:
                        bull += 1
                    else:
                        bear += 1
            stack = (bull - bear) / total if total > 0 else 0.0

            # Price vs SMAs
            pv_num = sum((1.0 / p) * (1 if close_val > sma_vals[p] else -1) for p in sma_vals)
            pv_den = sum(1.0 / p for p in sma_vals)
            pv = pv_num / pv_den if pv_den > 0 else 0.0

            by_date[d] = (tsi + stack + pv) / 3.0

        results[ticker] = by_date
    return results


def build_signals(data_5m, daily_data, combined_grads, t_mode="sma50"):
    """Build ORB signals with quality scoring.

    t_mode:
      "sma50"     - T=1 when price vs SMA50 agrees (current)
      "sma50+grad" - T=1 when price vs SMA50 AND combinedGrad agrees (proposed)
      "grad_only" - T=1 when combinedGrad agrees (no SMA50 check)
    """
    all_signals = []
    for ticker in sorted(data_5m.keys()):
        df = data_5m[ticker]
        orb = compute_orb(df)
        if orb.empty:
            continue

        df_mkt = _market_hours(_to_ny(df))
        dd = daily_data.get(ticker)
        if dd is None:
            continue

        # SMA50 for trend
        sma50 = dd["Close"].rolling(50).mean()
        sma50_lookup = {}
        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            if pd.notna(sma50.iloc[i]):
                sma50_lookup[d] = float(sma50.iloc[i])

        # Monday range — keyed by ISO (year, week) from week_start date
        mon = compute_monday_range(dd)
        mon_lookup = {}
        if not mon.empty:
            for _, mr in mon.iterrows():
                ws = mr["week_start"]
                key = ws.isocalendar()[:2] if hasattr(ws, 'isocalendar') else ws
                mon_lookup[key] = mr

        # Volume SMA
        vol = dd["Volume"].rolling(20).mean()
        vol_lookup = {}
        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            if pd.notna(vol.iloc[i]):
                vol_lookup[d] = float(vol.iloc[i])

        orb_lookup = {row["date"]: row for _, row in orb.iterrows()}
        dates = df_mkt.index.date
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

            entry = float(row["Close"])

            # --- Quality scoring ---
            sma_val = sma50_lookup.get(date)
            cg = combined_grads.get(ticker, {}).get(date)

            # T factor based on mode
            if t_mode == "sma50":
                t_aligned = sma_val is not None and (
                    (direction == "long" and entry > sma_val) or
                    (direction == "short" and entry < sma_val)
                )
            elif t_mode == "sma50+grad":
                t_aligned = sma_val is not None and cg is not None and (
                    (direction == "long" and entry > sma_val and cg > 0) or
                    (direction == "short" and entry < sma_val and cg < 0)
                )
            elif t_mode == "grad_only":
                t_aligned = cg is not None and (
                    (direction == "long" and cg > 0) or
                    (direction == "short" and cg < 0)
                )
            else:
                t_aligned = False

            # M factor
            iso_week = date.isocalendar()[:2]
            mon_data = mon_lookup.get(iso_week)
            day_open = float(df_mkt[df_mkt.index.date == date].iloc[0]["Open"])
            inside_monday = (mon_data is not None and
                             date.weekday() != 0 and
                             mon_data["mon_low"] <= day_open <= mon_data["mon_high"])

            # B factor
            breakout_pct = (entry - orb_high) / entry * 100 if direction == "long" else (orb_low - entry) / entry * 100
            is_breakout = breakout_pct >= 1.0

            # V factor
            avg_vol = vol_lookup.get(date)
            cur_vol = float(row["Volume"])
            is_vol_spike = avg_vol is not None and avg_vol > 0 and cur_vol / avg_vol >= 2.0

            quality = sum([t_aligned, inside_monday, is_breakout, is_vol_spike])

            all_signals.append({
                "ticker": ticker, "date": date,
                "time": ts.strftime("%H:%M:%S"),
                "direction": direction, "entry_price": entry,
                "orb_high": orb_high, "orb_low": orb_low,
                "quality": quality,
                "T": t_aligned, "M": inside_monday, "B": is_breakout, "V": is_vol_spike,
            })
            seen.add(date)

    return pd.DataFrame(all_signals)


def simulate(signals_df, data_5m, combined_grads, min_quality=2,
             max_hold=5, ko_buffer_pct=1.0, position_sek=500, same_day_only=False):
    """Simulate turbo + runner with Combined gradient > 0.2 for overnight."""
    buffer_frac = ko_buffer_pct / 100
    threshold = 0.2
    results = []

    filtered = signals_df[signals_df["quality"] >= min_quality]

    for _, sig in filtered.iterrows():
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

        hold_limit = 0 if same_day_only else max_hold
        for day_offset in range(hold_limit + 1):
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
                    day_data = day_data[day_data.index.strftime("%H:%M:%S") > sig_time]
                if len(day_data) == 0:
                    break

            done = False
            for _, candle in day_data.iterrows():
                if direction == "long":
                    if candle["Low"] <= ko_level:
                        outcome, exit_price, done = "ko", ko_level, True
                        break
                    if sim_date == trade_date and candle["Close"] < orb_low:
                        outcome, exit_price, done = "stop", float(candle["Close"]), True
                        break
                else:
                    if candle["High"] >= ko_level:
                        outcome, exit_price, done = "ko", ko_level, True
                        break
                    if sim_date == trade_date and candle["Close"] > orb_high:
                        outcome, exit_price, done = "stop", float(candle["Close"]), True
                        break

            if done:
                break

            eod_price = float(day_data.iloc[-1]["Close"])

            if same_day_only or day_offset >= max_hold:
                exit_price = eod_price
                break

            grad_val = combined_grads.get(ticker, {}).get(sim_date)
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
            if direction == "long":
                pnl = (exit_price - entry) / entry * position_sek
            else:
                pnl = (entry - exit_price) / entry * position_sek

        results.append({
            "ticker": ticker, "direction": direction, "outcome": outcome,
            "pnl": pnl, "days_held": days_held,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


def print_results(label, res, position_sek=500):
    if len(res) == 0:
        print(f"  {label:<50} {'NO TRADES':>7}")
        return
    n = len(res)
    ko = (res["outcome"] == "ko").mean() * 100
    wr = (res["pnl"] > 0).mean() * 100
    roi = res["pnl"].sum() / (n * position_sek) * 100
    pnl = res["pnl"].sum()
    ovr = (res["days_held"] > 1).mean() * 100
    res["days_held"].mean()

    long_res = res[res["direction"] == "long"]
    short_res = res[res["direction"] == "short"]
    l_wr = (long_res["pnl"] > 0).mean() * 100 if len(long_res) > 0 else 0
    s_wr = (short_res["pnl"] > 0).mean() * 100 if len(short_res) > 0 else 0
    l_roi = long_res["pnl"].sum() / (len(long_res) * position_sek) * 100 if len(long_res) > 0 else 0
    s_roi = short_res["pnl"].sum() / (len(short_res) * position_sek) * 100 if len(short_res) > 0 else 0

    print(f"  {label:<50} {n:>5} ({len(long_res)}L/{len(short_res)}S)  KO={ko:>5.1f}%  WR={wr:>5.1f}%  ROI={roi:>+6.1f}%  PnL={pnl:>+8.0f}  O/N={ovr:>5.1f}%")
    print(f"  {'':<50}    Long:  WR={l_wr:>5.1f}%  ROI={l_roi:>+6.1f}%")
    print(f"  {'':<50}    Short: WR={s_wr:>5.1f}%  ROI={s_roi:>+6.1f}%")


def main():
    print("=" * 110)
    print("  TREND FILTER COMPARISON: T quality factor definitions")
    print("  Testing whether requiring combinedGrad > 0 for T=1 improves results")
    print("=" * 110)

    data_5m = fetch_intraday_5m(ALL_TICKERS)
    daily_data = fetch_daily(ALL_TICKERS)

    print("\nPrecomputing combined gradients...")
    combined_grads = precompute_combined_grad(daily_data)

    # Build signals for each T mode
    t_modes = {
        "A. Current: price vs SMA50":          "sma50",
        "B. Proposed: SMA50 + combinedGrad>0": "sma50+grad",
        "C. Alternative: combinedGrad only":   "grad_only",
    }

    for min_q in [2]:
        print(f"\n{'=' * 110}")
        print(f"  MIN QUALITY >= {min_q}")
        print(f"{'=' * 110}")

        for sim_label, same_day in [("TURBO (same-day only)", True), ("RUNNER (combined grad > 0.2)", False)]:
            print(f"\n  --- {sim_label} ---")
            print(f"  {'T factor definition':<50} {'Trades':>5}         {'KO':>6}   {'WR':>6}   {'ROI':>7}   {'PnL':>9}   {'O/N':>6}")
            print("  " + "─" * 105)

            for label, mode in t_modes.items():
                sigs = build_signals(data_5m, daily_data, combined_grads, t_mode=mode)
                res = simulate(sigs, data_5m, combined_grads, min_quality=min_q, same_day_only=same_day)
                print_results(label, res)
            print()

    # ══════════════════════════════════════════════════════════════════════════
    # KEY ANALYSIS: Split signals by combined gradient direction
    # User question: should we avoid longs when all major SMAs point down?
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print("  CORE QUESTION: Performance split by combined gradient direction")
    print("  (Using current T=sma50 signal set, quality >= 2)")
    print(f"{'=' * 110}")

    sigs_all = build_signals(data_5m, daily_data, combined_grads, t_mode="sma50")

    # Add combined gradient value to each signal
    sigs_all["cg"] = sigs_all.apply(
        lambda r: combined_grads.get(r["ticker"], {}).get(r["date"], 0.0), axis=1)

    q2 = sigs_all[sigs_all["quality"] >= 2].copy()

    # Split: gradient agrees vs disagrees with trade direction
    q2["grad_agrees"] = ((q2["direction"] == "long") & (q2["cg"] > 0)) | \
                        ((q2["direction"] == "short") & (q2["cg"] < 0))

    agrees = q2[q2["grad_agrees"]]
    disagrees = q2[~q2["grad_agrees"]]

    for label, subset in [("ALL quality>=2", q2),
                          ("Gradient AGREES with direction", agrees),
                          ("Gradient DISAGREES with direction", disagrees)]:
        print(f"\n  --- {label} ---")
        for sim_label, same_day in [("Turbo", True), ("Runner", False)]:
            res = simulate(subset, data_5m, combined_grads, min_quality=0,
                          same_day_only=same_day)
            print_results(f"  {sim_label}", res)
        print()

    # Break it down further: longs only
    print(f"\n{'=' * 110}")
    print("  LONGS ONLY — gradient agrees vs disagrees")
    print(f"{'=' * 110}")

    longs = q2[q2["direction"] == "long"]
    longs_agree = longs[longs["grad_agrees"]]
    longs_disagree = longs[~longs["grad_agrees"]]

    for label, subset in [("ALL longs quality>=2", longs),
                          ("Longs + gradient > 0 (agrees)", longs_agree),
                          ("Longs + gradient < 0 (DISAGREES)", longs_disagree)]:
        print(f"\n  --- {label} ---")
        for sim_label, same_day in [("Turbo", True), ("Runner", False)]:
            res = simulate(subset, data_5m, combined_grads, min_quality=0,
                          same_day_only=same_day)
            print_results(f"  {sim_label}", res)
        print()

    # And shorts
    print(f"\n{'=' * 110}")
    print("  SHORTS ONLY — gradient agrees vs disagrees")
    print(f"{'=' * 110}")

    shorts = q2[q2["direction"] == "short"]
    shorts_agree = shorts[shorts["grad_agrees"]]
    shorts_disagree = shorts[~shorts["grad_agrees"]]

    for label, subset in [("ALL shorts quality>=2", shorts),
                          ("Shorts + gradient < 0 (agrees)", shorts_agree),
                          ("Shorts + gradient > 0 (DISAGREES)", shorts_disagree)]:
        print(f"\n  --- {label} ---")
        for sim_label, same_day in [("Turbo", True), ("Runner", False)]:
            res = simulate(subset, data_5m, combined_grads, min_quality=0,
                          same_day_only=same_day)
            print_results(f"  {sim_label}", res)
        print()


if __name__ == "__main__":
    main()
