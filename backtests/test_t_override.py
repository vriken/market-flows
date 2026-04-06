"""
Test override rules for when T=0 (gradient disagrees) but other factors suggest
a strong momentum move worth taking anyway.

Override candidates:
  1. Big breakout (>= 2%) + volume spike (>= 2x)
  2. Gap above/below Monday range at open
  3. Very large breakout alone (>= 3%)
  4. Breakout >= 2% + gap beyond Monday range
  5. All of B + V + M (everything except T)

Uses cached data to speed up repeated runs.
"""
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import pickle
from pathlib import Path

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

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

SMA_PERIODS = [20, 50, 100, 150, 200, 250, 300, 400, 500, 600]
LOOKBACK = 5


def cached_fetch(name, fetch_fn, *args, max_age_hours=4):
    """Cache data fetches to disk."""
    cache_file = CACHE_DIR / f"{name}.pkl"
    if cache_file.exists():
        age_h = (pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime, unit="s")).total_seconds() / 3600
        if age_h < max_age_hours:
            with open(cache_file, "rb") as f:
                print(f"  Using cached {name} ({age_h:.1f}h old)")
                return pickle.load(f)
    data = fetch_fn(*args)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    return data


def precompute_all(daily_data):
    """Precompute combined gradient, SMA50, vol SMA, monday range for all tickers."""
    combined_grads = {}
    sma50_lookup = {}
    vol_lookup = {}
    mon_lookup = {}

    for ticker, dd in daily_data.items():
        # SMA50
        sma50 = dd["Close"].rolling(50).mean()
        s50 = {}
        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            if pd.notna(sma50.iloc[i]):
                s50[d] = float(sma50.iloc[i])
        sma50_lookup[ticker] = s50

        # Volume SMA
        vol = dd["Volume"].rolling(20).mean()
        vl = {}
        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            if pd.notna(vol.iloc[i]):
                vl[d] = float(vol.iloc[i])
        vol_lookup[ticker] = vl

        # Monday range
        mon = compute_monday_range(dd)
        ml = {}
        if not mon.empty:
            for _, mr in mon.iterrows():
                ws = mr["week_start"]
                key = ws.isocalendar()[:2] if hasattr(ws, 'isocalendar') else ws
                ml[key] = {"mon_high": mr["mon_high"], "mon_low": mr["mon_low"]}
        mon_lookup[ticker] = ml

        # Combined gradient
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
            tsi_num = sum(grads[p] * (1.0 / p) for p in grads)
            tsi_den = sum(abs(grads[p]) * (1.0 / p) for p in grads)
            tsi = tsi_num / tsi_den if tsi_den > 0 else 0.0
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
            pv_num = sum((1.0 / p) * (1 if close_val > sma_vals[p] else -1) for p in sma_vals)
            pv_den = sum(1.0 / p for p in sma_vals)
            pv = pv_num / pv_den if pv_den > 0 else 0.0
            by_date[d] = (tsi + stack + pv) / 3.0
        combined_grads[ticker] = by_date

    return combined_grads, sma50_lookup, vol_lookup, mon_lookup


def build_all_signals(data_5m, combined_grads, sma50_lookup, vol_lookup, mon_lookup):
    """Build all ORB signals with raw factor values (no filtering)."""
    all_signals = []
    for ticker in sorted(data_5m.keys()):
        df = data_5m[ticker]
        orb = compute_orb(df)
        if orb.empty:
            continue
        df_mkt = _market_hours(_to_ny(df))
        if len(df_mkt) == 0:
            continue

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
            seen.add(date)

            # Raw factor values
            sma_val = sma50_lookup.get(ticker, {}).get(date)
            cg = combined_grads.get(ticker, {}).get(date, 0.0)
            avg_vol = vol_lookup.get(ticker, {}).get(date)
            cur_vol = float(row["Volume"])
            vol_ratio = cur_vol / avg_vol if avg_vol and avg_vol > 0 else 0.0

            iso_week = date.isocalendar()[:2]
            mon_data = mon_lookup.get(ticker, {}).get(iso_week)
            day_bars = df_mkt[df_mkt.index.date == date]
            day_open = float(day_bars.iloc[0]["Open"])

            breakout_pct = (entry - orb_high) / entry * 100 if direction == "long" else (orb_low - entry) / entry * 100

            # Factor booleans
            above_sma50 = sma_val is not None and entry > sma_val
            below_sma50 = sma_val is not None and entry < sma_val
            t_aligned = (direction == "long" and above_sma50 and cg > 0) or \
                       (direction == "short" and below_sma50 and cg < 0)
            inside_monday = (mon_data is not None and date.weekday() != 0 and
                           mon_data["mon_low"] <= day_open <= mon_data["mon_high"])
            beyond_monday = False
            if mon_data is not None and date.weekday() != 0:
                beyond_monday = (direction == "long" and day_open > mon_data["mon_high"]) or \
                               (direction == "short" and day_open < mon_data["mon_low"])
            is_breakout = breakout_pct >= 1.0
            is_big_breakout = breakout_pct >= 2.0
            is_huge_breakout = breakout_pct >= 3.0
            is_vol_spike = vol_ratio >= 2.0

            all_signals.append({
                "ticker": ticker, "date": date, "time": ts.strftime("%H:%M:%S"),
                "direction": direction, "entry_price": entry,
                "orb_high": orb_high, "orb_low": orb_low,
                "cg": cg, "breakout_pct": breakout_pct, "vol_ratio": vol_ratio,
                "T": t_aligned, "M": inside_monday, "B": is_breakout,
                "V": is_vol_spike, "B2": is_big_breakout, "B3": is_huge_breakout,
                "beyond_mon": beyond_monday,
            })

    return pd.DataFrame(all_signals)


def simulate_batch(signals_df, data_5m, combined_grads, runner=False,
                   max_hold=5, ko_buffer_pct=1.0, position_sek=500):
    """Simulate a batch of signals. Returns DataFrame with results."""
    buffer_frac = ko_buffer_pct / 100
    runner_threshold = 0.2
    results = []

    # Pre-build market-hours data
    mkt_cache = {}
    for ticker in signals_df["ticker"].unique():
        if ticker in data_5m:
            mkt_cache[ticker] = _market_hours(_to_ny(data_5m[ticker]))

    for _, sig in signals_df.iterrows():
        ticker = sig["ticker"]
        direction = sig["direction"]
        entry = sig["entry_price"]
        orb_high = sig["orb_high"]
        orb_low = sig["orb_low"]
        trade_date = sig["date"]

        df = mkt_cache.get(ticker)
        if df is None or len(df) == 0:
            continue

        dates_available = sorted(set(df.index.date))
        if trade_date not in dates_available:
            continue
        start_idx = dates_available.index(trade_date)
        ko_level = entry * (1 - buffer_frac) if direction == "long" else entry * (1 + buffer_frac)

        outcome = "close"
        exit_price = entry
        days_held = 0
        hold_limit = max_hold if runner else 0

        for day_offset in range(hold_limit + 1):
            if start_idx + day_offset >= len(dates_available):
                break
            sim_date = dates_available[start_idx + day_offset]
            days_held = day_offset + 1
            day_data = df[df.index.date == sim_date]
            if len(day_data) == 0:
                break

            if sim_date == trade_date:
                day_data = day_data[day_data.index.strftime("%H:%M:%S") > sig["time"]]
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
            if not runner or day_offset >= max_hold:
                exit_price = eod_price
                break

            grad_val = combined_grads.get(ticker, {}).get(sim_date)
            if grad_val is None:
                exit_price = eod_price
                break
            favorable = ((direction == "long" and grad_val > runner_threshold) or
                        (direction == "short" and grad_val < -runner_threshold))
            if not favorable:
                exit_price = eod_price
                break

        if outcome == "ko":
            pnl = -position_sek
        else:
            pnl = ((exit_price - entry) / entry * position_sek if direction == "long"
                  else (entry - exit_price) / entry * position_sek)

        results.append({
            "ticker": ticker, "direction": direction, "outcome": outcome,
            "pnl": pnl, "days_held": days_held,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


def print_row(label, res, position_sek=500):
    if len(res) == 0:
        print(f"  {label:<45} {'—':>5}  {'—':>5}  {'—':>5}  {'—':>6}  {'—':>6}  {'—':>8}  {'—':>9}")
        return
    n = len(res)
    longs = res[res["direction"] == "long"]
    shorts = res[res["direction"] == "short"]
    ko = (res["outcome"] == "ko").mean() * 100
    wr = (res["pnl"] > 0).mean() * 100
    roi = res["pnl"].sum() / (n * position_sek) * 100
    pnl = res["pnl"].sum()
    l_wr = (longs["pnl"] > 0).mean() * 100 if len(longs) > 0 else 0
    s_wr = (shorts["pnl"] > 0).mean() * 100 if len(shorts) > 0 else 0
    print(f"  {label:<45} {n:>5} {len(longs):>5}L {len(shorts):>4}S  {ko:>5.1f}%  {wr:>5.1f}%  {roi:>+7.1f}%  {pnl:>+8.0f}  L:{l_wr:>4.0f}% S:{s_wr:>4.0f}%")


def main():
    print("=" * 115)
    print("  T OVERRIDE TEST: Can we rescue strong momentum signals when T=0?")
    print("=" * 115)

    data_5m = cached_fetch("5m_data", fetch_intraday_5m, ALL_TICKERS)
    daily_data = cached_fetch("daily_data", fetch_daily, ALL_TICKERS)

    print("\nPrecomputing factors...")
    combined_grads, sma50_lookup, vol_lookup, mon_lookup = precompute_all(daily_data)

    print("Building signals...")
    sigs = build_all_signals(data_5m, combined_grads, sma50_lookup, vol_lookup, mon_lookup)
    print(f"  Total signals: {len(sigs)}")

    # Compute quality with current rules (T requires combined grad > 0)
    sigs["quality"] = sigs["T"].astype(int) + sigs["M"].astype(int) + sigs["B"].astype(int) + sigs["V"].astype(int)

    # Signals that have quality >= 2 with current rules
    baseline = sigs[sigs["quality"] >= 2]

    # Signals that are BLOCKED by T=0 but would have quality >= 2 if T were granted
    # These are the ones we want to rescue with override rules
    t0_signals = sigs[(~sigs["T"]) & (sigs["quality"] < 2)]
    # Their quality WITHOUT T
    t0_quality_without = t0_signals["M"].astype(int) + t0_signals["B"].astype(int) + t0_signals["V"].astype(int)
    # Would be quality >= 2 if T were overridden to 1
    would_qualify = t0_signals[t0_quality_without >= 1]  # +1 from override = quality 2

    print(f"  Baseline (quality >= 2): {len(baseline)} signals")
    print(f"  Blocked by T=0 but would qualify with override: {len(would_qualify)} signals")

    # Define override rules
    overrides = {
        "0. Baseline (no override)": baseline,
        "1. Override: breakout>=2% + vol spike": pd.concat([baseline, t0_signals[t0_signals["B2"] & t0_signals["V"]]]),
        "2. Override: breakout>=2% + beyond Monday": pd.concat([baseline, t0_signals[t0_signals["B2"] & t0_signals["beyond_mon"]]]),
        "3. Override: breakout>=3%": pd.concat([baseline, t0_signals[t0_signals["B3"]]]),
        "4. Override: beyond Monday + vol spike": pd.concat([baseline, t0_signals[t0_signals["beyond_mon"] & t0_signals["V"]]]),
        "5. Override: beyond Monday + breakout>=1%": pd.concat([baseline, t0_signals[t0_signals["beyond_mon"] & t0_signals["B"]]]),
        "6. Override: breakout>=2% (alone)": pd.concat([baseline, t0_signals[t0_signals["B2"]]]),
        "7. Override: vol spike>=2x + breakout>=1%": pd.concat([baseline, t0_signals[t0_signals["V"] & t0_signals["B"]]]),
    }

    # Remove duplicates (a signal might already be in baseline)
    for k in overrides:
        overrides[k] = overrides[k].drop_duplicates(subset=["ticker", "date"])

    header = f"  {'Rule':<45} {'All':>5} {'L':>5}  {'S':>4}   {'KO%':>5}   {'WR%':>5}   {'ROI%':>7}   {'PnL':>8}  {'WR by dir'}"
    sep = "  " + "─" * 110

    print(f"\n{'=' * 115}")
    print("  TURBO (same-day)")
    print(f"{'=' * 115}")
    print(header)
    print(sep)
    for label, sig_set in overrides.items():
        res = simulate_batch(sig_set, data_5m, combined_grads, runner=False)
        print_row(label, res)
    print()

    # Show what each override adds
    print(f"\n{'=' * 115}")
    print("  WHAT EACH OVERRIDE ADDS (signals rescued from T=0 block)")
    print(f"{'=' * 115}")
    print(header)
    print(sep)
    for label, sig_set in overrides.items():
        if label.startswith("0."):
            continue
        # Signals in this set but NOT in baseline
        added = sig_set[~sig_set.set_index(["ticker", "date"]).index.isin(
            baseline.set_index(["ticker", "date"]).index)]
        if len(added) == 0:
            print(f"  {label:<45} (no new signals)")
            continue
        res = simulate_batch(added, data_5m, combined_grads, runner=False)
        print_row(f"  ADDED by {label[3:]}", res)
    print()

    print(f"\n{'=' * 115}")
    print("  RUNNER (combined grad > 0.2 overnight)")
    print(f"{'=' * 115}")
    print(header)
    print(sep)
    for label, sig_set in overrides.items():
        res = simulate_batch(sig_set, data_5m, combined_grads, runner=True)
        print_row(label, res)

    # Would today's AMZN be captured?
    print(f"\n{'=' * 115}")
    print("  AMZN-LIKE SIGNALS: T=0 + big breakout + beyond Monday")
    print(f"{'=' * 115}")
    amzn_like = t0_signals[t0_signals["B2"] & t0_signals["beyond_mon"]]
    if len(amzn_like) > 0:
        print(f"\n  Found {len(amzn_like)} signals matching AMZN-today pattern:")
        for _, s in amzn_like.iterrows():
            print(f"    {s['ticker']:>6} {s['date']} {s['time']} {s['direction']:>5}  "
                  f"entry={s['entry_price']:>8.2f}  brkout={s['breakout_pct']:>+5.1f}%  "
                  f"vol={s['vol_ratio']:>4.1f}x  cg={s['cg']:>+.2f}")
        res = simulate_batch(amzn_like, data_5m, combined_grads, runner=False)
        if len(res) > 0:
            print("\n  Turbo results:")
            print_row("AMZN-like signals", res)
    else:
        print("  No matching signals found")


if __name__ == "__main__":
    main()
