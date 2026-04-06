"""
Test different combined gradient thresholds for the T quality factor.

Current: T=1 when price vs SMA50 AND combinedGrad > 0
Question: should we require combinedGrad > 0.1, 0.2, 0.3 etc.?
"""
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import pandas as pd
from orb_monday_range import (
    ALL_TICKERS,
    RESULTS_DIR,
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
    results = {}
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
        results[ticker] = by_date
    return results


def build_and_simulate(data_5m, daily_data, combined_grads, t_grad_threshold=0.0,
                       min_quality=2, runner=False, max_hold=5,
                       ko_buffer_pct=1.0, position_sek=500):
    """Build signals with given T threshold and simulate."""
    buffer_frac = ko_buffer_pct / 100
    runner_threshold = 0.2  # combined grad threshold for overnight hold

    all_results = []

    for ticker in sorted(data_5m.keys()):
        df = data_5m[ticker]
        orb = compute_orb(df)
        if orb.empty:
            continue

        df_mkt = _market_hours(_to_ny(df))
        dd = daily_data.get(ticker)
        if dd is None:
            continue

        sma50 = dd["Close"].rolling(50).mean()
        sma50_lookup = {}
        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            if pd.notna(sma50.iloc[i]):
                sma50_lookup[d] = float(sma50.iloc[i])

        mon = compute_monday_range(dd)
        mon_lookup = {}
        if not mon.empty:
            for _, mr in mon.iterrows():
                ws = mr["week_start"]
                key = ws.isocalendar()[:2] if hasattr(ws, 'isocalendar') else ws
                mon_lookup[key] = mr

        vol = dd["Volume"].rolling(20).mean()
        vol_lookup = {}
        for i in range(len(dd)):
            d = dd.index[i].date() if hasattr(dd.index[i], 'date') else dd.index[i]
            if pd.notna(vol.iloc[i]):
                vol_lookup[d] = float(vol.iloc[i])

        orb_lookup = {row["date"]: row for _, row in orb.iterrows()}
        dates = df_mkt.index.date
        dates_available = sorted(set(dates))
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

            # Quality scoring
            sma_val = sma50_lookup.get(date)
            cg = combined_grads.get(ticker, {}).get(date, 0.0)

            # T: price vs SMA50 AND combined gradient above threshold
            t_aligned = sma_val is not None and (
                (direction == "long" and entry > sma_val and cg > t_grad_threshold) or
                (direction == "short" and entry < sma_val and cg < -t_grad_threshold)
            )

            # M
            iso_week = date.isocalendar()[:2]
            mon_data = mon_lookup.get(iso_week)
            day_open = float(df_mkt[df_mkt.index.date == date].iloc[0]["Open"])
            inside_monday = (mon_data is not None and date.weekday() != 0 and
                             mon_data["mon_low"] <= day_open <= mon_data["mon_high"])

            # B
            breakout_pct = ((entry - orb_high) / entry * 100 if direction == "long"
                           else (orb_low - entry) / entry * 100)
            is_breakout = breakout_pct >= 1.0

            # V
            avg_vol = vol_lookup.get(date)
            cur_vol = float(row["Volume"])
            is_vol_spike = avg_vol is not None and avg_vol > 0 and cur_vol / avg_vol >= 2.0

            quality = sum([t_aligned, inside_monday, is_breakout, is_vol_spike])
            if quality < min_quality:
                continue

            # ── Simulate trade ──
            if date not in dates_available:
                continue
            start_idx = dates_available.index(date)
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
                day_data = df_mkt[df_mkt.index.date == sim_date]
                if len(day_data) == 0:
                    break

                if sim_date == date:
                    day_data = day_data[day_data.index.strftime("%H:%M:%S") > ts.strftime("%H:%M:%S")]
                    if len(day_data) == 0:
                        break

                done = False
                for _, candle in day_data.iterrows():
                    if direction == "long":
                        if candle["Low"] <= ko_level:
                            outcome, exit_price, done = "ko", ko_level, True
                            break
                        if sim_date == date and candle["Close"] < orb_low:
                            outcome, exit_price, done = "stop", float(candle["Close"]), True
                            break
                    else:
                        if candle["High"] >= ko_level:
                            outcome, exit_price, done = "ko", ko_level, True
                            break
                        if sim_date == date and candle["Close"] > orb_high:
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

            all_results.append({
                "ticker": ticker, "direction": direction, "outcome": outcome,
                "pnl": pnl, "days_held": days_held,
            })

    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


def main():
    print("=" * 100)
    print("  T FACTOR THRESHOLD: How strict should combinedGrad be for T=1?")
    print("  Testing combinedGrad > 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5")
    print("=" * 100)

    data_5m = fetch_intraday_5m(ALL_TICKERS)
    daily_data = fetch_daily(ALL_TICKERS)

    print("\nPrecomputing combined gradients...")
    combined_grads = precompute_combined_grad(daily_data)

    thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    # Also test "no combined grad check" (original T = price vs SMA50 only)
    # We do this by setting threshold to -999 (always passes)

    print(f"\n{'=' * 100}")
    print("  TURBO (same-day only), quality >= 2")
    print(f"{'=' * 100}")
    print(f"  {'T threshold':<25} {'Trades':>7} {'Longs':>7} {'Shorts':>7} {'KO%':>7} {'WR%':>7} {'ROI%':>8} {'PnL':>10} {'L_WR%':>7} {'L_ROI%':>8} {'S_WR%':>7} {'S_ROI%':>8}")
    print("  " + "─" * 95)

    turbo_results = []
    for thresh in [-999.0] + thresholds:
        label = "No grad check (old)" if thresh == -999.0 else f"combinedGrad > {thresh}"
        res = build_and_simulate(data_5m, daily_data, combined_grads,
                                 t_grad_threshold=thresh if thresh != -999.0 else -999.0,
                                 min_quality=2, runner=False)
        if len(res) == 0:
            print(f"  {label:<25} {'NO TRADES':>7}")
            continue

        n = len(res)
        longs = res[res["direction"] == "long"]
        shorts = res[res["direction"] == "short"]
        ko = (res["outcome"] == "ko").mean() * 100
        wr = (res["pnl"] > 0).mean() * 100
        roi = res["pnl"].sum() / (n * 500) * 100
        pnl = res["pnl"].sum()
        l_wr = (longs["pnl"] > 0).mean() * 100 if len(longs) > 0 else 0
        l_roi = longs["pnl"].sum() / (len(longs) * 500) * 100 if len(longs) > 0 else 0
        s_wr = (shorts["pnl"] > 0).mean() * 100 if len(shorts) > 0 else 0
        s_roi = shorts["pnl"].sum() / (len(shorts) * 500) * 100 if len(shorts) > 0 else 0

        print(f"  {label:<25} {n:>7} {len(longs):>7} {len(shorts):>7} {ko:>6.1f}% {wr:>6.1f}% {roi:>+7.1f}% {pnl:>+9.0f} {l_wr:>6.1f}% {l_roi:>+7.1f}% {s_wr:>6.1f}% {s_roi:>+7.1f}%")

        turbo_results.append({"threshold": thresh, "label": label, "trades": n,
                              "longs": len(longs), "shorts": len(shorts),
                              "ko_pct": ko, "wr_pct": wr, "roi_pct": roi, "pnl": pnl,
                              "long_wr": l_wr, "long_roi": l_roi, "short_wr": s_wr, "short_roi": s_roi})

    print(f"\n{'=' * 100}")
    print("  RUNNER (combined grad > 0.2 for overnight hold), quality >= 2")
    print(f"{'=' * 100}")
    print(f"  {'T threshold':<25} {'Trades':>7} {'Longs':>7} {'Shorts':>7} {'KO%':>7} {'WR%':>7} {'ROI%':>8} {'PnL':>10} {'L_WR%':>7} {'L_ROI%':>8} {'S_WR%':>7} {'S_ROI%':>8}")
    print("  " + "─" * 95)

    runner_results = []
    for thresh in [-999.0] + thresholds:
        label = "No grad check (old)" if thresh == -999.0 else f"combinedGrad > {thresh}"
        res = build_and_simulate(data_5m, daily_data, combined_grads,
                                 t_grad_threshold=thresh if thresh != -999.0 else -999.0,
                                 min_quality=2, runner=True)
        if len(res) == 0:
            print(f"  {label:<25} {'NO TRADES':>7}")
            continue

        n = len(res)
        longs = res[res["direction"] == "long"]
        shorts = res[res["direction"] == "short"]
        ko = (res["outcome"] == "ko").mean() * 100
        wr = (res["pnl"] > 0).mean() * 100
        roi = res["pnl"].sum() / (n * 500) * 100
        pnl = res["pnl"].sum()
        l_wr = (longs["pnl"] > 0).mean() * 100 if len(longs) > 0 else 0
        l_roi = longs["pnl"].sum() / (len(longs) * 500) * 100 if len(longs) > 0 else 0
        s_wr = (shorts["pnl"] > 0).mean() * 100 if len(shorts) > 0 else 0
        s_roi = shorts["pnl"].sum() / (len(shorts) * 500) * 100 if len(shorts) > 0 else 0

        print(f"  {label:<25} {n:>7} {len(longs):>7} {len(shorts):>7} {ko:>6.1f}% {wr:>6.1f}% {roi:>+7.1f}% {pnl:>+9.0f} {l_wr:>6.1f}% {l_roi:>+7.1f}% {s_wr:>6.1f}% {s_roi:>+7.1f}%")

        runner_results.append({"threshold": thresh, "label": label, "trades": n,
                               "longs": len(longs), "shorts": len(shorts),
                               "ko_pct": ko, "wr_pct": wr, "roi_pct": roi, "pnl": pnl,
                               "long_wr": l_wr, "long_roi": l_roi, "short_wr": s_wr, "short_roi": s_roi})

    # Save
    pd.DataFrame(turbo_results).to_csv(RESULTS_DIR / "trend_threshold_turbo.csv", index=False)
    pd.DataFrame(runner_results).to_csv(RESULTS_DIR / "trend_threshold_runner.csv", index=False)
    print(f"\n  Saved to {RESULTS_DIR}/trend_threshold_*.csv")


if __name__ == "__main__":
    main()
