"""Does timing the entry on intraday bars beat buying at the daily close? Same daily RSI(14)<30 signal, same exit
(the close 5 trading days after the signal day); only the entry moment changes.

    E0 close        buy at the signal day's close (the current rule)
    E1 next open    buy at the next session's first bar
    E2 early        buy on the first intraday bar that closes below the RSI-30 trigger price, whether or not the
                    day ends below it (you can't know that yet)
    E3 reversal     after the signal, buy on the first bar that closes above the previous bar's high within the next
                    2 sessions; no trade if it never happens
    E4 limit dip    next session, limit order half a day's typical move below the signal close; if unfilled, buy at
                    that session's close
Yahoo keeps 730 days of 1-hour bars and 60 days of 30-minute bars. Daily closes, RSI and the trigger are built
from the same intraday bars, because Yahoo's daily history is dividend-adjusted and its intraday prices are not.
Knock-outs are checked on intraday lows.
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from data import CACHE_DIR, daily
from indicators import prior_value, rsi
from short_screen import universe

RESULTS_DIR = Path(__file__).parent / "results"
HOLD = 5
N = 14
OVERSOLD = 30
BULL_LEVERAGE, OTHER_LEVERAGE, HIGH_VIX = 20, 5, 22
FINANCING = 0.065
SPREAD = 0.005
PERIODS = {"60m": "730d", "30m": "60d"}


def intraday(tickers, interval, chunk=50):
    frames, missing = {}, []
    for t in tickers:
        path = CACHE_DIR / f"{t}_{interval}_{date.today():%Y%m%d}.pkl"
        (frames.__setitem__(t, pd.read_pickle(path)) if path.exists() else missing.append(t))
    for exchange_tz, group in (("America/New_York", [t for t in missing if not t.endswith(".ST")]),
                               ("Europe/Stockholm", [t for t in missing if t.endswith(".ST")])):
        for i in range(0, len(group), chunk):
            batch = group[i:i + chunk]
            raw = yf.download(batch, period=PERIODS[interval], interval=interval, auto_adjust=True, progress=False,
                              group_by="ticker", threads=True, prepost=False)
            for t in batch:
                if t not in raw.columns.get_level_values(0):
                    continue
                df = raw[t].dropna(subset=["Close"]).copy()
                df.columns = [c.lower() for c in df.columns]
                if len(df) < 50:
                    continue
                df.index = df.index.tz_convert(exchange_tz)
                df.to_pickle(CACHE_DIR / f"{t}_{interval}_{date.today():%Y%m%d}.pkl")
                frames[t] = df
    return frames


def daily_from_bars(bars):
    local_day = bars.index.tz_localize(None).normalize()
    return bars.groupby(local_day).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                       close=("close", "last"))


def has_split_jump(d):
    return bool((np.log(d.close).diff().abs() > 0.4).any())


def daily_context(d, spy_above, vix):
    c = d.close
    change = c.diff()
    def smooth(s):
        return s.ewm(alpha=1 / N, adjust=False, min_periods=N).mean()
    gain, loss = smooth(change.clip(lower=0)), smooth((-change).clip(lower=0))
    ratio = OVERSOLD / (100 - OVERSOLD)
    ctx = pd.DataFrame({"close": c, "rsi": rsi(c), "vol": np.log(c).diff().rolling(20).std()})
    ctx["trigger"] = (c - (N - 1) * (gain / ratio - loss)).shift(1).where(ctx.rsi.shift(1) >= OVERSOLD)
    ctx["signal"] = (ctx.rsi < OVERSOLD) & (ctx.rsi.shift(1) >= OVERSOLD)
    bull = (prior_value(spy_above, d.index) == 1) & (prior_value(vix, d.index) < HIGH_VIX)
    ctx["leverage"] = np.where(bull, BULL_LEVERAGE, OTHER_LEVERAGE)
    return ctx


def turbo(entry, exit_price, lows_after, days_held, leverage):
    f0 = entry * (1 - 1 / leverage)
    financing = f0 * (1 + FINANCING * days_held / 365)
    if len(lows_after) and lows_after.min() <= financing:
        return -1.0, True
    return (exit_price - financing) / (entry - f0) * (1 - SPREAD) - 1, False


def trades_for(ticker, ctx, bars):
    local_day = bars.index.tz_localize(None).normalize()
    days = pd.DatetimeIndex(sorted(set(local_day) & set(ctx.index)))
    if len(days) < HOLD + 3:
        return []
    ctx = ctx.loc[days[0]:]
    pos = {d: k for k, d in enumerate(ctx.index)}
    by_day = {d: bars[local_day == d] for d in days}
    rows, busy_until = [], {}

    def record(variant, entry_day, entry_bar_time, entry_price, signal_day, confirmed):
        k = pos[signal_day]
        if k + HOLD >= len(ctx):
            return
        exit_day = ctx.index[k + HOLD]
        if exit_day not in by_day:
            return
        exit_price = ctx.close.iloc[k + HOLD]
        window = bars[(bars.index > entry_bar_time) & (local_day <= exit_day)]
        lev = ctx.leverage.iloc[k]
        t_ret, knocked = turbo(entry_price, exit_price, window.low.to_numpy(), (exit_day - entry_day).days, lev)
        t5, ko5 = turbo(entry_price, exit_price, window.low.to_numpy(), (exit_day - entry_day).days, 5)
        rows.append({"ticker": ticker, "variant": variant, "signal_day": signal_day, "confirmed": confirmed,
                     "entry": entry_price, "ret": exit_price / entry_price - 1, "leverage": lev,
                     "turbo": t_ret, "ko": knocked, "turbo5": t5, "ko5": ko5})

    for k, day in enumerate(ctx.index):
        if day not in by_day or k + 2 >= len(ctx):
            continue
        today_bars = by_day[day]
        trigger = ctx.trigger.iloc[k]
        if not np.isnan(trigger) and busy_until.get("E2", pd.Timestamp.min) < day:
            below = today_bars[today_bars.close < trigger]
            if len(below):
                bar = below.iloc[0]
                record("E2 early", day, below.index[0], bar.close, day, bool(ctx.signal.iloc[k]))
                busy_until["E2"] = ctx.index[min(k + HOLD, len(ctx) - 1)]
        if not ctx.signal.iloc[k] or busy_until.get("confirmed", pd.Timestamp.min) >= day:
            continue
        busy_until["confirmed"] = ctx.index[min(k + HOLD, len(ctx) - 1)]
        next_day = ctx.index[k + 1]
        if next_day not in by_day:
            continue
        next_bars = by_day[next_day]
        record("E0 close", day, today_bars.index[-1], ctx.close.iloc[k], day, True)
        record("E1 next open", next_day, next_bars.index[0] - pd.Timedelta(seconds=1), next_bars.open.iloc[0], day, True)

        after = pd.concat([today_bars.iloc[-1:], next_bars, by_day.get(ctx.index[k + 2], next_bars.iloc[:0])])
        broke = after.close.to_numpy()[1:] > after.high.to_numpy()[:-1]
        if broke.any():
            j = int(np.argmax(broke)) + 1
            record("E3 reversal", after.index[j].tz_localize(None).normalize(), after.index[j], after.close.iloc[j], day, True)

        limit = ctx.close.iloc[k] * (1 - 0.5 * ctx.vol.iloc[k])
        hit = next_bars[next_bars.low <= limit]
        if len(hit):
            fill = min(limit, hit.open.iloc[0]) if hit.index[0] == next_bars.index[0] else limit
            record("E4 limit dip", next_day, hit.index[0], fill, day, True)
        else:
            record("E4 limit dip", next_day, next_bars.index[-1], next_bars.close.iloc[-1], day, True)
    return rows


def summarise(t):
    return pd.Series({"trades": len(t), "win_%": (t.ret > 0).mean() * 100, "stock_%": t.ret.mean() * 100,
                      "turbo_rule_%": t.turbo.mean() * 100, "ko_rule_%": t.ko.mean() * 100,
                      "turbo5x_%": t.turbo5.mean() * 100, "ko5x_%": t.ko5.mean() * 100})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--interval", default="60m", choices=list(PERIODS))
    args = p.parse_args()
    bars = intraday(universe(), args.interval)
    spy = daily("SPY").close
    spy_above = (spy > spy.rolling(200).mean()).astype(float)
    vix = daily("^VIX").close
    rows, skipped = [], []
    for t, b in bars.items():
        d = daily_from_bars(b)
        if has_split_jump(d) or len(d) < 60:
            skipped.append(t)
            continue
        rows += trades_for(t, daily_context(d, spy_above, vix), b)
    print(f"skipped {len(skipped)} stocks with a split-sized jump or too little history: {', '.join(skipped[:15])}")
    trades = pd.DataFrame(rows)
    trades.to_csv(RESULTS_DIR / f"entry_timing_{args.interval}.csv", index=False)
    print(f"{args.interval} bars: {len(bars)} stocks, {trades.signal_day.min():%Y-%m-%d} → {trades.signal_day.max():%Y-%m-%d}, "
          f"{int((trades.variant == 'E0 close').sum())} confirmed signals\n")

    pd.set_option("display.width", 200)
    print("== All trades per entry rule (exit: close 5 trading days after the signal day)")
    print(trades.groupby("variant").apply(summarise, include_groups=False).round(2).to_string())

    e2 = trades[trades.variant == "E2 early"]
    print("\n== E2 early entries split by whether the day closed below the trigger")
    print(e2.groupby("confirmed").apply(summarise, include_groups=False).round(2).to_string())

    base = trades[trades.variant == "E0 close"].set_index(["ticker", "signal_day"])
    print("\n== Same signals, entry rule vs buying at the close (paired)")
    for variant in ("E1 next open", "E2 early", "E3 reversal", "E4 limit dip"):
        v = trades[(trades.variant == variant) & trades.confirmed].set_index(["ticker", "signal_day"])
        common = v.index.intersection(base.index)
        diff = (v.loc[common, "ret"] - base.loc[common, "ret"]) * 100
        tdiff = (v.loc[common, "turbo"] - base.loc[common, "turbo"]) * 100
        print(f"  {variant:13s} n={len(common):4d}  stock {diff.mean():+.2f} pts (t {diff.mean() / diff.std() * np.sqrt(len(diff)):+.1f})  "
              f"turbo at rule leverage {tdiff.mean():+.2f} pts  fill rate {len(common) / len(base):.0%}")


if __name__ == "__main__":
    main()
