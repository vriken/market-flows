"""Does an earnings report during the 5-day hold hurt the v2 dip-buy? Fixed before looking: split v2 signals
(RSI(14) closes below 30, price above its 200-day) by whether a report's first price reaction falls inside the
hold, 2015–26, on the S&P 500 + OMX Stockholm 30 + watchlist. A report before the open reacts that day; one after
the close reacts the next session. Yahoo's earnings history starts in mid-2014.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from data import CACHE_DIR, daily
from indicators import prior_value, rsi
from short_screen import prices, universe

RESULTS_DIR = Path(__file__).parent / "results"
START = pd.Timestamp("2015-01-01")
HOLD = 5
MARGIN, SPREAD = 0.025, 0.005


def earnings_dates(ticker):
    path = CACHE_DIR / f"{ticker}_earnings_{date.today():%Y%m%d}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    try:
        e = yf.Ticker(ticker).get_earnings_dates(limit=60)
        stamps = pd.Series(e.index if e is not None else [], dtype="datetime64[ns, America/New_York]")
    except Exception:
        stamps = pd.Series([], dtype="datetime64[ns, America/New_York]")
    stamps.to_pickle(path)
    return stamps


def reaction_days(stamps, trading_days):
    days = []
    for ts in stamps:
        day = pd.Timestamp(ts.date())
        k = trading_days.searchsorted(day)
        if ts.hour >= 12:
            k = trading_days.searchsorted(day, side="right")
        if k < len(trading_days):
            days.append(trading_days[k])
    return pd.DatetimeIndex(sorted(set(days)))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    data = prices(universe())
    spy, vix, tbill = daily("SPY").close, daily("^VIX").close, daily("^IRX").close / 100
    spy_up = (spy > spy.rolling(200).mean()).astype(float)
    closes = pd.DataFrame({t: d.close for t, d in data.items()}).sort_index()
    avg_fwd = (closes.shift(-HOLD) / closes - 1).mean(axis=1)
    rows, covered = [], 0
    for n, (t, d) in enumerate(data.items(), 1):
        stamps = earnings_dates(t)
        if len(stamps):
            covered += 1
        react = reaction_days(stamps, d.index)
        c, lo = d.close.to_numpy(), d.low.to_numpy()
        day = d.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        r = rsi(d.close).to_numpy()
        above = (d.close > d.close.rolling(200).mean()).to_numpy()
        bull = (prior_value(spy_up, d.index) == 1) & (prior_value(vix, d.index) < 22)
        rate = np.nan_to_num(prior_value(tbill, d.index))
        k = 1
        while k < len(c) - HOLD:
            if d.index[k] >= START and r[k] < 30 <= r[k - 1] and above[k]:
                window = d.index[k + 1:k + HOLD + 1]
                lev = 20.0 if bull[k] else 5.0
                f0 = c[k] * (1 - 1 / lev)
                path = np.arange(k + 1, k + HOLD + 1)
                fin = f0 * (1 + (rate[k] + MARGIN) * (day[path] - day[k]) / 365)
                ko = bool((lo[path] <= fin).any())
                ret = c[k + HOLD] / c[k] - 1
                rows.append({"ticker": t, "date": d.index[k], "has_dates": len(stamps) > 0,
                             "earnings_in_hold": bool(react.isin(window).any()), "ret": ret,
                             "vs_avg": ret - avg_fwd.get(d.index[k], np.nan), "ko": ko, "leverage": lev,
                             "turbo": -1.0 if ko else (c[k + HOLD] - fin[-1]) / (c[k] - f0) * (1 - SPREAD) - 1})
                k += HOLD + 1
            else:
                k += 1
        if n % 100 == 0:
            print(f"  {n} stocks done")
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS_DIR / "earnings_test.csv", index=False)
    t = t[t.has_dates]
    def summary(g):
        return pd.Series({"signals": len(g), "win_%": (g.ret > 0).mean() * 100, "stock_%": g.ret.mean() * 100,
                                       "vs_avg_stock_%": g.vs_avg.mean() * 100, "turbo_%": g.turbo.mean() * 100,
                                       "ko_%": g.ko.mean() * 100})
    print(f"\n{covered} of {len(data)} stocks have earnings dates; {len(t)} v2 signals 2015–26 on those stocks\n")
    pd.set_option("display.width", 200)
    print("== v2 signals by whether an earnings reaction falls inside the 5-day hold")
    print(t.groupby("earnings_in_hold").apply(summary, include_groups=False).round(2).to_string())
    print("\n== Same, by year block")
    t["block"] = np.where(t.date < "2021-01-01", "2015–20", "2021–26")
    print(t.groupby(["block", "earnings_in_hold"]).apply(summary, include_groups=False).round(2).to_string())
    ko_share = t[t.ko].earnings_in_hold.mean()
    print(f"\nShare of knock-outs that had earnings in the hold: {ko_share:.0%} "
          f"(vs {t.earnings_in_hold.mean():.0%} of all signals)")


if __name__ == "__main__":
    main()
