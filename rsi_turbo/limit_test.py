"""Buy below the signal close? Limit orders under the close of a v2 signal (RSI(14) below 30, price above its
200-day), S&P 500 + OMX Stockholm 30 + watchlist, 2007–14 vs 2015–26. A limit fills when a day's low reaches it, at
the limit or at that day's open if it gapped below. Exit for all: the first close with RSI back at 40, at the latest
5 trading days after entry. Turbo at 20x in a bull market (previous day's SPY above its 200-day and VIX below 22),
5x otherwise, with the financing level set from the fill price; an intraday fill can be knocked out the same day.
"Per signal" counts unfilled signals as 0.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import daily
from indicators import prior_value, rsi
from short_screen import prices, universe

RESULTS_DIR = Path(__file__).parent / "results"
START, OOS_START = pd.Timestamp("2007-01-01"), pd.Timestamp("2015-01-01")
WAIT, HOLD, EXIT_RSI = 5, 5, 40
MARGIN, SPREAD = 0.025, 0.005
RULES = ("L0 signal close", "L1 limit −1%", "L2 limit −2%", "L3 limit −3%", "L4 limit −1 typical day",
         "L5 limit −1% next day, else next close")


def fill(rule, k, c, o, lo, vol):
    """(entry index, entry price, filled intraday) or None when the order never fills."""
    if rule == "L0 signal close":
        return k, c[k], False
    depth = {"L1 limit −1%": 0.01, "L2 limit −2%": 0.02, "L3 limit −3%": 0.03,
             "L4 limit −1 typical day": vol[k], "L5 limit −1% next day, else next close": 0.01}[rule]
    limit = c[k] * (1 - depth)
    last = k + 1 if rule.startswith("L5") else min(k + WAIT, len(c) - 1)
    for j in range(k + 1, last + 1):
        if lo[j] <= limit:
            return j, min(o[j], limit), True
    if rule.startswith("L5"):
        return k + 1, c[k + 1], False
    return None


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    data = prices(universe())
    spy, vix, tbill = daily("SPY").close, daily("^VIX").close, daily("^IRX").close / 100
    spy_up = (spy > spy.rolling(200).mean()).astype(float)
    rows = []
    for _t, d in data.items():
        if len(d) < 300:
            continue
        c, o, lo = d.close.to_numpy(), d.open.to_numpy(), d.low.to_numpy()
        day = d.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        r = rsi(d.close).to_numpy()
        vol = np.log(d.close).diff().rolling(20).std().to_numpy()
        above = (d.close > d.close.rolling(200).mean()).to_numpy()
        bull = (prior_value(spy_up, d.index) == 1) & (prior_value(vix, d.index) < 22)
        rate = np.nan_to_num(prior_value(tbill, d.index))
        signals = [k for k in range(1, len(c) - WAIT - HOLD - 1) if d.index[k] >= START and r[k] < 30 <= r[k - 1] and above[k]]
        for rule in RULES:
            free = 0
            for k in signals:
                if k < free:
                    continue
                got = fill(rule, k, c, o, lo, vol)
                if got is None:
                    rows.append({"rule": rule, "date": d.index[k], "taken": False})
                    free = k + WAIT + 1
                    continue
                e, price, intraday = got
                lev = 20.0 if bull[k] else 5.0
                f0 = price * (1 - 1 / lev)
                x = next((j for j in range(e + 1, e + HOLD + 1) if r[j] >= EXIT_RSI), e + HOLD)
                path = np.arange(e if intraday else e + 1, x + 1)
                fin = f0 * (1 + (rate[e] + MARGIN) * np.maximum(day[path] - day[e], 0) / 365)
                hit = np.flatnonzero(lo[path] <= fin)
                knocked = hit.size > 0
                free = (path[hit[0]] if knocked else x) + 1
                exit_price = fin[hit[0]] if knocked else c[x]
                rows.append({"rule": rule, "date": d.index[k], "taken": True, "waited": e - k,
                             "entry_vs_signal": price / c[k] - 1, "stock": exit_price / price - 1, "ko": knocked,
                             "turbo": -1.0 if knocked else (c[x] - fin[-1]) / (price - f0) * (1 - SPREAD) - 1})
    trades = pd.DataFrame(rows)
    trades["period"] = np.where(trades.date < OOS_START, "2007–14", "2015–26")
    trades.to_csv(RESULTS_DIR / "limit_test.csv", index=False)

    def summary(g):
        t = g[g.taken]
        return pd.Series({"signals": len(g), "filled_%": g.taken.mean() * 100, "waited_days": t.waited.mean(),
                          "entry_vs_signal_%": t.entry_vs_signal.mean() * 100, "stock_%": t.stock.mean() * 100,
                          "win_%": (t.turbo > 0).mean() * 100, "turbo_%": t.turbo.mean() * 100, "ko_%": t.ko.mean() * 100,
                          "per_signal_%": t.turbo.sum() / len(g) * 100})
    pd.set_option("display.width", 220)
    print("== Limit orders below the signal close (turbo % per trade; per_signal counts unfilled signals as 0)")
    print(trades.groupby(["period", "rule"]).apply(summary, include_groups=False).round(2).to_string())


if __name__ == "__main__":
    main()
