"""Wait for a V before buying? v2 signals (RSI(14) closes below 30, price above its 200-day) across the S&P 500 +
OMX Stockholm 30 + watchlist, 2007–14 vs 2015–26. Each confirmation waits up to 5 trading days after the signal and
skips the trade if nothing confirms. Same exit for all: the first close with RSI back at 40, at the latest 5 trading
days after entry. Turbo at 20x in a bull market (previous day's SPY above its 200-day and VIX below 22), 5x
otherwise, knocked out on daily lows. "Per signal" counts skipped signals as 0, the fair comparison when waiting
means missing trades.
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
RULES = ("C0 signal close", "C1 first up day", "C2 close above previous high", "C3 RSI back above 30",
         "C4 close above 5-day average", "C5 2% off the low")


def entry_day(rule, k, c, h, lo, r, sma5):
    if rule == "C0 signal close":
        return k
    low = lo[k]
    for j in range(k + 1, min(k + WAIT, len(c) - 1) + 1):
        low = min(low, lo[j])
        if (rule == "C1 first up day" and c[j] > c[j - 1]) \
                or (rule == "C2 close above previous high" and c[j] > h[j - 1]) \
                or (rule == "C3 RSI back above 30" and r[j] >= 30) \
                or (rule == "C4 close above 5-day average" and c[j] > sma5[j]) \
                or (rule == "C5 2% off the low" and c[j] >= low * 1.02):
            return j
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
        c, h, lo = d.close.to_numpy(), d.high.to_numpy(), d.low.to_numpy()
        day = d.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        r = rsi(d.close).to_numpy()
        sma5 = d.close.rolling(5).mean().to_numpy()
        above = (d.close > d.close.rolling(200).mean()).to_numpy()
        bull = (prior_value(spy_up, d.index) == 1) & (prior_value(vix, d.index) < 22)
        rate = np.nan_to_num(prior_value(tbill, d.index))
        signals = [k for k in range(1, len(c) - WAIT - HOLD - 1) if d.index[k] >= START and r[k] < 30 <= r[k - 1] and above[k]]
        for rule in RULES:
            free = 0
            for k in signals:
                if k < free:
                    continue
                e = entry_day(rule, k, c, h, lo, r, sma5)
                if e is None:
                    rows.append({"rule": rule, "date": d.index[k], "taken": False})
                    free = k + WAIT + 1
                    continue
                lev = 20.0 if bull[k] else 5.0
                f0 = c[e] * (1 - 1 / lev)
                x = next((j for j in range(e + 1, e + HOLD + 1) if r[j] >= EXIT_RSI), e + HOLD)
                path = np.arange(e + 1, x + 1)
                fin = f0 * (1 + (rate[e] + MARGIN) * (day[path] - day[e]) / 365)
                hit = np.flatnonzero(lo[path] <= fin)
                knocked = hit.size > 0
                if knocked:
                    x = e + 1 + hit[0]
                free = x + 1
                rows.append({"rule": rule, "date": d.index[k], "taken": True, "waited": e - k, "held": x - e,
                             "entry_vs_signal": c[e] / c[k] - 1,
                             "stock": (fin[hit[0]] if knocked else c[x]) / c[e] - 1, "ko": knocked,
                             "turbo": -1.0 if knocked else (c[x] - fin[-1]) / (c[e] - f0) * (1 - SPREAD) - 1})
    trades = pd.DataFrame(rows)
    trades["period"] = np.where(trades.date < OOS_START, "2007–14", "2015–26")
    trades.to_csv(RESULTS_DIR / "confirm_test.csv", index=False)

    def summary(g):
        t = g[g.taken]
        return pd.Series({"signals": len(g), "bought_%": g.taken.mean() * 100, "waited_days": t.waited.mean(),
                          "entry_vs_signal_%": t.entry_vs_signal.mean() * 100, "win_%": (t.turbo > 0).mean() * 100,
                          "turbo_%": t.turbo.mean() * 100, "ko_%": t.ko.mean() * 100,
                          "per_signal_%": t.turbo.sum() / len(g) * 100})
    pd.set_option("display.width", 220)
    print("== Wait for a V before buying (turbo % per trade; per_signal counts skipped signals as 0)")
    print(trades.groupby(["period", "rule"]).apply(summary, include_groups=False).round(2).to_string())


if __name__ == "__main__":
    main()
