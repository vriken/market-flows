"""When to close a v2 position? Seven exit rules, fixed before running, on v2 signals (RSI(14) closes below 30 with
the price above its 200-day) across the S&P 500 + OMX Stockholm 30 + watchlist, 2007–14 vs 2015–26.
Turbo at 20x in a bull market (SPY above its 200-day, VIX below 22, previous day) and 5x otherwise; valued as
(price − financing) / (entry − financing) with the financing accruing daily; knocked out when a day's low touches
the financing level. Exits are decided on closing prices.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import daily
from indicators import prior_value, rsi
from short_screen import prices, universe

RESULTS_DIR = Path(__file__).parent / "results"
START, OOS_START = pd.Timestamp("2007-01-01"), pd.Timestamp("2015-01-01")
MARGIN, SPREAD = 0.025, 0.005
RULES = {
    "X1 after 5 days": None,
    "X2 take profit +3%": None,
    "X3 take profit +1 typical 5-day move": None,
    "X4 stop halfway to knock-out": None,
    "X5 RSI back to 40": None,
    "X6 winners run to RSI 50 / day 10": None,
    "X7 breakeven stop after +2%": None,
}


def exit_index(rule, k, c, r, vol, lev):
    entry = c[k]
    last = min(k + 5, len(c) - 1)
    for j in range(k + 1, last + 1):
        gain = c[j] / entry - 1
        if rule == "X2 take profit +3%" and gain >= 0.03:
            return j
        if rule == "X3 take profit +1 typical 5-day move" and gain >= vol * np.sqrt(5):
            return j
        if rule == "X4 stop halfway to knock-out" and gain <= -0.5 / lev:
            return j
        if rule == "X5 RSI back to 40" and r[j] >= 40:
            return j
        if rule == "X7 breakeven stop after +2%":
            peak = c[k + 1:j].max() / entry - 1 if j > k + 1 else -1
            if peak >= 0.02 and gain <= 0:
                return j
    if rule == "X6 winners run to RSI 50 / day 10" and c[last] > entry:
        for j in range(last, min(k + 10, len(c) - 1) + 1):
            if r[j] >= 50:
                return j
        return min(k + 10, len(c) - 1)
    return last


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    data = prices(universe())
    spy, vix, tbill = daily("SPY").close, daily("^VIX").close, daily("^IRX").close / 100
    spy_up = (spy > spy.rolling(200).mean()).astype(float)
    rows = []
    for t, d in data.items():
        if len(d) < 300:
            continue
        c, lo = d.close.to_numpy(), d.low.to_numpy()
        day = d.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        r = rsi(d.close).to_numpy()
        vol = np.log(d.close).diff().rolling(20).std().to_numpy()
        above = (d.close > d.close.rolling(200).mean()).to_numpy()
        bull = (prior_value(spy_up, d.index) == 1) & (prior_value(vix, d.index) < 22)
        rate = np.nan_to_num(prior_value(tbill, d.index))
        signals = [k for k in range(1, len(c) - 10) if d.index[k] >= START and r[k] < 30 <= r[k - 1] and above[k]]
        for rule in RULES:
            free = 0
            for k in signals:
                if k < free:
                    continue
                lev = 20.0 if bull[k] else 5.0
                f0 = c[k] * (1 - 1 / lev)
                j = exit_index(rule, k, c, r, vol[k], lev)
                path = np.arange(k + 1, j + 1)
                fin = f0 * (1 + (rate[k] + MARGIN) * (day[path] - day[k]) / 365)
                hit = np.flatnonzero(lo[path] <= fin)
                knocked = hit.size > 0
                if knocked:
                    j = k + 1 + hit[0]
                free = j + 1
                turbo = -1.0 if knocked else (c[j] - fin[-1]) / (c[k] - f0) * (1 - SPREAD) - 1
                rows.append({"rule": rule, "ticker": t, "date": d.index[k], "held": j - k,
                             "stock": (fin[hit[0]] if knocked else c[j]) / c[k] - 1, "turbo": turbo, "ko": knocked,
                             "leverage": lev})
    trades = pd.DataFrame(rows)
    trades["period"] = np.where(trades.date < OOS_START, "2007–14", "2015–26")
    trades.to_csv(RESULTS_DIR / "exit_test.csv", index=False)
    def summary(g):
        return pd.Series({"trades": len(g), "held": g.held.mean(), "win_%": (g.turbo > 0).mean() * 100,
                                       "stock_%": g.stock.mean() * 100, "turbo_%": g.turbo.mean() * 100,
                                       "ko_%": g.ko.mean() * 100, "turbo_per_day_%": g.turbo.mean() / g.held.mean() * 100})
    table = trades.groupby(["period", "rule"]).apply(summary, include_groups=False)
    pd.set_option("display.width", 200)
    print("== Exit rules on v2 signals (win_% counts turbo trades that made money)")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
