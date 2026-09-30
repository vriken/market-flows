"""RSI hyperparameter grid: RSI length, buy threshold, exit rule and trend filter, on the 36 tested large caps.

Ranked on 2000–14, judged on 2015–26, and the top picks re-checked on ~500 other S&P 500 / OMX Stockholm stocks
(2015–26). Entry at the close of the day RSI crosses below the threshold; turbo leverage 20x in a bull market
(SPY above its 200-day, VIX below 22, previous day) and 5x otherwise; knock-out on the daily low.
Holds differ between exit rules, so the ranking is turbo return per day held.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from data import daily
from indicators import prior_value, rsi
from short_screen import prices, universe
from swing import UNIVERSE as TESTED

RESULTS_DIR = Path(__file__).parent / "results"
LENGTHS = (2, 3, 5, 7, 10, 14, 21)
THRESHOLDS = (10, 15, 20, 25, 30, 35, 40)
EXITS = (("hold", 3), ("hold", 5), ("hold", 10), ("rsi above", 50), ("rsi above", 60), ("rsi above", 70))
MAX_HOLD = 21
OOS_START = pd.Timestamp("2015-01-01")
MARGIN, SPREAD = 0.025, 0.005
MIN_IS_TRADES = 100


class Stock:
    def __init__(self, ticker, d, spy_up, vix, tbill, start):
        d = d[d.index >= pd.Timestamp(start) - pd.Timedelta(days=400)]
        self.ticker = ticker
        self.dates = d.index
        self.close, self.low = d.close.to_numpy(), d.low.to_numpy()
        self.day = d.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        self.bull = (prior_value(spy_up, d.index) == 1) & (prior_value(vix, d.index) < 22)
        self.rate = np.nan_to_num(prior_value(tbill, d.index))
        self.above_200 = (d.close > d.close.rolling(200).mean()).to_numpy()
        self.oos = d.index >= OOS_START
        self.valid = d.index >= pd.Timestamp(start)
        self.rsi = {n: rsi(d.close, n).to_numpy() for n in LENGTHS}
        n = len(self.close)
        self.drift = {}
        for oos in (False, True):
            period = self.valid & (self.oos == oos)
            self.drift[oos] = np.array([np.nanmean(self.close[k:] / self.close[:n - k] - 1)
                                        if (period[:n - k]).any() else np.nan for k in range(MAX_HOLD + 1)])
            for k in range(1, MAX_HOLD + 1):
                starts = np.flatnonzero(period[:n - k])
                self.drift[oos][k] = np.mean(self.close[starts + k] / self.close[starts] - 1) if starts.size else np.nan

    def next_above(self, length, level):
        r = self.rsi[length]
        out = np.full(len(r), len(r), dtype=np.int64)
        nxt = len(r)
        for j in range(len(r) - 1, -1, -1):
            out[j] = nxt
            if r[j] > level:
                nxt = j
        return out

    def trades(self, length, threshold, exit_rule, trend):
        r = self.rsi[length]
        n = len(r)
        signal = (r < threshold) & (np.roll(r, 1) >= threshold) & self.valid
        signal[0] = False
        if trend:
            signal &= self.above_200
        entries = np.flatnonzero(signal[: n - 1])
        if not entries.size:
            return []
        kind, value = exit_rule
        nxt = self.next_above(length, value) if kind == "rsi above" else None
        rows, free = [], 0
        for k in entries:
            if k < free:
                continue
            exit_k = k + value if kind == "hold" else min(nxt[k], k + MAX_HOLD)
            if exit_k >= n:
                break
            free = exit_k + 1
            held = exit_k - k
            ret = self.close[exit_k] / self.close[k] - 1
            lev = 20.0 if self.bull[k] else 5.0
            f0 = self.close[k] * (1 - 1 / lev)
            path = np.arange(k + 1, exit_k + 1)
            fin = f0 * (1 + (self.rate[k] + MARGIN) * (self.day[path] - self.day[k]) / 365)
            knocked = bool((self.low[path] <= fin).any())
            turbo = -1.0 if knocked else (self.close[exit_k] - fin[-1]) / (self.close[k] - f0) * (1 - SPREAD) - 1
            rows.append((self.dates[k], bool(self.oos[k]), held, ret, ret - self.drift[bool(self.oos[k])][held], turbo, knocked))
        return rows


def summarise(rows):
    t = pd.DataFrame(rows, columns=["date", "oos", "held", "ret", "excess", "turbo", "ko"])
    out = {}
    for oos, g in t.groupby("oos"):
        p = "oos" if oos else "is"
        weekly = g.groupby(g.date.dt.to_period("W")).excess.mean()
        out.update({f"trades_{p}": len(g), f"held_{p}": g.held.mean(), f"win_{p}": (g.ret > 0).mean() * 100,
                    f"stock_{p}": g.ret.mean() * 100, f"excess_{p}": g.excess.mean() * 100,
                    f"t_{p}": weekly.mean() / (weekly.std() / np.sqrt(len(weekly))) if len(weekly) > 2 else np.nan,
                    f"turbo_{p}": g.turbo.mean() * 100, f"ko_{p}": g.ko.mean() * 100,
                    f"turbo_per_day_{p}": g.turbo.mean() / g.held.mean() * 100})
    return out


def run(stocks, configs):
    rows = []
    for length, threshold, exit_rule, trend in configs:
        trades = [tr for s in stocks for tr in s.trades(length, threshold, exit_rule, trend)]
        if trades:
            rows.append({"rsi_length": length, "buy_below": threshold, "exit": f"{exit_rule[0]} {exit_rule[1]}",
                         "above_200d": trend, **summarise(trades)})
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    spy, vix, tbill = daily("SPY").close, daily("^VIX").close, daily("^IRX").close / 100
    spy_up = (spy > spy.rolling(200).mean()).astype(float)
    tested = [Stock(t, daily(t), spy_up, vix, tbill, "2000-01-01") for t in TESTED]
    configs = list(product(LENGTHS, THRESHOLDS, EXITS, (False, True)))
    grid = run(tested, configs)
    grid.to_csv(RESULTS_DIR / "rsi_grid.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    cols = ["rsi_length", "buy_below", "exit", "above_200d", "trades_is", "held_is", "excess_is", "turbo_is", "ko_is",
            "turbo_per_day_is", "trades_oos", "excess_oos", "t_oos", "turbo_oos", "ko_oos", "turbo_per_day_oos"]
    yours = grid[(grid.rsi_length == 14) & (grid.buy_below == 30) & ~grid.above_200d]
    print("== Your RSI 14 / 30, each exit rule (36 large caps)")
    print(yours[cols].round(2).to_string(index=False))

    ranked = grid[grid.trades_is >= MIN_IS_TRADES].sort_values("turbo_per_day_is", ascending=False)
    print(f"\n== Top 20 of {len(grid)} by 2000–14 turbo return per day held (≥{MIN_IS_TRADES} trades), with 2015–26")
    print(ranked.head(20)[cols].round(2).to_string(index=False))
    top = ranked.head(50)
    print(f"\n   Of the 2000–14 top 50: {(top.turbo_per_day_oos > yours[yours.exit == 'hold 5'].turbo_per_day_oos.iloc[0]).sum()} "
          f"beat RSI 14/30/hold 5 on 2015–26 per day held")

    print("\n== By RSI length and by exit rule (median over the other settings, 2015–26)")
    print(grid.groupby("rsi_length")[["turbo_per_day_is", "turbo_per_day_oos", "excess_oos"]].median().round(3).to_string())
    print(grid.groupby("exit")[["held_oos", "turbo_per_day_is", "turbo_per_day_oos", "excess_oos"]].median().round(3).to_string())

    wide_tickers = [t for t in universe() if t not in TESTED]
    wide_data = prices(wide_tickers)
    wide = [Stock(t, d, spy_up, vix, tbill, "2015-01-01") for t, d in wide_data.items() if len(d) > 500]
    check = pd.concat([ranked.head(10), yours]).drop_duplicates(["rsi_length", "buy_below", "exit", "above_200d"])
    confirm = run(wide, [(r.rsi_length, r.buy_below, (r.exit.rsplit(" ", 1)[0], int(r.exit.rsplit(" ", 1)[1])), r.above_200d)
                         for r in check.itertuples()])
    confirm.to_csv(RESULTS_DIR / "rsi_grid_wide_check.csv", index=False)
    print(f"\n== Second check on {len(wide)} other stocks, 2015–26 only")
    print(confirm[["rsi_length", "buy_below", "exit", "above_200d", "trades_oos", "held_oos", "win_oos", "excess_oos", "t_oos",
                   "turbo_oos", "ko_oos", "turbo_per_day_oos"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
