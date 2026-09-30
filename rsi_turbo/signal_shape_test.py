"""What should the chart look like at the signal close? Splits v2 signals (RSI(14) closes below 30, price above its
200-day) by features of the signal day that are visible before buying, S&P 500 + OMX Stockholm 30 + watchlist,
2007–14 vs 2015–26. Rule v2.1 trade: buy at the signal close, sell at the first close with RSI back at 40, at the
latest after 5 days; turbo at 20x in a bull market, 5x otherwise, knocked out on daily lows. Tercile cut points
are taken from 2007–14 only.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import daily
from indicators import prior_value, rsi
from short_screen import prices, universe

RESULTS_DIR = Path(__file__).parent / "results"
START, OOS_START = pd.Timestamp("2007-01-01"), pd.Timestamp("2015-01-01")
HOLD, EXIT_RSI = 5, 40
MARGIN, SPREAD = 0.025, 0.005


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    data = prices(universe())
    spy, vix, tbill = daily("SPY").close, daily("^VIX").close, daily("^IRX").close / 100
    spy_up = (spy > spy.rolling(200).mean()).astype(float)
    rows = []
    for d in data.values():
        if len(d) < 300:
            continue
        c, o, h, lo, v = (d[col].to_numpy() for col in ("close", "open", "high", "low", "volume"))
        day = d.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        r = rsi(d.close).to_numpy()
        sma200 = d.close.rolling(200).mean().to_numpy()
        vol_avg = pd.Series(v).replace(0, np.nan).rolling(20).mean().shift(1).to_numpy()
        bull = (prior_value(spy_up, d.index) == 1) & (prior_value(vix, d.index) < 22)
        rate = np.nan_to_num(prior_value(tbill, d.index))
        free = 0
        for k in range(6, len(c) - HOLD - 1):
            if k < free or d.index[k] < START or not (r[k] < 30 <= r[k - 1] and c[k] > sma200[k]):
                continue
            lev = 20.0 if bull[k] else 5.0
            f0 = c[k] * (1 - 1 / lev)
            x = next((j for j in range(k + 1, k + HOLD + 1) if r[j] >= EXIT_RSI), k + HOLD)
            path = np.arange(k + 1, x + 1)
            fin = f0 * (1 + (rate[k] + MARGIN) * (day[path] - day[k]) / 365)
            hit = np.flatnonzero(lo[path] <= fin)
            knocked = hit.size > 0
            free = (path[hit[0]] if knocked else x) + 1
            span = h[k] - lo[k]
            rows.append({
                "date": d.index[k], "ko": knocked,
                "turbo": -1.0 if knocked else (c[x] - fin[-1]) / (c[k] - f0) * (1 - SPREAD) - 1,
                "close_in_range": (c[k] - lo[k]) / span if span > 0 else np.nan,
                "day_move": c[k] / c[k - 1] - 1, "drop_5d": c[k] / c[k - 5] - 1,
                "gap_down": o[k] < lo[k - 1], "above_200d": c[k] / sma200[k] - 1,
                "rsi": r[k], "volume_x": v[k] / vol_avg[k] if vol_avg[k] > 0 else np.nan,
            })
    t = pd.DataFrame(rows)
    t["period"] = np.where(t.date < OOS_START, "2007–14", "2015–26")
    early = t[t.period == "2007–14"]
    def terciles(col, names):
        cuts = early[col].quantile([1 / 3, 2 / 3]).to_numpy()
        return pd.cut(t[col], [-np.inf, *cuts, np.inf], labels=names), cuts
    splits = {}
    splits["Close in the day's range"] = pd.cut(t.close_in_range, [-0.01, 1 / 3, 2 / 3, 1.01],
                                                labels=["near the low", "middle", "near the high"])
    splits["Signal day's move"], c1 = terciles("day_move", ["big down day", "moderate", "mild or up"])
    splits["Drop over 5 days"], c2 = terciles("drop_5d", ["steepest", "middle", "mildest"])
    splits["Gap down at the open"] = t.gap_down.map({True: "gapped below yesterday's low", False: "no gap"})
    splits["Distance above the 200-day"], c3 = terciles("above_200d", ["just above", "middle", "far above"])
    splits["RSI at the signal"] = pd.cut(t.rsi, [0, 25, 30], labels=["under 25", "25–30"])
    splits["Volume vs 20-day average"] = pd.cut(t.volume_x, [0, 1, 2, np.inf], labels=["below average", "1–2x", "over 2x"])
    rows = []
    for name, groups in splits.items():
        for (period, group), g in t.groupby(["period", groups], observed=True):
            rows.append({"feature": name, "group": group, "period": period, "trades": len(g),
                         "turbo_%": g.turbo.mean() * 100, "ko_%": g.ko.mean() * 100, "win_%": (g.turbo > 0).mean() * 100})
    out = pd.DataFrame(rows).pivot_table(index=["feature", "group"], columns="period",
                                         values=["trades", "turbo_%", "ko_%"], observed=True, sort=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    print(f"{len(t)} v2.1 trades · all signals: turbo {t.groupby('period').turbo.mean().mul(100).round(2).to_dict()}")
    print(f"cut points (2007–14): day move {np.round(c1 * 100, 2)}%, 5-day drop {np.round(c2 * 100, 2)}%, "
          f"above 200-day {np.round(c3 * 100, 2)}%\n")
    print(out.round(1).to_string())
    t.to_csv(RESULTS_DIR / "signal_shape_test.csv", index=False)


if __name__ == "__main__":
    main()
