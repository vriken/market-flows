"""Parameter health: is each part of rule v2.1 still earning its place?

Every RSI(14) cross below 30 on the S&P 500 + OMX Stockholm 30 + watchlist since 2007 is traded with and without
each part of the rule, and the difference in turbo return per trade is tracked by year and over rolling 3-year
windows. Standard errors are clustered by signal week, since dips come in market-wide waves.

Read it as decided before looking: a part is "strong" when its last-3-year edge is positive with t >= 2, "holding up"
when positive with t below 2, and "not helping" when zero or negative. It is for spotting decay, not re-tuning: a part that
stops helping is a reason to test the alternative on data it hasn't seen, not to swap in the best-looking value.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import daily
from indicators import prior_value, rsi
from short_screen import prices, universe

HERE = Path(__file__).parent
OUTPUT = HERE / "results" / "param_health.csv"
START = pd.Timestamp("2007-01-01")
OVERSOLD, EXIT_RSI, HOLD, LONG_HOLD, TREND_SMA = 30, 40, 5, 10, 200
BULL_LEVERAGE, OTHER_LEVERAGE, HIGH_VIX = 20.0, 5.0, 22
MARGIN, SPREAD = 0.025, 0.005
WINDOW_YEARS = 3
STRONG_T = 2.0

PARTS = {
    "entry": ("RSI below 30 at the close", "stock return minus SPY over the hold, % points"),
    "trend": ("Only above the 200-day", "turbo return above minus below the 200-day"),
    "leverage": ("20x in a bull market, 5x otherwise", "turbo return at the rule's leverage minus the other one"),
    "exit": ("Sell when RSI is back at 40", "turbo return per day held minus holding the full 5 days"),
    "cap": ("Sell after 5 days at the latest", "turbo return per day held minus a 10-day limit"),
    "day_shape": ("Flag: signal closed near the day's low", "turbo return, bottom third of the range minus the rest"),
}


def turbo(c, lo, r, day, rate, k, leverage, cap, rsi_exit):
    """Turbo bought at close k: (return, exit index). Knocked out when a daily low reaches the financing level."""
    last = k + cap
    x = next((j for j in range(k + 1, last + 1) if r[j] >= EXIT_RSI), last) if rsi_exit else last
    f0 = c[k] * (1 - 1 / leverage)
    path = np.arange(k + 1, x + 1)
    financing = f0 * (1 + (rate[k] + MARGIN) * (day[path] - day[k]) / 365)
    hit = np.flatnonzero(lo[path] <= financing)
    if hit.size:
        return -1.0, path[hit[0]]
    return (c[x] - financing[-1]) / (c[k] - f0) * (1 - SPREAD) - 1, x


def signals(data, spy, vix, tbill):
    spy_up = (spy > spy.rolling(TREND_SMA).mean()).astype(float)
    rows = []
    for d in data.values():
        if len(d) < TREND_SMA + 50:
            continue
        c, lo, h = d.close.to_numpy(), d.low.to_numpy(), d.high.to_numpy()
        day = d.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        r = rsi(d.close).to_numpy()
        sma = d.close.rolling(TREND_SMA).mean().to_numpy()
        bull = (prior_value(spy_up, d.index) == 1) & (prior_value(vix, d.index) < HIGH_VIX)
        rate = np.nan_to_num(prior_value(tbill, d.index))
        spy_close = spy.reindex(d.index, method="ffill").to_numpy()
        for k in np.flatnonzero((r[1:] < OVERSOLD) & (r[:-1] >= OVERSOLD)) + 1:
            if d.index[k] < START or k + LONG_HOLD >= len(c) or np.isnan(sma[k]):
                continue
            lev, other = (BULL_LEVERAGE, OTHER_LEVERAGE) if bull[k] else (OTHER_LEVERAGE, BULL_LEVERAGE)
            base, x = turbo(c, lo, r, day, rate, k, lev, HOLD, True)
            full_hold, x_full = turbo(c, lo, r, day, rate, k, lev, HOLD, False)
            long_cap, x_long = turbo(c, lo, r, day, rate, k, lev, LONG_HOLD, True)
            span = h[k] - lo[k]
            rows.append({
                "date": d.index[k], "above": bool(c[k] > sma[k]), "base": base, "held": x - k,
                "other_leverage": turbo(c, lo, r, day, rate, k, other, HOLD, True)[0],
                "full_hold": full_hold, "full_hold_held": x_full - k, "long_cap": long_cap, "long_cap_held": x_long - k,
                "excess": c[x] / c[k] - spy_close[x] / spy_close[k],
                "near_low": (c[k] - lo[k]) / span < 1 / 3 if span > 0 else np.nan,
            })
    t = pd.DataFrame(rows)
    t["week"] = t.date.dt.to_period("W")
    t["year"] = t.date.dt.year
    return t


def clustered_mean(values, clusters):
    """Mean and its week-clustered standard error."""
    values = pd.Series(values.to_numpy(), index=clusters.to_numpy()).dropna()
    n = len(values)
    if n < 2:
        return np.nan, np.nan, n
    mean = values.mean()
    by_cluster = (values - mean).groupby(level=0).sum()
    return mean, np.sqrt((by_cluster ** 2).sum()) / n, n


def clustered_per_day_gain(t, alternative):
    """Paired difference in return per day held (mean return / mean days, as capital is recycled), with a
    week-clustered delta-method standard error."""
    n = len(t)
    if n < 2:
        return np.nan, np.nan, n
    y, h, z, g = t.base, t.held, t[alternative], t[f"{alternative}_held"]
    rule_rate, alt_rate = y.mean() / h.mean(), z.mean() / g.mean()
    influence = (y - rule_rate * h) / h.mean() - (z - alt_rate * g) / g.mean()
    by_cluster = influence.groupby(t.week.to_numpy()).sum()
    return rule_rate - alt_rate, np.sqrt((by_cluster ** 2).sum()) / n, n


def edge(part, t):
    """(edge, standard error, trades) in % points for one part on one slice of signals."""
    rule = t[t.above]
    if part == "entry":
        m, se, n = clustered_mean(rule.excess, rule.week)
    elif part == "trend":
        (a, se_a, na), (b, se_b, nb) = clustered_mean(rule.base, rule.week), clustered_mean(t[~t.above].base, t[~t.above].week)
        m, se, n = a - b, np.hypot(se_a, se_b), na + nb
    elif part == "day_shape":
        low, rest = rule[rule.near_low.eq(True)], rule[rule.near_low.eq(False)]
        (a, se_a, na), (b, se_b, nb) = clustered_mean(low.base, low.week), clustered_mean(rest.base, rest.week)
        m, se, n = a - b, np.hypot(se_a, se_b), na + nb
    elif part == "leverage":
        m, se, n = clustered_mean(rule.base - rule.other_leverage, rule.week)
    else:
        m, se, n = clustered_per_day_gain(rule, {"exit": "full_hold", "cap": "long_cap"}[part])
    return m * 100, se * 100, n


def health(t):
    years = range(int(t.year.min()), int(t.year.max()) + 1)
    rows = []
    for part in PARTS:
        for year in years:
            for window, first in (("year", year), (f"{WINDOW_YEARS}y", year - WINDOW_YEARS + 1)):
                m, se, n = edge(part, t[(t.year >= first) & (t.year <= year)])
                rows.append({"part": part, "year": year, "window": window, "edge": m, "se": se, "trades": n})
        m, se, n = edge(part, t)
        rows.append({"part": part, "year": int(t.year.max()), "window": "all", "edge": m, "se": se, "trades": n})
    out = pd.DataFrame(rows)
    out["t"] = out.edge / out.se
    return out


def verdicts(table):
    """Latest rolling window per part against the pre-declared reading, next to the full period."""
    last = table[table.window == f"{WINDOW_YEARS}y"].sort_values("year").groupby("part").tail(1).set_index("part")
    full = table[table.window == "all"].set_index("part")
    status = np.where(last.edge <= 0, "not helping", np.where(last.t >= STRONG_T, "strong", "holding up"))
    return pd.DataFrame({
        "part": [PARTS[p][0] for p in last.index], "key": last.index, "status": status,
        "last_3y_edge": last.edge, "last_3y_t": last.t, "full_edge": full.edge.reindex(last.index),
        "full_t": full.t.reindex(last.index), "measure": [PARTS[p][1] for p in last.index],
    }).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    t = signals(prices(universe()), daily("SPY").close, daily("^VIX").close, daily("^IRX").close / 100)
    table = health(t)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    pd.set_option("display.width", 200)
    print(f"{len(t)} signals, {t.above.sum()} above the 200-day, {t.date.min():%Y-%m-%d} to {t.date.max():%Y-%m-%d}\n")
    print(verdicts(table).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
