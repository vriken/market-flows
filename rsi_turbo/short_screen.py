"""Which stocks make a good short watch list? Tests four short screens, fixed in advance, on today's S&P 500 +
OMX Stockholm 30 + the watchlist, then lists the stocks passing the screen that held up.

Screens (entry at the close, short held a fixed number of days):
    downtrend        close < SMA50 < SMA200 and SMA gradient < -0.2
    failed bounce    downtrend and RSI(14) crosses back below 50
    rally in trend   downtrend and RSI(14) above 60
    momentum loser   6-month return in the worst 10% of the universe that day
Membership is today's, so stocks that collapsed and left the index are missing: shorts look worse than they were.
"Excess" is the short's return minus shorting the average stock in the universe over the same days.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from data import CACHE_DIR, _drop_inconsistent_ohlc, daily, index_universe
from indicators import combined_gradient, prior_value, rsi
from scan import load_watchlist

RESULTS_DIR = Path(__file__).parent / "results"
IS_END = pd.Timestamp("2015-12-31")
START = pd.Timestamp("2006-01-01")
HOLDS = (5, 10, 21)
LEVERAGES = (3, 5)
FINANCING_MARGIN = 0.025
SPREAD = 0.005
SCREENS = ("downtrend", "failed bounce", "rally in trend", "momentum loser")


def universe():
    return list(dict.fromkeys([*index_universe(), *load_watchlist()]))


def prices(tickers, chunk=100):
    frames, missing = {}, []
    for t in tickers:
        path = CACHE_DIR / f"{t}_20y_{date.today():%Y%m%d}.pkl"
        (frames.__setitem__(t, pd.read_pickle(path)) if path.exists() else missing.append(t))
    for i in range(0, len(missing), chunk):
        batch = missing[i:i + chunk]
        raw = yf.download(batch, period="20y", interval="1d", auto_adjust=True, progress=False, group_by="ticker",
                          threads=True)
        for t in batch:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].copy()
            df.columns = [c.lower() for c in df.columns]
            df = _drop_inconsistent_ohlc(df.dropna(subset=["close"]))
            if len(df) > 260:
                df.to_pickle(CACHE_DIR / f"{t}_20y_{date.today():%Y%m%d}.pkl")
                frames[t] = df
    return frames


def stock_features(d):
    c = d.close
    f = pd.DataFrame({"open": d.open, "high": d.high, "low": d.low, "close": c})
    f["sma50"], f["sma200"] = c.rolling(50).mean(), c.rolling(200).mean()
    f["grad"] = combined_gradient(c)
    f["rsi"] = rsi(c)
    f["ret126"] = c.pct_change(126)
    f["vol"] = np.log(c).diff().rolling(20).std()
    return f


def screen_masks(f, loser):
    downtrend = (f.close < f.sma50) & (f.sma50 < f.sma200) & (f.grad < -0.2)
    return {
        "downtrend": downtrend,
        "failed bounce": downtrend & (f.rsi < 50) & (f.rsi.shift(1) >= 50),
        "rally in trend": downtrend & (f.rsi > 60),
        "momentum loser": loser.reindex(f.index).fillna(False).astype(bool),
    }


def short_trades(f, mask, hold, rate, spy_above, avg_fwd):
    c, h = f.close.to_numpy(), f.high.to_numpy()
    days = f.index.to_numpy().astype("datetime64[D]").astype(np.int64)
    entries = np.flatnonzero(mask.to_numpy()[: len(c) - hold] & (f.index[: len(c) - hold] >= START))
    chosen, i = [], 0
    while i < entries.size:
        chosen.append(entries[i])
        i = np.searchsorted(entries, entries[i] + hold + 1)
    if not chosen:
        return None
    e = np.array(chosen)
    path = e[:, None] + np.arange(1, hold + 1)
    ret = 1 - c[e + hold] / c[e]
    out = pd.DataFrame({"date": f.index[e], "ret": ret, "vol": f.vol.to_numpy()[e],
                        "worst_up": np.log(h[path].max(axis=1) / c[e]),
                        "bull": prior_value(spy_above, f.index[e]) == 1})
    out["excess"] = ret + avg_fwd[hold].reindex(out.date).to_numpy()
    fin = (rate[e] - FINANCING_MARGIN)[:, None]
    cal = days[path] - days[e][:, None]
    for lev in LEVERAGES:
        f0 = c[e] * (1 + 1 / lev)
        financing = f0[:, None] * (1 + fin * cal / 365)
        knocked = (h[path] >= financing).any(axis=1)
        value = (financing[:, -1] - c[e + hold]) / (f0 - c[e])
        out[f"turbo{lev}"] = np.where(knocked, -1.0, value * (1 - SPREAD) - 1)
        out[f"ko{lev}"] = knocked
    return out


def summarise(t):
    row = {"trades": len(t), "win_%": (t.ret > 0).mean() * 100, "mean_%": t.ret.mean() * 100,
           "excess_%": t.excess.mean() * 100}
    for lev in LEVERAGES:
        row[f"turbo{lev}_%"] = t[f"turbo{lev}"].mean() * 100
        row[f"ko{lev}_%"] = t[f"ko{lev}"].mean() * 100
    return row


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    tickers = universe()
    data = prices(tickers)
    print(f"{len(data)} of {len(tickers)} stocks with ≥ 1 year of data")
    feats = {t: stock_features(d) for t, d in data.items()}
    closes = pd.DataFrame({t: f.close for t, f in feats.items()}).sort_index()
    ret126 = pd.DataFrame({t: f.ret126 for t, f in feats.items()}).reindex(closes.index)
    loser = ret126.rank(axis=1, pct=True) <= 0.10
    avg_fwd = {h: (closes.shift(-h) / closes - 1).mean(axis=1) for h in HOLDS}
    spy = daily("SPY").close
    spy_above = (spy > spy.rolling(200).mean()).astype(float)
    tbill = daily("^IRX").close / 100

    rows, pooled = [], {}
    for t, f in feats.items():
        rate = np.nan_to_num(prior_value(tbill, f.index))
        masks = screen_masks(f, loser[t])
        masks["short every stock"] = pd.Series(True, index=f.index)
        for name, mask in masks.items():
            for hold in HOLDS:
                tr = short_trades(f, mask, hold, rate, spy_above, avg_fwd)
                if tr is not None:
                    pooled.setdefault((name, hold), []).append(tr.assign(ticker=t))
    for (name, hold), parts in pooled.items():
        t = pd.concat(parts, ignore_index=True)
        t["period"] = np.where(t.date <= IS_END, "2006–15", "2016–26")
        t["market"] = np.where(t.bull, "bull", "bear")
        for split, groups in (("period", t.groupby("period")), ("market", t.groupby("market"))):
            for label, g in groups:
                rows.append({"screen": name, "hold": hold, "split": split, "slice": label, **summarise(g)})
        pooled[(name, hold)] = t
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "short_screens.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    for split in ("period", "market"):
        print(f"\n== Short screens by {split} (mean_% = the short's own return; excess_% = vs shorting the average stock)")
        print(table[table.split == split].drop(columns="split").round(2).to_string(index=False))

    print("\n== Upside move against the short within the hold, in units of daily vol × √hold (for sizing short turbos)")
    for name in SCREENS:
        t = pooled[(name, 5)]
        z = t.worst_up / (t.vol * np.sqrt(5))
        is_z, oos_z = z[t.period == "2006–15"].dropna(), z[t.period == "2016–26"].dropna()
        q = {p: np.quantile(is_z, 1 - p / 100) for p in (5, 10, 20)}
        print(f"  {name:15s} " + "  ".join(f"{p}%: z={v:.2f} (hit 2016–26 {(oos_z >= v).mean():.1%})" for p, v in q.items()))

    today = closes.index[-1]
    latest = []
    for t, f in feats.items():
        if f.index[-1] < today - pd.Timedelta(days=4):
            continue
        masks = screen_masks(f, loser[t])
        last = f.iloc[-1]
        latest.append({"ticker": t, **{name: bool(m.iloc[-1]) for name, m in masks.items()},
                       "close": last.close, "vs_sma50_%": (last.close / last.sma50 - 1) * 100,
                       "vs_sma200_%": (last.close / last.sma200 - 1) * 100, "gradient": last.grad, "rsi": last.rsi,
                       "ret_6m_%": last.ret126 * 100, "daily_vol_%": last.vol * 100})
    latest = pd.DataFrame(latest)
    latest.to_csv(RESULTS_DIR / "short_candidates_today.csv", index=False)
    print(f"\n== Passing each screen at the last close ({today:%Y-%m-%d}): " +
          ", ".join(f"{s} {int(latest[s].sum())}" for s in SCREENS))


if __name__ == "__main__":
    main()
