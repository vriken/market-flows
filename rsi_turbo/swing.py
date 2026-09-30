"""Swing backtest: RSI entries and baselines held a fixed number of days, with turbo payoffs.

Turbo model: long financing level F0 = S0·(1 − 1/L), knock-out when the daily low touches F
(barrier = financing level, payout 0). F accrues (T-bill + margin) daily; shorts accrue
(T-bill − margin). Round-trip spread is charged on the turbo price. FX and dividends ignored.

Regime filters are market-flows' historical classifier, read from the prior trading day.
Without FRED_API_KEY its credit dimension is the HYG/LQD proxy.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import UNIVERSE, daily
from indicators import combined_gradient, prior_value, rsi
from regime import REGIME_DIMENSIONS, load_regime_history, prior_state

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_MARKET_FLOWS = Path(__file__).parent.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tickers", nargs="+", default=UNIVERSE)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--oos-start", default="2015-01-01", help="first day of the out-of-sample period")
    p.add_argument("--rsi-length", type=int, default=14)
    p.add_argument("--oversold", nargs="+", type=float, default=[25, 30, 35])
    p.add_argument("--overbought", nargs="+", type=float, default=[70])
    p.add_argument("--trend-sma", type=int, default=200)
    p.add_argument("--gradient", type=float, default=0.2)
    p.add_argument("--vix-low", type=float, default=18)
    p.add_argument("--vix-high", type=float, default=22)
    p.add_argument("--holds", nargs="+", type=int, default=[3, 5, 10, 21])
    p.add_argument("--leverages", nargs="+", type=float, default=[3, 5, 10])
    p.add_argument("--spread", type=float, default=0.005, help="turbo round-trip spread, fraction of turbo price")
    p.add_argument("--financing-margin", type=float, default=0.025)
    p.add_argument("--min-trades", type=int, default=40, help="minimum in-sample trades to be ranked")
    p.add_argument("--report-hold", type=int, default=5)
    p.add_argument("--report-leverage", type=float, default=5)
    p.add_argument("--market-flows", type=Path, default=DEFAULT_MARKET_FLOWS, help="path to a market-flows clone")
    p.add_argument("--no-regime", action="store_true")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "swing_results.csv")
    args = p.parse_args(argv)
    if args.report_hold not in args.holds or args.report_leverage not in args.leverages:
        p.error("--report-hold and --report-leverage must be among --holds and --leverages")
    return args


def features(d, vix_close, tbill_pct, regime, args):
    f = d[["open", "high", "low", "close", "volume"]].copy()
    c = f.close
    f["rsi"] = rsi(c, args.rsi_length)
    f["rsi_prev"] = f.rsi.shift(1)
    f["sma50"] = c.rolling(50).mean()
    f["trend_sma"] = c.rolling(args.trend_sma).mean()
    f["grad"] = combined_gradient(c)
    f["vix_prev"] = prior_value(vix_close, f.index)
    f["rate"] = np.nan_to_num(prior_value(tbill_pct, f.index) / 100)
    spy = daily("SPY").close
    f["spy_above_200d"] = prior_value((spy > spy.rolling(200).mean()).astype(float), f.index) == 1
    if regime is not None:
        for dim in REGIME_DIMENSIONS:
            f[dim] = prior_state(regime[dim], f.index)
    return f[f.index >= args.start]


def setups(f, args):
    out = {}
    for level in args.oversold:
        crossed = (f.rsi < level) & (f.rsi_prev >= level)
        out[(f"rsi<{level:g}", 1)] = crossed
        out[(f"rsi<{level:g} above sma{args.trend_sma}", 1)] = crossed & (f.close > f.trend_sma)
        out[(f"rsi<{level:g}", -1)] = crossed
    out[("rsi<50", -1)] = (f.rsi < 50) & (f.rsi_prev >= 50)
    out[("new 20-day low", -1)] = f.close < f.low.rolling(20).min().shift(1)
    for level in args.overbought:
        crossed = (f.rsi > level) & (f.rsi_prev <= level)
        out[(f"rsi>{level:g}", 1)] = crossed
        out[(f"rsi>{level:g}", -1)] = crossed
    out[(f"gradient>{args.gradient:g}", 1)] = (f.close > f.sma50) & (f.grad > args.gradient)
    out[(f"gradient<-{args.gradient:g}", -1)] = (f.close <= f.sma50) & (f.grad < -args.gradient)
    everything = pd.Series(True, index=f.index)
    out[("always", 1)] = everything
    out[("always", -1)] = everything
    return {key: mask.to_numpy(dtype=bool) for key, mask in out.items()}


def filter_specs(regime, args):
    specs = [
        ("none", lambda f: np.ones(len(f), dtype=bool)),
        (f"vix<{args.vix_low:g}", lambda f: (f.vix_prev < args.vix_low).to_numpy()),
        (f"vix>={args.vix_high:g}", lambda f: (f.vix_prev >= args.vix_high).to_numpy()),
        ("spy>200d", lambda f: f.spy_above_200d.to_numpy(dtype=bool)),
        ("spy<200d", lambda f: ~f.spy_above_200d.to_numpy(dtype=bool)),
        (f"spy<200d & vix>={args.vix_high:g}",
         lambda f: ~f.spy_above_200d.to_numpy(dtype=bool) & (f.vix_prev >= args.vix_high).to_numpy()),
    ]
    if regime is not None:
        for dim in REGIME_DIMENSIONS:
            for state in sorted(regime[dim].dropna().unique()):
                name = f"{dim.removesuffix('_state').removesuffix('_label')}={state}"
                specs.append((name, lambda f, dim=dim, state=state: (f[dim] == state).to_numpy(dtype=bool)))
    return specs


class Symbol:
    def __init__(self, ticker, f, args):
        self.ticker = ticker
        self.f = f
        self.args = args
        self.close = f.close.to_numpy()
        self.low = f.low.to_numpy()
        self.high = f.high.to_numpy()
        self.rate = f.rate.to_numpy()
        self.day = f.index.to_numpy().astype("datetime64[D]").astype(np.int64)
        self.is_oos = f.index >= pd.Timestamp(args.oos_start)
        self.setups = setups(f, args)
        self.filters = {}
        self.drift = self._drift()

    def _drift(self):
        """Mean k-day return from every start day in each period: what owning the stock earned."""
        n = len(self.close)
        drift = {}
        for oos in (False, True):
            for k in self.args.holds:
                starts = np.flatnonzero(self.is_oos[: n - k] == oos)
                drift[oos, k] = np.mean(self.close[starts + k] / self.close[starts] - 1) if starts.size else np.nan
        return drift

    def filter_mask(self, name, build):
        if name not in self.filters:
            self.filters[name] = build(self.f)
        return self.filters[name]

    def trades(self, setup_key, filter_name, build_filter, hold):
        direction = setup_key[1]
        mask = self.setups[setup_key] & self.filter_mask(filter_name, build_filter)
        n = len(self.close)
        entries = np.flatnonzero(mask[: n - hold])
        if entries.size == 0:
            return None
        chosen, i = [], 0
        while i < entries.size:
            chosen.append(i)
            i = np.searchsorted(entries, entries[i] + hold + 1)
        entries = entries[chosen]
        exits = entries + hold
        path = entries[:, None] + np.arange(1, hold + 1)

        s0 = self.close[entries]
        ret = direction * (self.close[exits] / s0 - 1)
        oos = self.is_oos[entries]
        drift = np.where(oos, self.drift[True, hold], self.drift[False, hold])
        out = pd.DataFrame({"ticker": self.ticker, "entry_date": self.f.index[entries], "oos": oos,
                            "ret": ret, "excess": ret - direction * drift})

        cal_days = self.day[path] - self.day[entries][:, None]
        fin = self.rate[entries][:, None] + direction * self.args.financing_margin
        for lev in self.args.leverages:
            f0 = s0 * (1 - direction / lev)
            financing = f0[:, None] * (1 + fin * cal_days / 365)
            breached = self.low[path] <= financing if direction == 1 else self.high[path] >= financing
            value = (self.close[exits] - financing[:, -1]) / (s0 - f0)
            out[f"turbo{lev:g}"] = np.where(breached.any(axis=1), -1.0, value * (1 - self.args.spread) - 1)
            out[f"ko{lev:g}"] = breached.any(axis=1)
        return out


def clustered_t(trades):
    weekly = trades.groupby(trades.entry_date.dt.to_period("W")).excess.mean()
    if len(weekly) < 3 or weekly.std() == 0:
        return np.nan
    return weekly.mean() / (weekly.std() / np.sqrt(len(weekly)))


def summarise(trades, leverages):
    row = {
        "trades": len(trades),
        "win_%": (trades.ret > 0).mean() * 100,
        "mean_%": trades.ret.mean() * 100,
        "excess_%": trades.excess.mean() * 100,
        "t_excess": clustered_t(trades),
    }
    for lev in leverages:
        row[f"turbo{lev:g}_mean_%"] = trades[f"turbo{lev:g}"].mean() * 100
        row[f"turbo{lev:g}_median_%"] = trades[f"turbo{lev:g}"].median() * 100
        row[f"turbo{lev:g}_ko_%"] = trades[f"ko{lev:g}"].mean() * 100
    return row


def run(symbols, specs, args):
    rows = []
    for key in symbols[0].setups:
        for filter_name, build in specs:
            for hold in args.holds:
                parts = [t for s in symbols if (t := s.trades(key, filter_name, build, hold)) is not None]
                if not parts:
                    continue
                trades = pd.concat(parts, ignore_index=True)
                cfg = {"setup": key[0], "dir": "long" if key[1] == 1 else "short", "filter": filter_name,
                       "hold": hold}
                for oos, sub in trades.groupby("oos"):
                    rows.append({**cfg, "period": "OOS" if oos else "IS", **summarise(sub, args.leverages)})
    return pd.DataFrame(rows)


def paired(results):
    cfg = ["setup", "dir", "filter", "hold"]
    is_rows = results[results.period == "IS"]
    oos_rows = results[results.period == "OOS"]
    return is_rows.merge(oos_rows, on=cfg, how="outer", suffixes=("_is", "_oos"))


def report(results, regime, args):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    lev = f"turbo{args.report_leverage:g}"
    cols = ["trades", "excess_%", "t_excess", f"{lev}_mean_%", f"{lev}_ko_%"]
    both = paired(results)

    def show(df, keys=("setup", "dir", "filter", "hold")):
        return df[list(keys) + [f"{c}_{p}" for p in ("is", "oos") for c in cols]].round(2).to_string(index=False)

    oversold = [s for s in results.setup.unique() if s.startswith("rsi<")]
    focus = [("always", "long")] + [(s, "long") for s in oversold]

    print("== 1. Baseline: own every stock, any day ==")
    print(show(both[(both.setup == "always") & (both["filter"] == "none")]))

    print("\n== 2. RSI and gradient signals, no filter ==")
    print(show(both[(both.setup != "always") & (both["filter"] == "none")]))

    print(f"\n== 3. Top 25 by in-sample t (≥{args.min_trades} IS trades), with out-of-sample ==")
    ranked = both[(both.trades_is >= args.min_trades) & (both.setup != "always")]
    print(show(ranked.sort_values("t_excess_is", ascending=False).head(25)))

    if regime is not None:
        print("\n== 4. Regime coverage (market-flows classifier) ==")
        for dim in REGIME_DIMENSIONS:
            s = regime[dim].dropna()
            print(f"  {dim:17s} from {s.index.min():%Y-%m-%d}  " + ", ".join(f"{k} {v}" for k, v in s.value_counts().items()))

        for setup, direction in focus:
            print(f"\n== 5. {setup} {direction}, {args.report_hold}-day hold, split by entry filter ==")
            sub = both[(both.setup == setup) & (both.dir == direction) & (both.hold == args.report_hold)]
            print(show(sub, keys=("filter",)))

    print("\n== 6. Leverage vs hold, no filter ==")
    lev_cols = ["trades", "mean_%"] + [f"turbo{x:g}_{m}" for x in args.leverages for m in ("mean_%", "ko_%")]
    for setup, direction in focus:
        sub = results[(results.setup == setup) & (results.dir == direction) & (results["filter"] == "none")]
        print(f"-- {setup} {direction}")
        print(sub[["period", "hold"] + lev_cols].sort_values(["period", "hold"]).round(1).to_string(index=False))


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    vix_close, tbill = daily("^VIX").close, daily("^IRX").close
    regime = None if args.no_regime else load_regime_history(args.market_flows, args.start, pd.Timestamp.today())
    symbols = [Symbol(t, features(daily(t), vix_close, tbill, regime, args), args) for t in args.tickers]
    print(f"{len(symbols)} symbols, daily {args.start} → {max(s.f.index.max() for s in symbols):%Y-%m-%d}, "
          f"out-of-sample from {args.oos_start}, RSI({args.rsi_length})\n")

    results = run(symbols, filter_specs(regime, args), args)
    results.to_csv(args.output, index=False)
    report(results, regime, args)


if __name__ == "__main__":
    main()
