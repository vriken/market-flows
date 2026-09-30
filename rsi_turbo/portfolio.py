"""Portfolio simulation of the RSI<30 turbo strategy: stake size, position limit and leverage rule.

Marked daily on the combined US + Stockholm calendar. Each position is a share of current equity, bought at
the signal day's close, worth 0 once the day's low touches its financing level, otherwise sold at the close
after the hold (spread charged on the way out). "Bull" means SPY above its 200-day and VIX below the
threshold, both from the previous day. SEK and USD amounts are added as if they were the same currency.
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import swing
from data import UNIVERSE, daily
from indicators import prior_value
from scan import KO_PCT, KO_Z

LEVERAGE_RULES = {
    "5x": (5, 5),
    "10x": (10, 10),
    "20x": (20, 20),
    "10x bull / 5x else": (10, 5),
    "20x bull / 5x else": (20, 5),
    "20% KO risk bull / 5x else": ("risk", 20, 5),
    "10% KO risk bull / 5x else": ("risk", 10, 5),
    "20% KO risk always": ("risk", 20, "risk"),
}
MAX_LEVERAGE = 30
RANKINGS = ("drawdown", "rsi", "random")
RESULTS_DIR = Path(__file__).parent / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--stakes", nargs="+", type=float, default=[2, 5, 10, 20], help="percent of equity per position")
    p.add_argument("--max-positions", nargs="+", type=int, default=[3, 5, 10])
    p.add_argument("--rules", nargs="+", default=list(LEVERAGE_RULES), choices=list(LEVERAGE_RULES))
    p.add_argument("--rank", default="drawdown", choices=RANKINGS,
                   help="which signals get the free slots: deepest 52-week drawdown, lowest RSI, or random")
    p.add_argument("--hold", type=int, default=5)
    p.add_argument("--high-vix", type=float, default=22)
    p.add_argument("--spread", type=float, default=0.005)
    p.add_argument("--financing-margin", type=float, default=0.025)
    p.add_argument("--idle", default="cash", choices=["cash", "spy", "qqq"],
                   help="where money not in a turbo sits: cash earning nothing, or SPY (a core index holding)")
    p.add_argument("--universe", default="tested", choices=["tested", "wide"],
                   help="tested: the 36 large caps; wide: S&P 500 + OMX Stockholm 30 + watchlist (Yahoo 20y)")
    p.add_argument("--trend-filter", action="store_true", help="only signals with the price above its 200-day")
    p.add_argument("--exit-rsi", type=float, default=None,
                   help="also sell at the close once RSI(14) is back at or above this level (before the hold ends)")
    p.add_argument("--max-close-in-range", type=float, default=None,
                   help="skip signals whose close sits above this share of the day's high-low range (0-1)")
    p.add_argument("--max-day-move", type=float, default=None, help="skip signals whose day move is above this, e.g. -0.0164")
    p.add_argument("--min-above-200d", type=float, default=None, help="skip signals closer than this above the 200-day")
    p.add_argument("--tag", default="", help="suffix for the output files")
    return p.parse_args(argv)


def prepare(args):
    swing_args = swing.parse_args(["--no-regime", "--holds", str(args.hold), "--report-hold", str(args.hold),
                                   "--start", args.start])
    vix, tbill, spy = daily("^VIX").close, daily("^IRX").close, daily("SPY").close
    spy_above = (spy > spy.rolling(200).mean()).astype(float)
    if args.universe == "wide":
        from short_screen import prices, universe
        frames = prices(universe())
    else:
        frames = {t: daily(t) for t in UNIVERSE}
    setup = ("rsi<30 above sma200", 1) if args.trend_filter else ("rsi<30", 1)
    symbols, signals = {}, {}
    for ticker, frame in frames.items():
        f_all = swing.features(frame, vix, tbill, None, swing_args)
        if len(f_all) < 250:
            continue
        s = swing.Symbol(ticker, f_all, swing_args)
        f = s.f
        bull = (prior_value(spy_above, f.index) == 1) & (f.vix_prev < args.high_vix).to_numpy()
        drawdown = (f.close / f.high.rolling(252, min_periods=20).max() - 1).to_numpy()
        vol = np.log(f.close).diff().rolling(20).std().to_numpy()
        symbols[ticker] = {"close": s.close, "low": s.low, "day": s.day, "rate": s.rate, "bull": bull, "vol": vol,
                           "rsi": f.rsi.to_numpy(),
                           "index": {d: k for k, d in enumerate(f.index)}}
        span = (f.high - f.low).replace(0, np.nan)
        keep = np.ones(len(f), dtype=bool)
        if args.max_close_in_range is not None:
            keep &= (((f.close - f.low) / span) <= args.max_close_in_range).to_numpy()
        if args.max_day_move is not None:
            keep &= (f.close.pct_change() <= args.max_day_move).to_numpy()
        if args.min_above_200d is not None:
            keep &= ((f.close / f.trend_sma - 1) >= args.min_above_200d).to_numpy()
        for k in np.flatnonzero(s.setups[setup] & keep):
            signals.setdefault(f.index[k], []).append((ticker, k, drawdown[k], f.rsi.iloc[k]))
    calendar = pd.DatetimeIndex(sorted(set().union(*(sym["index"] for sym in symbols.values()))))
    idle_growth = np.ones(len(calendar))
    if args.idle != "cash":
        index = spy if args.idle == "spy" else daily("QQQ").close
        on_calendar = index.reindex(calendar).ffill()
        idle_growth = (on_calendar / on_calendar.shift(1)).fillna(1.0).to_numpy()
    return symbols, signals, calendar, idle_growth


def leverage_at_risk(risk_pct, vol, rate, args):
    """Highest leverage whose financing level keeps the estimated knock-out risk over the hold at risk_pct."""
    accrual = 1 + (rate + args.financing_margin) * args.hold * 7 / 5 / 365
    financing_ratio = np.exp(np.interp(risk_pct, KO_PCT, KO_Z) * vol * np.sqrt(args.hold)) / accrual
    return float(np.clip(1 / (1 - financing_ratio), 1.5, MAX_LEVERAGE)) if np.isfinite(vol) and vol > 0 else 5.0


def pick_leverage(rule, sym, k, args):
    spec = LEVERAGE_RULES[rule]
    if spec[0] == "risk":
        _, risk_pct, other = spec
        if sym["bull"][k] or other == "risk":
            return leverage_at_risk(risk_pct, sym["vol"][k], sym["rate"][k], args)
        return other
    high, other = spec
    return high if sym["bull"][k] else other


def simulate(symbols, signals, calendar, idle_growth, stake_pct, max_positions, rule, args):
    rng = np.random.default_rng(0)
    cash, positions, trades = args.capital, {}, []
    curve = np.empty(len(calendar))
    for n, date in enumerate(calendar):
        cash *= idle_growth[n]
        sold_today = set()
        for ticker in list(positions):
            p, sym = positions[ticker], symbols[ticker]
            k = sym["index"].get(date)
            if k is None:
                continue
            p["held"] += 1
            financing = p["f0"] * (1 + p["fin"] * (sym["day"][k] - p["day0"]) / 365)
            knocked = sym["low"][k] <= financing
            p["value"] = 0.0 if knocked else p["stake"] * (sym["close"][k] - financing) / (p["s0"] - p["f0"])
            recovered = args.exit_rsi is not None and sym["rsi"][k] >= args.exit_rsi
            if knocked or p["held"] >= args.hold or recovered:
                proceeds = 0.0 if knocked else p["value"] * (1 - args.spread)
                cash += proceeds
                trades.append({"ticker": ticker, "entry": p["entry"], "exit": date, "leverage": p["lev"],
                               "stake": p["stake"], "ret": proceeds / p["stake"] - 1, "knocked_out": knocked})
                sold_today.add(ticker)
                del positions[ticker]

        equity = cash + sum(p["value"] for p in positions.values())
        todays = [c for c in signals.get(date, []) if c[0] not in positions and c[0] not in sold_today]
        if args.rank == "random":
            rng.shuffle(todays)
        else:
            todays.sort(key=lambda c: c[2] if args.rank == "drawdown" else c[3])
        for ticker, k, _, _ in todays:
            stake = min(stake_pct / 100 * equity, cash)
            if len(positions) >= max_positions or stake < 0.001 * args.capital:
                break
            sym = symbols[ticker]
            lev = pick_leverage(rule, sym, k, args)
            s0 = sym["close"][k]
            positions[ticker] = {"entry": date, "stake": stake, "s0": s0, "f0": s0 * (1 - 1 / lev), "lev": lev,
                                 "fin": sym["rate"][k] + args.financing_margin, "day0": sym["day"][k],
                                 "held": 0, "value": stake}
            cash -= stake
        curve[n] = cash + sum(p["value"] for p in positions.values())
    return pd.Series(curve, index=calendar), pd.DataFrame(trades)


def metrics(curve, trades, capital):
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    final = curve.iloc[-1] / capital
    year_end = curve.groupby(curve.index.year).last()
    yearly = year_end / year_end.shift(1, fill_value=capital) - 1
    return {
        "final_x": final,
        "cagr_%": (final ** (1 / years) - 1) * 100 if final > 0 else -100.0,
        "max_drawdown_%": (curve / curve.cummax() - 1).min() * 100,
        "lowest_equity_x": curve.min() / capital,
        "worst_year_%": yearly.min() * 100,
        "worst_year": int(yearly.idxmin()),
        "losing_years": int((yearly < 0).sum()),
        "trades": len(trades),
        "knockout_%": trades.knocked_out.mean() * 100 if len(trades) else np.nan,
        "avg_trade_%": trades.ret.mean() * 100 if len(trades) else np.nan,
    }


def buy_and_hold(calendar, capital):
    spy = daily("SPY").close.reindex(calendar).ffill().dropna()
    return metrics(spy / spy.iloc[0] * capital, pd.DataFrame(), capital)


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    symbols, signals, calendar, idle_growth = prepare(args)
    busiest = max(signals.items(), key=lambda kv: len(kv[1]))
    print(f"{len(symbols)} stocks, {calendar[0]:%Y-%m-%d} → {calendar[-1]:%Y-%m-%d}, "
          f"{sum(len(v) for v in signals.values()):,} RSI<30 signals, busiest day {busiest[0]:%Y-%m-%d} "
          f"with {len(busiest[1])} signals, ranking by {args.rank}, idle money in {args.idle}\n")

    rows, curves = [], {}
    for stake, slots, rule in product(args.stakes, args.max_positions, args.rules):
        curve, trades = simulate(symbols, signals, calendar, idle_growth, stake, slots, rule, args)
        name = f"{rule} | {stake:g}% | {slots} slots"
        curves[name] = curve
        rows.append({"leverage": rule, "stake_%": stake, "max_positions": slots, **metrics(curve, trades, args.capital)})
    table = pd.DataFrame(rows).sort_values("cagr_%", ascending=False)
    suffix = f"_{args.tag}" if args.tag else ""
    table.to_csv(RESULTS_DIR / f"portfolio_grid{suffix}.csv", index=False)
    pd.DataFrame(curves).to_csv(RESULTS_DIR / f"portfolio_curves{suffix}.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    print("SPY buy and hold:", {k: round(v, 2) for k, v in buy_and_hold(calendar, args.capital).items()
                                if k in ("final_x", "cagr_%", "max_drawdown_%", "worst_year_%", "worst_year")})
    print("\n== All combinations, best growth first ==")
    print(table.round(2).to_string(index=False))

    print("\n== Best growth for each drawdown you can stomach ==")
    for limit in (-30, -40, -50, -60, -70, -80, -90):
        ok = table[table["max_drawdown_%"] >= limit]
        if not ok.empty:
            best = ok.iloc[0]
            print(f"  max drawdown ≥ {limit}%: {best.leverage}, {best['stake_%']:g}% stake, {best.max_positions} slots → "
                  f"CAGR {best['cagr_%']:.1f}%, drawdown {best['max_drawdown_%']:.0f}%, worst year "
                  f"{best['worst_year_%']:.0f}% ({best.worst_year})")


if __name__ == "__main__":
    main()
