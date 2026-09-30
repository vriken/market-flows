"""Part A: the Pine indicator as written, on 5-min bars (Yahoo only serves the last 60 days).

Entries are taken at the close of the breakout candle using only information available at
that close. The Pine version evaluates T against the *following* bar's close on history,
which is a one-bar peek that this replication removes.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import UNIVERSE, daily, intraday_5m
from indicators import combined_gradient, prior_value

VOL_SPIKE_MULT = 2.0
BREAKOUT_MIN_PCT = 1.0
GRAD_THRESHOLD = 0.2
MAX_RUNNER_DAYS = 5
TP_R_MULTIPLE = 1.0
MIN_QUALITIES = (0, 2, 3, 4)
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class Position:
    long: bool
    entry: float
    entry_time: pd.Timestamp
    orb_high: float
    orb_low: float
    tp: float
    flags: dict
    grad: float
    vix: float
    days_held: int = 0
    pending_close: bool = False
    tp_hit: bool = False


def day_context(d, vix_close):
    week = d.index.to_period("W")
    by_week = d.groupby(week)
    ctx = pd.DataFrame(index=d.index)
    ctx["first_of_week"] = ~week.duplicated()
    ctx["mon_high"] = by_week.high.transform("first")
    ctx["mon_low"] = by_week.low.transform("first")
    ctx["grad_prev"] = prior_value(combined_gradient(d.close), d.index)
    ctx["vix_prev"] = prior_value(vix_close, d.index)
    return ctx


def close_trade(pos, time, price, reason, ticker):
    sign = 1 if pos.long else -1
    ret = sign * (price / pos.entry - 1) * 100
    tp_ret = sign * (pos.tp / pos.entry - 1) * 100
    return {
        "ticker": ticker, "direction": "long" if pos.long else "short",
        "entry_time": pos.entry_time, "exit_time": time, "exit_reason": reason,
        "days_held": pos.days_held, "ret_pct": ret,
        "ret_tp_exit_pct": tp_ret if pos.tp_hit else ret,
        "quality": sum(pos.flags.values()), **pos.flags,
        "grad": pos.grad, "vix_prev": pos.vix,
    }


def run_symbol(ticker, bars, ctx, min_quality, runner):
    o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    trend_sma = bars.close.rolling(50).mean().to_numpy()
    vol_ratio = (bars.volume / bars.volume.rolling(20).mean().clip(lower=1)).to_numpy()
    times = bars.index
    days = bars.index.tz_localize(None).normalize()
    groups = pd.Series(np.arange(len(bars))).groupby(days).indices

    trades, pos = [], None
    for day in sorted(groups):
        if day not in ctx.index:
            continue
        idx = groups[day]
        first, last = idx[0], idx[-1]
        dc = ctx.loc[day]
        orb_hi, orb_lo = h[first], lo[first]
        mon_set = not dc.first_of_week
        inside_monday = mon_set and dc.mon_low <= o[first] <= dc.mon_high
        mon_range = dc.mon_high - dc.mon_low
        orb_in_monday = mon_set and orb_lo >= dc.mon_low and orb_hi <= dc.mon_high
        orb_upper_third = (orb_in_monday and mon_range > 0
                           and ((orb_hi + orb_lo) / 2 - dc.mon_low) / mon_range > 0.67)
        signaled = False

        for k in idx:
            if pos is not None:
                if k == first:
                    pos.days_held += 1
                if not pos.tp_hit and (h[k] >= pos.tp if pos.long else lo[k] <= pos.tp):
                    pos.tp_hit = True
                stopped = c[k] < pos.orb_low if pos.long else c[k] > pos.orb_high
                if stopped or (pos.pending_close and k == first):
                    trades.append(close_trade(pos, times[k], c[k], "stop" if stopped else "next_open", ticker))
                    pos = None

            if first < k < last and not signaled:
                bull, bear = lo[k] > orb_hi, h[k] < orb_lo
                if bull or bear:
                    breakout_pct = ((c[k] - orb_hi) if bull else (orb_lo - c[k])) / c[k] * 100
                    flags = {
                        "T": bool(c[k] > trend_sma[k] and dc.grad_prev > 0) if bull
                        else bool(c[k] <= trend_sma[k] and dc.grad_prev < 0),
                        "M": bool(inside_monday),
                        "B": bool(breakout_pct >= BREAKOUT_MIN_PCT),
                        "V": bool(vol_ratio[k] >= VOL_SPIKE_MULT),
                    }
                    short_within_monday = bear and mon_set and dc.mon_low < c[k] < dc.mon_high
                    if sum(flags.values()) >= min_quality and not (orb_upper_third or short_within_monday):
                        signaled = True
                        if pos is None:
                            risk = (c[k] - orb_lo) if bull else (orb_hi - c[k])
                            tp = c[k] + risk * TP_R_MULTIPLE if bull else c[k] - risk * TP_R_MULTIPLE
                            pos = Position(bull, c[k], times[k], orb_hi, orb_lo, tp, flags,
                                           dc.grad_prev, dc.vix_prev)

            if k == last and pos is not None and not pos.pending_close:
                pnl = (c[k] / pos.entry - 1) * (1 if pos.long else -1)
                grad_ok = pos.long and dc.grad_prev > GRAD_THRESHOLD or (not pos.long and dc.grad_prev < -GRAD_THRESHOLD)
                if not runner:
                    trades.append(close_trade(pos, times[k], c[k], "eod", ticker))
                    pos = None
                elif not (pnl > 0 and grad_ok and pos.days_held < MAX_RUNNER_DAYS):
                    pos.pending_close = True

    if pos is not None:
        trades.append(close_trade(pos, times[-1], c[-1], "end_of_data", ticker))
    return trades


def summarise(trades):
    r = trades.ret_pct
    rng = np.random.default_rng(0)
    boot = rng.choice(r.to_numpy(), size=(5000, len(r)), replace=True).mean(axis=1) if len(r) else np.array([np.nan])
    return pd.Series({
        "trades": len(r),
        "win_%": (r > 0).mean() * 100,
        "mean_%": r.mean(),
        "mean_90ci_lo": np.percentile(boot, 5),
        "mean_90ci_hi": np.percentile(boot, 95),
        "median_%": r.median(),
        "sum_%": r.sum(),
        "mean_if_full_exit_at_1R_%": trades.ret_tp_exit_pct.mean(),
        "stopped_%": (trades.exit_reason == "stop").mean() * 100,
        "avg_days_held": trades.days_held.mean(),
    })


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    vix_close = daily("^VIX").close
    prepared = {}
    for t in UNIVERSE:
        bars = intraday_5m(t)
        if len(bars):
            prepared[t] = (bars, day_context(daily(t), vix_close))
    first_day = min(b.index.min() for b, _ in prepared.values())
    last_day = max(b.index.max() for b, _ in prepared.values())
    print(f"Part A — {len(prepared)} symbols, 5-min bars {first_day:%Y-%m-%d} → {last_day:%Y-%m-%d}\n")

    rows, all_trades = [], []
    for runner in (True, False):
        for q in MIN_QUALITIES:
            trades = pd.DataFrame([tr for t, (bars, ctx) in prepared.items()
                                   for tr in run_symbol(t, bars, ctx, q, runner)])
            trades["min_quality"], trades["runner"] = q, runner
            all_trades.append(trades)
            rows.append(pd.concat([pd.Series({"runner": runner, "min_quality": q}),
                                   summarise(trades) if len(trades) else pd.Series(dtype=float)]))

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("Underlying % per trade (multiply by turbo leverage for rough turbo %, before spread)\n")
    print(table.round(2).to_string(index=False))

    base = all_trades[0]
    print("\nRunner mode, every first breakout (q≥0): does each quality flag help?")
    flag_rows = []
    for flag in ("T", "M", "B", "V"):
        for present in (True, False):
            sub = base[base[flag] == present]
            flag_rows.append({"flag": flag, "present": present, "trades": len(sub),
                              "win_%": (sub.ret_pct > 0).mean() * 100, "mean_%": sub.ret_pct.mean()})
    print(pd.DataFrame(flag_rows).round(3).to_string(index=False))

    base = base.assign(vix_regime=pd.cut(base.vix_prev, [0, 18, 22, 100], labels=["<18", "18-22", ">22"]))
    print("\nBy VIX regime (q≥0, runner):")
    print(base.groupby("vix_regime", observed=True).ret_pct.agg(["count", "mean", "median"]).round(3).to_string())
    print("\nBy direction (q≥0, runner):")
    print(base.groupby("direction").ret_pct.agg(["count", "mean", "median"]).round(3).to_string())

    pd.concat(all_trades).to_csv(RESULTS_DIR / "intraday_trades.csv", index=False)
    table.to_csv(RESULTS_DIR / "intraday_summary.csv", index=False)


if __name__ == "__main__":
    main()
