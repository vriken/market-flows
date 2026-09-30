"""RSI<30 turbo scanner: a morning run on confirmed closes, pre-close runs, and a 30-minute intraday watch.

Rule v2 (portfolio test 2007–26 on ~535 stocks): RSI(14) crosses below 30 at the close while the price is above
its 200-day average; leverage set by the previous day's market (SPY above its 200-day and VIX below 22 means bull);
sell after 5 trading days. The morning run scans the S&P 500 + OMX Stockholm 30 + watchlist and saves the stocks
close to a signal; the other runs only fetch those, the watchlist and open positions. Pre-close and intraday runs use
the live price as a provisional close (Yahoo's Stockholm prices can lag about 15 minutes), so their signals can
still disappear at the close; only the morning run records signals in the signal log. The journal holds only the
trades you log. Knock-out risk estimates use
constants fitted on RSI<30 dips in 36 stocks over 2000–14 and checked on 2015–26.
"""
import argparse
import fcntl
import subprocess
from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from data import index_universe, recent_daily
from indicators import prior_value, rsi

HERE = Path(__file__).parent
WATCHLIST = HERE / "watchlist.txt"
LIVE_DIR = HERE / "live"
JOURNAL = LIVE_DIR / "journal.csv"
SIGNALS = LIVE_DIR / "signals.csv"
LOCK = LIVE_DIR / "scan.lock"
PROVISIONAL = LIVE_DIR / "provisional.csv"
KO_ALERTS = LIVE_DIR / "ko_alerts.csv"
INTRADAY_LOG = LIVE_DIR / "intraday.log"
CANDIDATES = LIVE_DIR / "candidates.csv"

MODES = {
    "morning": {"markets": {"SE", "US"}, "provisional": False, "label": "Morning — yesterday's closes"},
    "se-preclose": {"markets": {"SE"}, "provisional": True, "label": "Stockholm before the 17:30 close"},
    "us-preclose": {"markets": {"US"}, "provisional": True, "label": "US before the 22:00 close"},
    "intraday": {"markets": None, "provisional": True, "label": "Intraday watch"},
}
SESSIONS = {"SE": ("Europe/Stockholm", time(9, 0), time(17, 30)), "US": ("America/New_York", time(9, 30), time(16, 0))}

MORNING_RUN = time(8, 15)
HOLD = 5
OVERSOLD = 30
EXIT_RSI = 40
WATCH_BELOW = 35
CANDIDATE_BELOW = 40
TREND_SMA = 200
WATCH_SHOWN = 15
BULL_LEVERAGE = 20
OTHER_LEVERAGE = 5
HIGH_VIX = 22
STAKE_GUIDE = "1–1.5% of the portfolio"
MAX_POSITIONS = 30
FINANCING_MARGIN = 0.025
SPREAD = 0.005
PAPER_CRITERIA = {"trades": 40, "win_pct": 50.0, "avg_stock_pct": 0.3, "max_ko_pct": 15.0}
VOL_LEN = 20
MIN_HISTORY = 250
KO_Z = np.array([-5.565, -4.153, -3.107, -2.364, -1.944, -1.577, -1.165, -0.903, -0.695])
KO_PCT = np.array([1, 2.5, 5, 10, 15, 20, 30, 40, 50])
SIGNAL_COLUMNS = ["signal_date", "ticker", "close", "rsi", "market", "leverage", "financing_level",
                  "est_ko_risk_%", "max_fin_10%_risk", "max_fin_20%_risk", "sell_on",
                  "earnings_in_hold", "rule", "result_stock_pct", "result_turbo_pct", "result_ko", "sold_on", "exit_reason",
                  "notes"]
JOURNAL_COLUMNS = ["bought_on", "ticker", "rule", "signal_date", "product", "product_financing", "ko_level", "qty",
                   "parity", "fill_price", "fill_fx", "fill_underlying", "leverage", "exit_price", "closed_on",
                   "result_pct", "knocked_out", "exit_reason", "notes"]
LINK_WINDOW_DAYS = 5


def load_watchlist():
    tickers = [line.split("#")[0].strip() for line in WATCHLIST.read_text().splitlines()]
    return list(dict.fromkeys(t for t in tickers if t))


def full_universe():
    try:
        wide = index_universe()
    except Exception:
        wide = []
    return list(dict.fromkeys([*load_watchlist(), *wide]))


def open_signals(signal_log, today_iso):
    return (signal_log.rule == "v2") & (signal_log.result_stock_pct == "") & (signal_log.sell_on >= today_iso)


def open_trades(journal):
    return (journal.exit_price == "") & (journal.knocked_out != "yes")


def focus_list(signal_log, journal, today_iso):
    """Watchlist, open positions, running signals and yesterday's candidates: what the pre-close and intraday runs fetch."""
    running = [*journal[open_trades(journal)].ticker, *signal_log[open_signals(signal_log, today_iso)].ticker]
    candidates = pd.read_csv(CANDIDATES).ticker.tolist() if CANDIDATES.exists() else []
    return list(dict.fromkeys([*load_watchlist(), *running, *candidates]))


def market_of(ticker):
    return "SE" if ticker.endswith(".ST") else "US"


def auto_mode(now):
    if now.hour < 12:
        return "morning"
    return "se-preclose" if now.hour < 19 else "us-preclose"


def open_markets(now):
    markets = set()
    for market, (tz, start, end) in SESSIONS.items():
        local = now.tz_localize(ZoneInfo("Europe/Stockholm")).astimezone(ZoneInfo(tz))
        if local.weekday() < 5 and start <= local.time() < end:
            markets.add(market)
    return markets


def ko_risk_pct(barrier_ratio, vol):
    z = np.log(barrier_ratio) / (vol * np.sqrt(HOLD))
    if z <= KO_Z[0]:
        return "<1%"
    if z >= KO_Z[-1]:
        return ">50%"
    return f"{np.interp(z, KO_Z, KO_PCT):.0f}%"


def max_financing(close, vol, risk_pct, accrual):
    z = np.interp(risk_pct, KO_PCT, KO_Z)
    return close * np.exp(z * vol * np.sqrt(HOLD)) / accrual


def market_state(data, today):
    spy = data["SPY"].close
    spy_vs_200d = (spy / spy.rolling(200).mean() - 1).dropna()
    def confirmed(s):
        return s[s.index < today]
    return spy_vs_200d, data["^VIX"].close, data["^IRX"].close.iloc[-1] / 100, confirmed


def analyse(ticker, d, spy_vs_200d, vix, rate, today, provisional):
    if len(d) < MIN_HISTORY:
        return {"ticker": ticker, "skip": f"under {MIN_HISTORY} days of data"}
    if not provisional:
        d = d[d.index < today]
    elif d.index[-1] != today:
        return {"ticker": ticker, "skip": "no bar today"}
    r = rsi(d.close)
    last = d.index[-1]
    close = float(d.close.iloc[-1])
    above_trend = bool(close > d.close.rolling(TREND_SMA).mean().iloc[-1])
    span = float(d.high.iloc[-1] - d.low.iloc[-1])
    close_in_range = (close - float(d.low.iloc[-1])) / span if span > 0 else np.nan
    vol = float(np.log(d.close).diff().rolling(VOL_LEN).std().iloc[-1])
    bull = float(prior_value(spy_vs_200d, [last])[0]) > 0 and float(prior_value(vix, [last])[0]) < HIGH_VIX
    leverage = BULL_LEVERAGE if bull else OTHER_LEVERAGE
    accrual = 1 + (rate + FINANCING_MARGIN) * HOLD * 7 / 5 / 365
    financing = close * (1 - 1 / leverage)
    return {
        "ticker": ticker, "skip": None, "date": last, "close": close, "vol": vol,
        "rsi": float(r.iloc[-1]), "above_trend": above_trend, "close_in_range": close_in_range,
        "day_move": close / float(d.close.iloc[-2]) - 1, "dip": bool(r.iloc[-1] < OVERSOLD <= r.iloc[-2]),
        "signal": bool(r.iloc[-1] < OVERSOLD <= r.iloc[-2]) and above_trend,
        "market": "bull" if bull else "caution", "leverage": leverage, "financing": financing,
        "ko_risk": ko_risk_pct(financing * accrual / close, vol),
        "max_fin_10": max_financing(close, vol, 10, accrual), "max_fin_20": max_financing(close, vol, 20, accrual),
        "sell_on": (last + pd.offsets.BDay(HOLD)).date(),
    }


def load_csv(path, columns):
    if path.exists():
        return pd.read_csv(path, dtype=str).fillna("").reindex(columns=columns).fillna("")
    return pd.DataFrame(columns=columns)


def earnings_in_hold(ticker, signal_day, sell_on):
    """First price reaction of a report inside (signal close, sell close]; before-noon New York stamps react that day."""
    try:
        stamps = yf.Ticker(ticker).get_earnings_dates(limit=8)
    except Exception:
        return "unknown"
    if stamps is None or stamps.empty:
        return "unknown"
    for ts in stamps.index:
        day = pd.Timestamp(ts.date())
        reaction = day if ts.hour < 12 else day + pd.offsets.BDay(1)
        if signal_day < reaction <= pd.Timestamp(sell_on):
            return f"yes ({day:%Y-%m-%d})"
    return "no"


def settle(signal_log, data, today, rate):
    """Fill in results for signals whose sell day has passed, at the leverage recorded on the signal day."""
    for i in signal_log.index[signal_log.rule == ""]:
        d = data.get(signal_log.at[i, "ticker"])
        if d is None:
            continue
        hist = d.close[d.index <= pd.Timestamp(signal_log.at[i, "signal_date"])]
        above = len(hist) >= TREND_SMA and hist.iloc[-1] > hist.rolling(TREND_SMA).mean().iloc[-1]
        signal_log.at[i, "rule"] = "v2" if above else "v1 only"
    for i in signal_log.index[(signal_log.rule == "v2") & (signal_log.result_stock_pct == "")
                              & (signal_log.signal_date < today.date().isoformat())]:
        d = data.get(signal_log.at[i, "ticker"])
        if d is None:
            continue
        d = d[d.index < today]
        day = pd.Timestamp(signal_log.at[i, "signal_date"])
        if day not in d.index:
            continue
        r = rsi(d.close)
        k = d.index.get_loc(day)
        held = k > 0 and r.iloc[k] < OVERSOLD <= r.iloc[k - 1] and d.close.iloc[k] > d.close.rolling(TREND_SMA).mean().iloc[k]
        if not held:
            signal_log.at[i, "rule"] = "didn't hold"
            signal_log.at[i, "sell_on"] = ""
            signal_log.at[i, "notes"] = (signal_log.at[i, "notes"] + "; " if signal_log.at[i, "notes"] else "") + \
                "live signal didn't hold at the close"
    for i in signal_log.index[(signal_log.earnings_in_hold == "") & (signal_log.result_stock_pct == "")]:
        signal_log.at[i, "earnings_in_hold"] = earnings_in_hold(signal_log.at[i, "ticker"],
                                                                pd.Timestamp(signal_log.at[i, "signal_date"]),
                                                                signal_log.at[i, "sell_on"])
    for i in signal_log.index[(signal_log.result_stock_pct == "") & signal_log.rule.isin(["v2", "v1 only"])]:
        row = signal_log.loc[i]
        d = data.get(row.ticker)
        if d is None:
            continue
        d = d[d.index < today]
        signal_day, last_day = pd.Timestamp(row.signal_date), pd.Timestamp(row.sell_on)
        window = d[(d.index > signal_day) & (d.index <= last_day)]
        if window.empty:
            continue
        entry, lev = float(row.close), float(row.leverage)
        f0 = entry * (1 - 1 / lev)
        financing = f0 * (1 + (rate + FINANCING_MARGIN) * (window.index - signal_day).days.to_numpy() / 365)
        knocked_at = np.flatnonzero(window.low.to_numpy() <= financing)
        recovered_at = np.flatnonzero(rsi(d.close).reindex(window.index).to_numpy() >= EXIT_RSI)
        if knocked_at.size and (not recovered_at.size or knocked_at[0] <= recovered_at[0]):
            j, reason = knocked_at[0], "knocked out"
        elif recovered_at.size:
            j, reason = recovered_at[0], f"RSI back to {EXIT_RSI}"
        elif d.index[-1] >= last_day:
            j, reason = len(window) - 1, f"day {HOLD}"
        else:
            continue
        exit_price = float(window.close.iloc[j])
        knocked = reason == "knocked out"
        turbo = -100.0 if knocked else ((exit_price - financing[j]) / (entry - f0) * (1 - SPREAD) - 1) * 100
        signal_log.loc[i, ["result_stock_pct", "result_turbo_pct", "result_ko", "sold_on", "exit_reason"]] = [
            f"{(exit_price / entry - 1) * 100:.2f}", f"{turbo:.1f}", "yes" if knocked else "no",
            window.index[j].date().isoformat(), reason]
    return signal_log


def linked(rows, others):
    """Rows whose (signal date, stock) also appears in others."""
    keys = set(zip(others.signal_date, others.ticker, strict=False))
    return np.array([k in keys for k in zip(rows.signal_date, rows.ticker, strict=False)], dtype=bool)


def exits_due(signal_log, journal, rows, today_iso, markets):
    """What to sell before this close: running v2 signals with RSI back to the exit level on the latest price or on
    their last day, and your v2 trades still open after the rule already exited."""
    rsi_now = {r["ticker"]: r["rsi"] for r in rows}
    def in_markets(df):
        return df.ticker.map(market_of).isin(markets)
    held = journal[open_trades(journal) & (journal.rule == "v2") & in_markets(journal)]
    products = {(t.signal_date, t.ticker): t.product for t in held.itertuples()}
    due = []
    for row in signal_log[(signal_log.rule == "v2") & (signal_log.signal_date < today_iso) & in_markets(signal_log)].itertuples():
        key, level = (row.signal_date, row.ticker), rsi_now.get(row.ticker)
        if row.result_stock_pct:
            why = f"rule exit {row.sold_on} ({row.exit_reason})" if key in products else None
        elif level is not None and level >= EXIT_RSI:
            why = f"RSI back to {level:.0f}"
        else:
            why = f"day {HOLD}" if row.sell_on == today_iso else None
        if why:
            due.append({"ticker": row.ticker, "signal_date": row.signal_date, "why": why, "yours": key in products,
                        "product": products.get(key, "")})
    return pd.DataFrame(due, columns=["ticker", "signal_date", "why", "yours", "product"])


def product_leverage(underlying, financing):
    return underlying / (underlying - financing) if 0 < financing < underlying else np.nan


def implied_fill(journal, i):
    """The stock price at your fill, implied from the product price, and the leverage it gives."""
    financing, paid, parity, fx = (pd.to_numeric(journal.at[i, c], errors="coerce")
                                   for c in ("product_financing", "fill_price", "parity", "fill_fx"))
    underlying = financing + paid * parity / fx
    if pd.notna(underlying):
        journal.at[i, "fill_underlying"] = f"{underlying:.2f}"
        journal.at[i, "leverage"] = f"{product_leverage(underlying, financing):.1f}"
    return journal


def record_price(journal, i):
    paid, got = (pd.to_numeric(journal.at[i, c], errors="coerce") for c in ("fill_price", "exit_price"))
    if paid > 0 and got >= 0:
        journal.at[i, "result_pct"] = f"{(got / paid - 1) * 100:.1f}"
        journal.at[i, "knocked_out"] = journal.at[i, "knocked_out"] or "no"
        journal.at[i, "exit_reason"] = journal.at[i, "exit_reason"] or "sold"
    return journal


def settle_trades(journal, data):
    """Your result comes from your own prices. A trade without an exit price closes as knocked out when a day's low
    since the purchase reaches its knock-out level: the stop-loss for a mini long, which pays back what is left
    above the financing level, else the financing level."""
    for i in journal.index:
        implied_fill(journal, i)
    for i in journal.index[journal.exit_price != ""]:
        record_price(journal, i)
    for i in journal.index[open_trades(journal)]:
        row = journal.loc[i]
        d = data.get(row.ticker)
        entry, financing = (pd.to_numeric(v, errors="coerce") for v in (row.fill_underlying, row.product_financing))
        if d is None or not entry > financing > 0:
            continue
        ko = pd.to_numeric(row.ko_level, errors="coerce") if row.ko_level else financing
        hit = d.index[(d.index >= pd.Timestamp(row.bought_on)) & (d.low.to_numpy() <= ko)]
        if len(hit):
            journal.loc[i, ["result_pct", "knocked_out", "closed_on", "exit_reason"]] = [
                f"{((ko - financing) / (entry - financing) - 1) * 100:.1f}", "yes", hit[0].date().isoformat(), "knocked out"]
    return journal


def unlink_revoked(journal, signal_log):
    """Trades on a signal that didn't hold at the close count as off-rule."""
    journal.loc[(journal.rule == "v2") & ~linked(journal, signal_log[signal_log.rule == "v2"]), "rule"] = "off-rule"
    return journal


def log_trade(signal_log, journal, ticker, day, product, financing, stop, fill_sek, qty, parity, usd_sek):
    """Record a trade you made. It links to the latest v2 signal on the same stock from the last few trading days
    when there is one; otherwise it is an off-rule trade."""
    day_iso = day.isoformat()
    earliest = (pd.Timestamp(day) - pd.offsets.BDay(LINK_WINDOW_DAYS)).date().isoformat()
    on_signal = signal_log[(signal_log.ticker == ticker) & (signal_log.rule == "v2") & (signal_log.signal_date <= day_iso)
                           & (signal_log.signal_date >= earliest)].signal_date
    row = {"bought_on": day_iso, "ticker": ticker, "rule": "v2" if len(on_signal) else "off-rule",
           "signal_date": on_signal.max() if len(on_signal) else "", "product": product,
           "product_financing": f"{financing}", "ko_level": f"{stop or financing}", "qty": str(qty),
           "parity": f"{parity:g}", "fill_price": f"{fill_sek}", "fill_fx": f"{1.0 if ticker.endswith('.ST') else usd_sek}"}
    journal = pd.concat([journal, pd.DataFrame([row])], ignore_index=True).reindex(columns=JOURNAL_COLUMNS).fillna("")
    journal = implied_fill(journal, journal.index[-1])
    if len(on_signal):
        return journal, f"Linked to the {row['signal_date']} v2 signal on {ticker}."
    return journal, f"No v2 signal on {ticker} in the last {LINK_WINDOW_DAYS} trading days: logged as an off-rule trade."


def close_trade(journal, i, exit_sek, day):
    journal.at[i, "exit_price"] = f"{exit_sek}"
    journal.at[i, "closed_on"] = day.isoformat()
    return record_price(journal, i)


def you_vs_model(signal_log, journal):
    """Model's paper result next to yours, per group; the model uses the rule's leverage, you your own product."""
    v2 = signal_log[signal_log.rule == "v2"]
    took = linked(v2, journal[journal.rule == "v2"])
    model_done = (v2.result_turbo_pct != "").to_numpy()
    yours = journal[journal.result_pct != ""]
    groups = {
        "Model: every v2 signal": (v2[model_done], "result_turbo_pct", "result_ko"),
        "v2 signals you took, model's result": (v2[model_done & took], "result_turbo_pct", "result_ko"),
        "v2 signals you took, your result": (yours[yours.rule == "v2"], "result_pct", "knocked_out"),
        "v2 signals you skipped, model's result": (v2[model_done & ~took], "result_turbo_pct", "result_ko"),
        "Your off-rule trades": (yours[yours.rule == "off-rule"], "result_pct", "knocked_out"),
    }
    rows = []
    for name, (g, pnl_col, ko_col) in groups.items():
        pnl = pd.to_numeric(g[pnl_col], errors="coerce")
        rows.append({"group": name, "trades": len(g), "win_pct": (pnl > 0).mean() * 100 if len(g) else np.nan,
                     "avg_turbo_pct": pnl.mean(), "knocked_out_pct": (g[ko_col] == "yes").mean() * 100 if len(g) else np.nan})
    on_signal = journal[journal.rule == "v2"].merge(v2[["signal_date", "ticker", "close", "leverage"]],
                                                     on=["signal_date", "ticker"], suffixes=("", "_rule"))
    fill = pd.to_numeric(on_signal.fill_underlying, errors="coerce")
    execution = {"linked_trades": len(on_signal),
                 "entry_vs_signal_close_pct": ((fill / pd.to_numeric(on_signal.close) - 1) * 100).mean(),
                 "your_leverage": pd.to_numeric(on_signal.leverage, errors="coerce").mean(),
                 "rule_leverage": pd.to_numeric(on_signal.leverage_rule, errors="coerce").mean()}
    return pd.DataFrame(rows), execution


def scorecard(signal_log, journal=None):
    """Settled v2 signals. With your journal: only the signals you took, at your own result where you closed one."""
    done = signal_log[(signal_log.rule == "v2") & (signal_log.result_stock_pct != "")]
    turbo, knocked = pd.to_numeric(done.result_turbo_pct), done.result_ko == "yes"
    if journal is not None:
        mine = journal[(journal.rule == "v2") & (journal.result_pct != "")].assign(
            yours=lambda j: pd.to_numeric(j.result_pct), yours_ko=lambda j: j.knocked_out == "yes")
        mine = mine.groupby(["signal_date", "ticker"], as_index=False).agg(yours=("yours", "mean"), yours_ko=("yours_ko", "any"))
        done = done[linked(done, journal[journal.rule == "v2"])].merge(mine, on=["signal_date", "ticker"], how="left")
        turbo = done.yours.fillna(pd.to_numeric(done.result_turbo_pct))
        knocked = done.yours_ko.fillna(done.result_ko == "yes").astype(bool)
    stock = pd.to_numeric(done.result_stock_pct)
    card = {"trades": len(done), "win_pct": (stock > 0).mean() * 100 if len(done) else np.nan,
            "avg_stock_pct": stock.mean(), "max_ko_pct": knocked.mean() * 100 if len(done) else np.nan,
            "avg_turbo_pct": turbo.mean()}
    card["passed"] = {k: (card[k] >= v if k != "max_ko_pct" else card[k] <= v) if not pd.isna(card[k]) else None
                      for k, v in PAPER_CRITERIA.items()}
    return card


def scorecard_line(card, label):
    c = PAPER_CRITERIA
    def fmt(v, spec):
        return "—" if pd.isna(v) else format(v, spec)
    return (f"{label}: {card['trades']}/{c['trades']} settled · win {fmt(card['win_pct'], '.0f')}% (≥{c['win_pct']:.0f}) · "
            f"avg stock {fmt(card['avg_stock_pct'], '+.2f')}% (≥{c['avg_stock_pct']}) · knocked out "
            f"{fmt(card['max_ko_pct'], '.0f')}% (≤{c['max_ko_pct']:.0f}) · avg turbo {fmt(card['avg_turbo_pct'], '+.1f')}%")


def record_signals(signal_log, signals):
    known = set(zip(signal_log.signal_date, signal_log.ticker, strict=False))
    new = [{
        "signal_date": s["date"].date().isoformat(), "ticker": s["ticker"], "close": f"{s['close']:.2f}",
        "rsi": f"{s['rsi']:.1f}", "market": s["market"], "leverage": s["leverage"],
        "financing_level": f"{s['financing']:.2f}", "est_ko_risk_%": s["ko_risk"],
        "max_fin_10%_risk": f"{s['max_fin_10']:.2f}", "max_fin_20%_risk": f"{s['max_fin_20']:.2f}",
        "sell_on": s["sell_on"].isoformat(), "rule": "v2",
        "earnings_in_hold": s.get("earnings", ""),
    } for s in signals if (s["date"].date().isoformat(), s["ticker"]) not in known]
    if new:
        signal_log = pd.concat([signal_log, pd.DataFrame(new)], ignore_index=True).reindex(columns=SIGNAL_COLUMNS).fillna("")
    return signal_log


def new_provisionals(signals, today_iso, now):
    logged = load_csv(PROVISIONAL, ["date", "ticker", "rsi", "logged_at"])
    seen = set(logged[logged.date == today_iso].ticker)
    fresh = [s for s in signals if s["ticker"] not in seen]
    rows = pd.DataFrame([{"date": today_iso, "ticker": s["ticker"], "rsi": f"{s['rsi']:.1f}",
                          "logged_at": f"{now:%H:%M}"} for s in fresh])
    if len(rows):
        pd.concat([logged, rows], ignore_index=True).to_csv(PROVISIONAL, index=False)
    return fresh


def unconfirmed_provisionals(signals, last_close_iso):
    logged = load_csv(PROVISIONAL, ["date", "ticker", "rsi", "logged_at"])
    confirmed = {s["ticker"] for s in signals}
    return [r for r in logged[logged.date == last_close_iso].itertuples() if r.ticker not in confirmed]


def near_knockout(journal, rows):
    """Open positions whose live price is within one typical day's move of their knock-out level."""
    by_ticker = {r["ticker"]: r for r in rows}
    warnings = {}
    for pos in journal[open_trades(journal)].itertuples():
        live = by_ticker.get(pos.ticker)
        level = pd.to_numeric(pos.ko_level or pos.product_financing, errors="coerce")
        if live is None or pd.isna(level) or level <= 0:
            continue
        gap = live["close"] / level - 1
        if gap <= live["vol"]:
            warnings[(pos.ticker, level)] = {"ticker": pos.ticker, "price": live["close"], "level": level, "gap": gap,
                                             "source": pos.product or "your product"}
    return list(warnings.values())


def new_ko_warnings(warnings, today_iso):
    sent = load_csv(KO_ALERTS, ["date", "ticker"])
    seen = set(sent[sent.date == today_iso].ticker)
    fresh = [w for w in warnings if w["ticker"] not in seen]
    if fresh:
        rows = pd.DataFrame([{"date": today_iso, "ticker": w["ticker"]} for w in fresh])
        pd.concat([sent, rows], ignore_index=True).to_csv(KO_ALERTS, index=False)
    return fresh


def new_exit_alerts(sells, today_iso):
    sent = load_csv(KO_ALERTS, ["date", "ticker"])
    seen = set(sent[sent.date == today_iso].ticker)
    fresh = sells[~("exit:" + sells.ticker).isin(seen)]
    if len(fresh):
        rows = pd.DataFrame({"date": today_iso, "ticker": "exit:" + fresh.ticker})
        pd.concat([sent, rows], ignore_index=True).to_csv(KO_ALERTS, index=False)
    return fresh


def notify(title, message):
    def escape(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", f'display notification "{escape(message)}" with title "{escape(title)}"'],
                   check=False)


def day_shape(s):
    """Where the signal day closed in its range. Bottom third was the stronger setup in the portfolio test
    (Sharpe +0.03 to +0.10 in both periods); top third meant it had already bounced and did worst."""
    where = s["close_in_range"]
    part = "bottom third" if where < 1 / 3 else "middle third" if where < 2 / 3 else "top third (already bounced — weaker)"
    return f"day {s['day_move']:+.1%}, closed in the {part} of its range"


def signal_lines(signals):
    lines = []
    for s in signals:
        lines += [f"  {s['ticker']:10s} price {s['close']:.2f}  RSI {s['rsi']:.1f}  market {s['market']} · {day_shape(s)}",
                  f"             {s['leverage']}x → financing ≈ {s['financing']:.2f}, est. knock-out risk over "
                  f"{HOLD} days {s['ko_risk']}",
                  f"             highest financing at 20% risk (typical for 20x) {s['max_fin_20']:.2f} · at 10% risk "
                  f"{s['max_fin_10']:.2f}",
                  f"             sell at the first close with RSI {EXIT_RSI}+, at the latest {s['sell_on']} · stake {STAKE_GUIDE}"
                  + (f" · EARNINGS IN THE HOLD {s['earnings'][4:]}" if s.get("earnings", "").startswith("yes") else "")]
    return lines


def ko_lines(warnings):
    return [f"  {w['ticker']:10s} price {w['price']:.2f} is {w['gap']:.1%} above the knock-out level {w['level']:.2f} "
            f"({w['source']})" for w in warnings]


def morning_missed(now):
    """A weekday past the launch agent's 08:15 morning run with no morning report yet: the Mac slept or was off."""
    report = LIVE_DIR / "reports" / f"{now.date().isoformat()}_morning.txt"
    return now.weekday() < 5 and now.time() >= MORNING_RUN and not report.exists()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", default="auto", choices=["auto", *MODES],
                   help="auto picks morning before 12:00, se-preclose before 19:00, us-preclose after; auto and "
                        "intraday run the morning scan first if today's hasn't run")
    p.add_argument("--no-notify", action="store_true")
    args = p.parse_args()
    LIVE_DIR.mkdir(exist_ok=True)
    (LIVE_DIR / "reports").mkdir(exist_ok=True)
    with LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        now = pd.Timestamp.now()
        mode = auto_mode(now) if args.mode == "auto" else args.mode
        if args.mode in ("auto", "intraday") and mode != "morning" and morning_missed(now):
            run("morning", now, args.no_notify)
        run(mode, now, args.no_notify)


def run(mode, now, quiet):
    settings = MODES[mode]
    markets = open_markets(now) if mode == "intraday" else settings["markets"]
    if not markets:
        return
    today = pd.Timestamp(date.today())
    today_iso = today.date().isoformat()

    signal_log = load_csv(SIGNALS, SIGNAL_COLUMNS)
    journal = load_csv(JOURNAL, JOURNAL_COLUMNS)
    pool = full_universe() if mode == "morning" else focus_list(signal_log, journal, today_iso)
    tickers = [t for t in pool if market_of(t) in markets]
    data = recent_daily(tickers + ["SPY", "^VIX", "^IRX"])
    spy_vs_200d, vix, rate, _ = market_state(data, today)
    rows, skipped = [], []
    for ticker in tickers:
        result = analyse(ticker, data[ticker], spy_vs_200d, vix, rate, today, settings["provisional"]) \
            if ticker in data else {"ticker": ticker, "skip": "no data"}
        (skipped if result["skip"] else rows).append(result)

    signals = [r for r in rows if r["signal"]]
    if mode != "intraday":
        for sig in signals:
            sig["earnings"] = earnings_in_hold(sig["ticker"], sig["date"], sig["sell_on"])
    below_trend_dips = [r for r in rows if r["dip"] and not r["above_trend"]]
    watch = sorted((r for r in rows if not r["signal"] and r["above_trend"] and r["rsi"] < WATCH_BELOW),
                   key=lambda r: r["rsi"])[:WATCH_SHOWN]
    if mode == "morning":
        pd.DataFrame({"ticker": [r["ticker"] for r in rows if r["above_trend"] and r["rsi"] < CANDIDATE_BELOW]}) \
            .to_csv(CANDIDATES, index=False)
    missed, fresh = [], []
    if settings["provisional"]:
        fresh = new_provisionals(signals, today_iso, now)
    else:
        signal_log = settle(record_signals(signal_log, signals), data, today, rate)
        signal_log.to_csv(SIGNALS, index=False)
        journal = settle_trades(unlink_revoked(journal, signal_log), data)
        journal.to_csv(JOURNAL, index=False)
        last_close = max((r["date"] for r in rows), default=today).date().isoformat()
        missed = unconfirmed_provisionals(signals, last_close)
    warnings = near_knockout(journal, rows) if settings["provisional"] else []
    fresh_warnings = new_ko_warnings(warnings, today_iso)

    in_scope = signal_log.ticker.map(market_of).isin(markets)
    sells = exits_due(signal_log, journal, rows, today_iso, markets)
    just_closed = signal_log[(signal_log.sold_on != "") & (signal_log.exit_reason != f"day {HOLD}") & in_scope
                             & (signal_log.sold_on >= (today - pd.offsets.BDay(1)).date().isoformat())] \
        if mode == "morning" else signal_log.iloc[:0]
    held = journal[open_trades(journal)]
    spy_now, vix_now = spy_vs_200d.iloc[-1], vix.iloc[-1]
    bull_now = spy_now > 0 and vix_now < HIGH_VIX

    if mode == "intraday":
        lines = [f"{now:%Y-%m-%d %H:%M} open: {', '.join(sorted(markets))}; "
                 f"below 30 now: {', '.join(s['ticker'] for s in signals) or 'none'}"]
        lines += [f"  new: {s['ticker']} RSI {s['rsi']:.1f} {s['leverage']}x, est. KO risk {s['ko_risk']}" for s in fresh]
        lines += ko_lines(fresh_warnings)
        fresh_exits = new_exit_alerts(sells, today_iso)
        lines += [f"  sell: {r.ticker} {r.why}" for r in fresh_exits.itertuples()]
        with INTRADAY_LOG.open("a") as log:
            log.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        parts = [f"LIKELY BUY {s['ticker']} {s['leverage']}x (RSI {s['rsi']:.1f})" for s in fresh]
        parts += [f"{w['ticker']} {w['gap']:.1%} above knock-out" for w in fresh_warnings]
        parts += [f"SELL {r.ticker} ({r.why})" for r in fresh_exits.itertuples()]
        if parts and not quiet:
            notify("RSI turbo: heads-up", " · ".join(parts))
        return

    lines = [f"RSI<30 turbo scan — {today_iso} {now:%H:%M} — {settings['label']} — {len(rows)} stocks checked",
             f"Market: SPY {spy_now:+.1%} vs 200-day, VIX {vix_now:.1f} ({'bull' if bull_now else 'caution'}; "
             f"the next session's signals use {BULL_LEVERAGE if bull_now else OTHER_LEVERAGE}x)",
             f"Open positions in your journal: {len(held)} of {MAX_POSITIONS}",
             scorecard_line(scorecard(signal_log), "Paper trading, every v2 signal"),
             scorecard_line(scorecard(signal_log, journal), "Signals you took"), ""]
    if signals:
        lines.append("LIKELY BUY — RSI below 30 on the live price, above the 200-day; confirm near the close:"
                     if settings["provisional"] else "BUY today — RSI(14) crossed below 30 at the last close, above the 200-day:")
        lines += signal_lines(signals)
    else:
        lines.append("No buy signals.")
    if below_trend_dips:
        lines += ["", f"No signal, below the 200-day ({len(below_trend_dips)}): "
                  + ", ".join(r["ticker"] for r in below_trend_dips[:20])]
    if missed:
        lines += ["", "Didn't hold at the close (RSI finished at 30 or above) — treat as no signal:"]
        lines += [f"  {r.ticker:10s} was {r.rsi} at {r.logged_at}" for r in missed]
    if len(sells):
        lines += ["", "SELL before the close:"]
        lines += [f"  {r.ticker:10s} signal {r.signal_date} · {r.why}" + ("  (yours)" if r.yours else "")
                  for r in sells.itertuples()]
    if len(just_closed):
        lines += ["", "Rule exit at the last close (sell at the open if you still hold):"]
        lines += [f"  {r.ticker:10s} {r.exit_reason} on {r.sold_on}, stock {r.result_stock_pct}%, turbo est. {r.result_turbo_pct}%"
                  for r in just_closed.itertuples()]
    if warnings:
        lines += ["", "Close to knock-out:"] + ko_lines(warnings)
    if watch:
        lines += ["", f"Watch — above the 200-day, RSI under {WATCH_BELOW}: " + ", ".join(f"{r['ticker']} {r['rsi']:.1f}" for r in watch)]
    if skipped:
        lines += ["", "Skipped: " + ", ".join(f"{s['ticker']} ({s['skip']})" for s in skipped)]
    report = "\n".join(lines)
    (LIVE_DIR / "reports" / f"{today_iso}_{mode}.txt").write_text(report + "\n")
    print(report)

    if not quiet:
        verb = "LIKELY BUY" if settings["provisional"] else "BUY"
        parts = [f"{verb} {s['ticker']} {s['leverage']}x" for s in signals] + [f"SELL {r.ticker}" for r in sells.itertuples()]
        parts += [f"{w['ticker']} near knock-out" for w in warnings]
        title = f"RSI turbo ({mode}): {len(signals)} buy, {len(sells)} sell" if parts else f"RSI turbo ({mode}): nothing"
        notify(title, " · ".join(parts) if parts else f"Market {'bull' if bull_now else 'caution'}; watching {len(watch)}")


if __name__ == "__main__":
    main()
