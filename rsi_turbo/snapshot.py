"""Live state shared by the local dashboard and the GitHub page: prices, signals and open positions."""
from datetime import date

import numpy as np
import pandas as pd

import scan

RISK_LINE = 20


def trigger_price(close_history):
    """Close that would take daily RSI(14) to 30 from the last completed close; None if RSI is already below 30."""
    n = 14
    change = close_history.diff()
    def smooth(s):
        return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    gain, loss = smooth(change.clip(lower=0)).iloc[-1], smooth((-change).clip(lower=0)).iloc[-1]
    last_rsi = 100 - 100 / (1 + gain / loss) if loss else 100.0
    if last_rsi < scan.OVERSOLD:
        return None
    ratio = scan.OVERSOLD / (100 - scan.OVERSOLD)
    return close_history.iloc[-1] - (n - 1) * (gain / ratio - loss)


def status_of(row):
    if row["confirmed"]:
        return "BUY — closed below 30"
    if row["live"] and row["signal"]:
        return "likely buy (live)"
    if not row["above_trend"]:
        return "below 200-day, no signal" if row["rsi"] < scan.WATCH_BELOW else ""
    if row["rsi"] < scan.OVERSOLD:
        return "below 30"
    return "watch" if row["rsi"] < scan.WATCH_BELOW else ""


def snapshot(data, tickers):
    today = pd.Timestamp(date.today())
    open_now = scan.open_markets(pd.Timestamp.now())
    spy_vs_200d, vix, rate, _ = scan.market_state(data, today)
    accrual = 1 + (rate + scan.FINANCING_MARGIN) * scan.HOLD * 7 / 5 / 365
    rows = []
    for ticker in tickers:
        d = data.get(ticker)
        if d is None or len(d) < scan.MIN_HISTORY:
            continue
        exchange = scan.market_of(ticker)
        bar_today = d.index[-1] == today
        in_session = bar_today and exchange in open_now
        last_close = scan.analyse(ticker, d, spy_vs_200d, vix, rate, today, provisional=bar_today and not in_session)
        row = scan.analyse(ticker, d, spy_vs_200d, vix, rate, today, provisional=True) if in_session else last_close
        if row["skip"] or last_close["skip"]:
            continue
        completed = d.close[d.index < today] if in_session else d.close
        trigger = trigger_price(completed)
        signal_ref = last_close if last_close["signal"] else row
        row = dict(row, exchange=exchange, live=in_session, confirmed=bool(last_close["signal"]), signal_ref=signal_ref,
                   change=row["close"] / d.close.iloc[-2] - 1, trigger=trigger,
                   to_trigger=None if trigger is None else trigger / row["close"] - 1,
                   max_fin_20=scan.max_financing(row["close"], row["vol"], RISK_LINE, accrual))
        if row["confirmed"] and in_session:
            row.update({k: last_close[k] for k in ("leverage", "sell_on")})
        row["status"] = status_of(row)
        rows.append(row)
    return pd.DataFrame(rows), spy_vs_200d, vix, rate


def positions(snap, signal_log, journal):
    """Open trades valued at the latest price. Turbo P&L is estimated from the underlying and ignores the small
    daily financing charge."""
    held = journal[scan.open_trades(journal)].copy()
    prices = snap.set_index("ticker").close if len(snap) else pd.Series(dtype=float)

    def num(col):
        return pd.to_numeric(held[col].where(held[col] != ""), errors="coerce")
    entry, financing = num("fill_underlying"), num("product_financing")
    held["price"] = held.ticker.map(prices)
    held["financing"] = financing
    held["ko"] = num("ko_level").fillna(financing)
    held["lev_entry"] = entry / (entry - financing)
    held["lev_now"] = held.price / (held.price - financing)
    held["gap"] = held.price / held.ko - 1
    held["turbo_pnl"] = np.where(held.price <= held.ko, -1.0, (held.price - financing) / (entry - financing) - 1)
    by_signal = {(r.signal_date, r.ticker): r for r in signal_log[signal_log.rule == "v2"].itertuples()}
    exits = [rule_exit(by_signal.get((r.signal_date, r.ticker))) for r in held.itertuples()]
    held["sell_by"], held["rule_exited"] = [e[0] for e in exits], [e[1] for e in exits]
    return held


def rule_exit(signal):
    """(text, whether the rule already exited) for a trade's linked signal."""
    if signal is None:
        return "your call", False
    if signal.result_stock_pct:
        return f"rule exited {signal.sold_on}", True
    return f"{signal.sell_on} ({np.busday_count(date.today(), pd.Timestamp(signal.sell_on).date())}d)", False
