"""Local dashboard for the RSI<30 turbo strategy: market state, what to do today, positions, watchlist, chart, journal.

The local.orb-backtest.dashboard launch agent keeps it at http://localhost:8501. Prices refresh every 5 minutes;
signals flagged "live" use today's unfinished bar and only count if they hold at the close.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import figures
import param_health
import scan
import snapshot
from data import recent_daily
from indicators import rsi

HERE = Path(__file__).parent
REFRESH = "5min"
CHART_BARS = 180
EDITABLE = ["product", "product_financing", "ko_level", "qty", "parity", "fill_price", "exit_price", "closed_on", "notes"]
COLORS = {"up": "#4fb3a9", "down": "#e06c6c", "trigger": "#56c7e8", "ko": "#e06c6c", "safe": "#e0a44f",
          "sma": "#8a9aa0", "rsi": "#b39ddb", "grid": "#26323a"}

st.set_page_config(page_title="RSI Turbo", page_icon=":material/trending_down:", layout="wide")


@st.cache_data(ttl=240, show_spinner="Fetching prices…")
def fetch(tickers):
    return recent_daily(list(tickers) + ["SPY", "^VIX", "^IRX"])


def load_journal():
    return scan.load_csv(scan.JOURNAL, scan.JOURNAL_COLUMNS)


def load_signals():
    return scan.load_csv(scan.SIGNALS, scan.SIGNAL_COLUMNS)


def write_journal(journal):
    """Save the journal and push it to the GitHub page in the background; failures go to live/publish.log."""
    journal.to_csv(scan.JOURNAL, index=False)
    with (scan.LIVE_DIR / "publish.log").open("a") as log:
        subprocess.Popen([sys.executable, str(HERE / "publish.py")], cwd=HERE, stdout=log, stderr=log)


def save_journal(edited):
    current = load_journal()
    current.loc[edited.index, EDITABLE] = edited[EDITABLE].fillna("").astype(str)
    write_journal(scan.settle_trades(current, {}))


def chart(ticker, d, row):
    d = d.iloc[-(CHART_BARS + 200):]
    r = rsi(d.close)
    sma200 = d.close.rolling(200).mean()
    crosses = (r < scan.OVERSOLD) & (r.shift(1) >= scan.OVERSOLD)
    d, r, sma200, crosses = (x.iloc[-CHART_BARS:] for x in (d, r, sma200, crosses))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name=ticker,
                                 increasing_line_color=COLORS["up"], decreasing_line_color=COLORS["down"],
                                 increasing_fillcolor=COLORS["up"], decreasing_fillcolor=COLORS["down"]), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=sma200, name="200-day", line=dict(color=COLORS["sma"], width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index[crosses], y=d.low[crosses] * 0.985, mode="markers", name="RSI crossed below 30",
                             marker=dict(symbol="triangle-up", size=11, color=COLORS["up"])), 1, 1)

    levels = [("financing at " + f"{row['leverage']}x", row["financing"], COLORS["ko"], "dot"),
              (f"highest financing, {snapshot.RISK_LINE}% risk", row["max_fin_20"], COLORS["safe"], "solid")]
    if row["trigger"] is not None:
        levels.insert(0, ("RSI 30 trigger", row["trigger"], COLORS["trigger"], "dash"))
    for text, level, color, dash in levels:
        fig.add_hline(y=level, line=dict(color=color, width=1.4, dash=dash), row=1, col=1,
                      annotation_text=f"{text}  {level:,.2f}", annotation_position="top left",
                      annotation_font=dict(color=color, size=11))

    fig.add_trace(go.Scatter(x=r.index, y=r, name="RSI(14)", line=dict(color=COLORS["rsi"], width=1.5)), 2, 1)
    for level in (30, 70):
        fig.add_hline(y=level, line=dict(color=COLORS["sma"], width=1, dash="dot"), row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e1e7e6"),
                      xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(gridcolor=COLORS["grid"], rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


def market_banner(spy_vs_200d, vix, rate):
    spy_now, vix_now = spy_vs_200d.iloc[-1], vix.iloc[-1]
    bull = spy_now > 0 and vix_now < scan.HIGH_VIX
    lev = scan.BULL_LEVERAGE if bull else scan.OTHER_LEVERAGE
    open_now = sorted(scan.open_markets(pd.Timestamp.now())) or ["none"]
    cols = st.columns(5)
    cols[0].metric("Market", "BULL" if bull else "CAUTION", f"{lev}x on new signals", delta_color="off")
    cols[1].metric("SPY vs 200-day", f"{spy_now:+.1%}")
    cols[2].metric("VIX", f"{vix_now:.1f}", f"bull below {scan.HIGH_VIX}", delta_color="off")
    cols[3].metric("Financing cost", f"{(rate + scan.FINANCING_MARGIN) * 100:.1f}%/yr", "T-bill + 2.5%",
                   delta_color="off")
    cols[4].metric("Markets open", " · ".join(open_now), f"updated {pd.Timestamp.now():%H:%M}", delta_color="off")


def save_and_refresh(journal, message):
    write_journal(journal)
    st.toast(message)
    st.rerun(scope="app")


def take_button(row, key, container):
    with container.popover("Bought", use_container_width=True), st.form(f"take_{key}", clear_on_submit=True):
        st.markdown(f"**{row['ticker']}** · rule: {row['leverage']}x, financing ≈ {row['financing']:.2f}")
        product = st.text_input("Product name", key=f"take_product_{key}")
        c1, c2 = st.columns(2)
        financing = c1.number_input("Financing level", value=float(round(row["financing"], 4)), format="%.4f",
                                    key=f"take_fin_{key}")
        stop = c2.number_input("Stop-loss (mini long, else 0)", value=0.0, format="%.4f", key=f"take_stop_{key}")
        c3, c4, c5 = st.columns(3)
        fill = c3.number_input("Price paid (SEK)", min_value=0.0, format="%.4f", key=f"take_fill_{key}")
        qty = c4.number_input("Quantity", min_value=0, step=1, key=f"take_qty_{key}")
        parity = c5.number_input("Parity", min_value=0.0, value=10.0, key=f"take_parity_{key}")
        if st.form_submit_button("Log this trade", type="primary"):
            if financing <= 0 or fill <= 0 or qty <= 0 or parity <= 0:
                st.error("Fill in the financing level, price paid, quantity and parity.")
                return
            signal_log = scan.record_signals(load_signals(), [row["signal_ref"]])
            signal_log.to_csv(scan.SIGNALS, index=False)
            journal, message = scan.log_trade(signal_log, load_journal(), row["ticker"], date.today(), product.strip(),
                                              financing, stop, fill, int(qty), parity, usd_sek())
            save_and_refresh(journal, message)


def close_button(i, row, container):
    with container.popover("Close", use_container_width=True), st.form(f"close_{i}", clear_on_submit=True):
        st.markdown(f"**{row['ticker']}** · {row['product'] or 'no product'} · bought {row['bought_on']}")
        exit_sek = st.number_input("Price sold (SEK)", min_value=0.0, format="%.4f", key=f"close_price_{i}")
        day = st.date_input("Date sold", value=date.today(), key=f"close_day_{i}")
        if st.form_submit_button("Close trade", type="primary"):
            journal = scan.close_trade(load_journal(), i, exit_sek, day)
            save_and_refresh(journal, f"Closed {row['ticker']}")


def rows_with_action(df, columns, action):
    """A table drawn row by row, so each row can end in its own button. columns: (header, width, row -> text)."""
    widths = [w for _, w, _ in columns] + [1.1]
    for cell, (header, _, _) in zip(st.columns(widths), columns, strict=False):
        cell.caption(header)
    for i, row in df.iterrows():
        cells = st.columns(widths, vertical_alignment="center")
        for cell, (_, _, fmt) in zip(cells, columns, strict=False):
            cell.markdown(fmt(row))
        action(i, row, cells[-1])


def signed(value, spec, colour=True):
    if pd.isna(value):
        return "—"
    text = format(value, spec)
    return f":{'green' if value >= 0 else 'red'}[{text}]" if colour else text


def signal_day_text(sig):
    where = sig["close_in_range"]
    if pd.isna(where):
        return "—"
    colour, part = ("green", "low") if where < 1 / 3 else ("orange", "middle") if where < 2 / 3 else ("red", "high")
    return f":{colour}[closed near {part}] · {sig['day_move']:+.1%}"


def act_today(snap, signal_log, journal):
    today_iso = date.today().isoformat()
    buys = snap[snap.confirmed | (snap.live & snap.signal)].sort_values("rsi") if len(snap) else snap
    sells = scan.exits_due(signal_log, journal, snap.to_dict("records") if len(snap) else [], today_iso, {"SE", "US"})
    st.subheader(f"Buy ({len(buys)})")
    if buys.empty:
        st.caption("No stock crossed below RSI 30. Check the watchlist for stocks close to their trigger.")
    else:
        rows_with_action(buys, [
            ("Stock", 1, lambda r: f"**{r['ticker']}**"),
            ("Signal", 1.8, lambda r: "closed below 30" if r["confirmed"] else ":orange[live — confirm at close]"),
            ("Price", 1, lambda r: f"{r['close']:,.2f}"),
            ("RSI", 0.7, lambda r: f"{r['rsi']:.1f}"),
            ("Signal day", 1.6, lambda r: signal_day_text(r["signal_ref"])),
            ("Leverage", 0.8, lambda r: f"{r['leverage']}x"),
            ("Financing at that leverage", 1.4, lambda r: f"{r['financing']:,.2f}"),
            ("Knock-out risk, 5 days", 1.2, lambda r: r["ko_risk"]),
            (f"Highest financing, {snapshot.RISK_LINE}% risk", 1.4, lambda r: f"{r['max_fin_20']:,.2f}"),
            ("Sell by", 1.1, lambda r: str(r["sell_on"])),
        ], lambda i, row, cell: take_button(row, f"{row['ticker']}_{i}", cell))
        st.caption(f"Stake {scan.STAKE_GUIDE}, max {scan.MAX_POSITIONS} open. \"Closed below 30\" crossed at the last "
                   "finished close: buy now. \"Live\" uses today's unfinished bar: buy near the close only if RSI is "
                   f"still below 30. Sell at the first close with RSI back at {scan.EXIT_RSI}, at the latest on the "
                   "sell-by day. Signal day: closing near the low was the stronger setup in the backtest; closing near "
                   "the high (already bounced) did worst.")
    st.subheader(f"Sell today ({len(sells)})")
    if sells.empty:
        st.caption(f"Nothing has RSI back at {scan.EXIT_RSI} or reaches its last day today.")
    else:
        st.dataframe(sells.assign(yours=sells.yours.map({True: "yes", False: "—"})), hide_index=True, width="stretch",
                     column_config={"ticker": "Stock", "signal_date": "Signal", "why": "Reason", "yours": "You hold it",
                                    "product": "Product"})
        st.caption("Sell before the close. Close your positions with their Close button under Open positions.")


def positions(snap, signal_log, journal):
    held = snapshot.positions(snap, signal_log, journal)
    st.subheader(f"Open positions ({len(held)} of {scan.MAX_POSITIONS})")
    if held.empty:
        st.caption("Use the Bought button on a buy signal, or \"Log a trade\" below, when you buy.")
        return
    rows_with_action(held, [
        ("Stock", 0.9, lambda r: f"**{r['ticker']}**"),
        ("Rule", 1, lambda r: ":orange[off-rule]" if r["rule"] == "off-rule" else r["rule"]),
        ("Product", 2, lambda r: r["product"] or "—"),
        ("Price", 1, lambda r: f"{r['price']:,.2f}"),
        ("Knock-out", 1, lambda r: f"{r['ko']:,.2f}"),
        ("Above KO", 0.9, lambda r: f":{'red' if r['gap'] < 0.02 else 'orange' if r['gap'] < 0.04 else 'green'}[{r['gap']:.1%}]"),
        ("Leverage entry → now", 1.3, lambda r: f"{r['lev_entry']:.1f}x → {r['lev_now']:.1f}x"),
        ("Turbo P&L (est.)", 1.1, lambda r: signed(r["turbo_pnl"], "+.1%")),
        ("Sell by", 1.1, lambda r: f":red[{r['sell_by']}]" if r["rule_exited"] else r["sell_by"]),
    ], close_button)
    st.caption("Every trade stays open until you close it, or the morning scan marks it knocked out. Off-rule trades "
               "have no rule exit. Leverage = price ÷ (price − financing level). Turbo P&L is estimated from the "
               "underlying and ignores the small daily financing charge; your broker's quote is the real value.")


def leverage_calculator(snap):
    with st.sidebar:
        st.divider()
        st.subheader("Leverage calculator")
        tickers = snap.sort_values("rsi").ticker.tolist() if len(snap) else []
        if not tickers:
            return
        ticker = st.selectbox("Stock", tickers, key="calc_ticker")
        row = snap.set_index("ticker").loc[ticker]
        price = st.number_input("Underlying price", value=float(row["close"]), format="%.2f", key="calc_price")
        financing = st.number_input("Product financing level", value=float(row["financing"]), format="%.2f",
                                    key="calc_fin")
        stop = st.number_input("Stop-loss level (mini long; 0 for a turbo)", value=0.0, format="%.2f", key="calc_stop")
        ko = stop if stop > 0 else financing
        if not 0 < financing < price or ko >= price:
            st.warning("The financing and stop-loss levels must be below the price for a long product.")
            return
        accrual = 1 + 0.065 * scan.HOLD * 7 / 5 / 365
        st.metric("Leverage", f"{price / (price - financing):.1f}x")
        st.metric("Distance to knock-out", f"{price / ko - 1:.1%}")
        st.metric(f"Est. knock-out risk over {scan.HOLD} days", scan.ko_risk_pct(ko * accrual / price, row["vol"]),
                  f"{row['vol']:.1%} daily volatility", delta_color="off")
        st.caption(f"Rule: {row['leverage']}x now ({row['market']} market). Highest financing at {snapshot.RISK_LINE}% risk: "
                   f"{row['max_fin_20']:.2f}.")


def scorecard_row(card, title):
    crit = scan.PAPER_CRITERIA
    st.markdown(f"**{title}**")
    cols = st.columns(5)
    def fmt(v, spec):
        return "—" if pd.isna(v) else format(v, spec)
    def verdict(ok):
        return "—" if ok is None else ("meets the bar" if ok else "below the bar")
    cols[0].metric("Settled trades", f"{card['trades']} / {crit['trades']}",
                   "enough to judge" if card["trades"] >= crit["trades"] else "keep going", delta_color="off")
    cols[1].metric("Win rate", fmt(card["win_pct"], ".0f") + "%", f"{verdict(card['passed']['win_pct'])} (≥{crit['win_pct']:.0f}%)",
                   delta_color="off")
    cols[2].metric("Avg stock move", fmt(card["avg_stock_pct"], "+.2f") + "%",
                   f"{verdict(card['passed']['avg_stock_pct'])} (≥+{crit['avg_stock_pct']}%)", delta_color="off")
    cols[3].metric("Knocked out", fmt(card["max_ko_pct"], ".0f") + "%",
                   f"{verdict(card['passed']['max_ko_pct'])} (≤{crit['max_ko_pct']:.0f}%)", delta_color="off")
    cols[4].metric("Avg turbo (est.)", fmt(card["avg_turbo_pct"], "+.1f") + "%", "at the recorded leverage", delta_color="off")


def paper_trading(signal_log, journal):
    st.subheader("Paper trading")
    st.caption("Decided in advance: after 40 settled v2 trades, go live small only if every bar is met. Results fill in "
               "automatically the morning after each sell day. \"Every v2 signal\" treats all signals as taken.")
    scorecard_row(scan.scorecard(signal_log), "Every v2 signal")
    scorecard_row(scan.scorecard(signal_log, journal), "Signals you took")


def watchlist_table(snap):
    st.subheader("Watchlist")
    st.caption("Your watchlist, open positions, and the stocks from the S&P 500 / OMX Stockholm 30 that the morning "
               "scan found above their 200-day with RSI under 40.")
    view = snap.sort_values("rsi")[["ticker", "exchange", "status", "close", "change", "rsi", "above_trend", "trigger",
                                    "to_trigger", "leverage", "ko_risk", "max_fin_20"]]
    st.dataframe(view, hide_index=True, width="stretch", height=420,
                 column_config={
                     "ticker": "Stock", "exchange": "Market", "status": "Status",
                     "close": st.column_config.NumberColumn("Price", format="%.2f"),
                     "change": st.column_config.NumberColumn("Today", format="percent"),
                     "rsi": st.column_config.ProgressColumn("RSI(14)", min_value=0, max_value=100, format="%.1f"),
                     "above_trend": st.column_config.CheckboxColumn("Above 200-day"),
                     "trigger": st.column_config.NumberColumn("Close below for a signal", format="%.2f"),
                     "to_trigger": st.column_config.NumberColumn("Distance", format="percent"),
                     "leverage": "Leverage", "ko_risk": "Est. KO risk at that leverage",
                     "max_fin_20": st.column_config.NumberColumn(f"Highest financing, {snapshot.RISK_LINE}% risk", format="%.2f"),
                 })


@st.fragment(run_every=REFRESH)
def live_view():
    signal_log, journal = load_signals(), load_journal()
    tickers = tuple(scan.focus_list(signal_log, journal, date.today().isoformat()))
    data = fetch(tickers)
    snap, spy_vs_200d, vix, rate = snapshot.snapshot(data, tickers)
    market_banner(spy_vs_200d, vix, rate)
    st.divider()
    act_today(snap, signal_log, journal)
    positions(snap, signal_log, journal)
    st.divider()
    paper_trading(signal_log, journal)
    st.divider()
    watchlist_table(snap)
    leverage_calculator(snap)
    st.subheader("Chart")
    options = snap.sort_values("rsi").ticker.tolist()
    ticker = st.selectbox("Stock", options, index=0, label_visibility="collapsed")
    if ticker:
        st.plotly_chart(chart(ticker, data[ticker], snap.set_index("ticker").loc[ticker]), width="stretch",
                        config={"displayModeBar": False})
        st.caption("Blue dashed: close below this and daily RSI goes under 30. Red dotted: financing level at the "
                   f"leverage the rule picks. Orange: highest financing at {snapshot.RISK_LINE}% knock-out risk over 5 days.")


@st.cache_data(ttl=300, show_spinner=False)
def usd_sek():
    return float(recent_daily(["SEK=X"], period="5d")["SEK=X"].close.iloc[-1])


def your_trades():
    st.subheader("Your trades vs the model")
    journal = load_journal()
    left, right = st.columns(2)
    with left, st.form("log_trade", clear_on_submit=True):
        st.markdown("**Log a trade you made**")
        known = sorted(set(scan.load_watchlist()) | set(journal.ticker))
        ticker = st.selectbox("Stock (Yahoo ticker)", known, index=None, accept_new_options=True,
                              placeholder="e.g. LLY or EQT.ST", key="log_ticker")
        day = st.date_input("Date bought", value=date.today(), key="log_day")
        product = st.text_input("Product name", placeholder="e.g. B LONGLLY AF SG", key="log_product")
        c1, c2 = st.columns(2)
        financing = c1.number_input("Financing level", min_value=0.0, format="%.4f", key="log_fin")
        stop = c2.number_input("Stop-loss level (mini long, else 0)", min_value=0.0, format="%.4f", key="log_stop")
        c3, c4, c5 = st.columns(3)
        fill = c3.number_input("Price paid (SEK)", min_value=0.0, format="%.4f", key="log_fill")
        qty = c4.number_input("Quantity", min_value=0, step=1, key="log_qty")
        parity = c5.number_input("Parity", min_value=0.0, value=10.0, key="log_parity")
        if st.form_submit_button("Log trade", type="primary"):
            if not ticker or financing <= 0 or fill <= 0 or qty <= 0 or parity <= 0:
                st.error("Fill in the stock, financing level, price paid, quantity and parity.")
            else:
                updated, message = scan.log_trade(load_signals(), journal, ticker.strip().upper(), day, product.strip(),
                                                  financing, stop, fill, int(qty), parity, usd_sek())
                write_journal(updated)
                st.success(message)
    with right:
        open_taken = journal[scan.open_trades(journal) | (journal.exit_price == "sold")]
        with st.form("close_trade", clear_on_submit=True):
            st.markdown("**Close a trade**")
            labels = {f"{r['ticker']} · {r['product'] or 'no product'} · bought {r['bought_on']}"
                      + (" · sold, price missing" if r["exit_price"] == "sold" else ""): i for i, r in open_taken.iterrows()}
            choice = st.selectbox("Open trade", list(labels), index=None, key="close_choice")
            exit_sek = st.number_input("Price sold (SEK)", min_value=0.0, format="%.4f", key="close_price")
            closed_day = st.date_input("Date sold", value=date.today(), key="close_day")
            if st.form_submit_button("Close trade"):
                if choice is None:
                    st.error("Pick a trade.")
                else:
                    write_journal(scan.close_trade(journal, labels[choice], exit_sek, closed_day))
                    st.success(f"Closed {choice.split(' · ')[0]}.")
        st.caption("A knocked-out product is closed automatically by the morning scan; log it here at 0 SEK if you "
                   "want the price recorded.")
    table, execution = scan.you_vs_model(load_signals(), load_journal())
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"group": "Group", "trades": "Trades",
                                "win_pct": st.column_config.NumberColumn("Win rate", format="%.0f%%"),
                                "avg_turbo_pct": st.column_config.NumberColumn("Avg turbo return", format="%+.1f%%"),
                                "knocked_out_pct": st.column_config.NumberColumn("Knocked out", format="%.0f%%")})
    if execution["linked_trades"]:
        st.caption(f"Execution on {execution['linked_trades']} signal trades: you bought on average "
                   f"{execution['entry_vs_signal_close_pct']:+.2f}% from the signal close, at {execution['your_leverage']:.1f}x "
                   f"vs the rule's {execution['rule_leverage']:.1f}x.")
    else:
        st.caption("Once you log trades on v2 signals, this shows how your entries and leverage differ from the rule. "
                   "The model's result uses the rule's leverage; yours uses your product and prices.")


def monthly_results():
    st.subheader("Your results by month")
    fig, totals = figures.monthly_results(load_journal())
    missing = (f" {totals['unpriced']} sold trades have no sale price yet; close them under \"Close a trade\" to count "
               "them.") if totals["unpriced"] else ""
    if fig is None:
        st.caption("Closed trades show up here by the month they closed." + missing)
        return
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"{totals['trades']} closed trades, {totals['pnl']:+,.0f} SEK, {totals['pnl'] / totals['staked']:+.1%} on "
               f"{totals['staked']:,.0f} SEK staked, before courtage. A trade counts in the month it closed; knocked-out "
               "products count as a full loss (a mini long as what was left above its financing level)." + missing)


def parameter_health():
    st.subheader("Parameter health")
    if not param_health.OUTPUT.exists():
        st.caption("Not computed yet.")
    else:
        table = pd.read_csv(param_health.OUTPUT)
        st.dataframe(param_health.verdicts(table), hide_index=True, width="stretch",
                     column_order=["part", "status", "last_3y_edge", "last_3y_t", "full_edge", "full_t", "measure"],
                     column_config={"part": "Part of the rule", "status": "Last 3 years",
                                    "last_3y_edge": st.column_config.NumberColumn("Edge, last 3 years", format="%+.2f"),
                                    "last_3y_t": st.column_config.NumberColumn("t", format="%.1f"),
                                    "full_edge": st.column_config.NumberColumn("Edge since 2007", format="%+.2f"),
                                    "full_t": st.column_config.NumberColumn("t ", format="%.1f"),
                                    "measure": "Edge measured as"})
        st.plotly_chart(figures.health_heatmap(table), width="stretch", config={"displayModeBar": False})
        st.plotly_chart(figures.health_trend(table), width="stretch", config={"displayModeBar": False})
        st.caption(f"Each part of the rule is traded with and without it on every RSI dip since 2007. Heatmap: that "
                   f"year's evidence (blue helped, red hurt). Lines: {param_health.WINDOW_YEARS}-year rolling edge, band "
                   "±2 standard errors (clustered by week). Strong = last 3 years positive with t ≥ 2; not helping = zero "
                   "or below. A part that stops helping is a reason to test the alternative on new data, not to swap in "
                   "the best-looking value. Updated "
                   f"{pd.Timestamp(param_health.OUTPUT.stat().st_mtime, unit='s'):%Y-%m-%d}.")
    if st.button("Recompute parameter health (about a minute)"):
        with st.spinner("Trading every RSI dip since 2007…"):
            out = subprocess.run([sys.executable, str(HERE / "param_health.py")], cwd=HERE, capture_output=True,
                                 text=True, timeout=900)
        if out.returncode:
            st.code(out.stderr[-2000:], language=None)
        else:
            st.rerun()


def journal_editor():
    st.subheader("Journal — your trades")
    journal = load_journal()
    if journal.empty:
        st.caption("Trades you log with the Bought button or \"Log a trade\" appear here.")
    else:
        locked = [c for c in scan.JOURNAL_COLUMNS if c not in EDITABLE]
        edited = st.data_editor(journal.iloc[::-1], hide_index=True, width="stretch", disabled=locked, key="journal",
                                column_config={"product_financing": st.column_config.TextColumn("Financing level"),
                                               "ko_level": st.column_config.TextColumn("Knock-out level"),
                                               "fill_price": st.column_config.TextColumn("Price paid (SEK)"),
                                               "exit_price": st.column_config.TextColumn("Price sold (SEK)"),
                                               "result_pct": st.column_config.TextColumn("Result %"),
                                               "signal_date": st.column_config.TextColumn("Linked signal")})
        if st.button("Save journal", type="primary"):
            save_journal(edited)
            st.toast("Journal saved")
    with st.expander("The model's signal log (every signal, paper-traded)"):
        st.dataframe(load_signals().iloc[::-1], hide_index=True, width="stretch")


def sidebar():
    with st.sidebar:
        st.header("RSI Turbo")
        st.markdown(f"**Rule v2** · buy when daily RSI(14) closes below {scan.OVERSOLD} **and the price is above its "
                    f"200-day** · {scan.BULL_LEVERAGE}x in a bull market (SPY above its 200-day, VIX below "
                    f"{scan.HIGH_VIX}), {scan.OTHER_LEVERAGE}x otherwise · sell at the first close with RSI back at "
                    f"{scan.EXIT_RSI}, at the latest after {scan.HOLD} trading days · stake "
                    f"{scan.STAKE_GUIDE}, max {scan.MAX_POSITIONS} open.")
        st.caption("Portfolio test on ~535 stocks (S&P 500 + OMX Stockholm 30), idle money in SPY, 20–30 × 1–1.5%: "
                   "20–22% a year 2007–26 vs 11% for SPY (max drawdown −57% to −60%); 23–24% since 2015 vs 14% "
                   "(−42% to −44%).")
        st.divider()
        mode = st.selectbox("Run a scan now", ["auto", "morning", "se-preclose", "us-preclose"])
        if st.button("Run scan"):
            with st.spinner("Scanning…"):
                out = subprocess.run([sys.executable, str(HERE / "scan.py"), "--mode", mode, "--no-notify"], cwd=HERE,
                                     capture_output=True, text=True, timeout=300)
            st.code(out.stdout or out.stderr, language=None)
            st.cache_data.clear()
        st.divider()
        new = st.text_input("Add to watchlist", placeholder="Yahoo ticker, e.g. EVO.ST or LLY").strip().upper()
        if st.button("Add") and new:
            if new in scan.load_watchlist():
                st.info(f"{new} is already on the watchlist.")
            elif recent_daily([new], period="3mo"):
                with scan.WATCHLIST.open("a") as f:
                    f.write(f"{new}\n")
                st.success(f"Added {new}.")
            else:
                st.error(f"Yahoo has no data for {new}. Stockholm tickers end in .ST, e.g. EQT.ST.")
        st.caption("To remove a stock, edit watchlist.txt.")


sidebar()
live_view()
st.divider()
your_trades()
st.divider()
monthly_results()
st.divider()
parameter_health()
st.divider()
journal_editor()
