"""Build the static RSI Turbo page for GitHub Pages: market state, today's signals, open positions, your results
against the model, and parameter health, from the signal log, journal and parameter health in live/. The GitHub
workflow copies those from the gh-pages branch, runs this with TZ=Europe/Stockholm, and publishes the result."""
import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

import figures
import param_health
import scan
import snapshot
from data import recent_daily

HERE = Path(__file__).parent
WATCH_ROWS = 25


def pct(value, spec="+.1%"):
    return "—" if pd.isna(value) else format(value, spec)


def number(value, spec=",.2f"):
    return "—" if pd.isna(value) else format(value, spec)


def tone(value):
    return "" if pd.isna(value) else ("pos" if value >= 0 else "neg")


def chart(fig, first):
    return fig.to_html(include_plotlyjs="cdn" if first else False, full_html=False,
                       config={"displayModeBar": False, "responsive": True})


def buys(snap):
    rows = snap[snap.confirmed | (snap.live & snap.signal)].sort_values("rsi") if len(snap) else snap
    return [{"ticker": r.ticker, "confirmed": r.confirmed, "price": number(r.close), "rsi": f"{r.rsi:.1f}",
             "day": "—" if pd.isna(r.signal_ref["close_in_range"]) else scan.day_shape(r.signal_ref), "leverage": f"{r.leverage}x", "financing": number(r.financing),
             "ko_risk": r.ko_risk, "max_fin": number(r.max_fin_20), "sell_by": str(r.sell_on)}
            for r in rows.itertuples()]


def open_positions(held):
    return [{"ticker": r.ticker, "rule": r.rule, "product": r.product or "—", "qty": r.qty, "bought": r.bought_on,
             "paid": r.fill_price, "price": number(r.price), "ko": number(r.ko), "gap": pct(r.gap),
             "gap_tone": "neg" if r.gap < 0.02 else "warn" if r.gap < 0.04 else "pos",
             "leverage": f"{r.lev_entry:.1f}x → {r.lev_now:.1f}x", "pnl": pct(r.turbo_pnl), "pnl_tone": tone(r.turbo_pnl),
             "sell_by": r.sell_by, "rule_exited": r.rule_exited} for r in held.itertuples()]


def watchlist(snap):
    rows = snap[snap.above_trend & (snap.rsi < scan.CANDIDATE_BELOW)].sort_values("rsi").head(WATCH_ROWS)
    return [{"ticker": r.ticker, "status": r.status or "", "price": number(r.close), "change": pct(r.change),
             "change_tone": tone(r.change), "rsi": f"{r.rsi:.1f}", "trigger": number(r.trigger),
             "distance": pct(r.to_trigger)} for r in rows.itertuples()]


def scorecards(signal_log, journal):
    def card(c, title):
        return {"title": title, "trades": c["trades"], "win": pct(c["win_pct"] / 100, ".0%"),
                "stock": pct(c["avg_stock_pct"] / 100, "+.2%"), "ko": pct(c["max_ko_pct"] / 100, ".0%"),
                "turbo": pct(c["avg_turbo_pct"] / 100)}
    return [card(scan.scorecard(signal_log), "Every v2 signal"), card(scan.scorecard(signal_log, journal), "Signals you took")]


def versus(signal_log, journal):
    table, execution = scan.you_vs_model(signal_log, journal)
    rows = [{"group": r.group, "trades": r.trades, "win": pct(r.win_pct / 100, ".0%"),
             "turbo": pct(r.avg_turbo_pct / 100), "turbo_tone": tone(r.avg_turbo_pct),
             "ko": pct(r.knocked_out_pct / 100, ".0%")} for r in table.itertuples()]
    return rows, execution


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=scan.LIVE_DIR / "page" / "index.html")
    args = p.parse_args()
    signal_log = scan.load_csv(scan.SIGNALS, scan.SIGNAL_COLUMNS)
    journal = scan.load_csv(scan.JOURNAL, scan.JOURNAL_COLUMNS)
    tickers = scan.focus_list(signal_log, journal, date.today().isoformat())
    data = recent_daily(tickers + ["SPY", "^VIX", "^IRX"])
    journal = scan.settle_trades(journal, data)
    snap, spy_vs_200d, vix, rate = snapshot.snapshot(data, tickers)
    spy_now, vix_now = spy_vs_200d.iloc[-1], vix.iloc[-1]
    bull = spy_now > 0 and vix_now < scan.HIGH_VIX
    records = snap.to_dict("records") if len(snap) else []
    sells = scan.exits_due(signal_log, journal, records, date.today().isoformat(), {"SE", "US"})
    held = snapshot.positions(snap, signal_log, journal)
    monthly, totals = figures.monthly_results(journal)
    health_path = scan.LIVE_DIR / "param_health.csv"
    health = pd.read_csv(health_path) if health_path.exists() else None
    versus_rows, execution = versus(signal_log, journal)

    figs = {"monthly": monthly}
    if health is not None:
        figs.update(heatmap=figures.health_heatmap(health), trend=figures.health_trend(health))
    shown = [(name, fig) for name, fig in figs.items() if fig is not None]
    charts = {name: chart(fig, first=i == 0) for i, (name, fig) in enumerate(shown)}

    env = Environment(loader=FileSystemLoader(HERE), autoescape=True)
    html = env.get_template("page.html.j2").render(
        updated=f"{pd.Timestamp.now():%Y-%m-%d %H:%M}",
        market={"bull": bull, "leverage": scan.BULL_LEVERAGE if bull else scan.OTHER_LEVERAGE, "spy": pct(spy_now),
                "vix": f"{vix_now:.1f}", "high_vix": scan.HIGH_VIX,
                "financing": f"{(rate + scan.FINANCING_MARGIN) * 100:.1f}%",
                "open": " · ".join(sorted(scan.open_markets(pd.Timestamp.now()))) or "none"},
        rule={"oversold": scan.OVERSOLD, "exit_rsi": scan.EXIT_RSI, "hold": scan.HOLD, "bull": scan.BULL_LEVERAGE,
              "other": scan.OTHER_LEVERAGE, "stake": scan.STAKE_GUIDE, "max_positions": scan.MAX_POSITIONS},
        buys=buys(snap), sells=sells.to_dict("records"), positions=open_positions(held), totals=totals, charts=charts,
        versus=versus_rows, execution=execution, scorecards=scorecards(signal_log, journal),
        health=param_health.verdicts(health).to_dict("records") if health is not None else [],
        health_updated=f"{pd.Timestamp(health_path.stat().st_mtime, unit='s'):%Y-%m-%d}" if health is not None else "",
        window=param_health.WINDOW_YEARS, watch=watchlist(snap), risk_line=snapshot.RISK_LINE,
        criteria=scan.PAPER_CRITERIA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"wrote {args.output} ({len(html) // 1024} kB)")


if __name__ == "__main__":
    main()
