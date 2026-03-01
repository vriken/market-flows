#!/usr/bin/env python3
"""
Market Flows — Weekly sector, commodity & index flow tracker.

Tracks:
  - CFTC Commitment of Traders positioning (commodities + financial futures)
  - Estimated ETF sector/index/commodity flows via AUM changes

Usage:
  python market_flows.py                          # Show all data
  python market_flows.py --update                 # Save AUM snapshot & show all
  python market_flows.py --cot-only               # COT data only
  python market_flows.py --etf-only               # ETF data only
  python market_flows.py --cot gold silver oil     # Only specific COT contracts
  python market_flows.py --etf SPY QQQ XLK XLE    # Only specific ETFs
  python market_flows.py --etf SPY --cot gold     # Mix and match
"""

import argparse
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import warnings

import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DATA_DIR = Path(__file__).parent / "data"


# ── Watchlists ────────────────────────────────────────────────────────────────

ETF_GROUPS = {
    "S&P 500 Sectors": {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLB": "Materials",
        "XLI": "Industrials",
        "XLY": "Consumer Disc",
        "XLP": "Consumer Staples",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication",
    },
    "Major Indices": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "IWM": "Russell 2000",
        "DIA": "Dow Jones",
        "EFA": "Intl Developed",
        "EEM": "Emerging Markets",
    },
    "Commodities & Crypto": {
        "GLD": "Gold",
        "SLV": "Silver",
        "CPER": "Copper",
        "PPLT": "Platinum",
        "USO": "WTI Crude",
        "BNO": "Brent Crude",
        "IBIT": "Bitcoin",
    },
    "Bonds": {
        "TLT": "20+ Yr Treasury",
        "IEF": "7-10 Yr Treasury",
        "HYG": "High Yield",
        "LQD": "Inv Grade Corp",
    },
    "Currency": {
        "UUP": "US Dollar",
    },
}

# COT contracts: (search_pattern, display_name, report_type)
#   report_type: "disagg" = commodities, "fin" = financial futures
COT_CONTRACTS = [
    # Commodities (disaggregated report — uses Managed Money)
    ("GOLD", "Gold", "disagg"),
    ("SILVER", "Silver", "disagg"),
    ("CRUDE OIL, LIGHT SWEET", "Crude Oil", "disagg"),
    ("NATURAL GAS", "Natural Gas", "disagg"),
    ("CORN", "Corn", "disagg"),
    ("SOYBEANS", "Soybeans", "disagg"),
    ("WHEAT-SRW", "Wheat", "disagg"),
    ("COPPER", "Copper", "disagg"),
    ("PLATINUM", "Platinum", "disagg"),
    # Financial futures (TFF report — uses Asset Managers / Leveraged Money)
    ("E-MINI S&P 500", "S&P 500", "fin"),
    ("NASDAQ MINI", "Nasdaq 100", "fin"),
    ("RUSSELL E-MINI", "Russell 2000", "fin"),
    ("EURO FX - C", "EUR/USD", "fin"),
    ("JAPANESE YEN", "JPY", "fin"),
    ("BRITISH POUND", "GBP", "fin"),
    ("UST BOND", "US T-Bonds", "fin"),
    ("UST 10Y NOTE", "10Y T-Notes", "fin"),
    ("VIX FUTURES", "VIX", "fin"),
    ("BITCOIN -", "Bitcoin", "fin"),
]


# ── COT Data ──────────────────────────────────────────────────────────────────

def _fetch_cftc_zip(report_type):
    """Download a CFTC annual zip and return a DataFrame with headers."""
    year = datetime.now().year
    if report_type == "disagg":
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
    else:
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    df.columns = df.columns.str.strip()

    date_col = next(
        (c for c in df.columns if "Report_Date" in c and "YYYY" in c), None
    )
    if date_col:
        df["date"] = pd.to_datetime(df[date_col])
    else:
        date_col = next(c for c in df.columns if "date" in c.lower())
        df["date"] = pd.to_datetime(df[date_col], format="mixed")

    return df


def fetch_cot(filter_contracts=None):
    """Fetch COT data from both CFTC reports and return summary rows."""
    contracts = COT_CONTRACTS
    if filter_contracts:
        lc = [f.lower() for f in filter_contracts]
        contracts = [
            c for c in contracts
            if any(term in c[1].lower() for term in lc)
        ]
        if not contracts:
            print(f"  No matching COT contracts. Available:")
            for _, name, _ in COT_CONTRACTS:
                print(f"    {name}")
            return []

    need_disagg = any(t == "disagg" for _, _, t in contracts)
    need_fin = any(t == "fin" for _, _, t in contracts)

    dfs = {}
    print("  Fetching COT data from CFTC...")
    if need_disagg:
        dfs["disagg"] = _fetch_cftc_zip("disagg")
    if need_fin:
        dfs["fin"] = _fetch_cftc_zip("fin")

    rows = []
    for pattern, label, rtype in contracts:
        df = dfs.get(rtype)
        if df is None:
            continue

        name_col = next(c for c in df.columns if "Market_and_Exchange" in c)
        oi_col = next(c for c in df.columns if "Open_Interest_All" in c)

        # Pick the right long/short columns based on report type
        if rtype == "disagg":
            long_col = next(
                c for c in df.columns
                if "M_Money" in c and "Long" in c and "All" in c
                and "Pct" not in c and "Change" not in c and "Spread" not in c
            )
            short_col = next(
                c for c in df.columns
                if "M_Money" in c and "Short" in c and "All" in c
                and "Pct" not in c and "Change" not in c and "Spread" not in c
            )
            trader_label = "Managed Money"
        else:
            # Financial futures: use Leveraged Money (hedge funds)
            long_col = next(
                c for c in df.columns
                if "Lev_Money" in c and "Long" in c and "All" in c
                and "Pct" not in c and "Change" not in c and "Spread" not in c
            )
            short_col = next(
                c for c in df.columns
                if "Lev_Money" in c and "Short" in c and "All" in c
                and "Pct" not in c and "Change" not in c and "Spread" not in c
            )
            trader_label = "Leveraged Money"

        mask = df[name_col].str.upper().str.contains(pattern, na=False)
        all_rows = df.loc[mask].sort_values("date", ascending=False)
        if all_rows.empty:
            continue

        # Compute net position for all weeks (for percentile)
        all_nets = (
            all_rows[long_col].astype(int) - all_rows[short_col].astype(int)
        )

        latest = all_rows.iloc[0]
        net = int(latest[long_col]) - int(latest[short_col])
        oi = int(latest[oi_col])
        net_pct = net / oi * 100 if oi else 0

        chg = None
        if len(all_rows) >= 2:
            prev = all_rows.iloc[1]
            prev_net = int(prev[long_col]) - int(prev[short_col])
            chg = net - prev_net

        # Percentile: where does current net sit in this year's range
        net_min = all_nets.min()
        net_max = all_nets.max()
        if net_max != net_min:
            percentile = (net - net_min) / (net_max - net_min) * 100
        else:
            percentile = 50.0

        rows.append({
            "contract": label,
            "net": net,
            "net_pct_oi": net_pct,
            "change": chg,
            "percentile": percentile,
            "weeks": len(all_nets),
            "date": latest["date"].strftime("%Y-%m-%d"),
            "trader_type": trader_label,
        })

    return rows


def print_cot(rows):
    if not rows:
        print("  No COT data found.")
        return

    print(f"  Report date: {rows[0]['date']}\n")

    # Group by commodities vs financials
    disagg = [r for r in rows if r["trader_type"] == "Managed Money"]
    fin = [r for r in rows if r["trader_type"] == "Leveraged Money"]

    for label, group in [("Commodities (Managed Money)", disagg),
                         ("Financials (Leveraged Money)", fin)]:
        if not group:
            continue
        print(f"  {label}")
        print(f"  {'Contract':<16} {'Net Pos':>11} {'% of OI':>8} {'Wk Chg':>10} {'YTD %ile':>8}  Signal")
        print(f"  {'─'*16} {'─'*11} {'─'*8} {'─'*10} {'─'*8}  {'─'*12}")

        for r in group:
            net_s = f"{r['net']:>+11,}"
            pct_s = f"{r['net_pct_oi']:>+7.1f}%"
            chg_s = f"{r['change']:>+10,}" if r["change"] is not None else f"{'—':>10}"
            ptile = f"{r['percentile']:>7.0f}%"

            # VIX is inverse: short VIX = bullish stocks, long VIX = bearish stocks
            inverse = r["contract"] == "VIX"

            sig = ""
            if r["percentile"] >= 90:
                sig = "!! Extreme low" if inverse else "!! Extreme high"
            elif r["percentile"] <= 10:
                sig = "!! Extreme high" if inverse else "!! Extreme low"
            elif r["change"] is not None:
                if r["change"] > 0 and r["net"] > 0:
                    sig = "▼ Bearish" if inverse else "▲ Bullish"
                elif r["change"] < 0 and r["net"] < 0:
                    sig = "▲ Bullish" if inverse else "▼ Bearish"
                elif r["change"] > 0:
                    sig = "↘ Reducing" if inverse else "↗ Covering"
                elif r["change"] < 0:
                    sig = "↗ Covering" if inverse else "↘ Reducing"

            print(f"  {r['contract']:<16} {net_s} {pct_s} {chg_s} {ptile}  {sig}")
        print()


# ── ETF Flows ─────────────────────────────────────────────────────────────────

def _load_history():
    p = DATA_DIR / "etf_aum.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_history(h):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "etf_aum.json").write_text(json.dumps(h, indent=2))


def fetch_etfs(filter_tickers=None, update=False):
    """Get current ETF data and estimate flows vs last saved snapshot.

    Flow formula:
        flow = current_aum - previous_aum * (1 + price_return)

    This separates actual inflows/outflows from market-driven AUM changes.
    Requires at least 2 snapshots (run with --update weekly).
    """
    hist = _load_history()
    today = datetime.now().strftime("%Y-%m-%d")

    # Build ticker dict: either filtered or full watchlist
    all_known = {t: n for grp in ETF_GROUPS.values() for t, n in grp.items()}
    if filter_tickers:
        tickers = {}
        for t in filter_tickers:
            t_upper = t.upper()
            tickers[t_upper] = all_known.get(t_upper, t_upper)
    else:
        tickers = all_known

    print(f"  Fetching data for {len(tickers)} ETFs...")

    results = []
    snapshot = {}

    for tick, name in tickers.items():
        try:
            tkr = yf.Ticker(tick)
            info = tkr.info
            aum = info.get("totalAssets")
            price = info.get("previousClose") or info.get("regularMarketPrice")

            # Weekly return
            prices = tkr.history(period="5d")
            wk_ret = None
            if len(prices) >= 2:
                wk_ret = prices["Close"].iloc[-1] / prices["Close"].iloc[0] - 1

            # Estimate flow vs previous snapshot
            flow = None
            prev = hist.get(tick)
            if prev and aum and prev.get("aum") and prev.get("price") and price:
                price_ret = price / prev["price"] - 1
                flow = aum - prev["aum"] * (1 + price_ret)

            if aum and price:
                snapshot[tick] = {"aum": aum, "price": price, "date": today}

            results.append({
                "ticker": tick,
                "name": name,
                "aum_b": aum / 1e9 if aum else None,
                "wk_ret": wk_ret,
                "flow_m": flow / 1e6 if flow else None,
            })
        except Exception:
            results.append({
                "ticker": tick, "name": name,
                "aum_b": None, "wk_ret": None, "flow_m": None,
            })

    if update:
        for t, s in snapshot.items():
            hist[t] = s
        _save_history(hist)
        print(f"  Saved AUM snapshot for {len(snapshot)} ETFs.\n")
    else:
        print()

    return results


def print_etfs(results, filter_tickers=None):
    has_flow = any(r["flow_m"] is not None for r in results)

    if filter_tickers:
        # Custom list — print as a single flat table
        print(f"\n  Custom ETFs")
        hdr = f"  {'Tick':<6} {'Name':<18} {'AUM ($B)':>9} {'Return':>8}"
        div = f"  {'─'*6} {'─'*18} {'─'*9} {'─'*8}"
        if has_flow:
            hdr += f" {'Est Flow ($M)':>14}"
            div += f" {'─'*14}"
        print(hdr)
        print(div)
        for r in sorted(results, key=lambda r: r["flow_m"] or 0, reverse=True):
            _print_etf_row(r, has_flow)
    else:
        # Default grouped view
        for group, tickers in ETF_GROUPS.items():
            grp = [r for r in results if r["ticker"] in tickers]
            grp.sort(key=lambda r: r["flow_m"] or 0, reverse=True)

            print(f"\n  {group}")
            hdr = f"  {'Tick':<6} {'Name':<18} {'AUM ($B)':>9} {'Return':>8}"
            div = f"  {'─'*6} {'─'*18} {'─'*9} {'─'*8}"
            if has_flow:
                hdr += f" {'Est Flow ($M)':>14}"
                div += f" {'─'*14}"
            print(hdr)
            print(div)
            for r in grp:
                _print_etf_row(r, has_flow)


def _print_etf_row(r, has_flow):
    aum_s = f"{r['aum_b']:>9.1f}" if r["aum_b"] else f"{'—':>9}"
    ret_s = (
        f"{r['wk_ret']*100:>+7.1f}%"
        if r["wk_ret"] is not None else f"{'—':>8}"
    )
    line = f"  {r['ticker']:<6} {r['name']:<18} {aum_s} {ret_s}"
    if has_flow:
        fl_s = (
            f"{r['flow_m']:>+13.0f}M"
            if r["flow_m"] is not None else f"{'—':>14}"
        )
        line += f" {fl_s}"
    print(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Weekly market-flow tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s                          Show all data
  %(prog)s --update                 Save AUM snapshot & show all
  %(prog)s --cot gold oil bitcoin   COT for specific contracts
  %(prog)s --etf SPY QQQ XLK       ETF data for specific tickers
  %(prog)s --etf AAPL MSFT         Works with any Yahoo Finance ticker
  %(prog)s --cot gold --etf GLD    Mix COT and ETF filters
""",
    )
    ap.add_argument("--update", action="store_true",
                    help="Save current AUM snapshot for flow calculations")
    ap.add_argument("--cot-only", action="store_true",
                    help="Only show COT data (all contracts)")
    ap.add_argument("--etf-only", action="store_true",
                    help="Only show ETF data (all tickers)")
    ap.add_argument("--cot", nargs="+", metavar="NAME",
                    help="Show specific COT contracts (e.g. gold oil bitcoin)")
    ap.add_argument("--etf", nargs="+", metavar="TICKER",
                    help="Show specific ETFs (e.g. SPY QQQ XLK)")
    args = ap.parse_args()

    # Determine what to show
    has_cot_filter = args.cot is not None
    has_etf_filter = args.etf is not None

    if has_cot_filter or has_etf_filter:
        show_cot = has_cot_filter
        show_etf = has_etf_filter
    else:
        show_cot = not args.etf_only
        show_etf = not args.cot_only

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║           WEEKLY MARKET FLOWS SUMMARY            ║")
    print(f"║           {datetime.now().strftime('%Y-%m-%d %H:%M'):^40} ║")
    print("╚══════════════════════════════════════════════════╝")

    if show_cot:
        print("\n━━━ COT Positioning ━━━\n")
        try:
            rows = fetch_cot(filter_contracts=args.cot)
            print_cot(rows)
        except Exception as e:
            print(f"  Error fetching COT data: {e}")

    etf_results = []
    if show_etf:
        print("\n━━━ ETF Flows ━━━\n")
        if not args.update and not has_etf_filter:
            print("  Tip: run with --update to save AUM data for flow estimates.\n")
        try:
            etf_results = fetch_etfs(
                filter_tickers=args.etf,
                update=args.update,
            )
            print_etfs(etf_results, filter_tickers=args.etf)
        except Exception as e:
            print(f"  Error fetching ETF data: {e}")

        if not any(r.get("flow_m") for r in etf_results):
            print("\n  Note: flow estimates need 2+ snapshots.")
            print("  Run with --update weekly to build history.")
            print("  For instant data: https://www.etfdb.com/etf/flow-tool/")

    print()


if __name__ == "__main__":
    main()
